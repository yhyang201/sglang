# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TurboQuant decode attention kernel.

Replaces standard K buffer load with:
  1. Load packed codebook indices (uint8)
  2. Unpack to per-coordinate centroid indices
  3. Codebook lookup → reconstructed unit key in rotated space
  4. Dot product with pre-rotated query, scaled by key norm

Optimization: ⟨q, ||k|| × Π^T × codebook[idx]⟩ = ||k|| × ⟨Π × q, codebook[idx]⟩
So we pre-rotate query ONCE outside the KV loop.
"""

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.triton_ops.decode_attention import (
    _fwd_kernel_stage2,
    tanh,
)


@triton.jit
def _turbo_quant_decode_stage1(
    Q_rotated,  # Pre-rotated query: [batch, q_heads, head_dim]
    K_Quant_Buffer,  # Packed indices: [total_slots, kv_heads, packed_dim]
    K_Norm_Buffer,  # Key norms: [total_slots, kv_heads]
    V_Buffer,  # Values: [total_slots, kv_heads, v_head_dim]
    Codebook,  # Lloyd-Max centroids: [2^bits]
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_kq_bs,  # k_quant_buffer stride dim0
    stride_kq_h,  # k_quant_buffer stride dim1
    stride_kn_bs,  # k_norm_buffer stride dim0
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    BITS: tl.constexpr,
    PACKED_DIM: tl.constexpr,
):
    """TurboQuant decode attention stage 1.

    For each query, iterates over KV cache blocks:
    1. Loads packed uint8 indices from K_Quant_Buffer
    2. Unpacks to per-coordinate 4-bit indices (for BITS=4)
    3. Looks up codebook centroids
    4. Computes dot product with pre-rotated query
    5. Scales by key norm and sm_scale
    6. Accumulates softmax(qk) × V
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q_rotated + off_q, mask=mask_d, other=0.0)

        # Offsets for packed indices (4-bit: 2 per byte)
        offs_packed = tl.arange(0, PACKED_DIM)

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )

            # ---- Load packed indices and unpack ----
            offs_kq = (
                kv_loc[:, None] * stride_kq_bs
                + cur_kv_head * stride_kq_h
                + offs_packed[None, :]
            )
            packed = tl.load(
                K_Quant_Buffer + offs_kq,
                mask=(offs_n[:, None] < split_kv_end)
                & (offs_packed[None, :] < PACKED_DIM),
                other=0,
            ).to(tl.uint8)

            # Unpack 4-bit: low nibble = even dims, high nibble = odd dims
            # For BITS=4: packed has shape [BLOCK_N, head_dim//2]
            # even_idx: indices for dims 0, 2, 4, ...
            # odd_idx: indices for dims 1, 3, 5, ...
            if BITS == 4:
                even_idx = packed & 0x0F  # [BLOCK_N, packed_dim]
                odd_idx = (packed >> 4) & 0x0F

                # Codebook lookup: interleave even/odd to get [BLOCK_N, head_dim]
                even_vals = tl.load(
                    Codebook + even_idx.to(tl.int32)
                )  # [BLOCK_N, packed_dim]
                odd_vals = tl.load(
                    Codebook + odd_idx.to(tl.int32)
                )  # [BLOCK_N, packed_dim]

                # We need to interleave even_vals and odd_vals to form k_rot
                # But Triton doesn't have great interleave support.
                # Instead: compute dot product separately for even and odd dims
                # q_even = q[0::2], q_odd = q[1::2]
                # dot = sum(q_even * even_vals) + sum(q_odd * odd_vals)
                offs_even = tl.arange(0, PACKED_DIM)
                q_even = tl.load(
                    Q_rotated
                    + cur_batch * stride_qbs
                    + cur_head * stride_qh
                    + offs_even * 2,
                    mask=offs_even * 2 < Lk,
                    other=0.0,
                )
                q_odd = tl.load(
                    Q_rotated
                    + cur_batch * stride_qbs
                    + cur_head * stride_qh
                    + offs_even * 2
                    + 1,
                    mask=offs_even * 2 + 1 < Lk,
                    other=0.0,
                )
                qk = tl.sum(q_even[None, :] * even_vals, 1) + tl.sum(
                    q_odd[None, :] * odd_vals, 1
                )
            else:
                # For other bit widths, fall back to full dequantization via Python
                # (handled outside kernel for now)
                qk = tl.zeros([BLOCK_N], dtype=tl.float32)

            # ---- Load key norms and scale ----
            offs_kn = kv_loc * stride_kn_bs + cur_kv_head
            k_norms = tl.load(
                K_Norm_Buffer + offs_kn,
                mask=offs_n < split_kv_end,
                other=0.0,
            ).to(tl.float32)

            qk = qk * k_norms * sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            # ---- Load V and accumulate ----
            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )
        tl.store(Att_Out + offs_mid_o, acc / e_sum, mask=mask_dv)

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv
        tl.store(Att_Lse + offs_mid_o_1, e_max + tl.log(e_sum))


def turbo_quant_decode_attention_fwd(
    q_rotated: torch.Tensor,
    k_quant_buffer: torch.Tensor,
    k_norm_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    codebook: torch.Tensor,
    o: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    num_kv_splits: torch.Tensor,
    max_kv_splits: int,
    sm_scale: float,
    logit_cap: float = 0.0,
    bits: int = 4,
):
    """Launch TurboQuant decode attention kernel.

    Args:
        q_rotated: Pre-rotated queries [batch, q_heads, head_dim] float16
        k_quant_buffer: Packed indices [total_slots, kv_heads, packed_dim] uint8
        k_norm_buffer: Key norms [total_slots, kv_heads] float16
        v_buffer: Values [total_slots, kv_heads, v_head_dim] float16/bf16
        codebook: Centroids [2^bits] float16
        o: Output [batch, q_heads, v_head_dim] float16/bf16
        kv_indptr: [batch+1] int32
        kv_indices: [total_kv_len] int32
        num_kv_splits: [batch] int32
        max_kv_splits: int
        sm_scale: attention scale factor (1/√d)
        logit_cap: logit capping value (0 = disabled)
        bits: quantization bits (currently only 4 supported in kernel)
    """
    batch, head_num = q_rotated.shape[0], q_rotated.shape[1]
    Lk = q_rotated.shape[-1]
    Lv = v_buffer.shape[-1]
    packed_dim = k_quant_buffer.shape[-1]

    BLOCK_N = 64
    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)
    PACKED_DIM = triton.next_power_of_2(packed_dim)

    kv_group_num = head_num // k_quant_buffer.shape[1]
    num_warps = 4 if kv_group_num == 1 else 2

    # Intermediate output for split-KV reduction
    att_out = torch.empty(
        (batch, head_num, max_kv_splits, Lv),
        dtype=torch.float32,
        device=q_rotated.device,
    )
    att_lse = torch.empty(
        (batch, head_num, max_kv_splits), dtype=torch.float32, device=q_rotated.device
    )

    grid = (batch, head_num, max_kv_splits)

    _turbo_quant_decode_stage1[grid](
        q_rotated,
        k_quant_buffer,
        k_norm_buffer,
        v_buffer,
        codebook,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q_rotated.stride(0),
        q_rotated.stride(1),
        k_quant_buffer.stride(0),
        k_quant_buffer.stride(1),
        k_norm_buffer.stride(0),
        v_buffer.stride(0),
        v_buffer.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N,
        MIN_BLOCK_KV=32,
        logit_cap=logit_cap,
        Lk=Lk,
        Lv=Lv,
        BITS=bits,
        PACKED_DIM=PACKED_DIM,
        num_warps=num_warps,
        num_stages=2,
    )

    # Stage 2: reduce across KV splits (reuse existing kernel)
    _fwd_kernel_stage2[(batch, head_num)](
        att_out,
        att_lse,
        o,
        num_kv_splits,
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        o.stride(0),
        o.stride(1),
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        MAX_KV_SPLITS=max_kv_splits,
    )
