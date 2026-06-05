"""Fused Q/K GemmaRMSNorm + NeoX RoPE + optional gate deinterleave.

Single Triton kernel replacing the 4-op sequence in
Qwen3_5AttentionDecoderLayer.forward_prepare_native:
  1. Q/Gate deinterleave (view + chunk)
  2. GemmaRMSNorm on Q (per-head)
  3. GemmaRMSNorm on K (per-head)
  4. NeoX RoPE on Q and K (partial or full rotation)

Two implementations controlled by SGLANG_OPT_FUSED_QK_RMSNORM_ROPE_GATE_V2:
  v1: all-fp32 path (slightly more precise but diverges from unfused path)
  v2: bf16 round-trip after RMSNorm before RoPE (matches unfused path exactly)
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# Persistent (graph-safe) output buffers. Per-call torch.empty here gets captured into
# the EAGLE draft cuda graphs; dynamic allocs alias across the ~35 captured graphs
# (shared mempool) -> graph-replay memory corruption -> IMA on sm_100 (GB200).
# Reusing a buffer per (shape,dtype,device) keeps capture allocation-free.
_QK_OUT_CACHE = {}
def _qk_buf(shape, dtype, device):
    k = (tuple(shape), dtype, device)
    b = _QK_OUT_CACHE.get(k)
    if b is None:
        b = torch.empty(shape, dtype=dtype, device=device)
        _QK_OUT_CACHE[k] = b
    return b


# ---------------------------------------------------------------------------
# v2 kernel — bf16 round-trip, full HEAD_BLOCK load, reload rotary for RoPE
# ---------------------------------------------------------------------------


@triton.jit
def _fused_qk_rmsnorm_rope_gate_v2_kernel(
    q_gate_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    gate_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    stride_qg_t,
    stride_k_t,
    stride_qo_t,
    stride_ko_t,
    stride_gate_t,
    stride_cos_t,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    HALF_ROTARY: tl.constexpr,
    EPS: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROT_HALF_BLOCK: tl.constexpr,
    HAS_PASS: tl.constexpr,
    HAS_GATE: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    is_k = head >= NUM_Q_HEADS
    local_head = tl.where(is_k, head - NUM_Q_HEADS, head)

    if is_k:
        in_base = k_ptr + token * stride_k_t + local_head * HEAD_DIM
        w_ptr = k_weight_ptr
        out_base = k_out_ptr + token * stride_ko_t + local_head * HEAD_DIM
    else:
        if HAS_GATE:
            in_base = q_gate_ptr + token * stride_qg_t + local_head * 2 * HEAD_DIM
        else:
            in_base = q_gate_ptr + token * stride_qg_t + local_head * HEAD_DIM
        w_ptr = q_weight_ptr
        out_base = q_out_ptr + token * stride_qo_t + local_head * HEAD_DIM

    # --- RMSNorm over the full head_dim ---
    head_offs = tl.arange(0, HEAD_BLOCK)
    head_mask = head_offs < HEAD_DIM
    x = tl.load(in_base + head_offs, mask=head_mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / HEAD_DIM
    inv_rms = tl.rsqrt(var + EPS)
    w = tl.load(w_ptr + head_offs, mask=head_mask, other=0.0).to(tl.float32)
    # bf16 round-trip: matches the unfused (norm → memory → RoPE) path exactly
    x_norm = (x * inv_rms * w).to(INPUT_DTYPE).to(tl.float32)

    # --- Pass-through tail [rotary_dim, head_dim): RMSNorm only ---
    if HAS_PASS:
        pass_mask = head_mask & (head_offs >= ROTARY_DIM)
        tl.store(out_base + head_offs, x_norm, mask=pass_mask)

    # --- Partial NeoX RoPE on first rotary_dim elements ---
    # Reload rotary halves on a smaller block (hits L1 cache, negligible cost)
    rot_offs = tl.arange(0, ROT_HALF_BLOCK)
    rot_mask = rot_offs < HALF_ROTARY
    x_r1 = tl.load(in_base + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    x_r2 = tl.load(in_base + HALF_ROTARY + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    w_r1 = tl.load(w_ptr + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    w_r2 = tl.load(w_ptr + HALF_ROTARY + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    x_r1 = (x_r1 * inv_rms * w_r1).to(INPUT_DTYPE).to(tl.float32)
    x_r2 = (x_r2 * inv_rms * w_r2).to(INPUT_DTYPE).to(tl.float32)

    pos = tl.load(positions_ptr + token).to(tl.int64)
    cache_off = pos * stride_cos_t
    cos = tl.load(
        cos_sin_cache_ptr + cache_off + rot_offs, mask=rot_mask, other=0.0
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + cache_off + HALF_ROTARY + rot_offs,
        mask=rot_mask,
        other=0.0,
    ).to(tl.float32)

    o1 = x_r1 * cos - x_r2 * sin
    o2 = x_r2 * cos + x_r1 * sin
    tl.store(out_base + rot_offs, o1, mask=rot_mask)
    tl.store(out_base + HALF_ROTARY + rot_offs, o2, mask=rot_mask)

    # --- Gate copy (Q heads only, verbatim) ---
    if HAS_GATE and not is_k:
        gate_in = in_base + HEAD_DIM
        gate_out = gate_out_ptr + token * stride_gate_t + local_head * HEAD_DIM
        g = tl.load(gate_in + head_offs, mask=head_mask, other=0.0)
        tl.store(gate_out + head_offs, g, mask=head_mask)


# ---------------------------------------------------------------------------
# v1 kernel — all-fp32 (original implementation, slightly more precise
# but diverges from unfused path due to no intermediate bf16 quantization)
# ---------------------------------------------------------------------------


@triton.jit
def _fused_qk_rmsnorm_rope_gate_v1_kernel(
    q_gate_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    gate_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    stride_qg_t,
    stride_k_t,
    stride_qo_t,
    stride_ko_t,
    stride_gate_t,
    stride_gate_h,
    stride_cos_t,
    eps,
    num_tokens,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    HALF_ROTARY: tl.constexpr,
    HAS_NOPE: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HAS_GATE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_t >= num_tokens:
        return

    rot_offs = tl.arange(0, HALF_ROTARY)

    pos = tl.load(positions_ptr + pid_t)
    cos_base = pos * stride_cos_t
    cos = tl.load(cos_sin_cache_ptr + cos_base + rot_offs).to(tl.float32)
    sin = tl.load(cos_sin_cache_ptr + cos_base + HALF_ROTARY + rot_offs).to(tl.float32)

    if pid_h < NUM_Q_HEADS:
        head_id = pid_h
        if HAS_GATE:
            q_base = pid_t * stride_qg_t + head_id * 2 * HEAD_DIM
        else:
            q_base = pid_t * stride_qg_t + head_id * HEAD_DIM

        q_r1 = tl.load(q_gate_ptr + q_base + rot_offs).to(tl.float32)
        q_r2 = tl.load(q_gate_ptr + q_base + HALF_ROTARY + rot_offs).to(tl.float32)
        var_acc = tl.sum(q_r1 * q_r1) + tl.sum(q_r2 * q_r2)

        if HAS_NOPE:
            nope_offs = tl.arange(0, NOPE_BLOCK)
            nope_mask = nope_offs < NOPE_DIM
            q_nope = tl.load(
                q_gate_ptr + q_base + ROTARY_DIM + nope_offs,
                mask=nope_mask,
                other=0.0,
            ).to(tl.float32)
            var_acc += tl.sum(q_nope * q_nope)

        if HAS_GATE:
            gate_base = q_base + HEAD_DIM
            g_r1 = tl.load(q_gate_ptr + gate_base + rot_offs)
            g_r2 = tl.load(q_gate_ptr + gate_base + HALF_ROTARY + rot_offs)
            gate_out_base = pid_t * stride_gate_t + head_id * stride_gate_h
            tl.store(gate_out_ptr + gate_out_base + rot_offs, g_r1)
            tl.store(gate_out_ptr + gate_out_base + HALF_ROTARY + rot_offs, g_r2)
            if HAS_NOPE:
                g_nope = tl.load(
                    q_gate_ptr + gate_base + ROTARY_DIM + nope_offs,
                    mask=nope_mask,
                )
                tl.store(
                    gate_out_ptr + gate_out_base + ROTARY_DIM + nope_offs,
                    g_nope,
                    mask=nope_mask,
                )

        q_var = var_acc / HEAD_DIM
        q_rstd = tl.math.rsqrt(q_var + eps)
        w_r1 = tl.load(q_weight_ptr + rot_offs).to(tl.float32) + 1.0
        w_r2 = tl.load(q_weight_ptr + HALF_ROTARY + rot_offs).to(tl.float32) + 1.0
        q_r1_n = q_r1 * q_rstd * w_r1
        q_r2_n = q_r2 * q_rstd * w_r2

        o1 = q_r1_n * cos - q_r2_n * sin
        o2 = q_r2_n * cos + q_r1_n * sin

        q_out_base = pid_t * stride_qo_t + head_id * HEAD_DIM
        tl.store(q_out_ptr + q_out_base + rot_offs, o1.to(q_out_ptr.dtype.element_ty))
        tl.store(
            q_out_ptr + q_out_base + HALF_ROTARY + rot_offs,
            o2.to(q_out_ptr.dtype.element_ty),
        )

        if HAS_NOPE:
            w_nope = (
                tl.load(
                    q_weight_ptr + ROTARY_DIM + nope_offs,
                    mask=nope_mask,
                    other=0.0,
                ).to(tl.float32)
                + 1.0
            )
            q_nope_n = q_nope * q_rstd * w_nope
            tl.store(
                q_out_ptr + q_out_base + ROTARY_DIM + nope_offs,
                q_nope_n.to(q_out_ptr.dtype.element_ty),
                mask=nope_mask,
            )
    else:
        kv_head_id = pid_h - NUM_Q_HEADS
        k_base = pid_t * stride_k_t + kv_head_id * HEAD_DIM

        k_r1 = tl.load(k_ptr + k_base + rot_offs).to(tl.float32)
        k_r2 = tl.load(k_ptr + k_base + HALF_ROTARY + rot_offs).to(tl.float32)
        var_acc = tl.sum(k_r1 * k_r1) + tl.sum(k_r2 * k_r2)

        if HAS_NOPE:
            nope_offs = tl.arange(0, NOPE_BLOCK)
            nope_mask = nope_offs < NOPE_DIM
            k_nope = tl.load(
                k_ptr + k_base + ROTARY_DIM + nope_offs,
                mask=nope_mask,
                other=0.0,
            ).to(tl.float32)
            var_acc += tl.sum(k_nope * k_nope)

        k_var = var_acc / HEAD_DIM
        k_rstd = tl.math.rsqrt(k_var + eps)
        w_r1 = tl.load(k_weight_ptr + rot_offs).to(tl.float32) + 1.0
        w_r2 = tl.load(k_weight_ptr + HALF_ROTARY + rot_offs).to(tl.float32) + 1.0
        k_r1_n = k_r1 * k_rstd * w_r1
        k_r2_n = k_r2 * k_rstd * w_r2

        o1 = k_r1_n * cos - k_r2_n * sin
        o2 = k_r2_n * cos + k_r1_n * sin

        k_out_base = pid_t * stride_ko_t + kv_head_id * HEAD_DIM
        tl.store(k_out_ptr + k_out_base + rot_offs, o1.to(k_out_ptr.dtype.element_ty))
        tl.store(
            k_out_ptr + k_out_base + HALF_ROTARY + rot_offs,
            o2.to(k_out_ptr.dtype.element_ty),
        )

        if HAS_NOPE:
            w_nope = (
                tl.load(
                    k_weight_ptr + ROTARY_DIM + nope_offs,
                    mask=nope_mask,
                    other=0.0,
                ).to(tl.float32)
                + 1.0
            )
            k_nope_n = k_nope * k_rstd * w_nope
            tl.store(
                k_out_ptr + k_out_base + ROTARY_DIM + nope_offs,
                k_nope_n.to(k_out_ptr.dtype.element_ty),
                mask=nope_mask,
            )


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def fused_qk_gemma_rmsnorm_rope_gate_v2(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: Optional[int] = None,
    has_gate: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """v2: bf16 round-trip after RMSNorm, matches unfused path precision.

    NOTE: q_weight / k_weight must be gemma_weight (already +1.0).
    """
    if rotary_dim is None:
        rotary_dim = head_dim

    T = q_gate.shape[0]
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim

    q_out = _qk_buf((T, q_size), q_gate.dtype, q_gate.device)
    k_out = _qk_buf((T, kv_size), k.dtype, k.device)

    if has_gate:
        gate_out = _qk_buf((T, num_q_heads, head_dim), q_gate.dtype, q_gate.device)
    else:
        gate_out = q_out  # dummy

    half_rotary = rotary_dim // 2
    head_block = triton.next_power_of_2(head_dim)
    rot_half_block = triton.next_power_of_2(half_rotary)
    num_warps = max(1, head_block // 64)

    grid = (T, num_q_heads + num_kv_heads)
    _fused_qk_rmsnorm_rope_gate_v2_kernel[grid](
        q_gate,
        k,
        q_out,
        k_out,
        gate_out,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        q_gate.stride(0),
        k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        gate_out.stride(0),
        cos_sin_cache.stride(0),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,
        HALF_ROTARY=half_rotary,
        EPS=eps,
        INPUT_DTYPE=tl.bfloat16 if q_gate.dtype == torch.bfloat16 else tl.float16,
        HEAD_BLOCK=head_block,
        ROT_HALF_BLOCK=rot_half_block,
        HAS_PASS=rotary_dim < head_dim,
        HAS_GATE=has_gate,
        num_warps=num_warps,
        num_stages=2,
    )

    return q_out, k_out, gate_out if has_gate else None


def fused_qk_gemma_rmsnorm_rope_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: Optional[int] = None,
    has_gate: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """v1: all-fp32 path. q_weight/k_weight are raw (kernel adds +1.0)."""
    if rotary_dim is None:
        rotary_dim = head_dim

    T = q_gate.shape[0]
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    HALF_ROTARY = rotary_dim // 2
    NOPE_DIM = head_dim - rotary_dim
    HAS_NOPE = NOPE_DIM > 0
    NOPE_BLOCK = triton.next_power_of_2(NOPE_DIM) if NOPE_DIM > 0 else 1

    q_out = _qk_buf((T, q_size), q_gate.dtype, q_gate.device)
    k_out = _qk_buf((T, kv_size), k.dtype, k.device)

    if has_gate:
        gate_out = _qk_buf((T, num_q_heads, head_dim), q_gate.dtype, q_gate.device)
    else:
        gate_out = q_out

    num_warps = 4 if head_dim <= 128 else 8

    grid = (T, num_q_heads + num_kv_heads)
    _fused_qk_rmsnorm_rope_gate_v1_kernel[grid](
        q_gate,
        k,
        q_out,
        k_out,
        gate_out,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        q_gate.stride(0),
        k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        gate_out.stride(0),
        gate_out.stride(1) if has_gate else 0,
        cos_sin_cache.stride(0),
        eps,
        T,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,
        HALF_ROTARY=HALF_ROTARY,
        HAS_NOPE=HAS_NOPE,
        NOPE_BLOCK=NOPE_BLOCK,
        NOPE_DIM=NOPE_DIM,
        HAS_GATE=has_gate,
        num_warps=num_warps,
    )

    return q_out, k_out, gate_out if has_gate else None
