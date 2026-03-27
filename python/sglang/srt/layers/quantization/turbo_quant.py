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
"""TurboQuant: KV cache compression via random rotation + Lloyd-Max quantization + QJL correction.

Reference: https://arxiv.org/abs/2504.19874 (ICLR 2026)

Algorithm:
  1. Rotate key vectors by a random orthogonal matrix Π (via QR decomposition)
  2. Quantize each coordinate independently using Lloyd-Max codebook for Beta distribution
  3. Store packed centroid indices + QJL sign bits + residual norm
  4. During attention: exploit orthogonal invariance ⟨q, Π^T·c⟩ = ⟨Π·q, c⟩
     to avoid per-token inverse rotation; add QJL correction via packed-bit ops
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch
from scipy import integrate, special

logger = logging.getLogger(__name__)


# ─── Lloyd-Max codebook computation for Beta distribution ───


def _beta_pdf(x: float, d: int) -> float:
    """PDF of a single coordinate after random orthogonal rotation of a unit vector in R^d.

    After rotation, each coordinate follows:
      f(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)
    on the interval [-1, 1].
    """
    if abs(x) >= 1.0:
        return 0.0
    alpha = (d - 1) / 2.0
    # Use log-gamma for numerical stability
    log_norm = (
        special.gammaln(d / 2.0) - 0.5 * math.log(math.pi) - special.gammaln(alpha)
    )
    if d > 3:
        log_val = log_norm + (alpha - 1) * math.log(max(1 - x * x, 1e-300))
    elif d == 3:
        log_val = log_norm  # (1-x²)^0 = 1
    else:
        # d=2: (1-x²)^(-0.5), handle carefully
        log_val = log_norm - 0.5 * math.log(max(1 - x * x, 1e-300))
    return math.exp(log_val)


def _compute_lloyd_max_codebook(d: int, bits: int, max_iter: int = 300) -> list[float]:
    """Compute optimal Lloyd-Max codebook for the Beta distribution induced by rotation in R^d.

    Args:
        d: dimension (head_dim)
        bits: number of quantization bits
        max_iter: maximum Lloyd-Max iterations

    Returns:
        List of 2^bits centroid values, sorted ascending.
    """
    n_levels = 1 << bits

    # The distribution is symmetric around 0, supported on [-1, 1].
    # The effective support shrinks as d grows (concentration).
    # Use std ≈ 1/√d as a guide for initial spread.
    std = 1.0 / math.sqrt(d)

    # Initialize centroids uniformly in [-3*std, 3*std]
    lo, hi = -3.0 * std, 3.0 * std
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    def pdf(x):
        return _beta_pdf(x, d)

    for _ in range(max_iter):
        # Compute boundaries (midpoints between adjacent centroids)
        boundaries = [-1.0]
        for i in range(n_levels - 1):
            boundaries.append((centroids[i] + centroids[i + 1]) / 2.0)
        boundaries.append(1.0)

        # Update centroids: c_i = E[X | b_{i-1} <= X < b_i]
        new_centroids = []
        for i in range(n_levels):
            a, b = boundaries[i], boundaries[i + 1]
            if b - a < 1e-15:
                new_centroids.append((a + b) / 2.0)
                continue
            num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
            den, _ = integrate.quad(pdf, a, b)
            if den < 1e-30:
                new_centroids.append((a + b) / 2.0)
            else:
                new_centroids.append(num / den)

        # Check convergence
        max_change = max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels))
        centroids = new_centroids
        if max_change < 1e-12:
            break

    return sorted(centroids)


# ─── Precomputed codebook cache (computed once per dimension/bits pair) ───

_codebook_cache: dict[tuple[int, int], torch.Tensor] = {}


def get_lloyd_max_codebook(
    head_dim: int, bits: int, device: torch.device = torch.device("cuda")
) -> torch.Tensor:
    """Get precomputed Lloyd-Max codebook, computing on first access.

    Returns:
        Tensor of shape [2^bits] with centroid values in float16.
    """
    key = (head_dim, bits)
    if key not in _codebook_cache:
        logger.info(
            f"Computing Lloyd-Max codebook for head_dim={head_dim}, bits={bits}..."
        )
        centroids = _compute_lloyd_max_codebook(head_dim, bits)
        _codebook_cache[key] = torch.tensor(
            centroids, dtype=torch.float16, device=device
        )
        logger.info(f"Codebook computed: {_codebook_cache[key].tolist()}")
    return _codebook_cache[key].to(device)


# ─── TurboQuant Configuration ───


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV cache compression."""

    bits: int = 3  # Quantization bits for keys (2, 3, or 4)
    seed: int = 42  # Random seed for rotation & QJL matrices
    use_qjl: bool = (
        False  # QJL residual correction (disabled by default; adds noise at d<=256)
    )
    head_dim: int = 128  # Will be set from model config

    # Precomputed tensors (initialized lazily)
    rotation_matrix: Optional[torch.Tensor] = None  # [head_dim, head_dim]
    qjl_matrix: Optional[torch.Tensor] = None  # [head_dim, head_dim]
    codebook: Optional[torch.Tensor] = None  # [2^bits]

    def initialize(self, head_dim: int, device: torch.device):
        """Initialize precomputed matrices and codebook for the given head dimension."""
        self.head_dim = head_dim

        # Generate rotation matrix via QR decomposition
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        gaussian = torch.randn(head_dim, head_dim, generator=gen, dtype=torch.float32)
        q, _ = torch.linalg.qr(gaussian)
        self.rotation_matrix = q.to(dtype=torch.float16, device=device)

        # Generate QJL projection matrix with entries ~ N(0, 1/d)
        # This normalization ensures the √(π/(2d)) scale factor is correct
        if self.use_qjl:
            gen_qjl = torch.Generator(device="cpu").manual_seed(self.seed + 1)
            self.qjl_matrix = (
                torch.randn(head_dim, head_dim, generator=gen_qjl, dtype=torch.float32)
                / math.sqrt(head_dim)
            ).to(dtype=torch.float16, device=device)

        # Get Lloyd-Max codebook
        self.codebook = get_lloyd_max_codebook(head_dim, self.bits, device)

        logger.info(
            f"TurboQuant initialized: head_dim={head_dim}, bits={self.bits}, "
            f"use_qjl={self.use_qjl}, codebook_size={len(self.codebook)}"
        )

    @property
    def packed_k_dim(self) -> int:
        """Number of uint8 bytes needed to store packed indices for one head vector.

        For 4-bit: head_dim / 2 (two indices per byte)
        For 3-bit: ceil(head_dim * 3 / 8)
        For 2-bit: head_dim / 4 (four indices per byte)
        """
        return math.ceil(self.head_dim * self.bits / 8)

    @property
    def qjl_dim(self) -> int:
        """Number of uint8 bytes for QJL sign bits (1 bit per dimension)."""
        return self.head_dim // 8

    @property
    def bytes_per_head_per_token(self) -> int:
        """Total bytes for quantized key per head per token."""
        total = self.packed_k_dim  # packed indices
        total += 2  # key norm (float16)
        if self.use_qjl:
            total += self.qjl_dim  # QJL sign bits
            total += 2  # residual norm (float16)
        return total


# ─── Encoding / Decoding functions ───


def turbo_quant_encode(
    keys: torch.Tensor,
    config: TurboQuantConfig,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Encode key vectors using TurboQuant.

    The algorithm normalizes keys to unit vectors before rotation, since the
    Lloyd-Max codebook is optimized for the Beta distribution on [-1, 1] that
    arises from rotating unit vectors.

    Args:
        keys: Key tensor of shape [num_tokens, num_kv_heads, head_dim] in float16/bf16
        config: TurboQuantConfig with precomputed matrices

    Returns:
        packed_indices: [num_tokens, num_kv_heads, packed_k_dim] uint8
        key_norms: [num_tokens, num_kv_heads] float16 — norms of original keys
        qjl_signs: [num_tokens, num_kv_heads, qjl_dim] uint8 (or None if no QJL)
        residual_norms: [num_tokens, num_kv_heads] float16 (or None if no QJL)
    """
    num_tokens, num_heads, head_dim = keys.shape
    assert config.rotation_matrix is not None, "TurboQuantConfig not initialized"
    assert config.codebook is not None

    # Cast to float32 for computation accuracy
    k = keys.float()

    # Step 1: Normalize to unit vectors
    key_norms = k.norm(dim=-1, keepdim=True)  # [T, H, 1]
    k_unit = k / key_norms.clamp(min=1e-8)  # [T, H, D]
    key_norms_out = key_norms.squeeze(-1).to(torch.float16)  # [T, H]

    # Step 2: Apply rotation — k_rot = k_unit @ Π^T
    Pi = config.rotation_matrix.float()  # [D, D]
    k_rot = torch.matmul(k_unit, Pi.t())  # [T, H, D] — each coord ~ Beta dist

    # Step 3: Lloyd-Max quantization per coordinate
    codebook = config.codebook.float()  # [2^bits]
    diffs = (k_rot.unsqueeze(-1) - codebook.view(1, 1, 1, -1)).abs()  # [T,H,D,C]
    indices = diffs.argmin(dim=-1)  # [T,H,D] — values in [0, 2^bits - 1]

    # Step 4: Pack indices into uint8
    packed_indices = _pack_indices(indices, config.bits)  # [T, H, packed_k_dim]

    # Step 5: QJL residual correction
    qjl_signs = None
    residual_norms = None
    if config.use_qjl and config.qjl_matrix is not None:
        # Compute residual in original (non-rotated, non-normalized) space
        k_rot_dequant = codebook[indices]  # [T, H, D] — quantized in rotated space
        k_unit_restored = torch.matmul(k_rot_dequant, Pi)  # [T, H, D]
        k_restored = k_unit_restored * key_norms  # scale back
        residual = k - k_restored  # [T, H, D]

        # Residual norm
        residual_norms = residual.norm(dim=-1).to(torch.float16)  # [T, H]

        # QJL sign bits: sign(S @ residual)
        S = config.qjl_matrix.float()  # [D, D]
        projected = torch.matmul(residual, S.t())  # [T, H, D]
        sign_bits = (projected >= 0).to(torch.uint8)  # [T, H, D]
        qjl_signs = _pack_bits(sign_bits)  # [T, H, D//8]

    return packed_indices, key_norms_out, qjl_signs, residual_norms


def _pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack quantization indices into uint8 bytes.

    Args:
        indices: [T, H, D] tensor with values in [0, 2^bits - 1]
        bits: number of bits per index

    Returns:
        packed: [T, H, packed_dim] uint8
    """
    T, H, D = indices.shape
    idx = indices.to(torch.uint8)

    if bits == 4:
        # Pack 2 indices per byte: low nibble = even, high nibble = odd
        packed = (idx[:, :, 1::2] << 4) | idx[:, :, 0::2]
        return packed

    elif bits == 2:
        # Pack 4 indices per byte
        packed = (
            idx[:, :, 0::4]
            | (idx[:, :, 1::4] << 2)
            | (idx[:, :, 2::4] << 4)
            | (idx[:, :, 3::4] << 6)
        )
        return packed

    elif bits == 3:
        # Pack 8 indices into 3 bytes (8 * 3 = 24 bits = 3 bytes)
        # Process in groups of 8 indices
        packed_dim = math.ceil(D * 3 / 8)
        packed = torch.zeros(T, H, packed_dim, dtype=torch.uint8, device=indices.device)

        # For simplicity, process 8 indices at a time into 3 bytes
        n_groups = D // 8
        for g in range(n_groups):
            base_idx = g * 8
            base_out = g * 3
            i = idx[:, :, base_idx : base_idx + 8]  # [T, H, 8], each 0-7

            # Pack 8 x 3-bit values into 3 bytes (24 bits)
            # byte0: i0[2:0] | i1[2:0] | i2[1:0]
            # byte1: i2[2] | i3[2:0] | i4[2:0] | i5[1:0]
            # byte2: i5[2] | i6[2:0] | i7[2:0] | pad[1:0]
            byte0 = i[:, :, 0] | (i[:, :, 1] << 3) | (i[:, :, 2] << 6)
            byte1 = (
                (i[:, :, 2] >> 2)
                | (i[:, :, 3] << 1)
                | (i[:, :, 4] << 4)
                | (i[:, :, 5] << 7)
            )
            byte2 = (i[:, :, 5] >> 1) | (i[:, :, 6] << 2) | (i[:, :, 7] << 5)

            packed[:, :, base_out] = byte0
            packed[:, :, base_out + 1] = byte1
            packed[:, :, base_out + 2] = byte2

        return packed

    else:
        raise ValueError(f"Unsupported bits: {bits}")


def _unpack_indices(packed: torch.Tensor, bits: int, head_dim: int) -> torch.Tensor:
    """Unpack uint8 bytes back to quantization indices.

    Args:
        packed: [T, H, packed_dim] uint8
        bits: number of bits per index
        head_dim: original dimension

    Returns:
        indices: [T, H, head_dim] uint8
    """
    T, H, _ = packed.shape
    mask = (1 << bits) - 1

    if bits == 4:
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        indices = torch.stack([low, high], dim=-1).reshape(T, H, -1)
        return indices[:, :, :head_dim]

    elif bits == 2:
        i0 = packed & 0x03
        i1 = (packed >> 2) & 0x03
        i2 = (packed >> 4) & 0x03
        i3 = (packed >> 6) & 0x03
        indices = torch.stack([i0, i1, i2, i3], dim=-1).reshape(T, H, -1)
        return indices[:, :, :head_dim]

    elif bits == 3:
        indices = torch.zeros(T, H, head_dim, dtype=torch.uint8, device=packed.device)
        n_groups = head_dim // 8
        for g in range(n_groups):
            base_out = g * 3
            base_idx = g * 8
            b0 = packed[:, :, base_out]
            b1 = packed[:, :, base_out + 1]
            b2 = packed[:, :, base_out + 2]

            indices[:, :, base_idx + 0] = b0 & 0x07
            indices[:, :, base_idx + 1] = (b0 >> 3) & 0x07
            indices[:, :, base_idx + 2] = ((b0 >> 6) | (b1 << 2)) & 0x07
            indices[:, :, base_idx + 3] = (b1 >> 1) & 0x07
            indices[:, :, base_idx + 4] = (b1 >> 4) & 0x07
            indices[:, :, base_idx + 5] = ((b1 >> 7) | (b2 << 1)) & 0x07
            indices[:, :, base_idx + 6] = (b2 >> 2) & 0x07
            indices[:, :, base_idx + 7] = (b2 >> 5) & 0x07

        return indices

    else:
        raise ValueError(f"Unsupported bits: {bits}")


def _pack_bits(bits_tensor: torch.Tensor) -> torch.Tensor:
    """Pack a binary tensor (0/1) into uint8 bytes, 8 bits per byte.

    Args:
        bits_tensor: [T, H, D] uint8 with values 0 or 1

    Returns:
        packed: [T, H, D//8] uint8
    """
    T, H, D = bits_tensor.shape
    assert D % 8 == 0
    reshaped = bits_tensor.view(T, H, D // 8, 8)
    # Pack: bit 0 in LSB, bit 7 in MSB
    powers = torch.tensor(
        [1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=bits_tensor.device
    )
    packed = (reshaped * powers).sum(dim=-1).to(torch.uint8)
    return packed


def _unpack_bits(packed: torch.Tensor, dim: int) -> torch.Tensor:
    """Unpack uint8 bytes to binary tensor.

    Args:
        packed: [T, H, D//8] uint8
        dim: original dimension D

    Returns:
        bits: [T, H, D] uint8 with values 0 or 1
    """
    T, H, _ = packed.shape
    expanded = packed.unsqueeze(-1)  # [T, H, D//8, 1]
    powers = torch.tensor(
        [1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=packed.device
    )
    bits = ((expanded & powers) > 0).to(torch.uint8)
    return bits.reshape(T, H, dim)


# ─── Reference decode (for testing) ───


def turbo_quant_decode_dot(
    query: torch.Tensor,
    packed_indices: torch.Tensor,
    key_norms: torch.Tensor,
    config: TurboQuantConfig,
    qjl_signs: Optional[torch.Tensor] = None,
    residual_norms: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference implementation: compute attention scores ⟨q, k⟩ from quantized keys.

    Args:
        query: [num_tokens_q, num_heads, head_dim] float16
        packed_indices: [num_tokens_k, num_kv_heads, packed_k_dim] uint8
        key_norms: [num_tokens_k, num_kv_heads] float16
        config: TurboQuantConfig
        qjl_signs: [num_tokens_k, num_kv_heads, qjl_dim] uint8 (optional)
        residual_norms: [num_tokens_k, num_kv_heads] float16 (optional)

    Returns:
        attn_scores: [num_tokens_q, num_heads, num_tokens_k] float32
    """
    Pi = config.rotation_matrix.float()  # [D, D]
    codebook = config.codebook.float()  # [C]

    # Unpack indices
    indices = _unpack_indices(
        packed_indices, config.bits, config.head_dim
    )  # [Tk, Hkv, D]

    # Reconstruct unit keys in rotated space
    k_rot = codebook[indices.long()]  # [Tk, Hkv, D]

    # Exploit orthogonal invariance: ⟨q, norm * Π^T·k_rot⟩ = norm * ⟨Π·q, k_rot⟩
    q = query.float()
    q_rot = torch.matmul(q, Pi.t())  # [Tq, Hq, D]

    # Handle GQA
    num_q_heads = q_rot.shape[1]
    num_kv_heads = k_rot.shape[1]
    if num_q_heads != num_kv_heads:
        group_size = num_q_heads // num_kv_heads
        k_rot_expanded = k_rot.unsqueeze(2).expand(-1, -1, group_size, -1)
        k_rot_expanded = k_rot_expanded.reshape(k_rot.shape[0], num_q_heads, -1)
        norms_expanded = key_norms.float().unsqueeze(2).expand(-1, -1, group_size)
        norms_expanded = norms_expanded.reshape(key_norms.shape[0], num_q_heads)
    else:
        k_rot_expanded = k_rot
        norms_expanded = key_norms.float()

    # Dot product in rotated space, scaled by key norms
    attn_scores = torch.einsum("qhd,khd->qhk", q_rot, k_rot_expanded)
    attn_scores = attn_scores * norms_expanded.permute(1, 0).unsqueeze(0)

    # QJL correction
    if config.use_qjl and qjl_signs is not None and residual_norms is not None:
        S = config.qjl_matrix.float()  # [D, D]
        D = config.head_dim

        # Compute S @ q (use full values, NOT sign)
        s_q = torch.matmul(q, S.t())  # [Tq, Hq, D]

        # Unpack QJL signs for keys: {0,1} → {-1,+1}
        k_signs = _unpack_bits(qjl_signs, D)  # [Tk, Hkv, D]
        k_signs_float = k_signs.float() * 2 - 1  # {-1, +1}

        if num_q_heads != num_kv_heads:
            k_signs_expanded = k_signs_float.unsqueeze(2).expand(-1, -1, group_size, -1)
            k_signs_expanded = k_signs_expanded.reshape(
                k_signs_float.shape[0], num_q_heads, -1
            )
            res_norms_expanded = (
                residual_norms.float().unsqueeze(2).expand(-1, -1, group_size)
            )
            res_norms_expanded = res_norms_expanded.reshape(
                residual_norms.shape[0], num_q_heads
            )
        else:
            k_signs_expanded = k_signs_float
            res_norms_expanded = residual_norms.float()

        # QJL dot: ⟨S@q, sign(S@residual)⟩
        qjl_dot = torch.einsum("qhd,khd->qhk", s_q, k_signs_expanded)

        # Correction: ||residual||_k × √(π/(2D)) × qjl_dot
        scale = math.sqrt(math.pi / (2.0 * D))
        correction = scale * res_norms_expanded.permute(1, 0).unsqueeze(0) * qjl_dot
        attn_scores = attn_scores + correction

    return attn_scores
