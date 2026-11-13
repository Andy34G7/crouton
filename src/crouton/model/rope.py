"""Rotary Position Embeddings (RoPE) with YaRN and NTK context-extension scaling."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _yarn_find_correction_dim(
    num_rotations: float,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2.0 * math.pi))) / (2.0 * math.log(base))


class RotaryEmbedding(nn.Module):
    """Precomputed sin/cos RoPE cache with YaRN or NTK-aware scaling for context extension."""

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        scale_factor: float = 1.0,
        scaling_type: str = "yarn",
        original_max_seq_len: int = 2048,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        if scaling_type not in ("yarn", "ntk"):
            raise ValueError(f"Unsupported scaling_type: {scaling_type!r}")

        self.head_dim = head_dim
        self.base_theta = theta
        self.scale_factor = scale_factor
        self.scaling_type = scaling_type
        self.mscale = 1.0

        dim_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        if scale_factor <= 1.0:
            inv_freq = 1.0 / (theta ** (dim_range / head_dim))
        elif scaling_type == "ntk":
            effective_theta = theta * (scale_factor ** (head_dim / max(2, head_dim - 2)))
            inv_freq = 1.0 / (effective_theta ** (dim_range / head_dim))
        else:
            # YaRN: wavelength-band ramp between interpolation (low freq) and extrapolation (high freq)
            inv_freq_extrap = 1.0 / (theta ** (dim_range / head_dim))
            inv_freq_interp = inv_freq_extrap / scale_factor
            half_dim = head_dim // 2
            low = max(0, math.floor(_yarn_find_correction_dim(beta_fast, head_dim, theta, original_max_seq_len)))
            high = min(
                half_dim - 1,
                math.ceil(_yarn_find_correction_dim(beta_slow, head_dim, theta, original_max_seq_len)),
            )
            if high <= low:
                high = min(half_dim - 1, low + 1)
            idx = torch.arange(half_dim, dtype=torch.float32)
            interp_factor = torch.clamp((idx - low) / max(float(high - low), 1e-3), 0.0, 1.0)
            inv_freq = inv_freq_interp * interp_factor + inv_freq_extrap * (1.0 - interp_factor)
            self.mscale = 0.1 * math.log(scale_factor) + 1.0

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._max_seq_len_cached = 0
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos() * self.mscale, persistent=False)
        self.register_buffer("sin_cached", emb.sin() * self.mscale, persistent=False)
        self._max_seq_len_cached = seq_len

    def forward(self, seq_len: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self._max_seq_len_cached:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(dtype=dtype),
            self.sin_cached[:seq_len].to(dtype=dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to q, k of shape [B, H, T, D]."""
    cos_u = cos.unsqueeze(0).unsqueeze(0)
    sin_u = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos_u) + (rotate_half(q) * sin_u)
    k_rot = (k * cos_u) + (rotate_half(k) * sin_u)
    return q_rot, k_rot
