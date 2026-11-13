"""Rotary Position Embeddings (RoPE) with context-extension scaling."""

from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Precomputed sin/cos RoPE cache with NTK-aware base scaling for context extension."""

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        scale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.head_dim = head_dim
        self.base_theta = theta
        self.scale_factor = scale_factor

        # NTK-aware base scaling when extending context (e.g. 2048 -> 4096 with scale_factor=2.0).
        effective_theta = theta * (scale_factor ** (head_dim / (head_dim - 2))) if scale_factor > 1.0 else theta
        inv_freq = 1.0 / (effective_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._max_seq_len_cached = 0
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
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
