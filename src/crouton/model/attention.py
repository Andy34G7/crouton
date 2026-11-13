"""Hybrid Sliding Window + Global Attention with sink tokens and compile-safe FlexAttention masks."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from crouton.config import CroutonConfig
from crouton.model.norm import RMSNorm
from crouton.model.rope import apply_rope

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    HAS_FLEX_ATTENTION = True
except ImportError:
    HAS_FLEX_ATTENTION = False
    create_block_mask = None
    flex_attention = None

_BLOCK_MASK_CACHE: OrderedDict[tuple[int, int, int, str, int | None], object] = OrderedDict()
_MAX_MASK_CACHE = 16


def build_swa_sink_mask(
    seq_len: int,
    window_size: int,
    sink_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Boolean attention mask [1, 1, T, T]: causal AND (within window OR sink token)."""
    q_idx = torch.arange(seq_len, device=device)[:, None]
    kv_idx = torch.arange(seq_len, device=device)[None, :]
    causal = q_idx >= kv_idx
    in_window = (q_idx - kv_idx) < window_size
    is_sink = kv_idx < sink_tokens
    mask = causal & (in_window | is_sink)
    return mask.unsqueeze(0).unsqueeze(0)


@torch.compiler.disable
def _get_cached_swa_block_mask(
    seq_len: int,
    window_size: int,
    sink_tokens: int,
    device: torch.device,
):
    key = (seq_len, window_size, sink_tokens, device.type, device.index)
    cached = _BLOCK_MASK_CACHE.get(key)
    if cached is not None:
        _BLOCK_MASK_CACHE.move_to_end(key)
        return cached

    def swa_sink_mask(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        causal = q_idx >= kv_idx
        in_window = (q_idx - kv_idx) < window_size
        is_sink = kv_idx < sink_tokens
        return causal & (in_window | is_sink)

    block_mask = create_block_mask(
        swa_sink_mask,
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
    )
    _BLOCK_MASK_CACHE[key] = block_mask
    if len(_BLOCK_MASK_CACHE) > _MAX_MASK_CACHE:
        _BLOCK_MASK_CACHE.popitem(last=False)
    return block_mask


class HybridSparseAttention(nn.Module):
    """GQA attention that alternates between Sliding Window (+ sinks) and Global attention."""

    def __init__(self, config: CroutonConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_global = config.is_global_layer(layer_idx)
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.sliding_window = config.sliding_window
        self.sink_tokens = config.sink_tokens
        self.use_flex = config.use_flex_attention and HAS_FLEX_ATTENTION

        self.wq = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps) if config.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps) if config.qk_norm else None

    def _flex_swa_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        block_mask = _get_cached_swa_block_mask(
            seq_len=q.shape[2],
            window_size=self.sliding_window,
            sink_tokens=self.sink_tokens,
            device=q.device,
        )
        return flex_attention(q, k, v, block_mask=block_mask, enable_gqa=self.n_rep > 1)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        q, k = apply_rope(q, k, cos, sin)

        # Avoid O(T^2) dense mask overhead when batch has no padding tokens (outside torch.compile).
        if attention_mask is not None and not torch.compiler.is_compiling() and bool(attention_mask.all().item()):
            attention_mask = None

        if attention_mask is not None:
            key_keep = attention_mask[:, None, None, :].bool()  # [B, 1, 1, T]
            if self.is_global:
                base_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=q.device))
                base_mask = base_mask.unsqueeze(0).unsqueeze(0)
            else:
                base_mask = build_swa_sink_mask(seq_len, self.sliding_window, self.sink_tokens, q.device)
            eye = torch.eye(seq_len, dtype=torch.bool, device=q.device).unsqueeze(0).unsqueeze(0)
            full_mask = (base_mask & key_keep) | eye
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=full_mask, enable_gqa=self.n_rep > 1)
        elif self.is_global:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.n_rep > 1)
        elif self.use_flex and q.is_cuda:
            out = self._flex_swa_attention(q, k, v)
        else:
            attn_mask = build_swa_sink_mask(seq_len, self.sliding_window, self.sink_tokens, q.device)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=self.n_rep > 1)

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        return self.wo(out)
