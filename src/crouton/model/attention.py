"""Hybrid Sliding Window + Global Attention with sink tokens and compile-safe FlexAttention masks."""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from crouton.config import CroutonConfig
from crouton.model.norm import RMSNorm
from crouton.model.rope import apply_rope

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    HAS_FLEX_ATTENTION = True
    _compiled_flex_attention = torch.compile(flex_attention, dynamic=True)
except ImportError:
    HAS_FLEX_ATTENTION = False
    create_block_mask = None
    flex_attention = None
    _compiled_flex_attention = None

_BLOCK_MASK_CACHE: OrderedDict[tuple[int, int, int, str, int | None], object] = OrderedDict()
_MAX_MASK_CACHE = 16


def _supports_cuda_flex_attention(device: torch.device, seq_len: int) -> bool:
    """Use FlexAttention only on Ampere+ CUDA GPUs for block-aligned or compiled lengths >= 128."""
    if not HAS_FLEX_ATTENTION or device.type != "cuda" or seq_len < 128:
        return False
    if not torch.compiler.is_compiling() and seq_len % 128 != 0:
        return False
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_capability(dev_idx)[0] >= 8


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


def _is_right_padded(attention_mask: torch.Tensor) -> bool:
    """Return True when every row has False/0 only in a trailing suffix (or no padding)."""
    m = attention_mask.bool()
    if m.size(1) <= 1:
        return True
    return not bool((m[:, 1:] & ~m[:, :-1]).any().item())


def build_padded_attn_mask(
    attention_mask: torch.Tensor,
    seq_len: int,
    is_global: bool,
    sliding_window: int,
    sink_tokens: int,
) -> torch.Tensor:
    """Build [B, 1, T, T] boolean causal/SWA mask without NaN rows."""
    key_keep = attention_mask[:, None, None, :].bool()
    if is_global:
        base_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=attention_mask.device))
        base_mask = base_mask.unsqueeze(0).unsqueeze(0)
    else:
        base_mask = build_swa_sink_mask(seq_len, sliding_window, sink_tokens, attention_mask.device)
    eye = torch.eye(seq_len, dtype=torch.bool, device=attention_mask.device).unsqueeze(0).unsqueeze(0)
    return (base_mask & key_keep) | eye


@lru_cache(maxsize=16)
def _get_swa_mask_mod(window_size: int, sink_tokens: int):
    def swa_sink_mask(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        causal = q_idx >= kv_idx
        in_window = (q_idx - kv_idx) < window_size
        is_sink = kv_idx < sink_tokens
        return causal & (in_window | is_sink)

    return swa_sink_mask


@torch.compiler.disable
def _get_cached_swa_block_mask(
    seq_len: int,
    window_size: int,
    sink_tokens: int,
    device: torch.device,
):
    dev_idx = device.index if device.index is not None else (torch.cuda.current_device() if device.type == "cuda" else 0)
    key = (seq_len, window_size, sink_tokens, device.type, dev_idx)
    cached = _BLOCK_MASK_CACHE.get(key)
    if cached is not None:
        _BLOCK_MASK_CACHE.move_to_end(key)
        return cached

    block_mask = create_block_mask(
        _get_swa_mask_mod(window_size, sink_tokens),
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

    def _flex_swa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: object | None = None,
    ) -> torch.Tensor:
        if block_mask is None:
            block_mask = _get_cached_swa_block_mask(
                seq_len=q.shape[2],
                window_size=self.sliding_window,
                sink_tokens=self.sink_tokens,
                device=q.device,
            )
        attn_fn = flex_attention if torch.compiler.is_compiling() else _compiled_flex_attention
        return attn_fn(q, k, v, block_mask=block_mask, enable_gqa=self.n_rep > 1)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        swa_block_mask: object | None = None,
        precomputed_mask: torch.Tensor | None = None,
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

        if precomputed_mask is not None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=precomputed_mask, enable_gqa=self.n_rep > 1)
        else:
            if attention_mask is not None and not torch.compiler.is_compiling() and _is_right_padded(attention_mask):
                attention_mask = None

            if attention_mask is not None:
                full_mask = build_padded_attn_mask(
                    attention_mask=attention_mask,
                    seq_len=seq_len,
                    is_global=self.is_global,
                    sliding_window=self.sliding_window,
                    sink_tokens=self.sink_tokens,
                )
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=full_mask, enable_gqa=self.n_rep > 1)
            elif self.is_global:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.n_rep > 1)
            elif self.use_flex and _supports_cuda_flex_attention(q.device, seq_len):
                out = self._flex_swa_attention(q, k, v, block_mask=swa_block_mask)
            else:
                attn_mask = build_swa_sink_mask(seq_len, self.sliding_window, self.sink_tokens, q.device)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=self.n_rep > 1)

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        return self.wo(out)
