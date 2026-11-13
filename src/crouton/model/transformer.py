"""Crouton transformer block and causal language model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from crouton.config import CroutonConfig
from crouton.model.attention import (
    HybridSparseAttention,
    _get_cached_swa_block_mask,
    _is_right_padded,
    _supports_cuda_flex_attention,
    build_padded_attn_mask,
    build_swa_sink_mask,
)
from crouton.model.moe import MergedLoRADenseMoE
from crouton.model.norm import RMSNorm
from crouton.model.rope import RotaryEmbedding


@dataclass
class CroutonOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    router_entropy: torch.Tensor | None = None


class CroutonBlock(nn.Module):
    """Pre-norm Transformer block with Hybrid Sparse Attention and Merged-LoRA Dense MoE."""

    def __init__(self, config: CroutonConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = HybridSparseAttention(config, layer_idx=layer_idx)
        self.moe_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.moe = MergedLoRADenseMoE(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        swa_block_mask: object | None = None,
        precomputed_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(
            self.attn_norm(x),
            cos,
            sin,
            attention_mask=attention_mask,
            swa_block_mask=swa_block_mask,
            precomputed_mask=precomputed_mask,
        )
        moe_out, router_weights = self.moe(self.moe_norm(x))
        x = x + moe_out
        return x, router_weights


class CroutonForCausalLM(nn.Module):
    """Causal LM with tied embeddings, hybrid SWA/Global attention, and ReLU^2 merged MoE."""

    def __init__(self, config: CroutonConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
            scale_factor=config.rope_scale_factor,
            scaling_type=config.rope_scaling_type,
            original_max_seq_len=config.rope_original_max_seq_len,
        )
        self.blocks = nn.ModuleList([CroutonBlock(config, layer_idx=i) for i in range(config.n_layers)])
        self._needs_flex_swa_mask = any(
            not block.attn.is_global and block.attn.use_flex for block in self.blocks
        )
        self._has_global_layers = any(block.attn.is_global for block in self.blocks)
        self._has_swa_layers = any(not block.attn.is_global for block in self.blocks)
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)

        if config.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            total -= self.tok_emb.weight.numel()
        return total

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> CroutonOutput:
        _, seq_len = input_ids.shape
        device = input_ids.device

        # Check right-padded masks and fetch any @torch.compiler.disable FlexAttention mask
        # before entering embedding/RoPE/transformer tensor ops so compiled graphs stay contiguous.
        # Under strictly causal attention with right-padding, valid queries (q < valid_len) never
        # attend to trailing pad keys (kv >= valid_len), avoiding O(T^2) 4D mask materialization.
        if attention_mask is not None and not torch.compiler.is_compiling() and _is_right_padded(attention_mask):
            attention_mask = None

        swa_block_mask = None
        global_padded_mask = None
        swa_padded_mask = None
        if attention_mask is None and self._has_swa_layers:
            if self._needs_flex_swa_mask and _supports_cuda_flex_attention(device, seq_len):
                swa_block_mask = _get_cached_swa_block_mask(
                    seq_len=seq_len,
                    window_size=self.config.sliding_window,
                    sink_tokens=self.config.sink_tokens,
                    device=device,
                )
            else:
                swa_padded_mask = build_swa_sink_mask(
                    seq_len=seq_len,
                    window_size=self.config.sliding_window,
                    sink_tokens=self.config.sink_tokens,
                    device=device,
                )
        elif attention_mask is not None:
            if self._has_global_layers:
                global_padded_mask = build_padded_attn_mask(
                    attention_mask, seq_len, is_global=True, sliding_window=self.config.sliding_window, sink_tokens=self.config.sink_tokens
                )
            if self._has_swa_layers:
                swa_padded_mask = build_padded_attn_mask(
                    attention_mask, seq_len, is_global=False, sliding_window=self.config.sliding_window, sink_tokens=self.config.sink_tokens
                )

        x = self.tok_emb(input_ids)
        rope_dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        cos, sin = self.rope(seq_len, dtype=rope_dtype)

        entropies = []
        for block in self.blocks:
            layer_mask = global_padded_mask if block.attn.is_global else swa_padded_mask
            x, rw = block(
                x,
                cos,
                sin,
                attention_mask=attention_mask,
                swa_block_mask=swa_block_mask,
                precomputed_mask=layer_mask,
            )
            with torch.no_grad():
                rw_f = rw.float().clamp(min=1e-8)
                entropies.append(-(rw_f * rw_f.log()).sum(dim=-1).mean())

        x = self.final_norm(x)
        if self.lm_head is None:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)

        logits_fp32: torch.Tensor | None = None
        if self.config.logit_softcap > 0.0:
            cap = self.config.logit_softcap
            logits_fp32 = cap * torch.tanh(logits.float() / cap)
            logits = logits_fp32.to(logits.dtype)

        loss = None
        if targets is not None:
            src_logits = logits_fp32 if logits_fp32 is not None else logits
            flat_logits = src_logits.reshape(-1, src_logits.size(-1))
            flat_targets = targets.reshape(-1)
            valid_mask = flat_targets != -100
            if not torch.compiler.is_compiling():
                if bool(valid_mask.any()):
                    # Only promote active supervised rows to fp32 when running eagerly (matches dpo.sequence_logps).
                    loss = F.cross_entropy(flat_logits[valid_mask].float(), flat_targets[valid_mask], reduction="mean")
                else:
                    loss = flat_logits[:1, :1].float().sum() * 0.0
            else:
                safe_targets = torch.where(valid_mask, flat_targets, 0)
                tok_loss = F.cross_entropy(flat_logits.float(), safe_targets, reduction="none")
                valid_f = valid_mask.float()
                loss = (tok_loss * valid_f).sum() / valid_f.sum().clamp(min=1.0)

        router_entropy = torch.stack(entropies).mean() if entropies else None
        return CroutonOutput(logits=logits, loss=loss, router_entropy=router_entropy)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int = 0,
        eos_token_ids: set[int] | list[int] | None = None,
        vocab_limit: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation with fp32 sampling, top-k filtering, and EOS stopping."""
        if input_ids.size(1) < 1:
            raise ValueError("input_ids must have seq_len >= 1")
        if vocab_limit is not None and vocab_limit < 1:
            raise ValueError("vocab_limit must be >= 1 when specified")
        was_training = self.training
        self.eval()
        try:
            eos_tensor = (
                torch.tensor(list(dict.fromkeys(int(t) for t in eos_token_ids)), dtype=torch.long, device=input_ids.device)
                if eos_token_ids
                else None
            )
            eos_fill = eos_tensor[:1].view(1, 1) if eos_tensor is not None else None
            ids = input_ids
            finished = torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device)

            for _ in range(max(0, max_new_tokens)):
                ctx_ids = ids[:, -self.config.max_seq_len :]
                logits = self(ctx_ids).logits[:, -1, :].float()
                if vocab_limit is not None and 0 < vocab_limit < logits.size(-1):
                    logits = logits[:, :vocab_limit]
                if temperature <= 0.0:
                    next_tok = logits.argmax(dim=-1, keepdim=True)
                else:
                    scaled = logits / temperature
                    if top_k > 0:
                        v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                        scaled = scaled.masked_fill(scaled < v[:, [-1]], float("-inf"))
                    probs = F.softmax(scaled, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1)
                if eos_fill is not None:
                    next_tok = torch.where(finished.unsqueeze(-1), eos_fill, next_tok)
                ids = torch.cat([ids, next_tok], dim=-1)
                if eos_tensor is not None:
                    finished = finished | torch.isin(next_tok.squeeze(-1), eos_tensor)
                    if bool(finished.all().item()):
                        break
            return ids
        finally:
            if was_training:
                self.train()
