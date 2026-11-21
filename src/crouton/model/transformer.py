"""Crouton transformer block and causal language model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

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
    logits: torch.Tensor | None
    loss: torch.Tensor | None = None
    router_entropy: torch.Tensor | None = None
    router_z_loss: torch.Tensor | None = None
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    hidden_states: torch.Tensor | None = None


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
        is_right_padded: bool = False,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        return_z_loss: bool = False,
        is_causal: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_res = self.attn(
            self.attn_norm(x),
            cos,
            sin,
            attention_mask=attention_mask,
            swa_block_mask=swa_block_mask,
            precomputed_mask=precomputed_mask,
            is_right_padded=is_right_padded,
            past_kv=past_kv,
            use_cache=use_cache,
            is_causal=is_causal,
        )
        if use_cache:
            attn_out, present_kv = attn_res  # type: ignore[misc]
        else:
            attn_out, present_kv = attn_res, None  # type: ignore[assignment]
        x = x + attn_out
        if return_z_loss:
            moe_out, router_weights, z_loss = self.moe(self.moe_norm(x), return_z_loss=True)  # type: ignore[misc]
        else:
            moe_out, router_weights = self.moe(self.moe_norm(x))  # type: ignore[misc]
            z_loss = None
        x = x + moe_out
        return x, router_weights, z_loss, present_kv


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
        resid_std = 0.02 / math.sqrt(2.0 * config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.wo.weight, mean=0.0, std=resid_std)
            nn.init.normal_(block.moe.w_down.weight, mean=0.0, std=resid_std)

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
        is_right_padded: bool = False,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        is_causal: bool = True,
        return_hidden_states: bool = False,
        skip_lm_head: bool = False,
    ) -> CroutonOutput:
        _, seq_len = input_ids.shape
        device = input_ids.device
        past_len = past_key_values[0][0].size(2) if past_key_values else 0

        # Check right-padded masks and fetch any @torch.compiler.disable FlexAttention mask
        # before entering embedding/RoPE/transformer tensor ops so compiled graphs stay contiguous.
        # Under strictly causal attention with right-padding, valid queries (q < valid_len) never
        # attend to trailing pad keys (kv >= valid_len), avoiding O(T^2) 4D mask materialization.
        # Under bidirectional attention (is_causal=False), pad keys MUST remain masked.
        if is_causal and attention_mask is not None and (
            is_right_padded or (not torch.compiler.is_compiling() and _is_right_padded(attention_mask))
        ):
            attention_mask = None

        swa_block_mask = None
        global_padded_mask = None
        swa_padded_mask = None
        if past_key_values is None:
            if attention_mask is None and self._has_swa_layers:
                if is_causal and self._needs_flex_swa_mask and _supports_cuda_flex_attention(device, seq_len):
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
                        is_causal=is_causal,
                    )
            elif attention_mask is not None:
                if self._has_global_layers:
                    global_padded_mask = build_padded_attn_mask(
                        attention_mask,
                        seq_len,
                        is_global=True,
                        sliding_window=self.config.sliding_window,
                        sink_tokens=self.config.sink_tokens,
                        is_causal=is_causal,
                    )
                if self._has_swa_layers:
                    swa_padded_mask = build_padded_attn_mask(
                        attention_mask,
                        seq_len,
                        is_global=False,
                        sliding_window=self.config.sliding_window,
                        sink_tokens=self.config.sink_tokens,
                        is_causal=is_causal,
                    )

        x = self.tok_emb(input_ids)
        cos, sin = self.rope(seq_len, dtype=torch.float32, offset=past_len)

        compute_z_loss = self.config.router_z_loss_coef > 0.0
        entropies = []
        z_losses = []
        present_kvs: list[tuple[torch.Tensor, torch.Tensor]] | None = [] if use_cache else None
        for idx, block in enumerate(self.blocks):
            layer_mask = global_padded_mask if block.attn.is_global else swa_padded_mask
            layer_past = past_key_values[idx] if past_key_values is not None else None
            x, rw, z_loss, present_kv = block(
                x,
                cos,
                sin,
                attention_mask=attention_mask,
                swa_block_mask=swa_block_mask,
                precomputed_mask=layer_mask,
                is_right_padded=is_right_padded,
                past_kv=layer_past,
                use_cache=use_cache,
                return_z_loss=compute_z_loss,
                is_causal=is_causal,
            )
            if present_kvs is not None:
                assert present_kv is not None
                present_kvs.append(present_kv)
            if z_loss is not None:
                z_losses.append(z_loss)
            with torch.no_grad():
                rw_f = rw.float().clamp(min=1e-8)
                entropies.append(-(rw_f * rw_f.log()).sum(dim=-1).mean())

        x = self.final_norm(x)
        router_z_loss = torch.stack(z_losses).mean() if z_losses else None
        router_entropy = torch.stack(entropies).mean() if entropies else None

        if skip_lm_head and targets is None:
            return CroutonOutput(
                logits=None,
                loss=None,
                router_entropy=router_entropy,
                router_z_loss=router_z_loss,
                past_key_values=present_kvs,
                hidden_states=x,
            )

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
            if router_z_loss is not None:
                loss = loss + self.config.router_z_loss_coef * router_z_loss

        return CroutonOutput(
            logits=logits,
            loss=loss,
            router_entropy=router_entropy,
            router_z_loss=router_z_loss,
            past_key_values=present_kvs,
            hidden_states=x if return_hidden_states else None,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int = 0,
        eos_token_ids: Sequence[int] | set[int] | None = None,
        vocab_limit: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation with fp32 sampling, top-k filtering, and EOS stopping."""
        if input_ids.ndim != 2 or input_ids.size(0) < 1 or input_ids.size(1) < 1:
            raise ValueError("input_ids must be 2D [B, T] with B >= 1 and T >= 1")
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        if vocab_limit is not None and (
            not isinstance(vocab_limit, int) or isinstance(vocab_limit, bool) or vocab_limit < 1
        ):
            raise ValueError("vocab_limit must be >= 1 when specified")
        was_training = self.training
        self.eval()
        try:
            ordered_eos = (
                sorted({int(t) for t in eos_token_ids})
                if isinstance(eos_token_ids, set)
                else list(dict.fromkeys(int(t) for t in eos_token_ids))
            ) if eos_token_ids else []
            eos_tensor = (
                torch.tensor(ordered_eos, dtype=torch.long, device=input_ids.device)
                if ordered_eos
                else None
            )
            eos_fill = eos_tensor[:1].view(1, 1) if eos_tensor is not None else None
            ids = input_ids
            finished = torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device)
            past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None

            for _ in range(max(0, max_new_tokens)):
                if past_key_values is not None and ids.size(1) <= self.config.max_seq_len:
                    out = self(ids[:, -1:], past_key_values=past_key_values, use_cache=True)
                else:
                    ctx_ids = ids[:, -self.config.max_seq_len :]
                    want_cache = ids.size(1) < self.config.max_seq_len
                    out = self(ctx_ids, use_cache=want_cache)
                past_key_values = getattr(out, "past_key_values", None)
                logits = out.logits[:, -1, :].float()
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
