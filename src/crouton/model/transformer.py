"""Crouton transformer block and causal language model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from crouton.config import CroutonConfig
from crouton.model.attention import HybridSparseAttention
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.attn_norm(x), cos, sin, attention_mask=attention_mask)
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
        x = self.tok_emb(input_ids)
        cos, sin = self.rope(seq_len, dtype=x.dtype)

        entropies = []
        for block in self.blocks:
            x, rw = block(x, cos, sin, attention_mask=attention_mask)
            rw_f = rw.float().clamp(min=1e-8)
            entropies.append(-(rw_f * rw_f.log()).sum(dim=-1).mean())

        x = self.final_norm(x)
        if self.lm_head is None:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)

        if self.config.logit_softcap > 0.0:
            cap = self.config.logit_softcap
            logits = cap * torch.tanh(logits / cap)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

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
    ) -> torch.Tensor:
        """Autoregressive generation with optional top-k sampling and EOS stopping."""
        was_training = self.training
        self.eval()
        eos_set = set(eos_token_ids) if eos_token_ids is not None else set()
        ids = input_ids
        finished = torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device)

        for _ in range(max_new_tokens):
            ctx_ids = ids[:, -self.config.max_seq_len :]
            logits = self(ctx_ids).logits[:, -1, :]
            if temperature <= 0.0:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                scaled = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                    scaled = scaled.masked_fill(scaled < v[:, [-1]], float("-inf"))
                probs = F.softmax(scaled, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_tok], dim=-1)
            if eos_set:
                for idx in range(ids.size(0)):
                    if int(next_tok[idx, 0].item()) in eos_set:
                        finished[idx] = True
                if finished.all():
                    break

        if was_training:
            self.train()
        return ids
