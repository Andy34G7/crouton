"""SPLADE-v3 sparse lexical retrieval head and losses for Crouton."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from crouton.config import CroutonConfig
from crouton.model.norm import RMSNorm
from crouton.model.transformer import CroutonForCausalLM


@dataclass
class SPLADEOutput:
    sparse_vector: torch.Tensor
    flops_loss: torch.Tensor
    l1_loss: torch.Tensor
    hidden_states: torch.Tensor | None = None


def quad_reg_weight(step: int, warmup_steps: int, max_weight: float) -> float:
    """Quadratic regularization warmup schedule used in SPLADE-v2 / SPLADE-v3."""
    if max_weight <= 0.0:
        return 0.0
    if warmup_steps <= 0 or step >= warmup_steps:
        return max_weight
    frac = max(0.0, float(step) / float(warmup_steps))
    return max_weight * (frac * frac)


def splade_infonce_loss(
    q_vec: torch.Tensor,
    pos_vec: torch.Tensor,
    neg_vec: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """In-batch (+ optional hard-negative) InfoNCE loss on sparse lexical dot products."""
    q_f = q_vec.float()
    p_f = pos_vec.float()
    scores = torch.matmul(q_f, p_f.transpose(0, 1))
    if neg_vec is not None:
        n_f = neg_vec.float()
        if n_f.ndim == 2:
            # [B, V] hard negatives (1 per query, also shared as in-batch negatives)
            neg_scores = torch.matmul(q_f, n_f.transpose(0, 1))
            scores = torch.cat([scores, neg_scores], dim=1)
        elif n_f.ndim == 3:
            # [B, K, V] K hard negatives per query
            hard_scores = torch.einsum("bv,bkv->bk", q_f, n_f)
            scores = torch.cat([scores, hard_scores], dim=1)
        else:
            raise ValueError(f"neg_vec must be 2D or 3D, got shape {tuple(n_f.shape)}")
    if temperature != 1.0:
        scores = scores / temperature
    labels = torch.arange(q_f.size(0), device=q_f.device, dtype=torch.long)
    return F.cross_entropy(scores, labels)


def splade_margin_mse_loss(
    q_vec: torch.Tensor,
    pos_vec: torch.Tensor,
    neg_vec: torch.Tensor,
    teacher_pos: torch.Tensor,
    teacher_neg: torch.Tensor,
) -> torch.Tensor:
    """SPLADE-v3 Margin-MSE cross-encoder distillation loss."""
    student_pos = (q_vec.float() * pos_vec.float()).sum(dim=-1)
    student_neg = (q_vec.float() * neg_vec.float()).sum(dim=-1)
    student_margin = student_pos - student_neg
    teacher_margin = teacher_pos.float() - teacher_neg.float()
    return F.mse_loss(student_margin, teacher_margin)


class SPLADESparseHead(nn.Module):
    """
    SPLADE Sparse Lexical & Expansion Representation Head:
        w_{i, j} = log(1 + ReLU(W * h_i + b)_j)
        v_j      = max_{i in valid_tokens} w_{i, j}

    Ties `decoder.weight` to the pretrained LM head when provided and zero-initializes
    the residual adapter so step-0 lexical alignment from pretraining is preserved.
    """

    def __init__(
        self,
        config: CroutonConfig,
        lm_head_weight: nn.Parameter | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = config.vocab_size
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.decoder = nn.Linear(config.d_model, config.vocab_size, bias=True)

        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.decoder.bias)
        if lm_head_weight is not None:
            self.decoder.weight = lm_head_weight

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.norm(hidden_states + self.proj(F.silu(hidden_states)))
        logits = self.decoder(h).float()
        term_weights = torch.log1p(F.relu(logits))

        if attention_mask is not None:
            mask = attention_mask.to(dtype=term_weights.dtype).unsqueeze(-1)
            term_weights = term_weights * mask

        sparse_vector, _ = torch.max(term_weights, dim=1)  # [B, V] fp32
        flops_loss = torch.sum(torch.mean(sparse_vector, dim=0).square())
        l1_loss = sparse_vector.sum(dim=-1).mean()
        return sparse_vector, flops_loss, l1_loss


class CroutonSPLADE(nn.Module):
    """Crouton backbone with bidirectional attention toggle and tied SPLADE sparse head."""

    def __init__(
        self,
        backbone: CroutonForCausalLM,
        tie_lm_head: bool = True,
        pad_token_id: int | None = None,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pad_token_id = pad_token_id
        self.bidirectional = bidirectional
        if tie_lm_head:
            lm_weight = backbone.tok_emb.weight if backbone.lm_head is None else backbone.lm_head.weight
        else:
            lm_weight = None
        self.splade_head = SPLADESparseHead(backbone.config, lm_head_weight=lm_weight)

    @staticmethod
    def format_bpe_text(text: str, prefix: str = " ") -> str:
        """Prepend a consistent prefix/space so word-initial BPE tokens align across queries and passages."""
        stripped = text.strip()
        return f"{prefix}{stripped}" if prefix else stripped

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_hidden_states: bool = False,
    ) -> SPLADEOutput:
        if attention_mask is None and self.pad_token_id is not None:
            attention_mask = input_ids.ne(self.pad_token_id)

        out = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            is_causal=not self.bidirectional,
            skip_lm_head=True,
        )
        assert out.hidden_states is not None
        sparse_vector, flops_loss, l1_loss = self.splade_head(out.hidden_states, attention_mask=attention_mask)
        return SPLADEOutput(
            sparse_vector=sparse_vector,
            flops_loss=flops_loss,
            l1_loss=l1_loss,
            hidden_states=out.hidden_states if return_hidden_states else None,
        )
