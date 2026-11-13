"""Merged-LoRA Dense MoE with 2-layer ReLU^2 FFN."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from crouton.config import CroutonConfig


class MergedLoRADenseMoE(nn.Module):
    """
    2-layer ReLU^2 MLP with a shared base and input-dependent merged low-rank experts.

    Every token activates all E experts via learned softmax router weights, avoiding token
    dropping while keeping compute and memory low via factorized LoRA projections:
        h = ReLU(x @ W_up^T + scale * sum_e w_e (x @ A_up[e] @ B_up[e]))^2
        y = h @ W_down^T + scale * sum_e w_e (h @ A_down[e] @ B_down[e])
    """

    def __init__(self, config: CroutonConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.n_experts = config.n_experts
        self.rank = config.expert_rank
        self.scale = config.expert_alpha / config.expert_rank
        self.router_jitter = config.router_jitter

        # Shared base 2-layer FFN
        self.w_up = nn.Linear(self.d_model, self.d_ff, bias=False)
        self.w_down = nn.Linear(self.d_ff, self.d_model, bias=False)

        # Learned softmax router
        self.router = nn.Linear(self.d_model, self.n_experts, bias=False)

        # Low-rank expert deltas (LoRA-style: A initialized kaiming/normal, B initialized near zero)
        self.up_A = nn.Parameter(torch.empty(self.n_experts, self.d_model, self.rank))
        self.up_B = nn.Parameter(torch.zeros(self.n_experts, self.rank, self.d_ff))
        self.down_A = nn.Parameter(torch.empty(self.n_experts, self.d_ff, self.rank))
        self.down_B = nn.Parameter(torch.zeros(self.n_experts, self.rank, self.d_model))

        self.reset_expert_parameters()

    def reset_expert_parameters(self) -> None:
        nn.init.normal_(self.up_A, mean=0.0, std=1.0 / math.sqrt(self.d_model))
        nn.init.normal_(self.down_A, mean=0.0, std=1.0 / math.sqrt(self.d_ff))
        nn.init.zeros_(self.up_B)
        nn.init.zeros_(self.down_B)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = self.router(x).float()
        if self.training and self.router_jitter > 0.0:
            noise = torch.empty_like(router_logits).uniform_(1.0 - self.router_jitter, 1.0 + self.router_jitter)
            router_logits = router_logits * noise
        weights = F.softmax(router_logits, dim=-1)  # [B, T, E] fp32

        # Up projection: shared base + weighted low-rank expert deltas
        up_hidden = self.w_up(x)
        up_a = torch.einsum("btd,edr->bter", x, self.up_A)
        up_a_weighted = up_a * weights.to(up_a.dtype).unsqueeze(-1)
        up_delta = torch.einsum("bter,erf->btf", up_a_weighted, self.up_B)
        act = F.relu(up_hidden + self.scale * up_delta).square()

        # Down projection: shared base + weighted low-rank expert deltas
        down_out = self.w_down(act)
        down_a = torch.einsum("btf,efr->bter", act, self.down_A)
        down_a_weighted = down_a * weights.to(down_a.dtype).unsqueeze(-1)
        down_delta = torch.einsum("bter,erd->btd", down_a_weighted, self.down_B)
        out = down_out + self.scale * down_delta

        return out, weights
