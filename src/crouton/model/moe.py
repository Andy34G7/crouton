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
        self.expert_b_init_std = config.expert_b_init_std

        # Shared base 2-layer FFN
        self.w_up = nn.Linear(self.d_model, self.d_ff, bias=False)
        self.w_down = nn.Linear(self.d_ff, self.d_model, bias=False)

        # Learned softmax router
        self.router = nn.Linear(self.d_model, self.n_experts, bias=False)

        # Low-rank expert deltas (LoRA-style: A normal with std=1/sqrt(fan_in), B zero or small normal)
        self.up_A = nn.Parameter(torch.empty(self.n_experts, self.d_model, self.rank))
        self.up_B = nn.Parameter(torch.zeros(self.n_experts, self.rank, self.d_ff))
        self.down_A = nn.Parameter(torch.empty(self.n_experts, self.d_ff, self.rank))
        self.down_B = nn.Parameter(torch.zeros(self.n_experts, self.rank, self.d_model))

        self.reset_expert_parameters()

    def reset_expert_parameters(self) -> None:
        nn.init.normal_(self.up_A, mean=0.0, std=1.0 / math.sqrt(self.d_model))
        nn.init.normal_(self.down_A, mean=0.0, std=1.0 / math.sqrt(self.d_ff))
        if self.expert_b_init_std > 0.0:
            nn.init.normal_(self.up_B, mean=0.0, std=self.expert_b_init_std)
            nn.init.normal_(self.down_B, mean=0.0, std=self.expert_b_init_std)
        else:
            nn.init.zeros_(self.up_B)
            nn.init.zeros_(self.down_B)

    def forward(
        self, x: torch.Tensor, return_z_loss: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = x.shape
        er = self.n_experts * self.rank
        router_logits = self.router(x).float()
        z_loss = torch.logsumexp(router_logits, dim=-1).square().mean() if return_z_loss else None
        if self.training and self.router_jitter > 0.0:
            noise = torch.empty_like(router_logits).uniform_(1.0 - self.router_jitter, 1.0 + self.router_jitter)
            router_logits = router_logits * noise
        weights = F.softmax(router_logits, dim=-1)  # [B, T, E] fp32
        w_typed = weights.to(x.dtype).unsqueeze(-1)

        # Up projection: shared base + weighted low-rank expert deltas via 2D GEMMs
        up_hidden = self.w_up(x)
        up_a_mat = self.up_A.permute(1, 0, 2).reshape(self.d_model, er)
        up_a = (x @ up_a_mat).view(bsz, seq_len, self.n_experts, self.rank)
        up_a_weighted = (up_a * w_typed).reshape(bsz, seq_len, er)
        up_delta = up_a_weighted @ self.up_B.reshape(er, self.d_ff)
        act = F.relu(up_hidden + self.scale * up_delta).square()

        # Down projection: shared base + weighted low-rank expert deltas via 2D GEMMs
        down_out = self.w_down(act)
        down_a_mat = self.down_A.permute(1, 0, 2).reshape(self.d_ff, er)
        down_a = (act @ down_a_mat).view(bsz, seq_len, self.n_experts, self.rank)
        down_a_weighted = (down_a * w_typed.to(down_a.dtype)).reshape(bsz, seq_len, er)
        down_delta = down_a_weighted @ self.down_B.reshape(er, self.d_model)
        out = down_out + self.scale * down_delta

        if return_z_loss:
            assert z_loss is not None
            return out, weights, z_loss
        return out, weights
