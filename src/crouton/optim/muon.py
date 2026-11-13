"""Muon optimizer (Newton-Schulz orthogonalization) + AdamW parameter grouping."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Compute approximate orthogonalization G -> U V^T via 5-step Newton-Schulz."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.is_cuda else G.float()
    if G.size(0) > G.size(1):
        X = X.T

    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X.to(dtype=G.dtype)


class Muon(Optimizer):
    """MomentUm Orthogonalized by Newton-schulz for 2D linear layer weights."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    raise ValueError(f"Muon expects 2D parameters, got shape {tuple(g.shape)}")

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf

                ortho = zeropower_via_newtonschulz5(update, steps=ns_steps)
                scale = math.sqrt(max(1.0, p.size(0) / p.size(1)))

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(ortho, alpha=-lr * scale)

        return loss


def build_optimizers(
    model: nn.Module,
    muon_lr: float = 2e-2,
    adamw_lr: float = 3e-4,
    weight_decay: float = 0.01,
) -> tuple[Muon, AdamW]:
    """
    Partition parameters into Muon (2D block linear weights, excluding router/head/emb)
    and AdamW (embeddings, norms, router, 3D LoRA expert tensors).
    Handles torch.compile OptimizedModule `_orig_mod.` prefixes transparently.
    """
    muon_params: list[torch.nn.Parameter] = []
    adamw_decay: list[torch.nn.Parameter] = []
    adamw_nodecay: list[torch.nn.Parameter] = []
    has_blocks = False

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        clean_name = name.removeprefix("_orig_mod.").removeprefix("module.")
        if clean_name.startswith("blocks."):
            has_blocks = True
        is_emb_or_head = "tok_emb" in clean_name or "lm_head" in clean_name
        is_router = "router" in clean_name
        if p.ndim == 2 and clean_name.startswith("blocks.") and not is_router and not is_emb_or_head:
            muon_params.append(p)
        elif p.ndim >= 2 and not is_emb_or_head:
            adamw_decay.append(p)
        else:
            adamw_nodecay.append(p)

    if has_blocks and not muon_params:
        raise RuntimeError("No 2D block parameters matched Muon routing; check parameter naming.")

    muon_opt = Muon(muon_params, lr=muon_lr, weight_decay=weight_decay)
    adamw_opt = AdamW(
        [
            {"params": adamw_decay, "weight_decay": weight_decay},
            {"params": adamw_nodecay, "weight_decay": 0.0},
        ],
        lr=adamw_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    return muon_opt, adamw_opt
