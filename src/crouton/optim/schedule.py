"""Warmup-Stable-Decay (WSD) learning rate schedule."""

from __future__ import annotations

import math

from torch.optim import Optimizer


def wsd_lr_multiplier(
    step: int,
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Compute WSD learning-rate multiplier (linear warmup in (0, 1], stable 1.0, cosine decay to min_lr_ratio)."""
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    decay_start = max(warmup_steps, total_steps - decay_steps)
    if step < decay_start:
        return 1.0
    if decay_steps <= 0 or step >= total_steps:
        return min_lr_ratio
    progress = float(step - decay_start) / float(max(1, total_steps - decay_start))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def set_optimizer_lr(optimizer: Optimizer, base_lr: float, multiplier: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = base_lr * multiplier
