"""Shared checkpointing and device helpers for training scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from crouton.config import CroutonConfig
from crouton.model import CroutonForCausalLM


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(
    path: str | Path,
    model: CroutonForCausalLM,
    config: CroutonConfig,
    step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "config": config.to_dict(),
        "step": step,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, out)


def load_checkpoint(
    path: str | Path,
    device: torch.device,
    override_seq_len: int | None = None,
    override_rope_scale: float | None = None,
) -> tuple[CroutonForCausalLM, CroutonConfig, int]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]
    if override_seq_len is not None:
        cfg_dict["max_seq_len"] = override_seq_len
    if override_rope_scale is not None:
        cfg_dict["rope_scale_factor"] = override_rope_scale
    cfg = CroutonConfig(**cfg_dict)
    model = CroutonForCausalLM(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    return model, cfg, int(ckpt.get("step", 0))
