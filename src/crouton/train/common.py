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


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap torch.compile OptimizedModule (or DDP) if present."""
    while hasattr(model, "_orig_mod") or hasattr(model, "module"):
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        elif hasattr(model, "module"):
            model = model.module
    return model


def strip_orig_mod_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalize state_dict keys saved from torch.compile'd modules."""
    prefix = "_orig_mod."
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    config: CroutonConfig,
    step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_model = unwrap_model(model)
    payload: dict[str, Any] = {
        "model": raw_model.state_dict(),
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
    override_rope_scaling_type: str | None = None,
) -> tuple[CroutonForCausalLM, CroutonConfig, int]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt["config"])
    if "rope_original_max_seq_len" not in cfg_dict:
        cfg_dict["rope_original_max_seq_len"] = cfg_dict.get("max_seq_len", 2048)
    if override_seq_len is not None:
        cfg_dict["max_seq_len"] = override_seq_len
    if override_rope_scale is not None:
        cfg_dict["rope_scale_factor"] = override_rope_scale
    if override_rope_scaling_type is not None:
        cfg_dict["rope_scaling_type"] = override_rope_scaling_type

    cfg = CroutonConfig(**cfg_dict)
    model = CroutonForCausalLM(cfg).to(device)
    state_dict = strip_orig_mod_prefix(ckpt["model"])
    model.load_state_dict(state_dict, strict=True)
    return model, cfg, int(ckpt.get("step", 0))
