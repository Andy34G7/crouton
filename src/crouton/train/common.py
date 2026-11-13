"""Shared checkpointing and device helpers for training scripts."""

from __future__ import annotations

from collections.abc import Iterator
import math
from pathlib import Path
from typing import Any, TypeVar

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crouton.config import CroutonConfig
from crouton.data.pretrain_data import resolve_token_id
from crouton.model import CroutonForCausalLM

_T = TypeVar("_T")


def bf16_supported(device: torch.device) -> bool:
    """Return True only when running on an Ampere+ CUDA GPU supporting native bfloat16."""
    return device.type == "cuda" and torch.cuda.is_bf16_supported()


def safe_decode(tokenizer, token_ids: list[int]) -> str:
    """Decode generated token IDs, filtering any out-of-range untrained embedding slots."""
    vsize = tokenizer.vocab_size
    filtered = [int(t) for t in token_ids if 0 <= int(t) < vsize]
    return tokenizer.decode(filtered)


def validate_tokenizer_vocab(tokenizer, model_cfg: CroutonConfig) -> None:
    """Fail fast if the tokenizer produces IDs beyond the model embedding table."""
    if tokenizer.vocab_size > model_cfg.vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size ({tokenizer.vocab_size}) exceeds model vocab_size ({model_cfg.vocab_size})."
        )


def next_or_restart(it: Iterator[_T], dataloader: DataLoader[_T]) -> tuple[_T, Iterator[_T]]:
    """Fetch next batch from an iterator, restarting cleanly on exhaustion or failing on empty loaders."""
    try:
        return next(it), it
    except StopIteration:
        fresh = iter(dataloader)
        try:
            return next(fresh), fresh
        except StopIteration as exc:
            raise RuntimeError("DataLoader yielded zero batches; verify dataset content and batch size.") from exc


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
    """Normalize state_dict keys saved from torch.compile or DDP wrappers."""
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        clean = k
        while clean.startswith("_orig_mod.") or clean.startswith("module."):
            clean = clean.removeprefix("_orig_mod.").removeprefix("module.")
        out[clean] = v
    return out


@torch.no_grad()
def evaluate_model_bpb(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    seq_len: int,
    device: torch.device,
    amp_ctx,
) -> dict[str, float]:
    """
    Shared strided sliding-window BPB / PPL / NLL evaluator used by both pretraining validation
    and offline eval_runner evaluation so validation curves match final eval identically.
    """
    if seq_len < 1:
        raise ValueError(f"seq_len ({seq_len}) must be >= 1")
    raw_model = unwrap_model(model)
    was_training = raw_model.training
    raw_model.eval()
    try:
        eos_id = resolve_token_id(tokenizer, "<|endoftext|>", fallback=0)
        total_nll_nats = 0.0
        total_tokens = 0
        total_bytes = 0
        stride = max(1, seq_len // 2)

        for text in texts:
            doc_ids = tokenizer.encode(text)
            if not doc_ids:
                continue
            ids = [eos_id] + doc_ids
            first_end = min(len(ids), seq_len + 1)
            x = torch.tensor([ids[: first_end - 1]], dtype=torch.long, device=device)
            y = torch.tensor([ids[1:first_end]], dtype=torch.long, device=device)
            with amp_ctx:
                out = raw_model(x, targets=y)
            n_tok = y.numel()
            total_nll_nats += float(out.loss.item()) * n_tok
            total_tokens += n_tok

            pos = first_end
            while pos < len(ids):
                end = min(len(ids), pos + stride)
                cont_len = end - pos
                win_start = max(0, end - (seq_len + 1))
                win = ids[win_start:end]
                x = torch.tensor([win[:-1]], dtype=torch.long, device=device)
                y = torch.tensor([win[1:]], dtype=torch.long, device=device)
                with amp_ctx:
                    logits = raw_model(x).logits[0, -cont_len:].float()
                loss = F.cross_entropy(logits, y[0, -cont_len:], reduction="sum")
                total_nll_nats += float(loss.item())
                total_tokens += cont_len
                pos = end

            total_bytes += len(text.encode("utf-8"))

        if total_tokens == 0 or total_bytes == 0:
            raise ValueError("Evaluation texts must contain at least one non-empty tokenizable document")

        avg_nats = total_nll_nats / total_tokens
        ppl = math.exp(min(avg_nats, 20.0))
        bpb = (total_nll_nats / math.log(2.0)) / total_bytes
        return {"loss_nats": avg_nats, "ppl": ppl, "bpb": bpb, "tokens": float(total_tokens)}
    finally:
        if was_training:
            raw_model.train()


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
