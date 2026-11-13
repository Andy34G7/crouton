"""Shared checkpointing and device helpers for training scripts."""

from __future__ import annotations

from collections.abc import Iterator
import math
import os
from pathlib import Path
from typing import Any, TypeVar
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crouton.config import CroutonConfig, REQUIRED_CONFIG_KEYS
from crouton.data.pretrain_data import (
    normalize_optional_subset,
    resolve_token_id,
    validate_sequence_arg,
)
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


def compute_flex_pad_multiple(device: torch.device, model_cfg: CroutonConfig) -> int:
    """Return block-alignment multiple for CUDA SM80+ FlexAttention batches."""
    return 128 if (device.type == "cuda" and model_cfg.use_flex_attention and torch.cuda.get_device_capability(device)[0] >= 8) else 1


def warn_unscaled_rope_extension(model_cfg: CroutonConfig, seq_len: int) -> None:
    """Warn when post-training expands max_seq_len beyond rope_original_max_seq_len without RoPE scaling."""
    if seq_len > model_cfg.rope_original_max_seq_len and model_cfg.rope_scale_factor <= 1.0:
        warnings.warn(
            f"seq_len ({seq_len}) exceeds rope_original_max_seq_len ({model_cfg.rope_original_max_seq_len}) "
            f"while rope_scale_factor={model_cfg.rope_scale_factor}; unscaled RoPE extrapolation may degrade "
            f"long-context quality without midtrain context extension.",
            UserWarning,
            stacklevel=2,
        )


def looks_like_dir_path(raw_path: str) -> bool:
    """Return True if a raw string uses '.', '..', or trailing directory separators that Path would normalize away."""
    s = raw_path.strip()
    if not s:
        return False
    if s in (".", ".."):
        return True
    seps = tuple(dict.fromkeys(sep for sep in (os.sep, os.altsep, "/") if sep))
    return s.endswith(seps) or any(s.endswith(f"{sep}.") or s.endswith(f"{sep}..") for sep in seps)


def strip_required_paths(**named_paths: str) -> tuple[str, ...]:
    """Validate and strip required path/identifier strings in argument order."""
    cleaned: list[str] = []
    for label, value in named_paths.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be non-empty")
        cleaned.append(value.strip())
    return tuple(cleaned)


def validate_output_dir(output_dir: str | Path) -> Path:
    """Validate that output_dir and its existing ancestors are directories, not regular files or broken symlinks."""
    if isinstance(output_dir, str):
        (clean_dir,) = strip_required_paths(output_dir=output_dir)
        out = Path(clean_dir)
    elif isinstance(output_dir, Path) and str(output_dir).strip():
        out = output_dir
    else:
        raise ValueError("output_dir must be non-empty")
    for candidate in (out, *out.parents):
        if (candidate.exists() and not candidate.is_dir()) or (candidate.is_symlink() and not candidate.exists()):
            raise ValueError(f"Output directory path {str(candidate)!r} exists and is not a directory")
    return out


def read_nonempty_lines(path: str | Path, label: str = "Text file") -> list[str]:
    """Validate path, read UTF-8 text lines, and reject empty/whitespace-only files."""
    if isinstance(path, str):
        (clean_path,) = strip_required_paths(**{label: path})
        fpath = Path(clean_path)
    elif isinstance(path, Path) and str(path).strip():
        fpath = path
    else:
        raise ValueError(f"{label} must be non-empty")
    lines = [line.strip() for line in fpath.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{label} {str(fpath)!r} contains no non-empty lines")
    return lines


def validate_tokenizer_vocab(tokenizer, model_cfg: CroutonConfig) -> None:
    """Fail fast if tokenizer vocab is empty or produces IDs beyond the model embedding table."""
    if tokenizer.vocab_size < 1:
        raise ValueError(f"Tokenizer vocab_size ({tokenizer.vocab_size}) must be >= 1.")
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
    validate_sequence_arg(texts, "Evaluation texts")
    if not texts or not all(isinstance(t, str) for t in texts) or not any(t.strip() for t in texts):
        raise ValueError("Evaluation texts must contain at least one non-empty string")
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
            doc_ids = tokenizer.encode(text, split_special_tokens=True)
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
        ppl = math.exp(min(avg_nats, 50.0))
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
    if isinstance(path, str):
        (clean_path,) = strip_required_paths(checkpoint_path=path)
        out = Path(clean_path)
    elif isinstance(path, Path) and str(path).strip():
        out = path
    else:
        raise ValueError("checkpoint_path must be non-empty")
    if out.is_dir() or (out.is_symlink() and not out.exists()):
        raise ValueError(f"Checkpoint output path {str(out)!r} must be a file path, not a directory")
    validate_output_dir(out.parent)
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


def load_checkpoint_payload(
    path: str | Path,
    override_seq_len: int | None = None,
    override_rope_scale: float | None = None,
    override_rope_scaling_type: str | None = None,
) -> tuple[dict[str, Any], CroutonConfig, int]:
    """Load checkpoint payload, validated CroutonConfig, and model state schema on CPU before GPU/network setup."""
    if isinstance(path, str):
        (clean_path,) = strip_required_paths(checkpoint_path=path)
        ckpt_path: str | Path = clean_path
    elif isinstance(path, Path) and str(path).strip():
        ckpt_path = path
    else:
        raise ValueError("checkpoint_path must be non-empty")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint {ckpt_path!r} payload must be a dict, got {type(ckpt).__name__}")
    raw_cfg = ckpt.get("config")
    raw_model = ckpt.get("model")
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"Checkpoint {ckpt_path!r} is missing a valid 'config' dict")
    if not isinstance(raw_model, dict) or not raw_model or not all(isinstance(k, str) for k in raw_model):
        raise ValueError(f"Checkpoint {ckpt_path!r} is missing a valid 'model' state_dict")
    missing_cfg = [k for k in REQUIRED_CONFIG_KEYS if k not in raw_cfg]
    if missing_cfg:
        raise ValueError(
            f"Checkpoint {ckpt_path!r} CroutonConfig is missing required key(s): {', '.join(missing_cfg)}"
        )
    cfg_dict = dict(raw_cfg)
    if "rope_original_max_seq_len" not in cfg_dict:
        cfg_dict["rope_original_max_seq_len"] = cfg_dict.get("max_seq_len", 2048)
    if override_seq_len is not None:
        cfg_dict["max_seq_len"] = override_seq_len
    if override_rope_scale is not None:
        cfg_dict["rope_scale_factor"] = override_rope_scale
    if override_rope_scaling_type is not None:
        cfg_dict["rope_scaling_type"] = override_rope_scaling_type
    try:
        cfg = CroutonConfig(**cfg_dict)
    except TypeError as exc:
        raise ValueError(f"Checkpoint {ckpt_path!r} contains invalid CroutonConfig fields: {exc}") from exc
    clean_sd = strip_orig_mod_prefix(raw_model)
    with torch.device("meta"):
        expected_sd = CroutonForCausalLM(cfg).state_dict()
    if clean_sd.keys() != expected_sd.keys():
        missing = sorted(expected_sd.keys() - clean_sd.keys())
        unexpected = sorted(clean_sd.keys() - expected_sd.keys())
        raise ValueError(
            f"Checkpoint {ckpt_path!r} state_dict keys do not match CroutonConfig "
            f"(missing={missing[:5]}, unexpected={unexpected[:5]})"
        )
    for k, expected_t in expected_sd.items():
        actual_t = clean_sd[k]
        if not isinstance(actual_t, torch.Tensor) or tuple(actual_t.shape) != tuple(expected_t.shape):
            raise ValueError(
                f"Checkpoint {ckpt_path!r} tensor {k!r} shape mismatch: expected {tuple(expected_t.shape)}, got "
                f"{tuple(actual_t.shape) if isinstance(actual_t, torch.Tensor) else type(actual_t).__name__}"
            )
    raw_step = ckpt.get("step", 0)
    if raw_step is None:
        step = 0
    elif isinstance(raw_step, bool) or (isinstance(raw_step, float) and not raw_step.is_integer()):
        raise ValueError(f"Checkpoint {ckpt_path!r} has invalid 'step' value {raw_step!r}")
    else:
        try:
            step = int(raw_step)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Checkpoint {ckpt_path!r} has invalid 'step' value {raw_step!r}") from exc
    return ckpt, cfg, step


def instantiate_checkpoint_model(
    ckpt: dict[str, Any],
    cfg: CroutonConfig,
    device: torch.device,
) -> CroutonForCausalLM:
    """Build CroutonForCausalLM on `device` from an already-loaded checkpoint payload."""
    model = CroutonForCausalLM(cfg).to(device)
    state_dict = strip_orig_mod_prefix(ckpt["model"])
    model.load_state_dict(state_dict, strict=True)
    return model


def load_checkpoint(
    path: str | Path,
    device: torch.device,
    override_seq_len: int | None = None,
    override_rope_scale: float | None = None,
    override_rope_scaling_type: str | None = None,
) -> tuple[CroutonForCausalLM, CroutonConfig, int]:
    ckpt, cfg, step = load_checkpoint_payload(
        path,
        override_seq_len=override_seq_len,
        override_rope_scale=override_rope_scale,
        override_rope_scaling_type=override_rope_scaling_type,
    )
    return instantiate_checkpoint_model(ckpt, cfg, device), cfg, step
