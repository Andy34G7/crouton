"""Configuration dataclasses for crouton models and pretraining."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
from typing import Any

import yaml

REQUIRED_CONFIG_KEYS = (
    "vocab_size",
    "d_model",
    "n_layers",
    "n_heads",
    "n_kv_heads",
    "d_ff",
    "n_experts",
    "expert_rank",
    "max_seq_len",
)

MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "tiny": {
        "d_model": 256,
        "n_layers": 8,
        "n_heads": 4,
        "n_kv_heads": 2,
        "d_ff": 768,
        "n_experts": 4,
        "expert_rank": 16,
        "expert_alpha": 16.0,
    },
    "small": {
        "d_model": 384,
        "n_layers": 10,
        "n_heads": 6,
        "n_kv_heads": 2,
        "d_ff": 1152,
        "n_experts": 4,
        "expert_rank": 16,
        "expert_alpha": 16.0,
    },
    "base": {
        "d_model": 512,
        "n_layers": 12,
        "n_heads": 8,
        "n_kv_heads": 2,
        "d_ff": 1536,
        "n_experts": 6,
        "expert_rank": 24,
        "expert_alpha": 24.0,
    },
    "medium": {
        "d_model": 640,
        "n_layers": 16,
        "n_heads": 10,
        "n_kv_heads": 2,
        "d_ff": 1792,
        "n_experts": 6,
        "expert_rank": 24,
        "expert_alpha": 24.0,
    },
    "large": {
        "d_model": 1024,
        "n_layers": 24,
        "n_heads": 16,
        "n_kv_heads": 4,
        "d_ff": 3072,
        "n_experts": 8,
        "expert_rank": 32,
        "expert_alpha": 32.0,
    },
}

_PRESET_ALIASES: dict[str, str] = {
    "14": "tiny",
    "15": "tiny",
    "28": "small",
    "30": "small",
    "50": "base",
    "85": "medium",
    "90": "medium",
    "300": "large",
}


def _require_int_fields(obj: Any, names: tuple[str, ...]) -> None:
    for name in names:
        val = getattr(obj, name)
        if not isinstance(val, int) or isinstance(val, bool):
            raise ValueError(f"{name} must be an integer, got {type(val).__name__}")


def _require_num_fields(obj: Any, names: tuple[str, ...]) -> None:
    for name in names:
        val = getattr(obj, name)
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValueError(f"{name} must be a finite number, got {type(val).__name__}")


def _require_bool_fields(obj: Any, names: tuple[str, ...]) -> None:
    for name in names:
        val = getattr(obj, name)
        if not isinstance(val, bool):
            raise ValueError(f"{name} must be a boolean, got {type(val).__name__}")


def _clean_yaml_path(path: str | Path) -> Path:
    if isinstance(path, str):
        if not path.strip():
            raise ValueError("YAML config path must be non-empty")
        return Path(path.strip())
    if isinstance(path, Path) and str(path).strip():
        return path
    raise ValueError("YAML config path must be non-empty")


@dataclass
class CroutonConfig:
    """Scalable model configuration from tiny prototypes (~14M) to ~300M params."""

    vocab_size: int = 32768
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4
    d_ff: int = 3072
    n_experts: int = 8
    expert_rank: int = 32
    expert_alpha: float = 32.0
    max_seq_len: int = 2048
    sliding_window: int = 512
    sink_tokens: int = 4
    global_layer_interval: int = 4
    rope_theta: float = 10000.0
    rope_scale_factor: float = 1.0
    rope_scaling_type: str = "yarn"  # "yarn" | "ntk"
    rope_original_max_seq_len: int = 2048
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    qk_norm: bool = True
    logit_softcap: float = 0.0
    router_jitter: float = 0.0
    router_z_loss_coef: float = 0.0
    expert_b_init_std: float = 0.0
    use_flex_attention: bool = True

    def __post_init__(self) -> None:
        _require_int_fields(
            self,
            (
                "vocab_size",
                "d_model",
                "n_layers",
                "n_heads",
                "n_kv_heads",
                "d_ff",
                "n_experts",
                "expert_rank",
                "max_seq_len",
                "sliding_window",
                "sink_tokens",
                "global_layer_interval",
                "rope_original_max_seq_len",
            ),
        )
        _require_num_fields(
            self,
            (
                "expert_alpha",
                "rope_theta",
                "rope_scale_factor",
                "norm_eps",
                "logit_softcap",
                "router_jitter",
                "router_z_loss_coef",
                "expert_b_init_std",
            ),
        )
        _require_bool_fields(self, ("tie_embeddings", "qk_norm", "use_flex_attention"))
        if not isinstance(self.rope_scaling_type, str):
            raise ValueError(f"rope_scaling_type must be a string, got {type(self.rope_scaling_type).__name__}")
        if (
            self.vocab_size < 1
            or self.d_model < 1
            or self.n_layers < 1
            or self.n_heads < 1
            or self.n_kv_heads < 1
            or self.d_ff < 1
        ):
            raise ValueError("vocab_size, d_model, n_layers, n_heads, n_kv_heads, and d_ff must all be >= 1")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be even for RoPE")
        if self.global_layer_interval < 1:
            raise ValueError("global_layer_interval must be >= 1")
        if self.sliding_window < 1 or self.sink_tokens < 0:
            raise ValueError("sliding_window must be >= 1 and sink_tokens >= 0")
        if self.expert_rank < 1 or self.n_experts < 1 or not math.isfinite(self.expert_alpha) or self.expert_alpha <= 0.0:
            raise ValueError("expert_rank and n_experts must be >= 1 and expert_alpha must be > 0")
        if (
            not math.isfinite(self.norm_eps)
            or self.norm_eps <= 0.0
            or not math.isfinite(self.logit_softcap)
            or self.logit_softcap < 0.0
            or not math.isfinite(self.router_jitter)
            or self.router_jitter < 0.0
            or self.router_jitter >= 1.0
            or not math.isfinite(self.router_z_loss_coef)
            or self.router_z_loss_coef < 0.0
            or not math.isfinite(self.expert_b_init_std)
            or self.expert_b_init_std < 0.0
        ):
            raise ValueError(
                "norm_eps must be > 0, logit_softcap/router_z_loss_coef/expert_b_init_std finite >= 0, "
                "and router_jitter finite in [0, 1)"
            )
        if self.max_seq_len < 1:
            raise ValueError(f"max_seq_len ({self.max_seq_len}) must be >= 1")
        if (
            not math.isfinite(self.rope_theta)
            or self.rope_theta <= 1.0
            or not math.isfinite(self.rope_scale_factor)
            or self.rope_scale_factor <= 0.0
            or self.rope_original_max_seq_len < 1
        ):
            raise ValueError("rope_theta must be finite > 1, rope_scale_factor finite > 0, and rope_original_max_seq_len >= 1")
        if self.rope_scaling_type not in ("yarn", "ntk"):
            raise ValueError(f"Unsupported rope_scaling_type: {self.rope_scaling_type!r}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def is_global_layer(self, layer_idx: int) -> bool:
        """Use global attention every `global_layer_interval` layers and on the final layer."""
        return (layer_idx % self.global_layer_interval == self.global_layer_interval - 1) or (
            layer_idx == self.n_layers - 1
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        out = _clean_yaml_path(path)
        if out.is_dir():
            raise ValueError(f"YAML output path {str(out)!r} must be a file path, not a directory")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: str | Path) -> CroutonConfig:
        src = _clean_yaml_path(path)
        try:
            data = yaml.safe_load(src.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Expected YAML mapping in {path}, got {type(data).__name__}")
        valid_fields = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - valid_fields)
        if unknown:
            raise ValueError(f"Invalid CroutonConfig fields in {path}: unexpected key(s) {', '.join(unknown)}")
        missing = [k for k in REQUIRED_CONFIG_KEYS if k not in data]
        if missing:
            raise ValueError(f"Missing required CroutonConfig keys in {path}: {', '.join(missing)}")
        cfg_data = dict(data)
        if "rope_original_max_seq_len" not in cfg_data:
            cfg_data["rope_original_max_seq_len"] = cfg_data["max_seq_len"]
        try:
            return cls(**cfg_data)
        except TypeError as exc:
            raise ValueError(f"Invalid CroutonConfig fields in {path}: {exc}") from exc

    @classmethod
    def from_preset(cls, name: str, **kwargs: Any) -> CroutonConfig:
        """Construct a CroutonConfig matching shortcake size classes: tiny, small, base, medium, large."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Preset name must be a non-empty string")
        clean = name.lower().strip()
        for prefix in ("crouton-", "shortcake-"):
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
        if clean.endswith("m") and clean[:-1].isdigit():
            clean = clean[:-1]
        preset_key = _PRESET_ALIASES.get(clean, clean)
        if preset_key not in MODEL_PRESETS:
            valid = list(MODEL_PRESETS) + [f"{k}m" for k in _PRESET_ALIASES]
            raise ValueError(f"Unknown model preset {name!r}. Choose from: {valid}")
        cfg_data = dict(MODEL_PRESETS[preset_key])
        cfg_data.update(kwargs)
        if "max_seq_len" in cfg_data and "rope_original_max_seq_len" not in cfg_data:
            cfg_data["rope_original_max_seq_len"] = cfg_data["max_seq_len"]
        return cls(**cfg_data)


@dataclass
class TrainConfig:
    """Pretraining hyperparameters."""

    output_dir: str = "checkpoints/pretrain"
    tokenizer_path: str = "artifacts/tokenizer/wp_bpe_ppig_32k.json"
    seq_len: int = 2048
    micro_batch_size: int = 4
    grad_accum_steps: int = 16
    total_steps: int = 20000
    warmup_steps: int = 500
    decay_steps: int = 2000
    muon_lr: float = 2e-2
    adamw_lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    bf16: bool = True
    compile_model: bool = True
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    seed: int = 42

    def __post_init__(self) -> None:
        if (
            not isinstance(self.output_dir, str)
            or not self.output_dir.strip()
            or not isinstance(self.tokenizer_path, str)
            or not self.tokenizer_path.strip()
        ):
            raise ValueError("output_dir and tokenizer_path must be non-empty")
        self.output_dir = self.output_dir.strip()
        self.tokenizer_path = self.tokenizer_path.strip()
        _require_int_fields(
            self,
            (
                "seq_len",
                "micro_batch_size",
                "grad_accum_steps",
                "total_steps",
                "warmup_steps",
                "decay_steps",
                "log_interval",
                "eval_interval",
                "save_interval",
                "seed",
            ),
        )
        _require_num_fields(self, ("muon_lr", "adamw_lr", "min_lr_ratio", "weight_decay", "grad_clip"))
        _require_bool_fields(self, ("bf16", "compile_model"))
        if (
            self.seq_len < 1
            or self.micro_batch_size < 1
            or self.grad_accum_steps < 1
            or self.total_steps < 1
            or self.log_interval < 1
        ):
            raise ValueError(
                "seq_len, micro_batch_size, grad_accum_steps, total_steps, and log_interval must all be >= 1"
            )
        if self.warmup_steps < 0 or self.decay_steps < 0 or self.eval_interval < 0 or self.save_interval < 0:
            raise ValueError("warmup_steps, decay_steps, eval_interval, and save_interval must be >= 0")
        if (
            not math.isfinite(self.muon_lr)
            or self.muon_lr <= 0.0
            or not math.isfinite(self.adamw_lr)
            or self.adamw_lr <= 0.0
            or not math.isfinite(self.min_lr_ratio)
            or not (0.0 <= self.min_lr_ratio <= 1.0)
        ):
            raise ValueError("muon_lr and adamw_lr must be finite > 0 and min_lr_ratio must be in [0, 1]")
        if (
            not math.isfinite(self.weight_decay)
            or self.weight_decay < 0.0
            or not math.isfinite(self.grad_clip)
            or self.grad_clip <= 0.0
        ):
            raise ValueError("weight_decay must be finite >= 0 and grad_clip must be finite > 0")
