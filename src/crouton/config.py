"""Configuration dataclasses for crouton models and pretraining."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import yaml


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
    use_flex_attention: bool = True

    def __post_init__(self) -> None:
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
        ):
            raise ValueError("norm_eps must be > 0 and logit_softcap / router_jitter must be finite >= 0")
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
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_yaml(cls, path: str | Path) -> CroutonConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Expected YAML mapping in {path}, got {type(data).__name__}")
        try:
            return cls(**data)
        except TypeError as exc:
            raise ValueError(f"Invalid CroutonConfig fields in {path}: {exc}") from exc


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
