"""Configuration dataclasses for crouton models, pretraining, and post-training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be even for RoPE")
        if self.global_layer_interval < 1:
            raise ValueError("global_layer_interval must be >= 1")
        if self.expert_rank < 1 or self.n_experts < 1:
            raise ValueError("expert_rank and n_experts must be >= 1")
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
        data = yaml.safe_load(Path(path).read_text())
        return cls(**data)


@dataclass
class TrainConfig:
    """Pretraining & mid-training hyperparameters."""

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


@dataclass
class PostTrainConfig:
    """Shared config for SFT, DPO/ORPO, and GRPO."""

    base_checkpoint: str = "checkpoints/midtrain/final.pt"
    output_dir: str = "checkpoints/sft"
    tokenizer_path: str = "artifacts/tokenizer/wp_bpe_ppig_32k.json"
    seq_len: int = 4096
    micro_batch_size: int = 2
    grad_accum_steps: int = 16
    total_steps: int = 2000
    warmup_steps: int = 100
    lr: float = 2e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    beta: float = 0.1
    bf16: bool = True
    seed: int = 42
