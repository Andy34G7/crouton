"""Stage 1 Post-Training: Annealing / Mid-training with 4096 context extension."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
from pathlib import Path

import torch
from tokenisers import WpBpePpigTokenizer
from torch.utils.data import DataLoader

from crouton.data import StreamingPackedDataset
from crouton.optim import build_optimizers, set_optimizer_lr, wsd_lr_multiplier
from crouton.train.common import (
    bf16_supported,
    get_device,
    instantiate_checkpoint_model,
    load_checkpoint_payload,
    next_or_restart,
    save_checkpoint,
    validate_tokenizer_vocab,
)


def _validate_midtrain_args(
    seq_len: int,
    total_steps: int,
    micro_batch_size: int,
    grad_accum_steps: int,
    warmup_steps: int,
    save_interval: int,
    muon_lr: float,
    adamw_lr: float,
    rope_scale_factor: float,
    dataset_name: str | None,
    subset: str | None,
    split: str,
    text_field: str,
    local_file: str | None,
) -> None:
    if seq_len < 1 or total_steps < 1 or micro_batch_size < 1 or grad_accum_steps < 1:
        raise ValueError("seq_len, total_steps, micro_batch_size, and grad_accum_steps must all be >= 1")
    if (
        warmup_steps < 0
        or save_interval < 0
        or not math.isfinite(muon_lr)
        or muon_lr <= 0.0
        or not math.isfinite(adamw_lr)
        or adamw_lr <= 0.0
        or not math.isfinite(rope_scale_factor)
        or rope_scale_factor <= 0.0
    ):
        raise ValueError(
            "warmup_steps and save_interval must be >= 0; muon_lr, adamw_lr, and rope_scale_factor must be finite > 0"
        )
    if dataset_name is not None and (not isinstance(dataset_name, str) or not dataset_name.strip()):
        raise ValueError("dataset_name must be non-empty when specified")
    if not isinstance(split, str) or not split.strip() or not isinstance(text_field, str) or not text_field.strip():
        raise ValueError("split and text_field must be non-empty")
    if subset is not None and not dataset_name:
        raise ValueError("subset requires dataset_name to be set")
    if local_file is not None and (dataset_name is not None or subset is not None):
        raise ValueError("local_file cannot be combined with dataset_name or subset")


def run_midtrain(
    base_checkpoint: str,
    tokenizer_path: str,
    output_dir: str,
    seq_len: int = 4096,
    rope_scale_factor: float = 2.0,
    rope_scaling_type: str = "yarn",
    total_steps: int = 2000,
    warmup_steps: int = 0,
    micro_batch_size: int = 2,
    grad_accum_steps: int = 16,
    muon_lr: float = 1e-2,
    adamw_lr: float = 1.5e-4,
    dataset_name: str | None = None,
    subset: str | None = None,
    split: str = "train",
    text_field: str = "text",
    local_file: str | None = None,
    bf16: bool = True,
    save_interval: int = 0,
    seed: int = 42,
) -> None:
    _validate_midtrain_args(
        seq_len=seq_len,
        total_steps=total_steps,
        micro_batch_size=micro_batch_size,
        grad_accum_steps=grad_accum_steps,
        warmup_steps=warmup_steps,
        save_interval=save_interval,
        muon_lr=muon_lr,
        adamw_lr=adamw_lr,
        rope_scale_factor=rope_scale_factor,
        dataset_name=dataset_name,
        subset=subset,
        split=split,
        text_field=text_field,
        local_file=local_file,
    )

    local_texts = None
    if local_file is not None:
        local_texts = [line.strip() for line in Path(local_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not local_texts:
            raise ValueError(f"Local mid-training file {local_file!r} contains no non-empty lines")

    torch.manual_seed(seed)
    device = get_device()
    use_bf16 = bf16 and bf16_supported(device)
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    ckpt, model_cfg, _ = load_checkpoint_payload(
        base_checkpoint,
        override_seq_len=seq_len,
        override_rope_scale=rope_scale_factor,
        override_rope_scaling_type=rope_scaling_type,
    )
    tokenizer = WpBpePpigTokenizer.from_pretrained(tokenizer_path)
    validate_tokenizer_vocab(tokenizer, model_cfg)
    model = instantiate_checkpoint_model(ckpt, model_cfg, device)

    sources = (
        [(dataset_name, subset, split, text_field)]
        if dataset_name
        else [
            ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "train", "text"),
            ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
        ]
    )
    dataset = StreamingPackedDataset(
        tokenizer=tokenizer,
        seq_len=seq_len,
        sources=sources,
        local_texts=local_texts,
        shuffle_seed=seed + 1000,
    )
    dataloader = DataLoader(dataset, batch_size=micro_batch_size)
    loader = iter(dataloader)

    muon_opt, adamw_opt = build_optimizers(model, muon_lr=muon_lr, adamw_lr=adamw_lr)
    model.train()

    for step in range(total_steps):
        mult = wsd_lr_multiplier(
            step=step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            decay_steps=max(1, total_steps - warmup_steps),
            min_lr_ratio=0.05,
        )
        set_optimizer_lr(muon_opt, muon_lr, mult)
        set_optimizer_lr(adamw_opt, adamw_lr, mult)

        muon_opt.zero_grad(set_to_none=True)
        adamw_opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(grad_accum_steps):
            (x, y), loader = next_or_restart(loader, dataloader)
            x, y = x.to(device), y.to(device)
            with amp_ctx:
                out = model(x, targets=y)
                loss = out.loss / grad_accum_steps
            loss.backward()
            total_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        muon_opt.step()
        adamw_opt.step()

        if step % 10 == 0:
            print(f"[midtrain step {step:5d}] ctx={seq_len} loss={total_loss:.4f} lr_mult={mult:.3f}")
        if save_interval > 0 and step > 0 and step % save_interval == 0:
            save_checkpoint(Path(output_dir) / f"step_{step}.pt", model, model_cfg, step)

    save_checkpoint(Path(output_dir) / "final.pt", model, model_cfg, total_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Annealing / Mid-training (4096 ctx).")
    parser.add_argument("--base-checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="artifacts/tokenizer/wp_bpe_ppig_32k.json")
    parser.add_argument("--output-dir", type=str, default="checkpoints/midtrain")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--rope-scale", type=float, default=2.0)
    parser.add_argument("--rope-scaling", type=str, default="yarn", choices=["yarn", "ntk"])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--muon-lr", type=float, default=1e-2)
    parser.add_argument("--adamw-lr", type=float, default=1.5e-4)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--local-file", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=0)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.local_file is not None and (
        args.dataset is not None or args.subset is not None or args.split != "train" or args.text_field != "text"
    ):
        parser.error("--local-file cannot be combined with --dataset, --subset, --split, or --text-field")
    try:
        _validate_midtrain_args(
            seq_len=args.seq_len,
            total_steps=args.steps,
            micro_batch_size=args.micro_batch,
            grad_accum_steps=args.grad_accum,
            warmup_steps=args.warmup_steps,
            save_interval=args.save_interval,
            muon_lr=args.muon_lr,
            adamw_lr=args.adamw_lr,
            rope_scale_factor=args.rope_scale,
            dataset_name=args.dataset,
            subset=args.subset,
            split=args.split,
            text_field=args.text_field,
            local_file=args.local_file,
        )
    except ValueError as exc:
        parser.error(str(exc))

    dataset_name = args.dataset.strip() if args.dataset is not None else None
    subset = None if args.subset in (None, "none", "None", "") else args.subset
    run_midtrain(
        base_checkpoint=args.base_checkpoint,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        rope_scale_factor=args.rope_scale,
        rope_scaling_type=args.rope_scaling,
        total_steps=args.steps,
        warmup_steps=args.warmup_steps,
        micro_batch_size=args.micro_batch,
        grad_accum_steps=args.grad_accum,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        dataset_name=dataset_name,
        subset=subset,
        split=args.split,
        text_field=args.text_field,
        local_file=args.local_file,
        bf16=not args.no_bf16,
        save_interval=args.save_interval,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
