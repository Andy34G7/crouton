"""Stage 1 Post-Training: Annealing / Mid-training with 4096 context extension."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from tokenisers import WpBpePpigTokenizer
from torch.utils.data import DataLoader

from crouton.data import StreamingPackedDataset
from crouton.optim import build_optimizers, set_optimizer_lr, wsd_lr_multiplier
from crouton.train.common import (
    bf16_supported,
    get_device,
    load_checkpoint,
    next_or_restart,
    save_checkpoint,
    validate_tokenizer_vocab,
)


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
    seed: int = 42,
) -> None:
    if seq_len < 1 or total_steps < 1 or micro_batch_size < 1 or grad_accum_steps < 1:
        raise ValueError("seq_len, total_steps, micro_batch_size, and grad_accum_steps must all be >= 1")
    torch.manual_seed(seed)
    device = get_device()
    use_bf16 = bf16_supported(device)
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    model, model_cfg, _ = load_checkpoint(
        base_checkpoint,
        device=device,
        override_seq_len=seq_len,
        override_rope_scale=rope_scale_factor,
        override_rope_scaling_type=rope_scaling_type,
    )
    tokenizer = WpBpePpigTokenizer.from_pretrained(tokenizer_path)
    validate_tokenizer_vocab(tokenizer, model_cfg)

    local_texts = None
    if local_file is not None:
        local_texts = [line.strip() for line in Path(local_file).read_text(encoding="utf-8").splitlines() if line.strip()]

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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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
        dataset_name=args.dataset,
        subset=subset,
        split=args.split,
        text_field=args.text_field,
        local_file=args.local_file,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
