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
from crouton.train.common import get_device, load_checkpoint, save_checkpoint


def run_midtrain(
    base_checkpoint: str,
    tokenizer_path: str,
    output_dir: str,
    seq_len: int = 4096,
    rope_scale_factor: float = 2.0,
    total_steps: int = 2000,
    micro_batch_size: int = 2,
    grad_accum_steps: int = 16,
    muon_lr: float = 1e-2,
    adamw_lr: float = 1.5e-4,
) -> None:
    device = get_device()
    use_bf16 = device.type == "cuda"
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    model, model_cfg, _ = load_checkpoint(
        base_checkpoint,
        device=device,
        override_seq_len=seq_len,
        override_rope_scale=rope_scale_factor,
    )
    tokenizer = WpBpePpigTokenizer.from_pretrained(tokenizer_path)
    dataset = StreamingPackedDataset(
        tokenizer=tokenizer,
        seq_len=seq_len,
        sources=[
            ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "train", "text"),
            ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
        ],
    )
    loader = iter(DataLoader(dataset, batch_size=micro_batch_size))

    muon_opt, adamw_opt = build_optimizers(model, muon_lr=muon_lr, adamw_lr=adamw_lr)
    model.train()

    for step in range(total_steps):
        # Pure WSD decay phase from 1.0 -> 0.05
        mult = wsd_lr_multiplier(
            step=step,
            total_steps=total_steps,
            warmup_steps=50,
            decay_steps=total_steps - 50,
            min_lr_ratio=0.05,
        )
        set_optimizer_lr(muon_opt, muon_lr, mult)
        set_optimizer_lr(adamw_opt, adamw_lr, mult)

        muon_opt.zero_grad(set_to_none=True)
        adamw_opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(grad_accum_steps):
            x, y = next(loader)
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
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    run_midtrain(
        base_checkpoint=args.base_checkpoint,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        rope_scale_factor=args.rope_scale,
        total_steps=args.steps,
    )


if __name__ == "__main__":
    main()
