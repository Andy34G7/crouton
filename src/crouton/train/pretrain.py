"""Pretraining loop for crouton: Muon + AdamW, WSD schedule, bf16, grad accum."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from tokenisers import WpBpePpigTokenizer
from torch.utils.data import DataLoader

from crouton.config import CroutonConfig, TrainConfig
from crouton.data import StreamingPackedDataset
from crouton.model import CroutonForCausalLM
from crouton.optim import build_optimizers, set_optimizer_lr, wsd_lr_multiplier
from crouton.train.common import get_device, save_checkpoint


def run_pretrain(model_cfg: CroutonConfig, train_cfg: TrainConfig) -> None:
    torch.manual_seed(train_cfg.seed)
    device = get_device()
    use_bf16 = train_cfg.bf16 and device.type == "cuda"
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    tokenizer = WpBpePpigTokenizer.from_pretrained(train_cfg.tokenizer_path)
    dataset = StreamingPackedDataset(
        tokenizer=tokenizer,
        seq_len=train_cfg.seq_len,
        sources=[
            ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
            ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "train", "text"),
        ],
    )
    loader = iter(DataLoader(dataset, batch_size=train_cfg.micro_batch_size))

    model = CroutonForCausalLM(model_cfg).to(device)
    if train_cfg.compile_model and device.type == "cuda":
        model = torch.compile(model)

    muon_opt, adamw_opt = build_optimizers(
        model,
        muon_lr=train_cfg.muon_lr,
        adamw_lr=train_cfg.adamw_lr,
        weight_decay=train_cfg.weight_decay,
    )

    model.train()
    for step in range(train_cfg.total_steps):
        mult = wsd_lr_multiplier(
            step=step,
            total_steps=train_cfg.total_steps,
            warmup_steps=train_cfg.warmup_steps,
            decay_steps=train_cfg.decay_steps,
            min_lr_ratio=train_cfg.min_lr_ratio,
        )
        set_optimizer_lr(muon_opt, train_cfg.muon_lr, mult)
        set_optimizer_lr(adamw_opt, train_cfg.adamw_lr, mult)

        muon_opt.zero_grad(set_to_none=True)
        adamw_opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_entropy = 0.0

        for _ in range(train_cfg.grad_accum_steps):
            x, y = next(loader)
            x, y = x.to(device), y.to(device)
            with amp_ctx:
                out = model(x, targets=y)
                loss = out.loss / train_cfg.grad_accum_steps
            loss.backward()
            total_loss += float(loss.item())
            if out.router_entropy is not None:
                total_entropy += float(out.router_entropy.item()) / train_cfg.grad_accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        muon_opt.step()
        adamw_opt.step()

        if step % train_cfg.log_interval == 0:
            print(
                f"[pretrain step {step:6d}] loss={total_loss:.4f} "
                f"router_ent={total_entropy:.3f} lr_mult={mult:.3f}"
            )

        if step > 0 and step % train_cfg.save_interval == 0:
            save_checkpoint(Path(train_cfg.output_dir) / f"step_{step}.pt", model, model_cfg, step)

    save_checkpoint(Path(train_cfg.output_dir) / "final.pt", model, model_cfg, train_cfg.total_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain crouton.")
    parser.add_argument("--model-config", type=str, default="configs/model_300m.yaml")
    parser.add_argument("--tokenizer", type=str, default="artifacts/tokenizer/wp_bpe_ppig_32k.json")
    parser.add_argument("--output-dir", type=str, default="checkpoints/pretrain")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    model_cfg = CroutonConfig.from_yaml(args.model_config)
    train_cfg = TrainConfig(
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer,
        seq_len=model_cfg.max_seq_len,
        micro_batch_size=args.micro_batch,
        grad_accum_steps=args.grad_accum,
        total_steps=args.steps,
        compile_model=not args.no_compile,
    )
    run_pretrain(model_cfg, train_cfg)


if __name__ == "__main__":
    main()
