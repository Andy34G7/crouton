"""Stage 2 Post-Training: Chat/Instruct SFT with token-weighted assistant loss masking."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from datasets import load_dataset
from tokenisers import WpBpePpigTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader

from crouton.data import SFTDataset
from crouton.optim import build_adamw_param_groups, set_optimizer_lr, wsd_lr_multiplier
from crouton.train.common import (
    get_device,
    load_checkpoint,
    next_or_restart,
    save_checkpoint,
    validate_tokenizer_vocab,
)

SFT_PRESETS: dict[str, tuple[str, str | None]] = {
    "smoltalk": ("HuggingFaceTB/smoltalk", "all"),
    "tulu3": ("allenai/tulu-3-sft-mixture", None),
}


def run_sft(
    base_checkpoint: str,
    tokenizer_path: str,
    output_dir: str,
    dataset_name: str = "HuggingFaceTB/smoltalk",
    subset: str | None = "all",
    seq_len: int = 4096,
    total_steps: int = 2000,
    micro_batch_size: int = 2,
    grad_accum_steps: int = 16,
    lr: float = 2e-5,
    max_examples: int = 50000,
    examples_override: list[list[dict[str, str]]] | None = None,
) -> None:
    device = get_device()
    use_bf16 = device.type == "cuda"
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    model, model_cfg, _ = load_checkpoint(
        base_checkpoint,
        device=device,
        override_seq_len=seq_len,
    )
    tokenizer = WpBpePpigTokenizer.from_pretrained(tokenizer_path)
    validate_tokenizer_vocab(tokenizer, model_cfg)

    if examples_override is not None:
        examples = examples_override
    else:
        raw = load_dataset(dataset_name, subset, split="train", streaming=True)
        examples = []
        for i, row in enumerate(raw):
            if i >= max_examples:
                break
            msgs = row.get("messages")
            if isinstance(msgs, list) and msgs:
                examples.append(msgs)

    if not examples:
        raise ValueError(f"SFT dataset '{dataset_name}' yielded zero valid 'messages' examples.")

    ds = SFTDataset(examples, tokenizer=tokenizer, seq_len=seq_len)
    loader = DataLoader(ds, batch_size=micro_batch_size, shuffle=True)
    it = iter(loader)

    opt = AdamW(build_adamw_param_groups(model, weight_decay=0.01), lr=lr, betas=(0.9, 0.95))
    model.train()

    for step in range(total_steps):
        mult = wsd_lr_multiplier(
            step,
            total_steps,
            warmup_steps=min(100, max(1, total_steps // 10)),
            decay_steps=max(1, total_steps - min(100, max(1, total_steps // 10))),
            min_lr_ratio=0.1,
        )
        set_optimizer_lr(opt, lr, mult)
        opt.zero_grad(set_to_none=True)
        total_nll = 0.0
        total_valid_tokens = 0

        for _ in range(grad_accum_steps):
            (x, y, attn_mask), it = next_or_restart(it, loader)
            x, y, attn_mask = x.to(device), y.to(device), attn_mask.to(device)
            n_valid = int((y != -100).sum().item())
            with amp_ctx:
                out = model(x, targets=y, attention_mask=attn_mask)
                sum_loss = out.loss * float(n_valid)
            sum_loss.backward()
            total_nll += float(sum_loss.item())
            total_valid_tokens += n_valid

        if total_valid_tokens > 0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(float(total_valid_tokens))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        avg_loss = total_nll / max(1, total_valid_tokens)
        if step % 10 == 0:
            print(f"[sft step {step:5d}] loss={avg_loss:.4f} valid_toks={total_valid_tokens} lr_mult={mult:.3f}")

    save_checkpoint(Path(output_dir) / "final.pt", model, model_cfg, total_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: Chat/Instruct SFT.")
    parser.add_argument("--base-checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="artifacts/tokenizer/wp_bpe_ppig_32k.json")
    parser.add_argument("--output-dir", type=str, default="checkpoints/sft")
    parser.add_argument("--preset", type=str, default=None, choices=["smoltalk", "tulu3"])
    parser.add_argument("--dataset", type=str, default="HuggingFaceTB/smoltalk")
    parser.add_argument("--subset", type=str, default="all")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-examples", type=int, default=50000)
    args = parser.parse_args()

    if args.preset is not None:
        dataset_name, subset = SFT_PRESETS[args.preset]
    else:
        dataset_name = args.dataset
        subset = None if args.subset in (None, "none", "None", "") else args.subset

    run_sft(
        base_checkpoint=args.base_checkpoint,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        dataset_name=dataset_name,
        subset=subset,
        seq_len=args.seq_len,
        total_steps=args.steps,
        micro_batch_size=args.micro_batch,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
