"""Stage 3a Post-Training: DPO / ORPO preference optimization."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import partial
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tokenisers import WpBpePpigTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader

from crouton.data import PreferenceDataset, collate_preference_batch
from crouton.optim import build_adamw_param_groups, set_optimizer_lr, wsd_lr_multiplier
from crouton.train.common import (
    bf16_supported,
    get_device,
    instantiate_checkpoint_model,
    load_checkpoint_payload,
    next_or_restart,
    save_checkpoint,
    validate_tokenizer_vocab,
)


def sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum log-probabilities over non-masked tokens (labels != -100)."""
    token_nll = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(labels)
    return -token_nll.sum(dim=-1)


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(1 - exp(x)) for x <= 0."""
    x = x.clamp(max=-1e-7)
    return torch.where(
        x > -0.6931471805599453,
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    logits = (policy_chosen_logps - policy_rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
    return -F.logsigmoid(beta * logits).mean()


def orpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    chosen_sft_loss: torch.Tensor,
    chosen_lengths: torch.Tensor,
    rejected_lengths: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Reference-free Odds Ratio Preference Optimization with stable log1mexp."""
    c_avg = policy_chosen_logps / chosen_lengths.clamp(min=1)
    r_avg = policy_rejected_logps / rejected_lengths.clamp(min=1)
    log_odds = (c_avg - r_avg) - _log1mexp(c_avg) + _log1mexp(r_avg)
    return chosen_sft_loss - beta * F.logsigmoid(log_odds).mean()


def _validate_dpo_args(
    seq_len: int,
    total_steps: int,
    micro_batch_size: int,
    grad_accum_steps: int,
    max_examples: int,
    lr: float,
    beta: float,
    save_interval: int,
    dataset_name: str,
    require_dataset: bool = True,
) -> None:
    if seq_len < 1 or total_steps < 1 or micro_batch_size < 1 or grad_accum_steps < 1 or max_examples < 1:
        raise ValueError("seq_len, total_steps, micro_batch_size, grad_accum_steps, and max_examples must be >= 1")
    if (
        not math.isfinite(lr)
        or lr <= 0.0
        or not math.isfinite(beta)
        or beta <= 0.0
        or save_interval < 0
    ):
        raise ValueError("lr and beta must be finite > 0 and save_interval must be >= 0")
    if require_dataset and (not isinstance(dataset_name, str) or not dataset_name.strip()):
        raise ValueError("dataset_name must be non-empty")


def run_preference_tuning(
    base_checkpoint: str,
    tokenizer_path: str,
    output_dir: str,
    dataset_name: str = "argilla/ultrafeedback-binarized-preferences-cleaned",
    subset: str | None = None,
    seq_len: int = 4096,
    total_steps: int = 1000,
    micro_batch_size: int = 1,
    grad_accum_steps: int = 16,
    lr: float = 5e-6,
    beta: float = 0.1,
    use_orpo: bool = False,
    max_examples: int = 20000,
    bf16: bool = True,
    save_interval: int = 0,
    seed: int = 42,
    pairs_override: list[dict[str, Any]] | None = None,
) -> None:
    _validate_dpo_args(
        seq_len=seq_len,
        total_steps=total_steps,
        micro_batch_size=micro_batch_size,
        grad_accum_steps=grad_accum_steps,
        max_examples=max_examples,
        lr=lr,
        beta=beta,
        save_interval=save_interval,
        dataset_name=dataset_name,
        require_dataset=pairs_override is None,
    )
    torch.manual_seed(seed)
    device = get_device()
    use_bf16 = bf16 and bf16_supported(device)
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    ckpt, model_cfg, _ = load_checkpoint_payload(base_checkpoint, override_seq_len=seq_len)
    tokenizer = WpBpePpigTokenizer.from_pretrained(tokenizer_path)
    validate_tokenizer_vocab(tokenizer, model_cfg)

    if pairs_override is not None:
        pairs = pairs_override
    else:
        raw = load_dataset(dataset_name.strip(), subset, split="train", streaming=True)
        pairs = []
        for row in raw:
            if "prompt" in row and "chosen" in row and "rejected" in row:
                pairs.append(
                    {
                        "prompt": row["prompt"],
                        "chosen": row["chosen"],
                        "rejected": row["rejected"],
                    }
                )
                if len(pairs) >= max_examples:
                    break

    if not pairs:
        raise ValueError(f"Preference dataset '{dataset_name}' yielded zero valid (prompt, chosen, rejected) pairs.")

    policy = instantiate_checkpoint_model(ckpt, model_cfg, device)
    ref_model = None
    if not use_orpo:
        ref_model = instantiate_checkpoint_model(ckpt, model_cfg, device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    ds = PreferenceDataset(pairs, tokenizer=tokenizer, seq_len=seq_len, pad_to_max=False)
    loader = DataLoader(
        ds,
        batch_size=micro_batch_size,
        shuffle=True,
        collate_fn=partial(collate_preference_batch, pad_id=ds.pad_id),
    )
    it = iter(loader)

    opt = AdamW(build_adamw_param_groups(policy, weight_decay=0.01), lr=lr, betas=(0.9, 0.95))
    policy.train()

    for step in range(total_steps):
        mult = wsd_lr_multiplier(
            step,
            total_steps,
            warmup_steps=min(50, max(1, total_steps // 10)),
            decay_steps=max(1, total_steps - min(50, max(1, total_steps // 10))),
            min_lr_ratio=0.1,
        )
        set_optimizer_lr(opt, lr, mult)
        opt.zero_grad(set_to_none=True)
        total_weighted_loss = 0.0
        total_pairs = 0
        for _ in range(grad_accum_steps):
            batch, it = next_or_restart(it, loader)
            c_ids = batch["chosen_ids"].to(device)
            c_lbl = batch["chosen_labels"].to(device)
            c_mask = batch["chosen_mask"].to(device)
            r_ids = batch["rejected_ids"].to(device)
            r_lbl = batch["rejected_labels"].to(device)
            r_mask = batch["rejected_mask"].to(device)
            valid_pair = ((c_lbl != -100).sum(dim=-1) > 0) & ((r_lbl != -100).sum(dim=-1) > 0)
            if not valid_pair.any():
                continue
            if not valid_pair.all():
                c_ids, c_lbl, c_mask = c_ids[valid_pair], c_lbl[valid_pair], c_mask[valid_pair]
                r_ids, r_lbl, r_mask = r_ids[valid_pair], r_lbl[valid_pair], r_mask[valid_pair]
            n_pairs = int(c_ids.size(0))

            with amp_ctx:
                c_out = policy(c_ids, attention_mask=c_mask)
                r_out = policy(r_ids, attention_mask=r_mask)
                p_c = sequence_logps(c_out.logits, c_lbl)
                p_r = sequence_logps(r_out.logits, r_lbl)

                if use_orpo:
                    c_len = (c_lbl != -100).sum(dim=-1)
                    r_len = (r_lbl != -100).sum(dim=-1)
                    chosen_sft_loss = (-p_c / c_len.clamp(min=1)).mean()
                    loss = orpo_loss(p_c, p_r, chosen_sft_loss, c_len, r_len, beta=beta)
                else:
                    with torch.no_grad():
                        ref_c = sequence_logps(ref_model(c_ids, attention_mask=c_mask).logits, c_lbl)
                        ref_r = sequence_logps(ref_model(r_ids, attention_mask=r_mask).logits, r_lbl)
                    loss = dpo_loss(p_c, p_r, ref_c, ref_r, beta=beta)

                weighted_loss = loss * float(n_pairs)

            weighted_loss.backward()
            total_weighted_loss += float(weighted_loss.item())
            total_pairs += n_pairs

        if total_pairs > 0:
            for p in policy.parameters():
                if p.grad is not None:
                    p.grad.div_(float(total_pairs))
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

        total_loss = total_weighted_loss / max(1, total_pairs)

        if step % 10 == 0:
            mode = "orpo" if use_orpo else "dpo"
            print(f"[{mode} step {step:5d}] loss={total_loss:.4f} lr_mult={mult:.3f}")
        if save_interval > 0 and step > 0 and step % save_interval == 0:
            save_checkpoint(Path(output_dir) / f"step_{step}.pt", policy, model_cfg, step)

    save_checkpoint(Path(output_dir) / "final.pt", policy, model_cfg, total_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3a: DPO / ORPO preference tuning.")
    parser.add_argument("--base-checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="artifacts/tokenizer/wp_bpe_ppig_32k.json")
    parser.add_argument("--output-dir", type=str, default="checkpoints/dpo")
    parser.add_argument("--dataset", type=str, default="argilla/ultrafeedback-binarized-preferences-cleaned")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--orpo", action="store_true", help="Use reference-free ORPO instead of DPO")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max-examples", type=int, default=20000)
    parser.add_argument("--save-interval", type=int, default=0)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        _validate_dpo_args(
            seq_len=args.seq_len,
            total_steps=args.steps,
            micro_batch_size=args.micro_batch,
            grad_accum_steps=args.grad_accum,
            max_examples=args.max_examples,
            lr=args.lr,
            beta=args.beta,
            save_interval=args.save_interval,
            dataset_name=args.dataset,
        )
    except ValueError as exc:
        parser.error(str(exc))

    subset = None if args.subset in (None, "none", "None", "") else args.subset
    run_preference_tuning(
        base_checkpoint=args.base_checkpoint,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        dataset_name=args.dataset.strip(),
        subset=subset,
        seq_len=args.seq_len,
        use_orpo=args.orpo,
        total_steps=args.steps,
        micro_batch_size=args.micro_batch,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        beta=args.beta,
        max_examples=args.max_examples,
        bf16=not args.no_bf16,
        save_interval=args.save_interval,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
