"""Stage 3a Post-Training: DPO / ORPO preference optimization."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tokenisers import WpBpePpigTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader

from crouton.data import PreferenceDataset
from crouton.optim import build_adamw_param_groups, set_optimizer_lr, wsd_lr_multiplier
from crouton.train.common import (
    bf16_supported,
    get_device,
    load_checkpoint,
    next_or_restart,
    save_checkpoint,
    validate_tokenizer_vocab,
)


def sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum log-probabilities over non-masked tokens (labels != -100)."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    mask = labels != -100
    safe_labels = labels.clamp(min=0)
    token_logps = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(dim=-1)


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
    seed: int = 42,
    pairs_override: list[dict[str, Any]] | None = None,
) -> None:
    if seq_len < 1 or total_steps < 1 or micro_batch_size < 1 or grad_accum_steps < 1 or max_examples < 1:
        raise ValueError("seq_len, total_steps, micro_batch_size, grad_accum_steps, and max_examples must be >= 1")
    torch.manual_seed(seed)
    device = get_device()
    use_bf16 = bf16_supported(device)
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    policy, model_cfg, _ = load_checkpoint(base_checkpoint, device=device, override_seq_len=seq_len)
    ref_model = None
    if not use_orpo:
        ref_model, _, _ = load_checkpoint(base_checkpoint, device=device, override_seq_len=seq_len)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    tokenizer = WpBpePpigTokenizer.from_pretrained(tokenizer_path)
    validate_tokenizer_vocab(tokenizer, model_cfg)

    if pairs_override is not None:
        pairs = pairs_override
    else:
        raw = load_dataset(dataset_name, subset, split="train", streaming=True)
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

    ds = PreferenceDataset(pairs, tokenizer=tokenizer, seq_len=seq_len)
    loader = DataLoader(ds, batch_size=micro_batch_size, shuffle=True)
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
            n_pairs = int(c_ids.size(0))

            with amp_ctx:
                c_out = policy(c_ids, targets=c_lbl if use_orpo else None, attention_mask=c_mask)
                r_out = policy(r_ids, attention_mask=r_mask)
                p_c = sequence_logps(c_out.logits, c_lbl)
                p_r = sequence_logps(r_out.logits, r_lbl)

                if use_orpo:
                    loss = orpo_loss(
                        p_c,
                        p_r,
                        c_out.loss,
                        (c_lbl != -100).sum(dim=-1),
                        (r_lbl != -100).sum(dim=-1),
                        beta=beta,
                    )
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    subset = None if args.subset in (None, "none", "None", "") else args.subset
    run_preference_tuning(
        base_checkpoint=args.base_checkpoint,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        subset=subset,
        seq_len=args.seq_len,
        use_orpo=args.orpo,
        total_steps=args.steps,
        micro_batch_size=args.micro_batch,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        beta=args.beta,
        max_examples=args.max_examples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
