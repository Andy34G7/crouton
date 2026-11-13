"""Stage 3b Post-Training: GRPO with per-token ratios and Schulman k3 KL."""

from __future__ import annotations

import argparse
import random
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from tokenisers import WpBpePpigTokenizer
from torch.optim import AdamW

from crouton.data.pretrain_data import resolve_token_id
from crouton.train.common import (
    bf16_supported,
    get_device,
    load_checkpoint,
    safe_decode,
    save_checkpoint,
    validate_tokenizer_vocab,
)

_HASHED_NUM_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_BARE_NUM_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")


def _clean_num_str(raw: str) -> str | None:
    cleaned = raw.replace(",", "").strip()
    return cleaned if cleaned and any(ch.isdigit() for ch in cleaned) else None


def extract_numeric_answer(text: str) -> str | None:
    hashed = _HASHED_NUM_RE.findall(text)
    if hashed:
        for val in reversed(hashed):
            cleaned = _clean_num_str(val)
            if cleaned is not None:
                return cleaned
    bare = _BARE_NUM_RE.findall(text)
    for val in reversed(bare):
        cleaned = _clean_num_str(val)
        if cleaned is not None:
            return cleaned
    return None


def make_numeric_verifier(expected: str) -> Callable[[str], float]:
    target = expected.replace(",", "").strip()

    def _verify(response: str) -> float:
        pred = extract_numeric_answer(response)
        if pred is None:
            return 0.0
        try:
            return 1.0 if abs(float(pred) - float(target)) < 1e-6 else 0.0
        except ValueError:
            return 1.0 if pred == target else 0.0

    return _verify


def build_builtin_math_tasks(n: int = 128, seed: int = 42) -> list[tuple[str, Callable[[str], float]]]:
    rng = random.Random(seed)
    tasks: list[tuple[str, Callable[[str], float]]] = []
    for _ in range(n):
        a = rng.randint(10, 999)
        b = rng.randint(10, 999)
        op = rng.choice(["+", "-", "*"])
        ans = a + b if op == "+" else (a - b if op == "-" else a * b)
        prompt = (
            f"<|im_start|>user\nCompute {a} {op} {b}. End with #### <answer>.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        tasks.append((prompt, make_numeric_verifier(str(ans))))
    return tasks


def load_gsm8k_tasks(max_examples: int = 1000) -> list[tuple[str, Callable[[str], float]]]:
    if max_examples < 1:
        raise ValueError("max_examples must be >= 1")
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    tasks: list[tuple[str, Callable[[str], float]]] = []
    for row in ds:
        q = str(row.get("question", "")).strip()
        ans = extract_numeric_answer(str(row.get("answer", "")))
        if q and ans is not None:
            prompt = (
                f"<|im_start|>user\n{q}\nEnd your final answer with #### <number>.<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            tasks.append((prompt, make_numeric_verifier(ans)))
            if len(tasks) >= max_examples:
                break
    return tasks


def _grpo_elementwise_loss(
    policy_logps: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    advantage: torch.Tensor,
    clip_eps: float,
    kl_beta: float,
) -> torch.Tensor:
    ratio = torch.exp((policy_logps - old_logps).clamp(min=-20.0, max=20.0))
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    pg_loss = -torch.min(unclipped, clipped)
    # Per-token Schulman k3 KL estimator: exp(ref - policy) - (ref - policy) - 1
    log_r = (ref_logps - policy_logps).clamp(min=-20.0, max=20.0)
    kl = torch.exp(log_r) - log_r - 1.0
    return pg_loss + kl_beta * kl


def grpo_step_loss(
    policy_logps: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    rewards: torch.Tensor,
    completion_mask: torch.Tensor | None = None,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> torch.Tensor:
    """
    Compute Group Relative Policy Optimization (GRPO) loss over G completions.
    Supports per-token logprobs [G, T] with optional boolean mask [G, T] (preferred)
    or 1D [G] logprobs.
    """
    std = rewards.std(unbiased=False)
    advantages = (rewards - rewards.mean()) / (std + 1e-6) if std > 1e-8 else torch.zeros_like(rewards)

    adv = advantages.unsqueeze(-1) if policy_logps.ndim == 2 else advantages
    mask = completion_mask.float() if completion_mask is not None else torch.ones_like(policy_logps)
    per_elem = _grpo_elementwise_loss(policy_logps, old_logps, ref_logps, adv, clip_eps, kl_beta)

    if per_elem.ndim == 2:
        return ((per_elem * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)).mean()
    return (per_elem * mask).sum() / mask.sum().clamp(min=1.0)


def _gather_completion_logps(
    model: torch.nn.Module,
    seq: torch.Tensor,
    prompt_len: int,
    max_seq_len: int,
) -> torch.Tensor:
    """Return per-token log-probabilities [T_gen] for generated tokens seq[0, prompt_len:]."""
    inp = seq[:, :-1]
    tgt = seq[:, prompt_len:]
    if inp.size(1) > max_seq_len:
        offset = inp.size(1) - max_seq_len
        inp = inp[:, -max_seq_len:]
        start_logit = max(0, prompt_len - 1 - offset)
    else:
        start_logit = prompt_len - 1
    logits = model(inp).logits[:, start_logit:, :].float()
    tgt = tgt[:, -logits.size(1) :]
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, -1, tgt.unsqueeze(-1)).squeeze(-1).squeeze(0)


def run_grpo(
    base_checkpoint: str,
    tokenizer_path: str,
    output_dir: str,
    prompts_with_verifiers: list[tuple[str, Callable[[str], float]]],
    group_size: int = 4,
    steps: int = 200,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    lr: float = 1e-6,
    kl_beta: float = 0.04,
    bf16: bool = True,
    save_interval: int = 0,
    seed: int = 42,
) -> None:
    if not prompts_with_verifiers:
        raise ValueError("prompts_with_verifiers must be non-empty")
    if steps < 1 or group_size < 1 or max_new_tokens < 1:
        raise ValueError("steps, group_size, and max_new_tokens must be >= 1")
    if temperature <= 0.0 or lr <= 0.0 or kl_beta < 0.0 or save_interval < 0:
        raise ValueError("temperature and lr must be > 0, and kl_beta / save_interval must be >= 0")
    torch.manual_seed(seed)

    device = get_device()
    use_bf16 = bf16 and bf16_supported(device)
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    policy, model_cfg, _ = load_checkpoint(base_checkpoint, device=device)
    if max_new_tokens >= model_cfg.max_seq_len:
        raise ValueError(
            f"max_new_tokens ({max_new_tokens}) must be < model max_seq_len ({model_cfg.max_seq_len})"
        )
    ref_model, _, _ = load_checkpoint(base_checkpoint, device=device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    tokenizer = WpBpePpigTokenizer.from_pretrained(tokenizer_path)
    validate_tokenizer_vocab(tokenizer, model_cfg)
    eos_ids = [
        resolve_token_id(tokenizer, "<|im_end|>", fallback=0),
        resolve_token_id(tokenizer, "<|endoftext|>", fallback=0),
    ]
    opt = AdamW(policy.parameters(), lr=lr, weight_decay=0.0)

    for step in range(steps):
        prompt, verifier = prompts_with_verifiers[step % len(prompts_with_verifiers)]
        raw_p_ids = tokenizer.encode(prompt) or [eos_ids[0]]
        max_prompt = model_cfg.max_seq_len - max_new_tokens
        raw_p_ids = raw_p_ids[-max_prompt:]
        p_ids = torch.tensor([raw_p_ids], dtype=torch.long, device=device)
        p_len = p_ids.size(1)

        completions: list[torch.Tensor] = []
        rewards: list[float] = []
        ref_tok_logps: list[torch.Tensor] = []

        policy.eval()
        for _ in range(group_size):
            with amp_ctx:
                seq = policy.generate(
                    p_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    eos_token_ids=eos_ids,
                    vocab_limit=tokenizer.vocab_size,
                )
            completions.append(seq)
            text = safe_decode(tokenizer, seq[0, p_len:].tolist())
            rewards.append(verifier(text))

            with torch.no_grad(), amp_ctx:
                ref_tok_logps.append(_gather_completion_logps(ref_model, seq, p_len, model_cfg.max_seq_len))

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        std = rewards_t.std(unbiased=False)
        advantages = (rewards_t - rewards_t.mean()) / (std + 1e-6) if std > 1e-8 else torch.zeros_like(rewards_t)

        policy.train()
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        # Accumulate per-completion gradients sequentially to reduce peak activation memory by group_size x.
        for g, seq in enumerate(completions):
            with amp_ctx:
                cur_lp = _gather_completion_logps(policy, seq, p_len, model_cfg.max_seq_len)
                loss_g = _grpo_elementwise_loss(
                    cur_lp,
                    cur_lp.detach(),
                    ref_tok_logps[g],
                    advantages[g],
                    clip_eps=0.2,
                    kl_beta=kl_beta,
                ).mean()
                scaled_loss = loss_g / float(group_size)
            scaled_loss.backward()
            step_loss += float(scaled_loss.item())

        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if step % 10 == 0:
            print(f"[grpo step {step:4d}] loss={step_loss:.4f} mean_reward={sum(rewards)/len(rewards):.3f}")
        if save_interval > 0 and step > 0 and step % save_interval == 0:
            save_checkpoint(Path(output_dir) / f"step_{step}.pt", policy, model_cfg, step)

    save_checkpoint(Path(output_dir) / "final.pt", policy, model_cfg, steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3b: GRPO reasoning RL.")
    parser.add_argument("--base-checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="artifacts/tokenizer/wp_bpe_ppig_32k.json")
    parser.add_argument("--output-dir", type=str, default="checkpoints/grpo")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--kl-beta", type=float, default=0.04)
    parser.add_argument("--builtin-math", action="store_true", help="Use built-in arithmetic tasks instead of GSM8K")
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=0)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.steps < 1 or args.group_size < 1 or args.max_new_tokens < 1 or args.max_examples < 1:
        parser.error("--steps, --group-size, --max-new-tokens, and --max-examples must all be >= 1")
    if args.temperature <= 0.0 or args.lr <= 0.0 or args.kl_beta < 0.0 or args.save_interval < 0:
        parser.error("--temperature and --lr must be > 0, and --kl-beta / --save-interval must be >= 0")

    tasks = (
        build_builtin_math_tasks(n=max(args.steps, 64), seed=args.seed)
        if args.builtin_math
        else load_gsm8k_tasks(max_examples=args.max_examples)
    )
    run_grpo(
        base_checkpoint=args.base_checkpoint,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        prompts_with_verifiers=tasks,
        group_size=args.group_size,
        steps=args.steps,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        lr=args.lr,
        kl_beta=args.kl_beta,
        bf16=not args.no_bf16,
        save_interval=args.save_interval,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
