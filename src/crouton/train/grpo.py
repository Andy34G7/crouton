"""Stage 3b Post-Training: GRPO for verifiable math/reasoning tasks."""

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
from crouton.train.common import get_device, load_checkpoint, save_checkpoint

_NUM_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)|(-?\d+(?:\.\d+)?)")


def extract_numeric_answer(text: str) -> str | None:
    matches = _NUM_RE.findall(text)
    if not matches:
        return None
    for hashed, bare in reversed(matches):
        if hashed:
            return hashed
        if bare:
            return bare
    return None


def make_numeric_verifier(expected: str) -> Callable[[str], float]:
    target = expected.strip()

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
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    tasks: list[tuple[str, Callable[[str], float]]] = []
    for i, row in enumerate(ds):
        if i >= max_examples:
            break
        q = str(row.get("question", "")).strip()
        ans = extract_numeric_answer(str(row.get("answer", "")))
        if q and ans is not None:
            prompt = (
                f"<|im_start|>user\n{q}\nEnd your final answer with #### <number>.<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            tasks.append((prompt, make_numeric_verifier(ans)))
    return tasks


def grpo_step_loss(
    policy_logps: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    rewards: torch.Tensor,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> torch.Tensor:
    """
    Compute Group Relative Policy Optimization (GRPO) loss over G completions.
    `policy_logps`, `old_logps`, `ref_logps`, `rewards` have shape [G].
    """
    std = rewards.std()
    advantages = (rewards - rewards.mean()) / (std + 1e-6) if std > 1e-8 else torch.zeros_like(rewards)
    ratio = torch.exp(policy_logps - old_logps)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    pg_loss = -torch.min(unclipped, clipped).mean()

    log_r = ref_logps - policy_logps
    kl = (torch.exp(log_r) - log_r - 1.0).mean()
    return pg_loss + kl_beta * kl


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
) -> None:
    if not prompts_with_verifiers:
        raise ValueError("prompts_with_verifiers must be non-empty")

    device = get_device()
    use_bf16 = device.type == "cuda"
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    policy, model_cfg, _ = load_checkpoint(base_checkpoint, device=device)
    ref_model, _, _ = load_checkpoint(base_checkpoint, device=device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    tokenizer = WpBpePpigTokenizer.from_pretrained(tokenizer_path)
    eos_ids = [
        resolve_token_id(tokenizer, "<|im_end|>", fallback=0),
        resolve_token_id(tokenizer, "<|endoftext|>", fallback=0),
    ]
    opt = AdamW(policy.parameters(), lr=lr, weight_decay=0.0)

    for step in range(steps):
        prompt, verifier = prompts_with_verifiers[step % len(prompts_with_verifiers)]
        p_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        p_len = p_ids.size(1)

        completions = []
        rewards = []
        old_logps = []
        ref_logps = []

        policy.eval()
        for _ in range(group_size):
            with amp_ctx:
                seq = policy.generate(
                    p_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    eos_token_ids=eos_ids,
                )
            completions.append(seq)
            text = tokenizer.decode(seq[0, p_len:].tolist())
            rewards.append(verifier(text))

            with torch.no_grad(), amp_ctx:
                p_logits = F.log_softmax(policy(seq[:, :-1]).logits[:, p_len - 1 :, :].float(), dim=-1)
                r_logits = F.log_softmax(ref_model(seq[:, :-1]).logits[:, p_len - 1 :, :].float(), dim=-1)
                tgt = seq[:, p_len:].unsqueeze(-1)
                old_logps.append(torch.gather(p_logits, -1, tgt).sum())
                ref_logps.append(torch.gather(r_logits, -1, tgt).sum())

        policy.train()
        opt.zero_grad(set_to_none=True)
        cur_logps = []
        for seq in completions:
            with amp_ctx:
                logits = F.log_softmax(policy(seq[:, :-1]).logits[:, p_len - 1 :, :].float(), dim=-1)
                tgt = seq[:, p_len:].unsqueeze(-1)
                cur_logps.append(torch.gather(logits, -1, tgt).sum())

        loss = grpo_step_loss(
            policy_logps=torch.stack(cur_logps),
            old_logps=torch.stack(old_logps),
            ref_logps=torch.stack(ref_logps),
            rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
            kl_beta=kl_beta,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if step % 10 == 0:
            print(f"[grpo step {step:4d}] loss={loss.item():.4f} mean_reward={sum(rewards)/len(rewards):.3f}")

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
    args = parser.parse_args()

    tasks = (
        build_builtin_math_tasks(n=max(args.steps, 64))
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
    )


if __name__ == "__main__":
    main()
