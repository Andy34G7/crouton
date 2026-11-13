"""Stage 3b Post-Training (Optional): GRPO for verifiable math/reasoning tasks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from tokenisers import WpBpePpigTokenizer
from torch.optim import AdamW

from crouton.train.common import get_device, load_checkpoint, save_checkpoint


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
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
    ratio = torch.exp(policy_logps - old_logps)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    pg_loss = -torch.min(unclipped, clipped).mean()

    # Unbiased Schulman KL estimator: exp(ref - policy) - (ref - policy) - 1
    log_r = ref_logps - policy_logps
    kl = (torch.exp(log_r) - log_r - 1.0).mean()
    return pg_loss + kl_beta * kl


@torch.no_grad()
def sample_completion(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
) -> torch.Tensor:
    ids = prompt_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(ids).logits[:, -1, :] / max(temperature, 1e-5)
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_tok], dim=-1)
    return ids


def run_grpo(
    base_checkpoint: str,
    tokenizer_path: str,
    output_dir: str,
    prompts_with_verifiers: list[tuple[str, Callable[[str], float]]],
    group_size: int = 4,
    steps: int = 200,
    lr: float = 1e-6,
) -> None:
    device = get_device()
    policy, model_cfg, _ = load_checkpoint(base_checkpoint, device=device)
    ref_model, _, _ = load_checkpoint(base_checkpoint, device=device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    tokenizer = WpBpePpigTokenizer.from_pretrained(tokenizer_path)
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
            seq = sample_completion(policy, p_ids, max_new_tokens=128)
            completions.append(seq)
            text = tokenizer.decode(seq[0, p_len:].tolist())
            rewards.append(verifier(text))

            with torch.no_grad():
                p_logits = F.log_softmax(policy(seq[:, :-1]).logits[:, p_len - 1 :, :], dim=-1)
                r_logits = F.log_softmax(ref_model(seq[:, :-1]).logits[:, p_len - 1 :, :], dim=-1)
                tgt = seq[:, p_len:].unsqueeze(-1)
                old_logps.append(torch.gather(p_logits, -1, tgt).sum())
                ref_logps.append(torch.gather(r_logits, -1, tgt).sum())

        policy.train()
        opt.zero_grad(set_to_none=True)
        cur_logps = []
        for seq in completions:
            logits = F.log_softmax(policy(seq[:, :-1]).logits[:, p_len - 1 :, :], dim=-1)
            tgt = seq[:, p_len:].unsqueeze(-1)
            cur_logps.append(torch.gather(logits, -1, tgt).sum())

        loss = grpo_step_loss(
            policy_logps=torch.stack(cur_logps),
            old_logps=torch.stack(old_logps),
            ref_logps=torch.stack(ref_logps),
            rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
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
    args = parser.parse_args()
    print(f"GRPO runner ready for checkpoint={args.base_checkpoint}; supply task verifiers via run_grpo().")


if __name__ == "__main__":
    main()
