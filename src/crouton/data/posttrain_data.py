"""Post-training data formatting for ChatML SFT and DPO/ORPO preference pairs."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from crouton.data.pretrain_data import resolve_token_id


def format_chatml_sft_with_mask(
    messages: list[dict[str, str]],
    tokenizer,
    seq_len: int,
    mask_earlier_assistant_turns: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode a multi-turn conversation in ChatML format with assistant-only loss masking.
    Returns (input_ids, targets, attention_mask) each of shape [seq_len].
    """
    pad_id = resolve_token_id(tokenizer, "<|pad|>", fallback=resolve_token_id(tokenizer, "<|endoftext|>", fallback=0))
    input_ids: list[int] = []
    targets: list[int] = []

    last_assistant_idx = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)

    for idx, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        header_ids = tokenizer.encode(f"<|im_start|>{role}\n")
        body_ids = tokenizer.encode(f"{content}<|im_end|>\n")

        input_ids.extend(header_ids)
        targets.extend([-100] * len(header_ids))

        input_ids.extend(body_ids)
        train_turn = role == "assistant" and (not mask_earlier_assistant_turns or idx == last_assistant_idx)
        if train_turn:
            targets.extend(body_ids)
        else:
            targets.extend([-100] * len(body_ids))

    input_ids = input_ids[: seq_len + 1]
    targets = targets[: seq_len + 1]

    x = input_ids[:-1]
    y = targets[1:]
    valid_len = len(x)
    attn_mask = [1] * valid_len
    if valid_len < seq_len:
        pad_len = seq_len - valid_len
        x = x + [pad_id] * pad_len
        y = y + [-100] * pad_len
        attn_mask = attn_mask + [0] * pad_len

    return (
        torch.tensor(x, dtype=torch.long),
        torch.tensor(y, dtype=torch.long),
        torch.tensor(attn_mask, dtype=torch.bool),
    )


def format_chatml_sft(
    messages: list[dict[str, str]],
    tokenizer,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible helper returning (input_ids, targets)."""
    x, y, _ = format_chatml_sft_with_mask(messages, tokenizer, seq_len)
    return x, y


class SFTDataset(Dataset):
    """In-memory SFT dataset wrapping ChatML conversations."""

    def __init__(self, examples: list[list[dict[str, str]]], tokenizer, seq_len: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return format_chatml_sft_with_mask(self.examples[idx], self.tokenizer, self.seq_len)


class PreferenceDataset(Dataset):
    """Dataset of (prompt, chosen, rejected) pairs for DPO/ORPO, preserving multi-turn history."""

    def __init__(self, pairs: list[dict[str, Any]], tokenizer, seq_len: int) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _to_messages(prompt: str | list[dict[str, str]], response: str | list[dict[str, str]]) -> list[dict[str, str]]:
        if isinstance(response, list) and all(isinstance(m, dict) and "role" in m and "content" in m for m in response):
            return response
        if isinstance(prompt, list) and all(isinstance(m, dict) and "role" in m and "content" in m for m in prompt):
            return [*prompt, {"role": "assistant", "content": str(response)}]
        return [{"role": "user", "content": str(prompt)}, {"role": "assistant", "content": str(response)}]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.pairs[idx]
        c_msgs = self._to_messages(item["prompt"], item["chosen"])
        r_msgs = self._to_messages(item["prompt"], item["rejected"])

        c_x, c_y, c_mask = format_chatml_sft_with_mask(
            c_msgs, self.tokenizer, self.seq_len, mask_earlier_assistant_turns=True
        )
        r_x, r_y, r_mask = format_chatml_sft_with_mask(
            r_msgs, self.tokenizer, self.seq_len, mask_earlier_assistant_turns=True
        )
        return {
            "chosen_ids": c_x,
            "chosen_labels": c_y,
            "chosen_mask": c_mask,
            "rejected_ids": r_x,
            "rejected_labels": r_y,
            "rejected_mask": r_mask,
        }
