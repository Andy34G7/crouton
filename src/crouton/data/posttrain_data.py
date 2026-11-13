"""Post-training data formatting for ChatML SFT and DPO/ORPO preference pairs."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


def format_chatml_sft(
    messages: list[dict[str, str]],
    tokenizer,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a multi-turn conversation in ChatML format with assistant-only loss masking.
    System, user, and role headers receive target=-100.
    """
    input_ids: list[int] = []
    targets: list[int] = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        header_ids = tokenizer.encode(f"<|im_start|>{role}\n")
        body_ids = tokenizer.encode(f"{content}<|im_end|>\n")

        input_ids.extend(header_ids)
        targets.extend([-100] * len(header_ids))

        input_ids.extend(body_ids)
        if role == "assistant":
            targets.extend(body_ids)
        else:
            targets.extend([-100] * len(body_ids))

    input_ids = input_ids[: seq_len + 1]
    targets = targets[: seq_len + 1]

    # Shift for next-token prediction
    x = input_ids[:-1]
    y = targets[1:]
    if len(x) < seq_len:
        pad_len = seq_len - len(x)
        x = x + [0] * pad_len
        y = y + [-100] * pad_len

    return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class SFTDataset(Dataset):
    """In-memory SFT dataset wrapping conversations."""

    def __init__(self, examples: list[list[dict[str, str]]], tokenizer, seq_len: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return format_chatml_sft(self.examples[idx], self.tokenizer, self.seq_len)


class PreferenceDataset(Dataset):
    """Dataset of (prompt, chosen, rejected) pairs for DPO/ORPO."""

    def __init__(self, pairs: list[dict[str, Any]], tokenizer, seq_len: int) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.pairs[idx]
        prompt = item["prompt"]
        c_x, c_y = format_chatml_sft(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": item["chosen"]}],
            self.tokenizer,
            self.seq_len,
        )
        r_x, r_y = format_chatml_sft(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": item["rejected"]}],
            self.tokenizer,
            self.seq_len,
        )
        return {
            "chosen_ids": c_x,
            "chosen_labels": c_y,
            "rejected_ids": r_x,
            "rejected_labels": r_y,
        }
