"""Post-training data formatting for ChatML SFT and DPO/ORPO preference pairs."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from crouton.data.pretrain_data import resolve_token_id


def is_chatml_messages(messages: Any) -> bool:
    """Return True when `messages` is a non-empty list of ChatML role/content dicts."""
    return (
        isinstance(messages, list)
        and bool(messages)
        and all(
            isinstance(m, dict)
            and isinstance(m.get("role"), str)
            and bool(m["role"].strip())
            and "content" in m
            and (m["content"] is None or isinstance(m["content"], str))
            for m in messages
        )
    )


def is_valid_preference_pair(item: Any) -> bool:
    """Return True when a preference record has valid prompt, chosen, and rejected fields."""
    if not isinstance(item, dict) or not {"prompt", "chosen", "rejected"} <= item.keys():
        return False
    for field in ("prompt", "chosen", "rejected"):
        val = item[field]
        if isinstance(val, list):
            if not is_chatml_messages(val):
                return False
        elif val is not None and not isinstance(val, str):
            return False
    return True


def _encode_chatml_messages(
    messages: list[dict[str, str]],
    tokenizer,
    mask_earlier_assistant_turns: bool = False,
) -> tuple[list[int], list[int]]:
    input_ids: list[int] = []
    targets: list[int] = []
    im_start_ids = tokenizer.encode("<|im_start|>")
    im_end_nl_ids = tokenizer.encode("<|im_end|>\n")
    last_assistant_idx = max(
        (i for i, m in enumerate(messages) if str(m.get("role", "")).strip() == "assistant"),
        default=-1,
    )

    for idx, msg in enumerate(messages):
        role = str(msg.get("role", "user")).strip() or "user"
        raw_content = msg.get("content")
        content = "" if raw_content is None else str(raw_content)
        header_ids = im_start_ids + tokenizer.encode(f"{role}\n", split_special_tokens=True)
        body_ids = tokenizer.encode(content, split_special_tokens=True) + im_end_nl_ids

        input_ids.extend(header_ids)
        targets.extend([-100] * len(header_ids))

        input_ids.extend(body_ids)
        train_turn = role == "assistant" and (not mask_earlier_assistant_turns or idx == last_assistant_idx)
        if train_turn:
            targets.extend(body_ids)
        else:
            targets.extend([-100] * len(body_ids))

    return input_ids, targets


def _pad_and_shift(
    input_ids: list[int],
    targets: list[int],
    seq_len: int,
    pad_id: int,
    pad_to_max: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_tokens = seq_len + 1
    if len(input_ids) > max_tokens:
        input_ids = input_ids[:max_tokens]
        targets = targets[:max_tokens]
    x = input_ids[:-1]
    y = targets[1:]
    valid_len = len(x)
    attn_mask = [1] * valid_len
    target_len = seq_len if pad_to_max else max(1, valid_len)
    if valid_len < target_len:
        pad_len = target_len - valid_len
        x = x + [pad_id] * pad_len
        y = y + [-100] * pad_len
        attn_mask = attn_mask + [0] * pad_len
    return (
        torch.tensor(x, dtype=torch.long),
        torch.tensor(y, dtype=torch.long),
        torch.tensor(attn_mask, dtype=torch.bool),
    )


def _truncate_keep_supervision(
    input_ids: list[int],
    targets: list[int],
    max_tokens: int,
) -> tuple[list[int], list[int]]:
    """Truncate sequence to max_tokens while preserving assistant supervision whenever present."""
    if len(input_ids) <= max_tokens:
        return input_ids, targets
    tail_targets = targets[-max_tokens:]
    if any(t != -100 for t in tail_targets[1:]):
        return input_ids[-max_tokens:], tail_targets
    last_sup = max((i for i, t in enumerate(targets) if t != -100), default=-1)
    if last_sup >= 0:
        end = last_sup + 1
        start = max(0, end - max_tokens)
        return input_ids[start:end], targets[start:end]
    return input_ids[:max_tokens], targets[:max_tokens]


def format_chatml_sft_with_mask(
    messages: list[dict[str, str]],
    tokenizer,
    seq_len: int,
    mask_earlier_assistant_turns: bool = False,
    pad_to_max: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode a multi-turn conversation in ChatML format with assistant-only loss masking.
    When an example exceeds `seq_len + 1` tokens, prefers keeping assistant supervision
    (left-truncating long prompts) so targets never collapse to all -100 when avoidable.
    Returns (input_ids, targets, attention_mask) each of shape [seq_len] when pad_to_max=True.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    pad_id = resolve_token_id(tokenizer, "<|pad|>", fallback=resolve_token_id(tokenizer, "<|endoftext|>", fallback=0))
    input_ids, targets = _encode_chatml_messages(messages, tokenizer, mask_earlier_assistant_turns)
    input_ids, targets = _truncate_keep_supervision(input_ids, targets, seq_len + 1)
    return _pad_and_shift(input_ids, targets, seq_len, pad_id, pad_to_max=pad_to_max)


def collate_sft_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    pad_id: int = 0,
    pad_to_multiple: int = 1,
    max_seq_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad SFT samples to batch max (optionally block-aligned for FlexAttention)."""
    if not batch:
        raise ValueError("Cannot collate empty SFT batch")
    max_len = max(1, *(int(x.size(0)) for x, _, _ in batch))
    target_len = max_len
    if pad_to_multiple > 1 and max_len >= pad_to_multiple:
        target_len = ((max_len + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
        if max_seq_len is not None:
            target_len = min(max_seq_len, target_len)
    if all(int(x.size(0)) == target_len for x, _, _ in batch):
        return (
            torch.stack([x for x, _, _ in batch], dim=0),
            torch.stack([y for _, y, _ in batch], dim=0),
            torch.stack([m for _, _, m in batch], dim=0),
        )
    bsz = len(batch)
    x_out = torch.full((bsz, target_len), pad_id, dtype=torch.long)
    y_out = torch.full((bsz, target_len), -100, dtype=torch.long)
    m_out = torch.zeros((bsz, target_len), dtype=torch.bool)
    for i, (x, y, m) in enumerate(batch):
        n = int(x.size(0))
        if n > 0:
            x_out[i, :n] = x
            y_out[i, :n] = y
            m_out[i, :n] = m
    return x_out, y_out, m_out


def collate_preference_batch(
    batch: list[dict[str, torch.Tensor]],
    pad_id: int = 0,
    pad_to_multiple: int = 1,
    max_seq_len: int | None = None,
) -> dict[str, torch.Tensor]:
    """Right-pad chosen and rejected pairs independently to batch-max lengths."""
    if not batch:
        raise ValueError("Cannot collate empty preference batch")
    c_x, c_y, c_m = collate_sft_batch(
        [(b["chosen_ids"], b["chosen_labels"], b["chosen_mask"]) for b in batch],
        pad_id=pad_id,
        pad_to_multiple=pad_to_multiple,
        max_seq_len=max_seq_len,
    )
    r_x, r_y, r_m = collate_sft_batch(
        [(b["rejected_ids"], b["rejected_labels"], b["rejected_mask"]) for b in batch],
        pad_id=pad_id,
        pad_to_multiple=pad_to_multiple,
        max_seq_len=max_seq_len,
    )
    return {
        "chosen_ids": c_x,
        "chosen_labels": c_y,
        "chosen_mask": c_m,
        "rejected_ids": r_x,
        "rejected_labels": r_y,
        "rejected_mask": r_m,
    }


class SFTDataset(Dataset):
    """In-memory SFT dataset wrapping ChatML conversations."""

    def __init__(
        self,
        examples: list[list[dict[str, str]]],
        tokenizer,
        seq_len: int,
        pad_to_max: bool = True,
    ) -> None:
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_to_max = pad_to_max
        self.pad_id = resolve_token_id(
            tokenizer, "<|pad|>", fallback=resolve_token_id(tokenizer, "<|endoftext|>", fallback=0)
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return format_chatml_sft_with_mask(
            self.examples[idx],
            self.tokenizer,
            self.seq_len,
            pad_to_max=self.pad_to_max,
        )


class PreferenceDataset(Dataset):
    """Dataset of (prompt, chosen, rejected) pairs for DPO/ORPO, preserving multi-turn history."""

    def __init__(
        self,
        pairs: list[dict[str, Any]],
        tokenizer,
        seq_len: int,
        pad_to_max: bool = True,
    ) -> None:
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_to_max = pad_to_max
        self.pad_id = resolve_token_id(
            tokenizer, "<|pad|>", fallback=resolve_token_id(tokenizer, "<|endoftext|>", fallback=0)
        )

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _to_messages(prompt: str | list[dict[str, str]], response: str | list[dict[str, str]]) -> list[dict[str, str]]:
        if prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError("Preference prompt must be a string, None, or non-empty ChatML message list")
        if response is not None and not isinstance(response, (str, list)):
            raise ValueError("Preference response must be a string, None, or non-empty ChatML message list")
        if isinstance(prompt, list) and not is_chatml_messages(prompt):
            raise ValueError("Preference prompt message list must be a non-empty list of {'role', 'content'} dicts")
        if isinstance(response, list):
            if not is_chatml_messages(response):
                raise ValueError("Preference response message list must be a non-empty list of {'role', 'content'} dicts")
            if is_chatml_messages(prompt) and response[: len(prompt)] != prompt:
                return [*prompt, *response]
            if isinstance(prompt, str) and prompt and str(response[0].get("role", "")).strip() == "assistant":
                return [{"role": "user", "content": prompt}, *response]
            return response
        if is_chatml_messages(prompt):
            return [*prompt, {"role": "assistant", "content": "" if response is None else str(response)}]
        return [
            {"role": "user", "content": "" if prompt is None else str(prompt)},
            {"role": "assistant", "content": "" if response is None else str(response)},
        ]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.pairs[idx]
        c_msgs = self._to_messages(item["prompt"], item["chosen"])
        r_msgs = self._to_messages(item["prompt"], item["rejected"])

        common_k = 0
        while common_k < len(c_msgs) and common_k < len(r_msgs) and c_msgs[common_k] == r_msgs[common_k]:
            common_k += 1

        if common_k == len(c_msgs) == len(r_msgs):
            c_x, c_y, c_mask = format_chatml_sft_with_mask(
                c_msgs,
                self.tokenizer,
                self.seq_len,
                mask_earlier_assistant_turns=True,
                pad_to_max=self.pad_to_max,
            )
            r_x, r_y, r_mask = c_x.clone(), c_y.clone(), c_mask.clone()
        elif common_k == 0:
            # No shared prefix exists (divergent prompts); truncate each side independently keeping assistant labels.
            c_x, c_y, c_mask = format_chatml_sft_with_mask(
                c_msgs,
                self.tokenizer,
                self.seq_len,
                mask_earlier_assistant_turns=True,
                pad_to_max=self.pad_to_max,
            )
            r_x, r_y, r_mask = format_chatml_sft_with_mask(
                r_msgs,
                self.tokenizer,
                self.seq_len,
                mask_earlier_assistant_turns=True,
                pad_to_max=self.pad_to_max,
            )
        else:
            shared_ids, _ = _encode_chatml_messages(c_msgs[:common_k], self.tokenizer, mask_earlier_assistant_turns=True)
            c_tail = c_msgs[common_k:]
            r_tail = r_msgs[common_k:]

            is_single_asst = (
                len(c_tail) == 1
                and len(r_tail) == 1
                and str(c_tail[0].get("role", "")).strip() == str(r_tail[0].get("role", "")).strip() == "assistant"
            )
            if is_single_asst:
                hdr = self.tokenizer.encode("<|im_start|>") + self.tokenizer.encode("assistant\n", split_special_tokens=True)
                shared_ids.extend(hdr)
                c_raw = "" if c_tail[0].get("content") is None else str(c_tail[0].get("content"))
                r_raw = "" if r_tail[0].get("content") is None else str(r_tail[0].get("content"))
                im_end_nl_ids = self.tokenizer.encode("<|im_end|>\n")
                c_comp_ids = self.tokenizer.encode(c_raw, split_special_tokens=True) + im_end_nl_ids
                r_comp_ids = self.tokenizer.encode(r_raw, split_special_tokens=True) + im_end_nl_ids
                c_comp_tgt = list(c_comp_ids)
                r_comp_tgt = list(r_comp_ids)
            else:
                c_comp_ids, c_comp_tgt = _encode_chatml_messages(c_tail, self.tokenizer, mask_earlier_assistant_turns=True)
                r_comp_ids, r_comp_tgt = _encode_chatml_messages(r_tail, self.tokenizer, mask_earlier_assistant_turns=True)

            # Synchronize shared-prompt truncation across chosen/rejected so both condition on identical prompt tokens.
            max_tokens = self.seq_len + 1
            max_comp = max(len(c_comp_ids), len(r_comp_ids))
            if len(shared_ids) + max_comp > max_tokens:
                keep_prompt = max(0, max_tokens - max_comp)
                if keep_prompt == 0 and shared_ids and max_comp >= max_tokens:
                    keep_prompt = min(len(shared_ids), max_tokens // 2)
                comp_budget = max_tokens - keep_prompt
                if len(c_comp_ids) > comp_budget:
                    if is_single_asst:
                        c_comp_ids, c_comp_tgt = c_comp_ids[:comp_budget], c_comp_tgt[:comp_budget]
                    else:
                        c_comp_ids, c_comp_tgt = _truncate_keep_supervision(c_comp_ids, c_comp_tgt, comp_budget)
                if len(r_comp_ids) > comp_budget:
                    if is_single_asst:
                        r_comp_ids, r_comp_tgt = r_comp_ids[:comp_budget], r_comp_tgt[:comp_budget]
                    else:
                        r_comp_ids, r_comp_tgt = _truncate_keep_supervision(r_comp_ids, r_comp_tgt, comp_budget)
                shared_ids = shared_ids[-keep_prompt:] if keep_prompt > 0 else []

            shared_tgt = [-100] * len(shared_ids)
            c_x, c_y, c_mask = _pad_and_shift(
                shared_ids + c_comp_ids, shared_tgt + c_comp_tgt, self.seq_len, self.pad_id, pad_to_max=self.pad_to_max
            )
            r_x, r_y, r_mask = _pad_and_shift(
                shared_ids + r_comp_ids, shared_tgt + r_comp_tgt, self.seq_len, self.pad_id, pad_to_max=self.pad_to_max
            )

        return {
            "chosen_ids": c_x,
            "chosen_labels": c_y,
            "chosen_mask": c_mask,
            "rejected_ids": r_x,
            "rejected_labels": r_y,
            "rejected_mask": r_mask,
        }
