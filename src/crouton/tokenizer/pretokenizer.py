"""Pretokenization rules for crouton's wp-bpe-ppig tokenizer."""

from __future__ import annotations

from typing import Any

# Llama-3 / cl100k-style regex pattern (handles contractions, letter runs,
# 1-3 digit chunks before individual digit splitting, punctuation, and newlines).
LLAMA3_SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

DEFAULT_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|pad|>",
]


def build_crouton_pretokenizer(individual_digits: bool = True) -> dict[str, Any]:
    """
    Build a Hugging Face Sequence pretokenizer config compatible with `tokenisers`.

    Combines a Llama-3-style regex Split (`behavior="Isolated"`) with optional
    individual digit splitting (`Digits(individual_digits=True)`).
    """
    pretokenizers: list[dict[str, Any]] = [
        {
            "type": "Split",
            "pattern": {"Regex": LLAMA3_SPLIT_PATTERN},
            "behavior": "Isolated",
            "invert": False,
        }
    ]
    if individual_digits:
        pretokenizers.append(
            {
                "type": "Digits",
                "individual_digits": True,
            }
        )
    return {
        "hf": {
            "type": "Sequence",
            "pretokenizers": pretokenizers,
        }
    }
