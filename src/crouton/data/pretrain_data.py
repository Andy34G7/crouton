"""Pretraining and mid-training streaming data loader with sequence packing."""

from __future__ import annotations

import itertools
from typing import Iterator

import torch
from torch.utils.data import IterableDataset


class StreamingPackedDataset(IterableDataset):
    """
    Streams text documents from HuggingFace datasets (or an iterator of strings),
    encodes with wp-bpe-ppig, appends EOS, and packs into fixed (seq_len + 1) chunks.
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int,
        sources: list[tuple[str, str | None, str, str]] | None = None,
        eos_token: str = "<|endoftext|>",
        local_texts: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.sources = sources or [
            ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
        ]
        self.local_texts = local_texts
        eos_ids = tokenizer.encode(eos_token)
        self.eos_id = eos_ids[0] if eos_ids else 0

    def _iter_texts(self) -> Iterator[str]:
        if self.local_texts is not None:
            while True:
                for t in self.local_texts:
                    yield t
            return

        from datasets import load_dataset

        iters = []
        for name, subset, split, field in self.sources:
            ds = load_dataset(name, subset, split=split, streaming=True)
            iters.append((row.get(field, "") for row in ds))

        # Round-robin across sources indefinitely
        for gen in itertools.cycle(iters):
            try:
                text = next(gen)
            except StopIteration:
                continue
            if isinstance(text, str) and text.strip():
                yield text

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        buf: list[int] = []
        chunk_size = self.seq_len + 1
        for text in self._iter_texts():
            ids = self.tokenizer.encode(text)
            buf.extend(ids)
            buf.append(self.eos_id)
            while len(buf) >= chunk_size:
                chunk = buf[:chunk_size]
                buf = buf[chunk_size:]
                t = torch.tensor(chunk, dtype=torch.long)
                yield t[:-1], t[1:]
