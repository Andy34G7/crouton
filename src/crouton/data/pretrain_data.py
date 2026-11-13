"""Pretraining and mid-training streaming data loader with sequence packing."""

from __future__ import annotations

from typing import Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info


def resolve_token_id(tokenizer, token: str, fallback: int = 0) -> int:
    tid = tokenizer.token_to_id(token)
    if tid is not None:
        return int(tid)
    encoded = tokenizer.encode(token)
    return int(encoded[0]) if encoded else fallback


class StreamingPackedDataset(IterableDataset):
    """
    Streams text documents from HuggingFace datasets (or local texts),
    encodes with wp-bpe-ppig, appends EOS, and packs into fixed (seq_len + 1) chunks.
    Supports multi-worker DataLoader sharding and terminates cleanly on stream exhaustion.
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
            ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "train", "text"),
        ]
        self.local_texts = (
            [t for t in local_texts if isinstance(t, str) and t.strip()]
            if local_texts is not None
            else None
        )
        self.eos_id = resolve_token_id(tokenizer, eos_token, fallback=0)

    def _iter_texts(self) -> Iterator[str]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        if self.local_texts is not None:
            if not self.local_texts:
                return
            doc_idx = 0
            while True:
                for t in self.local_texts:
                    if doc_idx % num_workers == worker_id and t.strip():
                        yield t
                    doc_idx += 1

        from datasets import load_dataset

        iters = [iter(load_dataset(name, subset, split=split, streaming=True)) for name, subset, split, _ in self.sources]
        fields = [field for _, _, _, field in self.sources]
        exhausted: set[int] = set()
        step = 0
        doc_idx = 0

        while len(exhausted) < len(iters):
            stream_i = step % len(iters)
            step += 1
            if stream_i in exhausted:
                continue
            try:
                row = next(iters[stream_i])
            except StopIteration:
                exhausted.add(stream_i)
                continue
            text = row.get(fields[stream_i], "")
            if isinstance(text, str) and text.strip():
                if doc_idx % num_workers == worker_id:
                    yield text
                doc_idx += 1

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
