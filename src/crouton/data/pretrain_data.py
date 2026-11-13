"""Pretraining and mid-training streaming data loader with sequence packing."""

from __future__ import annotations

from typing import Any, Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info


def normalize_optional_subset(subset: str | None) -> str | None:
    """Normalize optional HF subset names so blank or 'none' markers map cleanly to None."""
    if subset is None:
        return None
    if not isinstance(subset, str):
        raise ValueError("subset must be a string when specified")
    cleaned = subset.strip()
    return None if cleaned in ("", "none", "None") else cleaned


def validate_sequence_arg(value: Any, name: str) -> None:
    """Require a list or tuple sequence container (rejecting bare strings, bytes, and single-use iterators)."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")


def validate_hf_sources(sources: Any) -> list[tuple[str, str | None, str, str]]:
    """Validate and normalize a sequence of (dataset_name, subset, split, text_field) tuples."""
    validate_sequence_arg(sources, "sources")
    normalized: list[tuple[str, str | None, str, str]] = []
    for entry in sources:
        if not isinstance(entry, (tuple, list)) or len(entry) != 4:
            raise ValueError("Each source entry must be a 4-tuple of (dataset_name, subset, split, text_field)")
        name, subset, split, field = entry
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(split, str)
            or not split.strip()
            or not isinstance(field, str)
            or not field.strip()
        ):
            raise ValueError("Each source entry must have non-empty strings for dataset_name, split, and text_field")
        normalized.append((name.strip(), normalize_optional_subset(subset), split.strip(), field.strip()))
    return normalized


def resolve_token_id(tokenizer, token: str, fallback: int = 0) -> int:
    tid = tokenizer.token_to_id(token)
    if tid is not None:
        return int(tid)
    encoded = tokenizer.encode(token)
    return int(encoded[0]) if len(encoded) == 1 else fallback


def resolve_pad_token_id(tokenizer) -> int:
    """Resolve <|pad|>, falling back to <|endoftext|> and finally 0."""
    return resolve_token_id(tokenizer, "<|pad|>", fallback=resolve_token_id(tokenizer, "<|endoftext|>", fallback=0))


class StreamingPackedDataset(IterableDataset):
    """
    Streams text documents from HuggingFace datasets (or cycles non-empty local_texts),
    encodes with wp-bpe-ppig, appends EOS, and packs into fixed (seq_len + 1) chunks.
    Supports multi-worker DataLoader sharding and terminates cleanly on HF stream exhaustion.
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int,
        sources: list[tuple[str, str | None, str, str]] | None = None,
        eos_token: str = "<|endoftext|>",
        local_texts: list[str] | None = None,
        shuffle_seed: int | None = None,
        shuffle_buffer: int = 1000,
    ) -> None:
        super().__init__()
        if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if not isinstance(shuffle_buffer, int) or isinstance(shuffle_buffer, bool) or shuffle_buffer < 1:
            raise ValueError("shuffle_buffer must be >= 1")
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.sources = (
            [
                ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
                ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "train", "text"),
            ]
            if sources is None
            else validate_hf_sources(sources)
        )
        if local_texts is not None:
            validate_sequence_arg(local_texts, "local_texts")
        if local_texts is None and not self.sources:
            raise ValueError("StreamingPackedDataset sources must be non-empty when local_texts is None")
        self.local_texts = (
            [t for t in local_texts if isinstance(t, str) and t.strip()]
            if local_texts is not None
            else None
        )
        self.eos_id = resolve_token_id(tokenizer, eos_token, fallback=0)
        self.shuffle_seed = shuffle_seed
        self.shuffle_buffer = shuffle_buffer
        self._iter_epoch = 0

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

        epoch_offset = (
            ((int(worker.seed) - worker_id + self._iter_epoch) % (2**31 - 1))
            if worker is not None
            else self._iter_epoch
        ) * max(1, len(self.sources))
        self._iter_epoch += 1
        iters = []
        sharded_flags = []
        for idx, (name, subset, split, _) in enumerate(self.sources):
            ds = load_dataset(name, subset, split=split, streaming=True)
            sharded = False
            if num_workers > 1 and getattr(ds, "n_shards", 1) >= num_workers:
                ds = ds.shard(num_shards=num_workers, index=worker_id)
                sharded = True
            if self.shuffle_seed is not None:
                worker_mix = worker_id if sharded else 0
                ds = ds.shuffle(
                    seed=self.shuffle_seed + epoch_offset + idx + worker_mix,
                    buffer_size=self.shuffle_buffer,
                )
            sharded_flags.append(sharded)
            iters.append(iter(ds))
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
                if sharded_flags[stream_i] or doc_idx % num_workers == worker_id:
                    yield text
                if not sharded_flags[stream_i]:
                    doc_idx += 1

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        buf: list[int] = []
        pos = 0
        chunk_size = self.seq_len + 1
        for text in self._iter_texts():
            ids = self.tokenizer.encode(text, split_special_tokens=True)
            buf.extend(ids)
            buf.append(self.eos_id)
            while len(buf) - pos >= chunk_size:
                chunk = buf[pos : pos + chunk_size]
                pos += chunk_size
                t = torch.tensor(chunk, dtype=torch.long)
                yield t[:-1], t[1:]
            if pos >= max(chunk_size, len(buf) // 2):
                buf = buf[pos:]
                pos = 0
