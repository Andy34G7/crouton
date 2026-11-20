"""Prepare sharded uint16 binary pretraining datasets (shortcake-compatible 4-dataset mix)."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from tokenisers import WpBpePpigTokenizer

from crouton.data.pretrain_data import resolve_token_id

DATASETS_CONFIG: list[dict[str, Any]] = [
    {"name": "HuggingFaceTB/SmolLM-Corpus", "config": "fineweb-edu-dedup", "column": "text", "weight": 0.40},
    {"name": "HuggingFaceTB/SmolLM-Corpus", "config": "cosmopedia-v2", "column": "text", "weight": 0.30},
    {"name": "HuggingFaceTB/SmolLM-Corpus", "config": "python-edu", "column": "content", "weight": 0.20},
    {"name": "codeparrot/codeparrot-clean", "config": None, "column": "content", "weight": 0.10},
]

_WORKER_TOKENIZER: WpBpePpigTokenizer | None = None
_WORKER_EOS_ID: int = 0


def _init_worker(tokenizer_path: str, eos_id: int) -> None:
    global _WORKER_TOKENIZER, _WORKER_EOS_ID
    _WORKER_TOKENIZER = WpBpePpigTokenizer.from_pretrained(tokenizer_path)
    _WORKER_EOS_ID = eos_id


def _encode_chunk_worker(texts: list[str]) -> list[np.ndarray]:
    assert _WORKER_TOKENIZER is not None
    out: list[np.ndarray] = []
    for text in texts:
        ids = _WORKER_TOKENIZER.encode(text, split_special_tokens=True)
        if not ids:
            continue
        arr = np.empty(len(ids) + 1, dtype=np.uint16)
        arr[:-1] = ids
        arr[-1] = _WORKER_EOS_ID
        out.append(arr)
    return out


class ShardWriter:
    """Streams uint16 token arrays into fixed-size binary shards on disk with bounded RAM."""

    def __init__(self, output_dir: str | Path, prefix: str = "train", shard_size: int = 100_000_000) -> None:
        if shard_size < 1:
            raise ValueError("shard_size must be >= 1")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.shard_size = shard_size
        self.buffer = np.empty(shard_size, dtype=np.uint16)
        self.pos = 0
        self.shard_idx = 0
        self.total_tokens = 0
        self.shard_files: list[str] = []

    def add(self, tokens: np.ndarray) -> None:
        arr = np.asarray(tokens, dtype=np.uint16)
        n = len(arr)
        offset = 0
        while offset < n:
            space = self.shard_size - self.pos
            take = min(space, n - offset)
            self.buffer[self.pos : self.pos + take] = arr[offset : offset + take]
            self.pos += take
            offset += take
            self.total_tokens += take
            if self.pos == self.shard_size:
                self._flush()

    def _flush(self) -> None:
        if self.pos == 0:
            return
        filename = f"{self.prefix}_{self.shard_idx:04d}.bin"
        filepath = self.output_dir / filename
        self.buffer[: self.pos].tofile(filepath)
        self.shard_files.append(filename)
        print(f"  [Shard] Wrote {self.pos:,} tokens -> {filepath}", flush=True)
        self.shard_idx += 1
        self.pos = 0

    def finalize(self) -> list[str]:
        self._flush()
        return list(self.shard_files)


def _open_stream(target: dict[str, Any]):
    from datasets import load_dataset

    kwargs = {"name": target["config"]} if target.get("config") else {}
    return iter(load_dataset(target["name"], split="train", streaming=True, **kwargs))


def _encode_batch(
    texts: list[str],
    tokenizer: WpBpePpigTokenizer,
    eos_id: int,
    pool: ProcessPoolExecutor | None,
    workers: int,
) -> list[np.ndarray]:
    if pool is None or len(texts) < workers * 4:
        out: list[np.ndarray] = []
        for text in texts:
            ids = tokenizer.encode(text, split_special_tokens=True)
            if not ids:
                continue
            arr = np.empty(len(ids) + 1, dtype=np.uint16)
            arr[:-1] = ids
            arr[-1] = eos_id
            out.append(arr)
        return out

    chunk_size = max(1, (len(texts) + workers - 1) // workers)
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    merged: list[np.ndarray] = []
    for part in pool.map(_encode_chunk_worker, chunks):
        merged.extend(part)
    return merged


def prepare_bin_data(
    tokenizer_path: str = "artifacts/tokenizer/wp_bpe_ppig_32k.json",
    output_dir: str = "data",
    target_tokens: int = 2_000_000_000,
    shard_size: int = 100_000_000,
    batch_docs: int = 1024,
    val_ratio: float = 0.01,
    test_ratio: float = 0.005,
    max_val_tokens: int = 20_000_000,
    max_test_tokens: int = 10_000_000,
    workers: int = 0,
    seed: int = 42,
    datasets_config: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Weighted-interleaved streaming tokenization into sharded uint16 .bin files."""
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    if target_tokens < 1 or shard_size < 1 or batch_docs < 1:
        raise ValueError("target_tokens, shard_size, and batch_docs must be >= 1")

    tokenizer = WpBpePpigTokenizer.from_pretrained(tokenizer_path)
    vocab_size = int(tokenizer.vocab_size)
    if vocab_size > 65535:
        raise ValueError(f"Tokenizer vocab_size ({vocab_size}) exceeds uint16 max (65535)")
    eos_id = resolve_token_id(tokenizer, "<|endoftext|>", fallback=0)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    configs = DATASETS_CONFIG if datasets_config is None else datasets_config
    streams: list[dict[str, Any]] = []
    weights_list: list[float] = []
    for target in configs:
        label = f"{target['name']} ({target['config']})" if target.get("config") else target["name"]
        try:
            it = _open_stream(target)
            streams.append({"target": target, "iter": it, "label": label, "docs": 0})
            weights_list.append(float(target["weight"]))
            print(f"Opened stream: {label} (weight={target['weight']:.2f})", flush=True)
        except Exception as exc:
            print(f"Warning: Could not open stream '{label}': {exc}", flush=True)

    if not streams:
        raise RuntimeError("No dataset streams could be opened.")

    weights = np.array(weights_list, dtype=np.float64)
    weights /= weights.sum()

    train_writer = ShardWriter(out_dir, prefix="train", shard_size=shard_size)
    val_chunks: list[np.ndarray] = []
    test_chunks: list[np.ndarray] = []
    val_tokens = 0
    test_tokens = 0

    target_val = min(max_val_tokens, max(500_000, int(target_tokens * val_ratio)))
    target_test = min(max_test_tokens, max(250_000, int(target_tokens * test_ratio)))

    eff_workers = workers if workers > 0 else min(4, os.cpu_count() or 1)
    pool = (
        ProcessPoolExecutor(max_workers=eff_workers, initializer=_init_worker, initargs=(tokenizer_path, eos_id))
        if eff_workers > 1
        else None
    )

    t0 = time.time()
    total_docs = 0
    last_log = t0
    try:
        while train_writer.total_tokens < target_tokens and streams:
            idx = int(rng.choice(len(streams), p=weights))
            st = streams[idx]
            col = st["target"]["column"]

            texts: list[str] = []
            exhausted = False
            while len(texts) < batch_docs:
                try:
                    sample = next(st["iter"])
                except StopIteration:
                    exhausted = True
                    break
                except Exception as exc:
                    print(f"Warning: Stream error on '{st['label']}': {exc}", flush=True)
                    exhausted = True
                    break
                text = sample.get(col, "")
                if isinstance(text, str) and text.strip():
                    texts.append(text)

            if exhausted:
                print(f"Stream exhausted: {st['label']} ({st['docs']:,} docs read)", flush=True)
                streams.pop(idx)
                weights = np.delete(weights, idx)
                if len(weights) > 0:
                    weights /= weights.sum()

            if not texts:
                continue

            encoded = _encode_batch(texts, tokenizer, eos_id, pool, eff_workers)
            st["docs"] += len(encoded)
            total_docs += len(encoded)

            train_batch_arrays: list[np.ndarray] = []
            for arr in encoded:
                r = rng.random()
                if val_tokens < target_val and r < val_ratio:
                    val_chunks.append(arr)
                    val_tokens += len(arr)
                elif test_tokens < target_test and r < (val_ratio + test_ratio):
                    test_chunks.append(arr)
                    test_tokens += len(arr)
                else:
                    train_batch_arrays.append(arr)

            if train_batch_arrays:
                batch_cat = np.concatenate(train_batch_arrays)
                remaining = target_tokens - train_writer.total_tokens
                if len(batch_cat) > remaining:
                    batch_cat = batch_cat[:remaining]
                train_writer.add(batch_cat)

            now = time.time()
            if now - last_log >= 15.0:
                rate = (train_writer.total_tokens + val_tokens + test_tokens) / max(now - t0, 1e-5)
                pct = 100.0 * train_writer.total_tokens / target_tokens
                print(
                    f"[prepare] {train_writer.total_tokens:,}/{target_tokens:,} ({pct:.1f}%) "
                    f"val={val_tokens:,} docs={total_docs:,} rate={rate:,.0f} tok/s",
                    flush=True,
                )
                last_log = now
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    train_shards = train_writer.finalize()
    val_arr = np.concatenate(val_chunks) if val_chunks else np.array([], dtype=np.uint16)
    test_arr = np.concatenate(test_chunks) if test_chunks else np.array([], dtype=np.uint16)
    val_path = out_dir / "val.bin"
    test_path = out_dir / "test.bin"
    val_arr.tofile(val_path)
    test_arr.tofile(test_path)

    elapsed = time.time() - t0
    meta = {
        "vocab_size": vocab_size,
        "eos_token_id": eos_id,
        "train_tokens": int(train_writer.total_tokens),
        "val_tokens": int(len(val_arr)),
        "test_tokens": int(len(test_arr)),
        "train_shards": train_shards,
        "shard_size": shard_size,
        "total_docs": total_docs,
        "elapsed_sec": round(elapsed, 1),
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"Done in {elapsed:.1f}s | train={train_writer.total_tokens:,} ({len(train_shards)} shards) "
        f"| val={len(val_arr):,} | test={len(test_arr):,} | meta={meta_path}",
        flush=True,
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare sharded uint16 binary pretraining datasets for crouton.")
    parser.add_argument("--tokenizer", type=str, default="artifacts/tokenizer/wp_bpe_ppig_32k.json")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--target-tokens", type=int, default=2_000_000_000)
    parser.add_argument("--shard-size", type=int, default=100_000_000)
    parser.add_argument("--batch-docs", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=0, help="Parallel tokenization processes (0 = auto up to 4)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_bin_data(
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        target_tokens=args.target_tokens,
        shard_size=args.shard_size,
        batch_docs=args.batch_docs,
        workers=args.workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
