"""Train a 32k wp-bpe-ppig tokenizer with custom pretokenization rules."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from tokenisers import WpBpePpigTokenizer

from crouton.tokenizer.pretokenizer import DEFAULT_SPECIAL_TOKENS, build_crouton_pretokenizer
from crouton.train.common import looks_like_dir_path, normalize_optional_subset, validate_sequence_arg


DEFAULT_TOKENIZER_SOURCES = [
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
    ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "train", "text"),
]
MIN_VOCAB_SIZE = 256 + sum(len(t.encode("utf-8")) > 1 for t in DEFAULT_SPECIAL_TOKENS)


def iter_hf_corpus(
    sources: list[tuple[str, str | None, str, str]],
    max_docs: int,
) -> list[str]:
    validate_sequence_arg(sources, "sources")
    if not sources:
        raise ValueError("sources must be non-empty")
    if max_docs < 1:
        raise ValueError(f"max_docs ({max_docs}) must be >= 1")
    from datasets import load_dataset

    iters = [iter(load_dataset(name, subset, split=split, streaming=True)) for name, subset, split, _ in sources]
    fields = [field for _, _, _, field in sources]

    texts: list[str] = []
    idx = 0
    exhausted = set()
    while len(texts) < max_docs and len(exhausted) < len(iters):
        stream_i = idx % len(iters)
        idx += 1
        if stream_i in exhausted:
            continue
        try:
            row = next(iters[stream_i])
        except StopIteration:
            exhausted.add(stream_i)
            continue
        val = row.get(fields[stream_i], "")
        if isinstance(val, str) and val.strip():
            texts.append(val)
    return texts


def read_local_corpus(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line.strip()]


def _validate_tokenizer_output_path(output_path: str | Path) -> Path:
    out_str = str(output_path).strip() if isinstance(output_path, (str, Path)) else ""
    if not out_str:
        raise ValueError("output_path must be non-empty")
    out = Path(out_str)
    for parent in out.parents:
        if (parent.exists() and not parent.is_dir()) or (parent.is_symlink() and not parent.exists()):
            raise ValueError(f"Tokenizer output parent path {str(parent)!r} exists and is not a directory")
    if out.suffix == ".json":
        if (isinstance(output_path, str) and looks_like_dir_path(out_str)) or out.is_dir() or (
            out.is_symlink() and not out.exists()
        ):
            raise ValueError(f"Tokenizer JSON output path {out_str!r} must be a file path, not a directory")
    elif (out.exists() and not out.is_dir()) or (out.is_symlink() and not out.exists()):
        raise ValueError(f"Tokenizer output directory {out_str!r} exists and is not a directory")
    return out


def train_crouton_tokenizer(
    corpus: list[str],
    output_path: str | Path,
    vocab_size: int = 32768,
    length_penalty: float = 0.35,
    min_frequency: int = 2,
    individual_digits: bool = True,
) -> WpBpePpigTokenizer:
    """Train WpBpePpigTokenizer with crouton pretokenizer and export to `output_path`."""
    out = _validate_tokenizer_output_path(output_path)
    if (
        vocab_size < MIN_VOCAB_SIZE
        or min_frequency < 1
        or not math.isfinite(length_penalty)
        or length_penalty < 0.0
    ):
        raise ValueError(
            f"vocab_size must be >= {MIN_VOCAB_SIZE}, min_frequency >= 1, and length_penalty finite >= 0"
        )
    if isinstance(corpus, str):
        corpus = [line for line in corpus.splitlines() if line.strip()] or ([corpus] if corpus.strip() else [])
    else:
        corpus = [t for t in corpus if isinstance(t, str) and t.strip()]
    if not corpus:
        raise ValueError("Training corpus must contain at least one non-empty document")

    pretokenizer = build_crouton_pretokenizer(individual_digits=individual_digits)
    tok = WpBpePpigTokenizer.train(
        corpus=corpus,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        length_penalty=length_penalty,
        special_tokens=DEFAULT_SPECIAL_TOKENS,
        pretokenizer=pretokenizer,
    )
    if out.suffix == ".json":
        out.parent.mkdir(parents=True, exist_ok=True)
        tok.export_hf_json(str(out))
    else:
        out.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(str(out))
    return tok


def main() -> None:
    parser = argparse.ArgumentParser(description="Train crouton wp-bpe-ppig tokenizer.")
    parser.add_argument("--output", type=str, default="artifacts/tokenizer/wp_bpe_ppig_32k.json")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--length-penalty", type=float, default=0.35)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--no-individual-digits", action="store_true")
    parser.add_argument("--local-file", type=str, default=None, help="Optional local UTF-8 text file")
    parser.add_argument("--dataset", type=str, default=None, help="Optional single HF dataset override")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--max-docs", type=int, default=50000)
    args = parser.parse_args()
    output_path = args.output.strip()
    if not output_path:
        parser.error("--output must be non-empty")
    try:
        _validate_tokenizer_output_path(output_path)
    except ValueError as exc:
        parser.error(str(exc))
    if (
        args.vocab_size < MIN_VOCAB_SIZE
        or args.min_frequency < 1
        or not math.isfinite(args.length_penalty)
        or args.length_penalty < 0.0
        or args.max_docs < 1
    ):
        parser.error(
            f"--vocab-size must be >= {MIN_VOCAB_SIZE}, --min-frequency and --max-docs >= 1, and --length-penalty finite >= 0"
        )
    if args.dataset is not None and not args.dataset.strip():
        parser.error("--dataset must be a non-empty dataset name")
    split = args.split.strip()
    text_field = args.text_field.strip()
    if not split or not text_field:
        parser.error("--split and --text-field must be non-empty")
    local_file = args.local_file.strip() if args.local_file is not None else None
    if local_file is not None:
        if not local_file:
            parser.error("--local-file must be a non-empty file path")
        if args.dataset is not None or args.subset is not None or split != "train" or text_field != "text":
            parser.error("--local-file cannot be combined with --dataset, --subset, --split, or --text-field")
    if args.dataset is None and local_file is None and (
        args.subset is not None or split != "train" or text_field != "text"
    ):
        parser.error("--subset, non-default --split, and non-default --text-field require --dataset")

    if local_file is not None:
        corpus = read_local_corpus(local_file)
        if not corpus or not any(line.strip() for line in corpus):
            raise ValueError(f"Local tokenizer corpus {local_file!r} contains no non-empty text")
    elif args.dataset is not None:
        subset = normalize_optional_subset(args.subset)
        corpus = iter_hf_corpus(
            sources=[(args.dataset.strip(), subset, split, text_field)],
            max_docs=args.max_docs,
        )
    else:
        corpus = iter_hf_corpus(sources=DEFAULT_TOKENIZER_SOURCES, max_docs=args.max_docs)

    tok = train_crouton_tokenizer(
        corpus=corpus,
        output_path=output_path,
        vocab_size=args.vocab_size,
        length_penalty=args.length_penalty,
        min_frequency=args.min_frequency,
        individual_digits=not args.no_individual_digits,
    )
    print(f"Saved wp-bpe-ppig tokenizer (vocab_size={tok.vocab_size}) to {output_path}")


if __name__ == "__main__":
    main()
