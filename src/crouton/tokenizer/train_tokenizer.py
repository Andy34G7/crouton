"""Train a 32k wp-bpe-ppig tokenizer with custom pretokenization rules."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from tokenisers import WpBpePpigTokenizer

from crouton.tokenizer.pretokenizer import DEFAULT_SPECIAL_TOKENS, build_crouton_pretokenizer


DEFAULT_TOKENIZER_SOURCES = [
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
    ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "train", "text"),
]


def iter_hf_corpus(
    sources: list[tuple[str, str | None, str, str]],
    max_docs: int,
) -> list[str]:
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
    lines = [line for line in text.splitlines() if line.strip()]
    return lines if lines else ([text] if text else [])


def train_crouton_tokenizer(
    corpus: list[str],
    output_path: str | Path,
    vocab_size: int = 32768,
    length_penalty: float = 0.35,
    min_frequency: int = 2,
    individual_digits: bool = True,
) -> WpBpePpigTokenizer:
    """Train WpBpePpigTokenizer with crouton pretokenizer and export to `output_path`."""
    if isinstance(corpus, str):
        corpus = [line for line in corpus.splitlines() if line.strip()] or [corpus]

    pretokenizer = build_crouton_pretokenizer(individual_digits=individual_digits)
    tok = WpBpePpigTokenizer.train(
        corpus=corpus,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        length_penalty=length_penalty,
        special_tokens=DEFAULT_SPECIAL_TOKENS,
        pretokenizer=pretokenizer,
    )
    out = Path(output_path)
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

    if args.local_file:
        corpus = read_local_corpus(args.local_file)
    elif args.dataset:
        subset = None if args.subset in (None, "none", "None", "") else args.subset
        corpus = iter_hf_corpus(
            sources=[(args.dataset, subset, args.split, args.text_field)],
            max_docs=args.max_docs,
        )
    else:
        corpus = iter_hf_corpus(sources=DEFAULT_TOKENIZER_SOURCES, max_docs=args.max_docs)

    tok = train_crouton_tokenizer(
        corpus=corpus,
        output_path=args.output,
        vocab_size=args.vocab_size,
        length_penalty=args.length_penalty,
        min_frequency=args.min_frequency,
        individual_digits=not args.no_individual_digits,
    )
    print(f"Saved wp-bpe-ppig tokenizer (vocab_size={tok.vocab_size}) to {args.output}")


if __name__ == "__main__":
    main()
