"""Train a 32k wp-bpe-ppig tokenizer with custom pretokenization rules."""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenisers import WpBpePpigTokenizer

from crouton.tokenizer.pretokenizer import DEFAULT_SPECIAL_TOKENS, build_crouton_pretokenizer


def iter_hf_corpus(
    dataset_name: str,
    subset: str | None,
    split: str,
    text_field: str,
    max_docs: int,
) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, subset, split=split, streaming=True)
    texts: list[str] = []
    for i, row in enumerate(ds):
        if i >= max_docs:
            break
        val = row.get(text_field, "")
        if isinstance(val, str) and val.strip():
            texts.append(val)
    return texts


def train_crouton_tokenizer(
    corpus: list[str] | str,
    output_path: str | Path,
    vocab_size: int = 32768,
    length_penalty: float = 0.35,
    min_frequency: int = 2,
    individual_digits: bool = True,
) -> WpBpePpigTokenizer:
    """Train WpBpePpigTokenizer with crouton pretokenizer and save to `output_path`."""
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
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))
    return tok


def main() -> None:
    parser = argparse.ArgumentParser(description="Train crouton wp-bpe-ppig tokenizer.")
    parser.add_argument("--output", type=str, default="artifacts/tokenizer/wp_bpe_ppig_32k.json")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--length-penalty", type=float, default=0.35)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--no-individual-digits", action="store_true")
    parser.add_argument("--local-file", type=str, default=None, help="Optional local UTF-8 text file")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", type=str, default="sample-10BT")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--max-docs", type=int, default=50000)
    args = parser.parse_args()

    if args.local_file:
        corpus: list[str] | str = Path(args.local_file).read_text(encoding="utf-8")
    else:
        corpus = iter_hf_corpus(
            dataset_name=args.dataset,
            subset=args.subset,
            split=args.split,
            text_field=args.text_field,
            max_docs=args.max_docs,
        )

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
