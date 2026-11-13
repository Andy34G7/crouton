"""Evaluation runner: Bits-per-Byte (BPB) / PPL + lm-eval-harness comparison."""

from __future__ import annotations

import argparse
import math

import torch
from tokenisers import WpBpePpigTokenizer

from crouton.train.common import get_device, load_checkpoint


@torch.no_grad()
def evaluate_bpb(
    checkpoint_path: str,
    tokenizer_path: str,
    texts: list[str],
    seq_len: int = 2048,
) -> dict[str, float]:
    """
    Compute tokenizer-agnostic Bits-Per-Byte (BPB) and Perplexity (PPL)
    for fair comparison against shortcake.
    """
    device = get_device()
    model, _, _ = load_checkpoint(checkpoint_path, device=device)
    model.eval()
    tokenizer = WpBpePpigTokenizer.from_pretrained(tokenizer_path)

    total_nll_nats = 0.0
    total_tokens = 0
    total_bytes = 0

    for text in texts:
        ids = tokenizer.encode(text)
        if len(ids) < 2:
            continue
        for start in range(0, len(ids) - 1, seq_len):
            chunk = ids[start : start + seq_len + 1]
            if len(chunk) < 2:
                continue
            x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
            out = model(x, targets=y)
            n_tok = y.numel()
            total_nll_nats += float(out.loss.item()) * n_tok
            total_tokens += n_tok
        total_bytes += len(text.encode("utf-8"))

    avg_nats = total_nll_nats / max(1, total_tokens)
    ppl = math.exp(min(avg_nats, 20.0))
    bpb = (total_nll_nats / math.log(2.0)) / max(1, total_bytes)
    return {"loss_nats": avg_nats, "ppl": ppl, "bpb": bpb, "tokens": float(total_tokens)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate crouton checkpoint (BPB / PPL).")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="artifacts/tokenizer/wp_bpe_ppig_32k.json")
    parser.add_argument("--val-file", type=str, required=True, help="UTF-8 validation text file")
    parser.add_argument("--seq-len", type=int, default=2048)
    args = parser.parse_args()

    with open(args.val_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    metrics = evaluate_bpb(args.checkpoint, args.tokenizer, texts, seq_len=args.seq_len)
    print(f"Eval metrics: {metrics}")


if __name__ == "__main__":
    main()
