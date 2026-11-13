"""Evaluation runner: Bits-per-Byte (BPB) / PPL + lm-eval-harness adapter."""

from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tokenisers import WpBpePpigTokenizer

from crouton.data.pretrain_data import resolve_token_id
from crouton.train.common import get_device, load_checkpoint

try:
    import lm_eval
    from lm_eval.api.model import LM as LMEvalBase

    HAS_LM_EVAL = True
except ImportError:
    HAS_LM_EVAL = False
    lm_eval = None
    LMEvalBase = object  # type: ignore[assignment,misc]


class CroutonLM(LMEvalBase):  # type: ignore[valid-type,misc]
    """lm-eval-harness LM adapter wrapping CroutonForCausalLM + WpBpePpigTokenizer."""

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
        batch_size: int = 1,
        max_gen_toks: int = 256,
        device: str | None = None,
    ) -> None:
        if HAS_LM_EVAL:
            super().__init__()
        self.device = torch.device(device) if device else get_device()
        self.model, self.model_cfg, _ = load_checkpoint(checkpoint_path, device=self.device)
        self.model.eval()
        self.tokenizer = WpBpePpigTokenizer.from_pretrained(tokenizer_path)
        self._batch_size = batch_size
        self._max_gen_toks = max_gen_toks
        self._eot_id = resolve_token_id(self.tokenizer, "<|endoftext|>", fallback=0)

    @property
    def eot_token_id(self) -> int:
        return self._eot_id

    @property
    def max_length(self) -> int:
        return self.model_cfg.max_seq_len

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        return list(self.tokenizer.encode(string))

    def tok_decode(self, tokens: list[int], **kwargs) -> str:
        return self.tokenizer.decode(tokens)

    @torch.no_grad()
    def _score_pair(self, context_ids: list[int], cont_ids: list[int]) -> tuple[float, bool]:
        if not cont_ids:
            return 0.0, True
        full = (context_ids + cont_ids)[-(self.max_length + 1) :]
        if len(full) < 2:
            return 0.0, True
        cont_len = min(len(cont_ids), len(full) - 1)
        x = torch.tensor([full[:-1]], dtype=torch.long, device=self.device)
        y = torch.tensor(full[1:], dtype=torch.long, device=self.device)

        use_bf16 = self.device.type == "cuda"
        amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
        with amp_ctx:
            logits = self.model(x).logits[0]  # [T, V]

        cont_logits = logits[-cont_len:].float()
        cont_targets = y[-cont_len:]
        log_probs = F.log_softmax(cont_logits, dim=-1)
        tok_logps = torch.gather(log_probs, dim=-1, index=cont_targets.unsqueeze(-1)).squeeze(-1)
        is_greedy = bool((cont_logits.argmax(dim=-1) == cont_targets).all().item())
        return float(tok_logps.sum().item()), is_greedy

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = []
        for req in requests:
            ctx, cont = req.args
            ctx_ids = self.tok_encode(ctx) if ctx else [self.eot_token_id]
            cont_ids = self.tok_encode(cont)
            results.append(self._score_pair(ctx_ids, cont_ids))
        return results

    def loglikelihood_rolling(self, requests) -> list[float]:
        results: list[float] = []
        for req in requests:
            (text,) = req.args
            ids = [self.eot_token_id] + self.tok_encode(text)
            total = 0.0
            for start in range(0, max(1, len(ids) - 1), self.max_length):
                ctx = ids[max(0, start) : start + 1]
                cont = ids[start + 1 : start + 1 + self.max_length]
                ll, _ = self._score_pair(ctx, cont)
                total += ll
            results.append(total)
        return results

    def generate_until(self, requests) -> list[str]:
        outputs: list[str] = []
        for req in requests:
            ctx, gen_kwargs = req.args
            until = gen_kwargs.get("until", []) if isinstance(gen_kwargs, dict) else []
            max_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks) if isinstance(gen_kwargs, dict) else self.max_gen_toks
            temp = gen_kwargs.get("temperature", 0.0) if isinstance(gen_kwargs, dict) else 0.0

            ctx_ids = self.tok_encode(ctx)
            if not ctx_ids:
                ctx_ids = [self.eot_token_id]
            x = torch.tensor([ctx_ids[-self.max_length :]], dtype=torch.long, device=self.device)
            seq = self.model.generate(
                x,
                max_new_tokens=int(max_toks),
                temperature=float(temp),
                eos_token_ids=[self.eot_token_id],
            )
            text = self.tok_decode(seq[0, x.size(1) :].tolist())
            for stop in until:
                if stop in text:
                    text = text.split(stop, 1)[0]
            outputs.append(text)
        return outputs


@torch.no_grad()
def evaluate_bpb(
    checkpoint_path: str,
    tokenizer_path: str,
    texts: list[str],
    seq_len: int = 2048,
) -> dict[str, float]:
    """
    Compute tokenizer-agnostic Bits-Per-Byte (BPB) and Perplexity (PPL)
    for fair comparison across tokenizers (e.g., crouton vs shortcake).
    """
    device = get_device()
    model, _, _ = load_checkpoint(checkpoint_path, device=device)
    model.eval()
    tokenizer = WpBpePpigTokenizer.from_pretrained(tokenizer_path)
    use_bf16 = device.type == "cuda"
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

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
            with amp_ctx:
                out = model(x, targets=y)
            n_tok = y.numel()
            total_nll_nats += float(out.loss.item()) * n_tok
            total_tokens += n_tok
        total_bytes += len(text.encode("utf-8"))

    avg_nats = total_nll_nats / max(1, total_tokens)
    ppl = math.exp(min(avg_nats, 20.0))
    bpb = (total_nll_nats / math.log(2.0)) / max(1, total_bytes)
    return {"loss_nats": avg_nats, "ppl": ppl, "bpb": bpb, "tokens": float(total_tokens)}


def evaluate_lm_eval(
    checkpoint_path: str,
    tokenizer_path: str,
    tasks: list[str],
    limit: int | None = None,
) -> dict[str, Any]:
    if not HAS_LM_EVAL:
        raise RuntimeError("lm-eval is not installed. Install with: uv sync --extra eval")
    lm = CroutonLM(checkpoint_path=checkpoint_path, tokenizer_path=tokenizer_path)
    results = lm_eval.simple_evaluate(model=lm, tasks=tasks, limit=limit)
    return results.get("results", {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate crouton checkpoint (BPB/PPL and/or lm-eval-harness).")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--compare-checkpoint", type=str, default=None, help="Optional baseline checkpoint to compare")
    parser.add_argument("--tokenizer", type=str, default="artifacts/tokenizer/wp_bpe_ppig_32k.json")
    parser.add_argument("--val-file", type=str, default=None, help="UTF-8 validation text file for BPB/PPL")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated lm-eval-harness tasks")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    if not args.val_file and not args.tasks:
        parser.error("Provide at least one of --val-file or --tasks")

    report: dict[str, Any] = {"primary": {}}
    if args.val_file:
        texts = [line.strip() for line in Path(args.val_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        report["primary"]["bpb"] = evaluate_bpb(args.checkpoint, args.tokenizer, texts, seq_len=args.seq_len)
        if args.compare_checkpoint:
            report.setdefault("compare", {})["bpb"] = evaluate_bpb(
                args.compare_checkpoint, args.tokenizer, texts, seq_len=args.seq_len
            )

    if args.tasks:
        task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
        report["primary"]["lm_eval"] = evaluate_lm_eval(args.checkpoint, args.tokenizer, task_list, limit=args.limit)
        if args.compare_checkpoint:
            report.setdefault("compare", {})["lm_eval"] = evaluate_lm_eval(
                args.compare_checkpoint, args.tokenizer, task_list, limit=args.limit
            )

    print(json.dumps(report, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
