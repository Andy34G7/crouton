"""Evaluation runner: Bits-per-Byte (BPB) / PPL + lm-eval-harness adapter."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tokenisers import WpBpePpigTokenizer

from crouton.data.pretrain_data import resolve_token_id
from crouton.train.common import (
    bf16_supported,
    evaluate_model_bpb,
    get_device,
    instantiate_checkpoint_model,
    load_checkpoint_payload,
    read_nonempty_lines,
    safe_decode,
    strip_required_paths,
    validate_output_dir,
    validate_tokenizer_vocab,
)

try:
    import lm_eval
    from lm_eval.api.model import LM as LMEvalBase

    HAS_LM_EVAL = True
except ImportError:
    HAS_LM_EVAL = False
    lm_eval = None
    LMEvalBase = object  # type: ignore[assignment,misc]


class _TokenizerCompat:
    """Expose HuggingFace-style metadata attributes on top of WpBpePpigTokenizer for lm-eval."""

    def __init__(self, tokenizer: WpBpePpigTokenizer, tokenizer_path: str, eot_id: int) -> None:
        self._tokenizer = tokenizer
        self.name_or_path = tokenizer_path
        self.eos_token = "<|endoftext|>"
        self.eos_token_id = eot_id
        self.pad_token = "<|endoftext|>"
        self.pad_token_id = eot_id
        self.bos_token = "<|endoftext|>"
        self.bos_token_id = eot_id
        self.unk_token = None
        self.unk_token_id = None

    def encode(self, text: str, **kwargs) -> list[int]:
        return list(self._tokenizer.encode(text))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)


class CroutonLM(LMEvalBase):  # type: ignore[valid-type,misc]
    """lm-eval-harness LM adapter wrapping CroutonForCausalLM + WpBpePpigTokenizer."""

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
        batch_size: int = 1,
        max_gen_toks: int = 256,
        device: str | None = None,
        bf16: bool = True,
    ) -> None:
        if batch_size < 1 or max_gen_toks < 1:
            raise ValueError("batch_size and max_gen_toks must be >= 1")
        checkpoint_path, tokenizer_path = strip_required_paths(
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
        )
        ckpt, self.model_cfg, _ = load_checkpoint_payload(checkpoint_path)
        raw_tokenizer = WpBpePpigTokenizer.from_pretrained(tokenizer_path)
        validate_tokenizer_vocab(raw_tokenizer, self.model_cfg)
        resolved_device = torch.device(device) if device else get_device()
        if HAS_LM_EVAL:
            super().__init__()
        self._device = resolved_device
        self._bf16 = bf16
        self.model = instantiate_checkpoint_model(ckpt, self.model_cfg, self._device)
        self.model.eval()
        self._eot_id = resolve_token_id(raw_tokenizer, "<|endoftext|>", fallback=0)
        self._im_end_id = resolve_token_id(raw_tokenizer, "<|im_end|>", fallback=self._eot_id)
        self.tokenizer = _TokenizerCompat(raw_tokenizer, tokenizer_path=tokenizer_path, eot_id=self._eot_id)
        self._tokenizer_path = tokenizer_path
        self._batch_size = batch_size
        self._max_gen_toks = max_gen_toks

    def _amp_ctx(self):
        return torch.amp.autocast("cuda", dtype=torch.bfloat16) if (self._bf16 and bf16_supported(self.device)) else nullcontext()

    @property
    def tokenizer_name(self) -> str:
        return Path(self._tokenizer_path).name

    @property
    def device(self) -> torch.device:
        return self._device

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
        return safe_decode(self.tokenizer, tokens)

    def apply_chat_template(self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
        parts: list[str] = []
        for msg in chat_history:
            role = str(msg.get("role", "user")).strip() or "user"
            raw_content = msg.get("content")
            content = "" if raw_content is None else str(raw_content)
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

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

        with self._amp_ctx():
            logits = self.model(x).logits[0]  # [T, V]

        cont_logits = logits[-cont_len:].float()
        cont_targets = y[-cont_len:]
        log_probs = F.log_softmax(cont_logits, dim=-1)
        tok_logps = torch.gather(log_probs, dim=-1, index=cont_targets.unsqueeze(-1)).squeeze(-1)
        is_greedy = bool((cont_logits.argmax(dim=-1) == cont_targets).all().item())
        return float(tok_logps.sum().item()), is_greedy

    @torch.no_grad()
    def _score_long_pair(self, ctx_ids: list[int], cont_ids: list[int]) -> tuple[float, bool]:
        first_take = min(
            len(cont_ids),
            max(1, self.max_length - len(ctx_ids)) if len(ctx_ids) < self.max_length else max(1, self.max_length // 2),
        )
        ll_total, all_greedy = self._score_pair(ctx_ids, cont_ids[:first_take])
        ids = ctx_ids + cont_ids
        pos = len(ctx_ids) + first_take
        stride = max(1, self.max_length // 2)
        while pos < len(ids):
            end = min(len(ids), pos + stride)
            chunk_cont = ids[pos:end]
            ctx_start = max(0, pos - max(1, self.max_length + 1 - len(chunk_cont)))
            chunk_ll, chunk_greedy = self._score_pair(ids[ctx_start:pos], chunk_cont)
            ll_total += chunk_ll
            all_greedy = all_greedy and chunk_greedy
            pos = end
        return ll_total, all_greedy

    @torch.no_grad()
    def _score_batch(self, pairs: list[tuple[list[int], list[int]]]) -> list[tuple[float, bool]]:
        if len(pairs) == 1:
            return [self._score_pair(*pairs[0])]
        seqs: list[tuple[list[int], list[int], int]] = []
        max_t = 1
        for ctx_ids, cont_ids in pairs:
            full = (ctx_ids + cont_ids)[-(self.max_length + 1) :]
            cont_len = min(len(cont_ids), len(full) - 1)
            seqs.append((full[:-1], full[1:], cont_len))
            max_t = max(max_t, len(full) - 1)

        bsz = len(seqs)
        x_batch = torch.full((bsz, max_t), self.eot_token_id, dtype=torch.long, device=self.device)
        mask = torch.zeros((bsz, max_t), dtype=torch.bool, device=self.device)
        for i, (inp, _, _) in enumerate(seqs):
            n = len(inp)
            x_batch[i, :n] = torch.tensor(inp, dtype=torch.long, device=self.device)
            mask[i, :n] = True

        attn_mask = None if bool(mask.all().item()) else mask
        with self._amp_ctx():
            logits = self.model(x_batch, attention_mask=attn_mask).logits

        out: list[tuple[float, bool]] = []
        for i, (_, tgt_list, cont_len) in enumerate(seqs):
            seq_len = len(tgt_list)
            cont_logits = logits[i, seq_len - cont_len : seq_len].float()
            cont_targets = torch.tensor(tgt_list[-cont_len:], dtype=torch.long, device=self.device)
            log_probs = F.log_softmax(cont_logits, dim=-1)
            tok_logps = torch.gather(log_probs, dim=-1, index=cont_targets.unsqueeze(-1)).squeeze(-1)
            is_greedy = bool((cont_logits.argmax(dim=-1) == cont_targets).all().item())
            out.append((float(tok_logps.sum().item()), is_greedy))
        return out

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool] | None] = [None] * len(requests)
        short_items: list[tuple[int, list[int], list[int]]] = []
        for idx, req in enumerate(requests):
            ctx, cont = req.args
            if not cont:
                results[idx] = (0.0, True)
                continue

            if not ctx:
                ctx_ids = [self.eot_token_id]
                cont_ids = self.tok_encode(cont)
            else:
                ctx_ids = self.tok_encode(ctx)
                whole_ids = self.tok_encode(ctx + cont)
                prefix = 0
                for a, b in zip(ctx_ids, whole_ids):
                    if a != b:
                        break
                    prefix += 1
                if prefix == 0:
                    ctx_ids = [self.eot_token_id]
                    cont_ids = whole_ids
                elif prefix < len(whole_ids):
                    ctx_ids = ctx_ids[:prefix]
                    cont_ids = whole_ids[prefix:]
                else:
                    ctx_ids = ctx_ids[: max(0, prefix - 1)] or [self.eot_token_id]
                    cont_ids = whole_ids[-1:]

            if not cont_ids or (len(ctx_ids) + len(cont_ids)) < 2:
                results[idx] = (0.0, True)
            elif len(cont_ids) > self.max_length:
                results[idx] = self._score_long_pair(ctx_ids, cont_ids)
            else:
                short_items.append((idx, ctx_ids, cont_ids))

        short_items.sort(key=lambda item: len(item[1]) + len(item[2]))
        for start in range(0, len(short_items), self.batch_size):
            chunk = short_items[start : start + self.batch_size]
            scores = self._score_batch([(c_ids, t_ids) for _, c_ids, t_ids in chunk])
            for (orig_idx, _, _), score in zip(chunk, scores):
                results[orig_idx] = score

        return [r if r is not None else (0.0, True) for r in results]

    def loglikelihood_rolling(self, requests) -> list[float]:
        results: list[float] = []
        stride = max(1, self.max_length // 2)
        for req in requests:
            (text,) = req.args
            ids = [self.eot_token_id] + list(self.tokenizer.encode(text, split_special_tokens=True))
            if len(ids) < 2:
                results.append(0.0)
                continue

            # Initial window predicts tokens [1 .. min(len(ids), max_length + 1))
            first_end = min(len(ids), self.max_length + 1)
            ll_first, _ = self._score_pair(ids[:1], ids[1:first_end])
            total = ll_first

            # Subsequent overlapping windows keep full context up to max_length + 1 total tokens.
            pos = first_end
            while pos < len(ids):
                end = min(len(ids), pos + stride)
                cont = ids[pos:end]
                ctx_start = max(0, pos - max(1, self.max_length + 1 - len(cont)))
                ctx = ids[ctx_start:pos]
                ll_chunk, _ = self._score_pair(ctx, cont)
                total += ll_chunk
                pos = end

            results.append(total)
        return results

    def generate_until(self, requests) -> list[str]:
        outputs: list[str] = []
        amp_ctx = self._amp_ctx()
        gen_eos = list(dict.fromkeys([self._eot_id, self._im_end_id]))
        eos_set = set(gen_eos)
        for req in requests:
            ctx, gen_kwargs = req.args
            kwargs = gen_kwargs if isinstance(gen_kwargs, dict) else {}
            raw_until = kwargs.get("until")
            if raw_until is None:
                until: list[str] = []
            elif isinstance(raw_until, str):
                until = [raw_until]
            else:
                until = [str(s) for s in raw_until if s is not None]

            raw_max = kwargs.get("max_gen_toks")
            max_toks = self.max_gen_toks if raw_max is None else int(raw_max)
            if max_toks <= 0:
                outputs.append("")
                continue

            do_sample = bool(kwargs.get("do_sample", False))
            raw_temp = kwargs.get("temperature")
            temp = (1.0 if raw_temp is None else float(raw_temp)) if do_sample else 0.0
            raw_topk = kwargs.get("top_k")
            top_k = 0 if raw_topk is None else int(raw_topk)

            max_prompt = max(1, self.max_length - min(max_toks, max(1, self.max_length // 2)))
            ctx_ids = self.tok_encode(ctx) or [self.eot_token_id]
            x = torch.tensor([ctx_ids[-max_prompt:]], dtype=torch.long, device=self.device)
            with amp_ctx:
                seq = self.model.generate(
                    x,
                    max_new_tokens=max_toks,
                    temperature=temp,
                    top_k=top_k,
                    eos_token_ids=gen_eos,
                    vocab_limit=self.tokenizer.vocab_size,
                )
            gen_ids = seq[0, x.size(1) :].tolist()
            while gen_ids and gen_ids[-1] in eos_set:
                gen_ids = gen_ids[:-1]
            text = self.tok_decode(gen_ids)
            stop_indices = [text.find(stop) for stop in until if stop and stop in text]
            if stop_indices:
                text = text[: min(stop_indices)]
            outputs.append(text)
        return outputs


@torch.no_grad()
def evaluate_bpb(
    checkpoint_path: str,
    tokenizer_path: str,
    texts: list[str],
    seq_len: int | None = None,
    bf16: bool = True,
) -> dict[str, float]:
    """
    Compute tokenizer-agnostic Bits-Per-Byte (BPB) and Perplexity (PPL)
    for fair comparison across tokenizers (e.g., crouton vs shortcake).
    Prepends EOS so every token in each document is scored.
    """
    checkpoint_path, tokenizer_path = strip_required_paths(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
    )
    if seq_len is not None and seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if not texts or not all(isinstance(t, str) for t in texts) or not any(t.strip() for t in texts):
        raise ValueError("Evaluation texts must contain at least one non-empty string")
    ckpt, model_cfg, _ = load_checkpoint_payload(checkpoint_path)
    eff_seq_len = seq_len if seq_len is not None else model_cfg.max_seq_len
    if eff_seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    tokenizer = WpBpePpigTokenizer.from_pretrained(tokenizer_path)
    validate_tokenizer_vocab(tokenizer, model_cfg)
    device = get_device()
    model = instantiate_checkpoint_model(ckpt, model_cfg, device)
    model.eval()
    use_bf16 = bf16 and bf16_supported(device)
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
    return evaluate_model_bpb(model, tokenizer, texts, eff_seq_len, device, amp_ctx)


def evaluate_lm_eval(
    checkpoint_path: str,
    tokenizer_path: str,
    tasks: list[str],
    limit: int | None = None,
    batch_size: int = 1,
    bf16: bool = True,
) -> dict[str, Any]:
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError("batch_size and limit must be >= 1")
    if not tasks or any(not isinstance(t, str) or not t.strip() for t in tasks):
        raise ValueError("tasks must contain at least one non-empty task name")
    clean_tasks = [t.strip() for t in tasks]
    if not HAS_LM_EVAL:
        raise RuntimeError("lm-eval is not installed. Install with: uv sync --extra dev --extra eval")
    checkpoint_path, tokenizer_path = strip_required_paths(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
    )
    lm = CroutonLM(checkpoint_path=checkpoint_path, tokenizer_path=tokenizer_path, batch_size=batch_size, bf16=bf16)
    results = lm_eval.simple_evaluate(model=lm, tasks=clean_tasks, limit=limit)
    return results.get("results", {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate crouton checkpoint (BPB/PPL and/or lm-eval-harness).")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--compare-checkpoint", type=str, default=None, help="Optional crouton baseline checkpoint")
    parser.add_argument("--tokenizer", type=str, default="artifacts/tokenizer/wp_bpe_ppig_32k.json")
    parser.add_argument("--val-file", type=str, default=None, help="UTF-8 validation text file for BPB/PPL")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated lm-eval-harness tasks")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=None, help="Override BPB seq_len (defaults to checkpoint max_seq_len)")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.strip()
    tokenizer_path = args.tokenizer.strip()
    if not checkpoint_path or not tokenizer_path:
        parser.error("--checkpoint and --tokenizer must be non-empty")

    compare_checkpoint = args.compare_checkpoint.strip() if args.compare_checkpoint is not None else None
    if compare_checkpoint is not None and not compare_checkpoint:
        parser.error("--compare-checkpoint must be a non-empty path")

    val_file = args.val_file.strip() if args.val_file is not None else None
    if val_file is not None and not val_file:
        parser.error("--val-file must be a non-empty path")

    output_json = args.output_json.strip() if args.output_json is not None else None
    if output_json is not None:
        if not output_json:
            parser.error("--output-json must be a non-empty path")
        if output_json.endswith(os.sep) or (os.altsep is not None and output_json.endswith(os.altsep)):
            parser.error("--output-json must be a file path, not a directory")
        out_json_path = Path(output_json)
        if out_json_path.is_dir() or (out_json_path.is_symlink() and not out_json_path.exists()):
            parser.error("--output-json must be a file path, not a directory")
        try:
            validate_output_dir(out_json_path.parent)
        except ValueError as exc:
            parser.error(f"--output-json parent directory is invalid: {exc}")

    if val_file is None and args.tasks is None:
        parser.error("Provide at least one of --val-file or --tasks")
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1) or (args.seq_len is not None and args.seq_len < 1):
        parser.error("--batch-size, --limit, and --seq-len must all be >= 1")

    task_list: list[str] = []
    if args.tasks is not None:
        task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
        if not task_list:
            parser.error("--tasks must contain at least one non-empty task name")
        if not HAS_LM_EVAL:
            raise RuntimeError("lm-eval is not installed. Install with: uv sync --extra dev --extra eval")

    texts: list[str] | None = read_nonempty_lines(val_file, label="Validation file") if val_file is not None else None

    # Pre-validate tokenizer and both checkpoints before running potentially long evaluations.
    tokenizer = WpBpePpigTokenizer.from_pretrained(tokenizer_path)
    for ckpt_label, ckpt_path in (("primary", checkpoint_path), ("compare", compare_checkpoint)):
        if ckpt_path is None:
            continue
        _, ckpt_cfg, _ = load_checkpoint_payload(ckpt_path)
        validate_tokenizer_vocab(tokenizer, ckpt_cfg)
        if texts is not None and args.seq_len is None and ckpt_cfg.max_seq_len < 1:
            raise ValueError(f"{ckpt_label} checkpoint max_seq_len must be >= 1")

    use_bf16 = not args.no_bf16
    report: dict[str, Any] = {"primary": {}}
    if texts is not None:
        report["primary"]["bpb"] = evaluate_bpb(
            checkpoint_path, tokenizer_path, texts, seq_len=args.seq_len, bf16=use_bf16
        )
        if compare_checkpoint is not None:
            report.setdefault("compare", {})["bpb"] = evaluate_bpb(
                compare_checkpoint, tokenizer_path, texts, seq_len=args.seq_len, bf16=use_bf16
            )

    if task_list:
        report["primary"]["lm_eval"] = evaluate_lm_eval(
            checkpoint_path, tokenizer_path, task_list, limit=args.limit, batch_size=args.batch_size, bf16=use_bf16
        )
        if compare_checkpoint is not None:
            report.setdefault("compare", {})["lm_eval"] = evaluate_lm_eval(
                compare_checkpoint, tokenizer_path, task_list, limit=args.limit, batch_size=args.batch_size, bf16=use_bf16
            )

    print(json.dumps(report, indent=2))
    if output_json is not None:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
