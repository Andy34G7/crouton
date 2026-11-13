"""Pretraining loop for crouton: Muon + AdamW, WSD schedule, bf16, grad accum, periodic val."""

from __future__ import annotations

import argparse
import math
from contextlib import nullcontext
from pathlib import Path

import torch
from tokenisers import WpBpePpigTokenizer
from torch.utils.data import DataLoader

from crouton.config import CroutonConfig, TrainConfig
from crouton.data import StreamingPackedDataset, resolve_token_id
from crouton.model import CroutonForCausalLM
from crouton.optim import build_optimizers, set_optimizer_lr, wsd_lr_multiplier
from crouton.train.common import get_device, next_or_restart, save_checkpoint, unwrap_model, validate_tokenizer_vocab


@torch.no_grad()
def run_validation_bpb(
    model: torch.nn.Module,
    tokenizer: WpBpePpigTokenizer,
    val_texts: list[str],
    seq_len: int,
    device: torch.device,
    amp_ctx,
) -> tuple[float, float]:
    raw_model = unwrap_model(model)
    was_training = raw_model.training
    raw_model.eval()
    eos_id = resolve_token_id(tokenizer, "<|endoftext|>", fallback=0)
    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0

    for text in val_texts:
        doc_ids = tokenizer.encode(text)
        if not doc_ids:
            continue
        ids = [eos_id] + doc_ids
        for start in range(0, len(ids) - 1, seq_len):
            chunk = ids[start : start + seq_len + 1]
            if len(chunk) < 2:
                continue
            x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
            with amp_ctx:
                out = raw_model(x, targets=y)
            n_tok = y.numel()
            total_nll += float(out.loss.item()) * n_tok
            total_tokens += n_tok
        total_bytes += len(text.encode("utf-8"))

    if was_training:
        raw_model.train()
    avg_loss = total_nll / max(1, total_tokens)
    bpb = (total_nll / math.log(2.0)) / max(1, total_bytes)
    return avg_loss, bpb


def run_pretrain(
    model_cfg: CroutonConfig,
    train_cfg: TrainConfig,
    local_file: str | None = None,
    val_file: str | None = None,
) -> None:
    torch.manual_seed(train_cfg.seed)
    device = get_device()
    use_bf16 = train_cfg.bf16 and device.type == "cuda"
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    tokenizer = WpBpePpigTokenizer.from_pretrained(train_cfg.tokenizer_path)
    validate_tokenizer_vocab(tokenizer, model_cfg)
    local_texts = None
    if local_file is not None:
        local_texts = [line.strip() for line in Path(local_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    val_texts = None
    if val_file is not None:
        val_texts = [line.strip() for line in Path(val_file).read_text(encoding="utf-8").splitlines() if line.strip()]

    dataset = StreamingPackedDataset(
        tokenizer=tokenizer,
        seq_len=train_cfg.seq_len,
        sources=[
            ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
            ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "train", "text"),
        ],
        local_texts=local_texts,
        shuffle_seed=train_cfg.seed,
    )
    dataloader = DataLoader(dataset, batch_size=train_cfg.micro_batch_size)
    loader = iter(dataloader)

    raw_model = CroutonForCausalLM(model_cfg).to(device)
    muon_opt, adamw_opt = build_optimizers(
        raw_model,
        muon_lr=train_cfg.muon_lr,
        adamw_lr=train_cfg.adamw_lr,
        weight_decay=train_cfg.weight_decay,
    )
    model: torch.nn.Module = (
        torch.compile(raw_model) if (train_cfg.compile_model and device.type == "cuda") else raw_model
    )

    model.train()
    for step in range(train_cfg.total_steps):
        mult = wsd_lr_multiplier(
            step=step,
            total_steps=train_cfg.total_steps,
            warmup_steps=train_cfg.warmup_steps,
            decay_steps=train_cfg.decay_steps,
            min_lr_ratio=train_cfg.min_lr_ratio,
        )
        set_optimizer_lr(muon_opt, train_cfg.muon_lr, mult)
        set_optimizer_lr(adamw_opt, train_cfg.adamw_lr, mult)

        muon_opt.zero_grad(set_to_none=True)
        adamw_opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_entropy = 0.0

        for _ in range(train_cfg.grad_accum_steps):
            (x, y), loader = next_or_restart(loader, dataloader)
            x, y = x.to(device), y.to(device)
            with amp_ctx:
                out = model(x, targets=y)
                loss = out.loss / train_cfg.grad_accum_steps
            loss.backward()
            total_loss += float(loss.item())
            if out.router_entropy is not None:
                total_entropy += float(out.router_entropy.item()) / train_cfg.grad_accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        muon_opt.step()
        adamw_opt.step()

        if step % train_cfg.log_interval == 0:
            print(
                f"[pretrain step {step:6d}] loss={total_loss:.4f} "
                f"router_ent={total_entropy:.3f} lr_mult={mult:.3f}"
            )

        if val_texts and train_cfg.eval_interval > 0 and step > 0 and step % train_cfg.eval_interval == 0:
            val_loss, val_bpb = run_validation_bpb(
                model, tokenizer, val_texts, train_cfg.seq_len, device, amp_ctx
            )
            print(f"[pretrain val step {step:6d}] val_loss={val_loss:.4f} val_bpb={val_bpb:.4f}")

        if step > 0 and step % train_cfg.save_interval == 0:
            save_checkpoint(Path(train_cfg.output_dir) / f"step_{step}.pt", model, model_cfg, step)

    save_checkpoint(Path(train_cfg.output_dir) / "final.pt", model, model_cfg, train_cfg.total_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain crouton.")
    parser.add_argument("--model-config", type=str, default="configs/model_300m.yaml")
    parser.add_argument("--tokenizer", type=str, default="artifacts/tokenizer/wp_bpe_ppig_32k.json")
    parser.add_argument("--output-dir", type=str, default="checkpoints/pretrain")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--decay-steps", type=int, default=2000)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--muon-lr", type=float, default=2e-2)
    parser.add_argument("--adamw-lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--local-file", type=str, default=None)
    parser.add_argument("--val-file", type=str, default=None)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    model_cfg = CroutonConfig.from_yaml(args.model_config)
    if args.vocab_size is not None:
        model_cfg.vocab_size = args.vocab_size
    if args.seq_len is not None:
        model_cfg.max_seq_len = args.seq_len
        model_cfg.rope_original_max_seq_len = args.seq_len
    train_cfg = TrainConfig(
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer,
        seq_len=args.seq_len or model_cfg.max_seq_len,
        micro_batch_size=args.micro_batch,
        grad_accum_steps=args.grad_accum,
        total_steps=args.steps,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        compile_model=not args.no_compile,
        seed=args.seed,
    )
    run_pretrain(model_cfg, train_cfg, local_file=args.local_file, val_file=args.val_file)


if __name__ == "__main__":
    main()
