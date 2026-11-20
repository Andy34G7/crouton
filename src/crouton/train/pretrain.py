"""Pretraining loop for crouton: Muon + AdamW, WSD schedule, bf16, grad accum, periodic val."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from tokenisers import WpBpePpigTokenizer
from torch.utils.data import DataLoader

from crouton.config import CroutonConfig, TrainConfig
from crouton.data import ShardedMemmapDataset, StreamingPackedDataset
from crouton.model import CroutonForCausalLM
from crouton.optim import build_optimizers, set_optimizer_lr, wsd_lr_multiplier
from crouton.train.common import (
    bf16_supported,
    evaluate_model_bpb,
    get_device,
    instantiate_checkpoint_model,
    load_checkpoint_payload,
    next_or_restart,
    read_nonempty_lines,
    save_checkpoint,
    strip_required_paths,
    validate_output_dir,
    validate_tokenizer_vocab,
)


@torch.no_grad()
def run_validation_bpb(
    model: torch.nn.Module,
    tokenizer: WpBpePpigTokenizer,
    val_texts: list[str],
    seq_len: int,
    device: torch.device,
    amp_ctx,
) -> tuple[float, float]:
    metrics = evaluate_model_bpb(model, tokenizer, val_texts, seq_len, device, amp_ctx)
    return metrics["loss_nats"], metrics["bpb"]


@torch.no_grad()
def run_validation_memmap(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    amp_ctx,
    eval_batches: int = 20,
) -> float:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    it = iter(val_loader)
    try:
        for _ in range(eval_batches):
            x, y = next(it)
            x, y = x.to(device), y.to(device)
            with amp_ctx:
                out = model(x, targets=y)
            total_loss += float(out.loss.item())
    finally:
        if was_training:
            model.train()
    return total_loss / max(1, eval_batches)


def run_pretrain(
    model_cfg: CroutonConfig,
    train_cfg: TrainConfig,
    local_file: str | None = None,
    val_file: str | None = None,
    data_dir: str | None = None,
    resume_checkpoint: str | None = None,
) -> None:
    model_cfg.__post_init__()
    train_cfg.__post_init__()
    if train_cfg.seq_len > model_cfg.max_seq_len:
        raise ValueError(
            f"train_cfg.seq_len ({train_cfg.seq_len}) must be <= model_cfg.max_seq_len ({model_cfg.max_seq_len})"
        )
    if data_dir is not None and local_file is not None:
        raise ValueError("data_dir and local_file cannot be combined")
    validate_output_dir(train_cfg.output_dir)
    torch.manual_seed(train_cfg.seed)
    device = get_device()
    use_bf16 = train_cfg.bf16 and bf16_supported(device)
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    local_texts = read_nonempty_lines(local_file, label="Local training file") if local_file is not None else None
    val_texts = read_nonempty_lines(val_file, label="Validation file") if val_file is not None else None

    tokenizer = None
    val_memmap_loader = None
    if data_dir is not None:
        dataset = ShardedMemmapDataset(
            data_path=data_dir,
            seq_len=train_cfg.seq_len,
            split="train",
            seed=train_cfg.seed,
        )
        data_root = Path(data_dir) if Path(data_dir).is_dir() else Path(data_dir).parent
        if val_texts is None and (data_root / "val.bin").is_file():
            val_ds = ShardedMemmapDataset(
                data_path=data_root / "val.bin",
                seq_len=train_cfg.seq_len,
                split="val",
                seed=train_cfg.seed + 1,
            )
            val_memmap_loader = DataLoader(val_ds, batch_size=train_cfg.micro_batch_size)
        if val_texts is not None:
            tokenizer = WpBpePpigTokenizer.from_pretrained(train_cfg.tokenizer_path)
            validate_tokenizer_vocab(tokenizer, model_cfg)
    else:
        tokenizer = WpBpePpigTokenizer.from_pretrained(train_cfg.tokenizer_path)
        validate_tokenizer_vocab(tokenizer, model_cfg)
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

    if resume_checkpoint is not None:
        ckpt, ckpt_cfg, _ = load_checkpoint_payload(resume_checkpoint)
        if ckpt_cfg.max_seq_len < model_cfg.max_seq_len:
            ckpt_cfg.max_seq_len = model_cfg.max_seq_len
        model_cfg = ckpt_cfg
        raw_model = instantiate_checkpoint_model(ckpt, model_cfg, device)
    else:
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

        if train_cfg.eval_interval > 0 and step > 0 and step % train_cfg.eval_interval == 0:
            if val_texts and tokenizer is not None:
                val_loss, val_bpb = run_validation_bpb(
                    model, tokenizer, val_texts, train_cfg.seq_len, device, amp_ctx
                )
                print(f"[pretrain val step {step:6d}] val_loss={val_loss:.4f} val_bpb={val_bpb:.4f}")
            elif val_memmap_loader is not None:
                val_loss = run_validation_memmap(model, val_memmap_loader, device, amp_ctx)
                print(f"[pretrain val step {step:6d}] val_loss={val_loss:.4f}")

        if train_cfg.save_interval > 0 and step > 0 and step % train_cfg.save_interval == 0:
            save_checkpoint(Path(train_cfg.output_dir) / f"step_{step}.pt", model, model_cfg, step)

    save_checkpoint(Path(train_cfg.output_dir) / "final.pt", model, model_cfg, train_cfg.total_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain crouton.")
    parser.add_argument("--model-config", type=str, default="configs/model_300m.yaml")
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Optional size preset matching shortcake (tiny/5m, small/16m, base/50m, medium/110m, large/350m)",
    )
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
    parser.add_argument("--data-dir", type=str, default=None, help="Directory or .bin path of uint16 memmap shards")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Checkpoint .pt to resume pretraining from")
    parser.add_argument("--local-file", type=str, default=None)
    parser.add_argument("--val-file", type=str, default=None)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    local_file = args.local_file.strip() if args.local_file is not None else None
    if local_file is not None and not local_file:
        parser.error("--local-file must be a non-empty file path")
    val_file = args.val_file.strip() if args.val_file is not None else None
    if val_file is not None and not val_file:
        parser.error("--val-file must be a non-empty file path")
    data_dir = args.data_dir.strip() if args.data_dir is not None else None
    if data_dir is not None and not data_dir:
        parser.error("--data-dir must be a non-empty path")
    resume_checkpoint = args.resume_checkpoint.strip() if args.resume_checkpoint is not None else None
    if resume_checkpoint is not None and not resume_checkpoint:
        parser.error("--resume-checkpoint must be a non-empty file path")
    if data_dir is not None and local_file is not None:
        parser.error("--data-dir cannot be combined with --local-file")

    if args.vocab_size is not None and args.vocab_size < 1:
        parser.error("--vocab-size must be >= 1")
    if args.seq_len is not None and args.seq_len < 1:
        parser.error("--seq-len must be >= 1")
    try:
        model_config_path, output_dir, tokenizer_path = strip_required_paths(
            **{
                "--model-config": args.model_config,
                "--output-dir": args.output_dir,
                "--tokenizer": args.tokenizer,
            }
        )
        validate_output_dir(output_dir)
        model_cfg = (
            CroutonConfig.from_preset(args.preset)
            if args.preset is not None
            else CroutonConfig.from_yaml(model_config_path)
        )
        if args.vocab_size is not None:
            model_cfg.vocab_size = args.vocab_size
        if args.seq_len is not None:
            model_cfg.max_seq_len = args.seq_len
            model_cfg.rope_original_max_seq_len = args.seq_len
        model_cfg.__post_init__()
        eff_seq_len = args.seq_len if args.seq_len is not None else model_cfg.max_seq_len
        train_cfg = TrainConfig(
            output_dir=output_dir,
            tokenizer_path=tokenizer_path,
            seq_len=eff_seq_len,
            micro_batch_size=args.micro_batch,
            grad_accum_steps=args.grad_accum,
            total_steps=args.steps,
            warmup_steps=args.warmup_steps,
            decay_steps=args.decay_steps,
            muon_lr=args.muon_lr,
            adamw_lr=args.adamw_lr,
            bf16=not args.no_bf16,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            compile_model=not args.no_compile,
            seed=args.seed,
        )
    except ValueError as exc:
        parser.error(str(exc))

    run_pretrain(
        model_cfg,
        train_cfg,
        local_file=local_file,
        val_file=val_file,
        data_dir=data_dir,
        resume_checkpoint=resume_checkpoint,
    )


if __name__ == "__main__":
    main()
