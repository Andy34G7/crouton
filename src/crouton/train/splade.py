"""SPLADE-v3 sparse retrieval training for Crouton (bidirectional MNTP + InfoNCE + Margin-MSE + FLOPS/L1)."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tokenisers import WpBpePpigTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from crouton.data.pretrain_data import resolve_pad_token_id
from crouton.model.splade import (
    CroutonSPLADE,
    quad_reg_weight,
    splade_infonce_loss,
    splade_margin_mse_loss,
)
from crouton.optim import build_adamw_param_groups, set_optimizer_lr, wsd_lr_multiplier
from crouton.train.common import (
    bf16_supported,
    get_device,
    instantiate_checkpoint_model,
    load_checkpoint_payload,
    next_or_restart,
    strip_required_paths,
    validate_output_dir,
    validate_tokenizer_vocab,
)


class RetrievalTripletDataset(Dataset):
    """Triplet dataset yielding (query, pos, neg[, pos_score, neg_score]) with BPE-aligned prefixes."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: WpBpePpigTokenizer,
        max_query_len: int = 64,
        max_doc_len: int = 256,
        query_prefix: str = " ",
        doc_prefix: str = " ",
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    def __len__(self) -> int:
        return len(self.records)

    def _encode(self, text: str, max_len: int, prefix: str) -> list[int]:
        formatted = CroutonSPLADE.format_bpe_text(text, prefix=prefix)
        ids = self.tokenizer.encode(formatted, split_special_tokens=True)[:max_len]
        return ids if ids else [0]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        item: dict[str, Any] = {
            "q_ids": self._encode(str(rec["query"]), self.max_query_len, self.query_prefix),
            "p_ids": self._encode(str(rec["pos"]), self.max_doc_len, self.doc_prefix),
            "n_ids": self._encode(str(rec["neg"]), self.max_doc_len, self.doc_prefix),
        }
        if "pos_score" in rec and "neg_score" in rec and rec["pos_score"] is not None and rec["neg_score"] is not None:
            item["pos_score"] = float(rec["pos_score"])
            item["neg_score"] = float(rec["neg_score"])
        return item


def _pad_sequences(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(s) for s in seqs)
    bsz = len(seqs)
    ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = True
    return ids, mask


def collate_triplets(batch: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    q_ids, q_mask = _pad_sequences([b["q_ids"] for b in batch], pad_id)
    p_ids, p_mask = _pad_sequences([b["p_ids"] for b in batch], pad_id)
    n_ids, n_mask = _pad_sequences([b["n_ids"] for b in batch], pad_id)
    out = {
        "q_ids": q_ids,
        "q_mask": q_mask,
        "p_ids": p_ids,
        "p_mask": p_mask,
        "n_ids": n_ids,
        "n_mask": n_mask,
    }
    if all("pos_score" in b and "neg_score" in b for b in batch):
        out["pos_score"] = torch.tensor([b["pos_score"] for b in batch], dtype=torch.float32)
        out["neg_score"] = torch.tensor([b["neg_score"] for b in batch], dtype=torch.float32)
    return out


def load_splade_records(
    local_jsonl: str | None = None,
    dataset_name: str = "sentence-transformers/msmarco-bm25",
    subset: str | None = "triplet",
    max_examples: int = 100000,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if local_jsonl is not None:
        with open(local_jsonl, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("query") and row.get("pos") and row.get("neg"):
                    records.append(row)
                if len(records) >= max_examples:
                    break
        if not records:
            raise ValueError(f"No valid (query, pos, neg) records found in {local_jsonl}")
        return records

    from datasets import load_dataset

    kwargs = {"name": subset} if subset else {}
    ds = load_dataset(dataset_name, split="train", streaming=True, **kwargs)
    for row in ds:
        q = row.get("query") or row.get("anchor")
        p = row.get("positive") or row.get("pos")
        n = row.get("negative") or row.get("neg")
        if isinstance(q, str) and isinstance(p, str) and isinstance(n, str) and q.strip() and p.strip() and n.strip():
            rec: dict[str, Any] = {"query": q, "pos": p, "neg": n}
            if "pos_score" in row and "neg_score" in row:
                rec["pos_score"] = row["pos_score"]
                rec["neg_score"] = row["neg_score"]
            elif "score" in row and isinstance(row["score"], (int, float)):
                rec["pos_score"] = float(row["score"])
                rec["neg_score"] = 0.0
            records.append(rec)
        if len(records) >= max_examples:
            break
    if not records:
        raise RuntimeError(f"No valid retrieval triplets loaded from {dataset_name}")
    return records


def save_splade_checkpoint(path: str | Path, model: CroutonSPLADE, step: int) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": model.backbone.config.to_dict(),
            "step": step,
            "bidirectional": model.bidirectional,
            "pad_token_id": model.pad_token_id,
            "arch": "CroutonSPLADE",
        },
        out,
    )


def run_mntp_adaptation(
    splade_model: CroutonSPLADE,
    loader: DataLoader,
    device: torch.device,
    amp_ctx,
    steps: int,
    lr: float,
    mask_token_id: int,
    mask_prob: float = 0.2,
) -> None:
    """Short bidirectional Masked Next-Token Prediction (LLM2Vec-style) warmup before SPLADE training."""
    if steps <= 0:
        return
    backbone = splade_model.backbone
    opt = AdamW(build_adamw_param_groups(backbone, weight_decay=0.01), lr=lr, betas=(0.9, 0.95))
    it = iter(loader)
    backbone.train()
    for step in range(steps):
        batch, it = next_or_restart(it, loader)
        ids = batch["p_ids"].to(device)
        attn_mask = batch["p_mask"].to(device)
        if ids.size(1) < 2:
            continue
        corrupt = (torch.rand_like(ids.float()) < mask_prob) & attn_mask
        corrupt[:, 0] = False
        if not corrupt.any():
            corrupt[:, -1] = attn_mask[:, -1]
        masked_ids = torch.where(corrupt, mask_token_id, ids)
        targets = torch.full_like(ids, -100)
        # Predict token t from position t-1 under bidirectional context
        targets[:, :-1] = torch.where(corrupt[:, 1:], ids[:, 1:], -100)

        opt.zero_grad(set_to_none=True)
        with amp_ctx:
            out = backbone(masked_ids, targets=targets, attention_mask=attn_mask, is_causal=False)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"[mntp step {step:5d}] loss={float(out.loss.item()):.4f}", flush=True)


def run_splade_train(
    base_checkpoint: str,
    tokenizer_path: str,
    output_dir: str,
    records: list[dict[str, Any]] | None = None,
    local_jsonl: str | None = None,
    dataset_name: str = "sentence-transformers/msmarco-bm25",
    subset: str | None = "triplet",
    max_examples: int = 100000,
    mntp_steps: int = 0,
    total_steps: int = 5000,
    batch_size: int = 32,
    grad_accum_steps: int = 2,
    max_query_len: int = 64,
    max_doc_len: int = 256,
    lr: float = 3e-5,
    lambda_q: float = 0.05,
    lambda_d: float = 0.04,
    reg_warmup_steps: int = 1500,
    query_reg_type: str = "l1",
    margin_mse_weight: float = 0.5,
    temperature: float = 1.0,
    bidirectional: bool = True,
    bf16: bool = True,
    save_interval: int = 0,
    seed: int = 42,
) -> Path:
    validate_output_dir(output_dir)
    torch.manual_seed(seed)
    device = get_device()
    use_bf16 = bf16 and bf16_supported(device)
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    ckpt, model_cfg, _ = load_checkpoint_payload(base_checkpoint)
    tokenizer = WpBpePpigTokenizer.from_pretrained(tokenizer_path)
    validate_tokenizer_vocab(tokenizer, model_cfg)
    pad_id = resolve_pad_token_id(tokenizer)

    backbone = instantiate_checkpoint_model(ckpt, model_cfg, device)
    model = CroutonSPLADE(backbone, tie_lm_head=True, pad_token_id=pad_id, bidirectional=bidirectional).to(device)

    data_records = (
        records
        if records is not None
        else load_splade_records(
            local_jsonl=local_jsonl,
            dataset_name=dataset_name,
            subset=subset,
            max_examples=max_examples,
        )
    )
    ds = RetrievalTripletDataset(data_records, tokenizer, max_query_len=max_query_len, max_doc_len=max_doc_len)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(ds) >= batch_size,
        collate_fn=lambda b: collate_triplets(b, pad_id),
    )

    if mntp_steps > 0 and bidirectional:
        run_mntp_adaptation(model, loader, device, amp_ctx, steps=mntp_steps, lr=lr, mask_token_id=pad_id)

    opt = AdamW(build_adamw_param_groups(model, weight_decay=0.01), lr=lr, betas=(0.9, 0.95))
    it = iter(loader)
    model.train()

    warmup_steps = min(300, max(1, total_steps // 10))
    decay_steps = max(1, total_steps - warmup_steps)

    for step in range(total_steps):
        mult = wsd_lr_multiplier(step, total_steps, warmup_steps=warmup_steps, decay_steps=decay_steps, min_lr_ratio=0.1)
        set_optimizer_lr(opt, lr, mult)
        lq = quad_reg_weight(step, reg_warmup_steps, lambda_q)
        ld = quad_reg_weight(step, reg_warmup_steps, lambda_d)

        opt.zero_grad(set_to_none=True)
        tot_loss = 0.0
        tot_rank = 0.0
        tot_l0_q = 0.0
        tot_l0_d = 0.0

        for _ in range(grad_accum_steps):
            batch, it = next_or_restart(it, loader)
            q_ids, q_mask = batch["q_ids"].to(device), batch["q_mask"].to(device)
            p_ids, p_mask = batch["p_ids"].to(device), batch["p_mask"].to(device)
            n_ids, n_mask = batch["n_ids"].to(device), batch["n_mask"].to(device)

            with amp_ctx:
                q_out = model(q_ids, attention_mask=q_mask)
                p_out = model(p_ids, attention_mask=p_mask)
                n_out = model(n_ids, attention_mask=n_mask)

                rank_loss = splade_infonce_loss(
                    q_out.sparse_vector, p_out.sparse_vector, n_out.sparse_vector, temperature=temperature
                )
                if "pos_score" in batch and "neg_score" in batch and margin_mse_weight > 0.0:
                    mmse = splade_margin_mse_loss(
                        q_out.sparse_vector,
                        p_out.sparse_vector,
                        n_out.sparse_vector,
                        batch["pos_score"].to(device),
                        batch["neg_score"].to(device),
                    )
                    rank_loss = rank_loss + margin_mse_weight * mmse

                q_reg = q_out.l1_loss if query_reg_type == "l1" else q_out.flops_loss
                d_reg = 0.5 * (p_out.flops_loss + n_out.flops_loss)
                loss = (rank_loss + lq * q_reg + ld * d_reg) / grad_accum_steps

            loss.backward()
            tot_loss += float(loss.item())
            tot_rank += float(rank_loss.item()) / grad_accum_steps
            with torch.no_grad():
                tot_l0_q += float((q_out.sparse_vector > 0).float().sum(dim=-1).mean().item()) / grad_accum_steps
                tot_l0_d += float((p_out.sparse_vector > 0).float().sum(dim=-1).mean().item()) / grad_accum_steps

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 20 == 0:
            print(
                f"[splade step {step:5d}] loss={tot_loss:.4f} rank={tot_rank:.4f} "
                f"l0_q={tot_l0_q:.1f} l0_d={tot_l0_d:.1f} lq={lq:.4f} ld={ld:.4f} lr_mult={mult:.3f}",
                flush=True,
            )
        if save_interval > 0 and step > 0 and step % save_interval == 0:
            save_splade_checkpoint(Path(output_dir) / f"step_{step}.pt", model, step)

    final_path = Path(output_dir) / "splade_final.pt"
    save_splade_checkpoint(final_path, model, total_steps)
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CroutonSPLADE sparse lexical retriever.")
    parser.add_argument("--base-checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="artifacts/tokenizer/wp_bpe_ppig_32k.json")
    parser.add_argument("--output-dir", type=str, default="checkpoints/splade")
    parser.add_argument("--local-jsonl", type=str, default=None, help="Optional local JSONL with query/pos/neg")
    parser.add_argument("--dataset", type=str, default="sentence-transformers/msmarco-bm25")
    parser.add_argument("--subset", type=str, default="triplet")
    parser.add_argument("--max-examples", type=int, default=100000)
    parser.add_argument("--mntp-steps", type=int, default=300, help="Bidirectional MNTP warmup steps")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-query-len", type=int, default=64)
    parser.add_argument("--max-doc-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--lambda-q", type=float, default=0.05)
    parser.add_argument("--lambda-d", type=float, default=0.04)
    parser.add_argument("--reg-warmup-steps", type=int, default=1500)
    parser.add_argument("--query-reg-type", type=str, default="l1", choices=["l1", "flops"])
    parser.add_argument("--margin-mse-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--causal", action="store_true", help="Keep causal attention instead of bidirectional")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--save-interval", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_ckpt, tok_path, out_dir = strip_required_paths(
        **{
            "--base-checkpoint": args.base_checkpoint,
            "--tokenizer": args.tokenizer,
            "--output-dir": args.output_dir,
        }
    )
    run_splade_train(
        base_checkpoint=base_ckpt,
        tokenizer_path=tok_path,
        output_dir=out_dir,
        local_jsonl=args.local_jsonl,
        dataset_name=args.dataset,
        subset=None if args.subset in ("", "none", "None") else args.subset,
        max_examples=args.max_examples,
        mntp_steps=args.mntp_steps,
        total_steps=args.steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        max_query_len=args.max_query_len,
        max_doc_len=args.max_doc_len,
        lr=args.lr,
        lambda_q=args.lambda_q,
        lambda_d=args.lambda_d,
        reg_warmup_steps=args.reg_warmup_steps,
        query_reg_type=args.query_reg_type,
        margin_mse_weight=args.margin_mse_weight,
        temperature=args.temperature,
        bidirectional=not args.causal,
        bf16=not args.no_bf16,
        save_interval=args.save_interval,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
