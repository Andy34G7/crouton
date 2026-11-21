"""Evaluate CroutonSPLADE sparse retrieval (nDCG@10, MRR@10, Recall@100, L0 sparsity, top expanded terms)."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from tokenisers import WpBpePpigTokenizer

from crouton.config import CroutonConfig
from crouton.data.pretrain_data import resolve_pad_token_id
from crouton.model import CroutonForCausalLM, CroutonSPLADE
from crouton.train.common import get_device


def load_splade_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[CroutonSPLADE, CroutonConfig]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = CroutonConfig(**ckpt["config"])
    backbone = CroutonForCausalLM(cfg)
    pad_id = ckpt.get("pad_token_id", 0)
    bidirectional = bool(ckpt.get("bidirectional", True))
    model = CroutonSPLADE(backbone, tie_lm_head=True, pad_token_id=pad_id, bidirectional=bidirectional)
    state = ckpt["model"]
    if any(k.startswith("splade_head.") or k.startswith("backbone.") for k in state):
        model.load_state_dict(state, strict=True)
    else:
        model.backbone.load_state_dict(state, strict=True)
    return model.to(device).eval(), cfg


@torch.no_grad()
def encode_texts_splade(
    model: CroutonSPLADE,
    tokenizer: WpBpePpigTokenizer,
    texts: list[str],
    device: torch.device,
    max_len: int = 256,
    batch_size: int = 32,
    prefix: str = " ",
) -> torch.Tensor:
    pad_id = model.pad_token_id if model.pad_token_id is not None else resolve_pad_token_id(tokenizer)
    vecs: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        seqs = []
        for t in chunk:
            ids = tokenizer.encode(CroutonSPLADE.format_bpe_text(t, prefix=prefix), split_special_tokens=True)[:max_len]
            seqs.append(ids if ids else [0])
        mlen = max(len(s) for s in seqs)
        ids_t = torch.full((len(seqs), mlen), pad_id, dtype=torch.long, device=device)
        mask_t = torch.zeros((len(seqs), mlen), dtype=torch.bool, device=device)
        for i, s in enumerate(seqs):
            ids_t[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            mask_t[i, : len(s)] = True
        out = model(ids_t, attention_mask=mask_t)
        vecs.append(out.sparse_vector.cpu())
    return torch.cat(vecs, dim=0)


def top_expanded_terms(
    sparse_vec: torch.Tensor,
    tokenizer: WpBpePpigTokenizer,
    top_k: int = 12,
) -> list[tuple[str, float]]:
    vals, idxs = torch.topk(sparse_vec, k=min(top_k, sparse_vec.numel()))
    out: list[tuple[str, float]] = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        if v <= 0.0:
            break
        tok = tokenizer.id_to_token(int(i)) or tokenizer.decode([int(i)])
        out.append((tok, round(float(v), 3)))
    return out


def compute_ranking_metrics(
    q_vecs: torch.Tensor,
    d_vecs: torch.Tensor,
    relevant_doc_indices: list[set[int]],
) -> dict[str, float]:
    """Compute MRR@10, nDCG@10, Recall@10, Recall@100, and query/doc L0 sparsity."""
    scores = torch.matmul(q_vecs.float(), d_vecs.float().transpose(0, 1))
    n_q = scores.size(0)
    k_max = min(100, scores.size(1))
    _, top_indices = torch.topk(scores, k=k_max, dim=-1)

    mrr10 = 0.0
    ndcg10 = 0.0
    rec10 = 0.0
    rec100 = 0.0

    for q_idx in range(n_q):
        rel = relevant_doc_indices[q_idx]
        if not rel:
            continue
        ranked = top_indices[q_idx].tolist()
        top10 = ranked[:10]

        for rank, doc_id in enumerate(top10, start=1):
            if doc_id in rel:
                mrr10 += 1.0 / rank
                break

        dcg = 0.0
        for rank, doc_id in enumerate(top10, start=1):
            if doc_id in rel:
                dcg += 1.0 / math.log2(rank + 1.0)
        ideal_hits = min(len(rel), 10)
        idcg = sum(1.0 / math.log2(r + 1.0) for r in range(1, ideal_hits + 1))
        if idcg > 0.0:
            ndcg10 += dcg / idcg

        rec10 += sum(1 for d in top10 if d in rel) / len(rel)
        rec100 += sum(1 for d in ranked if d in rel) / len(rel)

    l0_q = float((q_vecs > 0).float().sum(dim=-1).mean().item())
    l0_d = float((d_vecs > 0).float().sum(dim=-1).mean().item())
    return {
        "mrr@10": mrr10 / max(1, n_q),
        "ndcg@10": ndcg10 / max(1, n_q),
        "recall@10": rec10 / max(1, n_q),
        "recall@100": rec100 / max(1, n_q),
        "l0_q": l0_q,
        "l0_d": l0_d,
    }


def evaluate_beir_dataset(
    model: CroutonSPLADE,
    tokenizer: WpBpePpigTokenizer,
    device: torch.device,
    dataset_name: str = "scifact",
    max_queries: int = 300,
    max_docs: int = 5000,
    encode_fn=None,
) -> dict[str, Any]:
    """Evaluate on a BEIR benchmark (e.g., scifact, nfcorpus, fiqa) via HuggingFace `BeIR/<dataset>`."""
    from datasets import load_dataset

    corpus_ds = load_dataset(f"BeIR/{dataset_name}", "corpus", split="corpus")
    queries_ds = load_dataset(f"BeIR/{dataset_name}", "queries", split="queries")
    qrels_ds = load_dataset(f"BeIR/{dataset_name}-qrels", split="test")

    q_id_to_text = {str(r["_id"]): str(r["text"]) for r in queries_ds}
    qrels_map: dict[str, set[str]] = {}
    for r in qrels_ds:
        if int(r.get("score", 1)) > 0:
            qid = str(r["query-id"])
            did = str(r["corpus-id"])
            qrels_map.setdefault(qid, set()).add(did)

    active_qids = [qid for qid in qrels_map if qid in q_id_to_text][:max_queries]
    must_keep_dids = {did for qid in active_qids for did in qrels_map[qid]}

    doc_ids: list[str] = []
    doc_texts: list[str] = []
    for r in corpus_ds:
        did = str(r["_id"])
        if did in must_keep_dids:
            title = str(r.get("title", "")).strip()
            body = str(r.get("text", "")).strip()
            doc_ids.append(did)
            doc_texts.append(f"{title}. {body}".strip(". ") if title else body)

    for r in corpus_ds:
        if len(doc_ids) >= max_docs:
            break
        did = str(r["_id"])
        if did not in must_keep_dids:
            title = str(r.get("title", "")).strip()
            body = str(r.get("text", "")).strip()
            doc_ids.append(did)
            doc_texts.append(f"{title}. {body}".strip(". ") if title else body)

    did_to_idx = {did: i for i, did in enumerate(doc_ids)}
    queries = [q_id_to_text[qid] for qid in active_qids]
    rel_indices = [{did_to_idx[d] for d in qrels_map[qid] if d in did_to_idx} for qid in active_qids]

    encode_fn = encode_fn or encode_texts_splade
    q_vecs = encode_fn(model, tokenizer, queries, device=device, max_len=64)
    d_vecs = encode_fn(model, tokenizer, doc_texts, device=device, max_len=256)
    metrics = compute_ranking_metrics(q_vecs, d_vecs, rel_indices)
    metrics["n_queries"] = float(len(queries))
    metrics["n_docs"] = float(len(doc_texts))
    return metrics


@torch.no_grad()
def encode_texts_hf_splade(model, tokenizer, texts: list[str], device: torch.device, max_len: int = 256, batch_size: int = 32) -> torch.Tensor:
    """Reference SPLADE encoder for HF MLM checkpoints (naver/splade-v3 etc.): max-pool log1p(ReLU(logits))."""
    if device.type != "cuda":
        batch_size = min(batch_size, 8)  # (B, T, 30522) fp32 logits exceed a few GB on CPU otherwise
    vecs: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(texts[start : start + batch_size], max_length=max_len, truncation=True, padding=True, return_tensors="pt").to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(**enc).logits
        vecs.append((torch.log1p(torch.relu(logits.float())) * enc["attention_mask"].unsqueeze(-1)).amax(dim=1))
    return torch.cat(vecs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CroutonSPLADE sparse retrieval.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--hf-baseline", type=str, default=None, help="Score an HF SPLADE model (e.g. naver/splade-v3) on the same pool instead")
    parser.add_argument("--tokenizer", type=str, default="artifacts/tokenizer/wp_bpe_ppig_32k.json")
    parser.add_argument("--beir-task", type=str, default="scifact", help="BEIR dataset name (e.g., scifact, nfcorpus, fiqa)")
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--max-docs", type=int, default=5000)
    args = parser.parse_args()

    device = get_device()
    if args.hf_baseline:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        hf_tok = AutoTokenizer.from_pretrained(args.hf_baseline)
        hf_model = AutoModelForMaskedLM.from_pretrained(args.hf_baseline).to(device).eval()
        metrics = evaluate_beir_dataset(
            hf_model, hf_tok, device=device, dataset_name=args.beir_task,
            max_queries=args.max_queries, max_docs=args.max_docs, encode_fn=encode_texts_hf_splade,
        )
        print(f"\n=== {args.hf_baseline} BEIR {args.beir_task} ({int(metrics['n_queries'])} queries, {int(metrics['n_docs'])} docs) ===")
        for k in ("ndcg@10", "mrr@10", "recall@10", "recall@100", "l0_q", "l0_d"):
            print(f"  {k:12s}: {metrics[k]:.4f}")
        return
    if not args.checkpoint:
        parser.error("--checkpoint is required unless --hf-baseline is given")
    tokenizer = WpBpePpigTokenizer.from_pretrained(args.tokenizer)
    model, _ = load_splade_model(args.checkpoint, device=device)

    sample_queries = ["what is a binary search tree", "python function to sort a list"]
    s_vecs = encode_texts_splade(model, tokenizer, sample_queries, device=device, max_len=64)
    for q, v in zip(sample_queries, s_vecs):
        print(f"Query: {q!r} -> top terms: {top_expanded_terms(v, tokenizer, top_k=10)}")

    metrics = evaluate_beir_dataset(
        model,
        tokenizer,
        device=device,
        dataset_name=args.beir_task,
        max_queries=args.max_queries,
        max_docs=args.max_docs,
    )
    print(f"\n=== BEIR {args.beir_task} ({int(metrics['n_queries'])} queries, {int(metrics['n_docs'])} docs) ===")
    for k in ("ndcg@10", "mrr@10", "recall@10", "recall@100", "l0_q", "l0_d"):
        print(f"  {k:12s}: {metrics[k]:.4f}")


if __name__ == "__main__":
    main()
