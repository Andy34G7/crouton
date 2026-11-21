# crouton

A scalable PyTorch language model family (from ~14M prototypes up to ~300M parameters) implementing **Merged-LoRA Dense MoE (Mixture of Experts) with ReLU^2** and **Hybrid Sparse Attention**, trained with a **`wp-bpe-ppig`** tokenizer and evaluated with tokenizer-agnostic Bits-Per-Byte (BPB) and `lm-eval-harness` (enabling comparison against another `crouton` checkpoint via `--compare-checkpoint`, or against external baselines such as `shortcake` via BPB and benchmark task scores).

## Setup & Installation

- **Python**: `>=3.11`
- **Rust toolchain**: `rustc` and `cargo` are required on `PATH` when building the editable PyO3 extension in `../tokenisers`.
- **Sibling dependency**: Requires [`../tokenisers`](../tokenisers) checked out alongside this repository (`pyproject.toml` links `../tokenisers` via `[tool.uv.sources]`).

```bash
# Recommended (using uv)
uv sync --extra dev --extra eval   # includes pytest and lm-eval>=0.4.0
source .venv/bin/activate          # or prefix commands below with `uv run`

# Alternative (pip)
pip install -e ../tokenisers -e ".[dev,eval]"
```

## Model Configurations

`CroutonConfig` ([`src/crouton/config.py`](src/crouton/config.py)) exposes discrete integer hyperparameters (`d_model`, `n_layers`, `n_heads`, `n_kv_heads`, `d_ff`, `n_experts`, `expert_rank`) subject to `d_model % n_heads == 0`, `n_heads % n_kv_heads == 0`, and even `head_dim`. Shipped presets in `configs/`:

| Preset | Total Params | Non-Embedding Params | `d_model` | `n_layers` | `n_heads` / `n_kv_heads` | `d_ff` | `n_experts` (rank) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [`configs/model_tiny.yaml`](configs/model_tiny.yaml) | 14,169,344 (~14.2M) | 5,780,736 | 256 | 8 | 4 / 2 | 768 | 4 (16) |
| [`configs/model_small.yaml`](configs/model_small.yaml) | 84,691,584 (~84.7M) | 63,720,064 | 640 | 16 | 10 / 2 | 1792 | 6 (24) |
| [`configs/model_300m.yaml`](configs/model_300m.yaml) | 298,045,440 (~298.0M) | 264,491,008 | 1024 | 24 | 16 / 4 | 3072 | 8 (32) |

## Architecture

- **Tied Embeddings**: Input and output token embeddings are tied by default (`tie_embeddings=True` in `CroutonConfig`).
- **Merged-LoRA Dense MoE (ReLU^2)** ([`src/crouton/model/moe.py`](src/crouton/model/moe.py)): End-to-end differentiable 2-layer MLP with no token dropping, injecting router-weighted low-rank (LoRA) expert deltas (`up_A`, `up_B`, `down_A`, `down_B`) scaled by `scale = expert_alpha / expert_rank` into both `w_up` (pre-nonlinearity) and `w_down`:
  - `act = ReLU(w_up(x) + scale * sum_e w_e * (x @ up_A[e] @ up_B[e]))^2`
  - `y = w_down(act) + scale * sum_e w_e * (act @ down_A[e] @ down_B[e])`
  - Router entropy (`router_entropy`) is computed per forward pass and logged during pretraining, with no auxiliary load-balancing loss.
- **Hybrid Sparse Attention** ([`src/crouton/model/attention.py`](src/crouton/model/attention.py)): Grouped-Query Attention (GQA) with RMSNorm QK-Norm (`qk_norm=True`), alternating Sliding Window Attention (SWA; `sliding_window=512`, `sink_tokens=4`) and Global Full Attention every `global_layer_interval=4` layers (3:1 ratio; final layer `n_layers - 1` is always Global via `is_global_layer`).
  - **Dispatch**: When no padding `attention_mask` is passed — or when right-padded SFT/DPO batches are unwrapped outside `torch.compile` — Global layers use causal Scaled Dot-Product Attention (SDPA) and SWA layers use cached `FlexAttention` block masks on Ampere+ (`sm_80+`) CUDA when eligible (`use_flex_attention=True`, `seq_len >= 128`, and 128-aligned outside `torch.compile`, with SFT/DPO collators aligning right-padding to 128), falling back to a shared model-level precomputed mask (`build_swa_sink_mask`) with SDPA otherwise. For non-right-padded masks (e.g., left-padded) — or when a padding `attention_mask` is passed inside `torch.compile` — Global and SWA layers use shared model-level `build_padded_attn_mask` tensors with SDPA.
- **Context Length & RoPE Scaling** ([`src/crouton/model/rope.py`](src/crouton/model/rope.py)): Pretrained at `2048` tokens and extendable to `4096` in mid-training via YaRN (`rope_scaling_type="yarn"`, wavelength-band ramp + `mscale`) or NTK-aware base scaling (`"ntk"`).

## Tokenizer

- **Algorithm**: `WpBpePpigTokenizer` (`wp-bpe-ppig`) from the sibling [`../tokenisers`](../tokenisers) repo (`vocab_size=32768`, `length_penalty=0.35`, `min_frequency=2`), registering 4 special tokens: `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, and `<|pad|>`.
- **Pretokenization** ([`src/crouton/tokenizer/pretokenizer.py`](src/crouton/tokenizer/pretokenizer.py)): Hugging Face (HF) `Sequence` pretokenizer (`{"hf": ...}`) combining a Llama-3/cl100k-style regex `Split` (`behavior="Isolated"`) and `Digits(individual_digits=True)` (toggleable via `--no-individual-digits`).
- **Training Corpus** ([`src/crouton/tokenizer/train_tokenizer.py`](src/crouton/tokenizer/train_tokenizer.py)): Streams 50,000 interleaved documents from `HuggingFaceFW/fineweb-edu` (`sample-10BT`) and `HuggingFaceTB/smollm-corpus` (`cosmopedia-v2`) by default (or `--local-file`), exporting a single HF JSON file when `--output` ends in `.json` (or a directory otherwise).

## Training & Post-Training Pipeline

1. **Tokenizer Training**:

```bash
python -m crouton.tokenizer.train_tokenizer \
  --output artifacts/tokenizer/wp_bpe_ppig_32k.json \
  --vocab-size 32768
```

2. **Pretraining** (`Muon` for 2D `blocks.*` weights excluding router/embeddings/head; `AdamW` for 3D LoRA expert tensors, router, embeddings, norms, and untied head; Warmup-Stable-Decay (WSD) schedule; `bf16` autocast and `torch.compile` on CUDA; checkpoints unwrap `_orig_mod.` automatically):

```bash
python -m crouton.train.pretrain \
  --model-config configs/model_300m.yaml \
  --tokenizer artifacts/tokenizer/wp_bpe_ppig_32k.json \
  --output-dir checkpoints/pretrain
```

3. **Stage 1 — Annealing / Mid-training** (WSD cosine decay from `1.0 -> 0.05` with `--warmup-steps 0` default + `4096` YaRN RoPE context extension on `cosmopedia-v2` + `fineweb-edu`, or custom `--dataset` / `--local-file`):

```bash
python -m crouton.train.midtrain \
  --base-checkpoint checkpoints/pretrain/final.pt \
  --tokenizer artifacts/tokenizer/wp_bpe_ppig_32k.json \
  --output-dir checkpoints/midtrain \
  --seq-len 4096 --rope-scale 2.0 --rope-scaling yarn
```

4. **Stage 2 — Chat / Instruct SFT (Supervised Fine-Tuning)** (`AdamW` with token-weighted gradient accumulation, ChatML assistant-only loss masking, left-truncation preserving assistant turns, and `attention_mask` padding on `--preset smoltalk` (`HuggingFaceTB/smoltalk`) or `--preset tulu3` (`allenai/tulu-3-sft-mixture`)):

```bash
python -m crouton.train.sft \
  --base-checkpoint checkpoints/midtrain/final.pt \
  --tokenizer artifacts/tokenizer/wp_bpe_ppig_32k.json \
  --output-dir checkpoints/sft \
  --preset smoltalk --seq-len 4096
```

5. **Stage 3a — Preference Tuning (DPO / ORPO)** (`AdamW`, `seq_len=4096`, synchronized prompt truncation & multi-turn history on `argilla/ultrafeedback-binarized-preferences-cleaned`; add `--orpo --output-dir checkpoints/orpo` for reference-free Odds Ratio Preference Optimization):

```bash
python -m crouton.train.dpo \
  --base-checkpoint checkpoints/sft/final.pt \
  --tokenizer artifacts/tokenizer/wp_bpe_ppig_32k.json \
  --output-dir checkpoints/dpo \
  --seq-len 4096
```

6. **Stage 3b — Reasoning RL via GRPO (Group Relative Policy Optimization, Optional)** (per-token clipped policy ratios and Schulman k3 KL estimator on `openai/gsm8k` with numeric answer verification preferring `#### <answer>`, or `--builtin-math` for offline arithmetic tasks):

```bash
python -m crouton.train.grpo \
  --base-checkpoint checkpoints/dpo/final.pt \
  --tokenizer artifacts/tokenizer/wp_bpe_ppig_32k.json \
  --output-dir checkpoints/grpo
```

7. **Evaluation** (run `lm-eval-harness` benchmarks via `--tasks`, and/or compute validation loss, Perplexity (PPL), and Bits-Per-Byte (BPB) via `--val-file` using one UTF-8 document per non-empty line; defaults to checkpoint `max_seq_len`; pass `checkpoints/grpo/final.pt` if Stage 3b was run, and `--compare-checkpoint` for side-by-side comparison):

```bash
python -m crouton.eval.eval_runner \
  --checkpoint checkpoints/dpo/final.pt \
  --tokenizer artifacts/tokenizer/wp_bpe_ppig_32k.json \
  --tasks hellaswag,arc_easy,piqa

# Optional BPB / PPL evaluation on a UTF-8 validation file (one document per line):
# python -m crouton.eval.eval_runner --checkpoint checkpoints/dpo/final.pt --val-file val.txt
```

## SPLADE Backbone Adaptation (Experimental)

`src/crouton/model/splade.py`, `src/crouton/train/splade.py`, and `src/crouton/eval/eval_splade.py` contain a small experimental test exploring how `crouton` adapts when used as a SPLADE sparse-retrieval backbone (switching to bidirectional attention with a short MNTP warmup and a tied SPLADE vocabulary head, plus a `--hf-baseline` flag for quick side-by-side BEIR checks). Full document retrieval benchmarks and broader SPLADE comparisons are intended to live in a separate repository.

## Verification & Testing

Run the unit and end-to-end smoke test suite (covering SWA+sink masks, YaRN/NTK RoPE, `attention_mask`, `generate()`, `Muon`+`AdamW`+`WSD`, `_orig_mod.` checkpoint loading, `wp-bpe-ppig` pretokenization, and all training/eval stages):

```bash
pytest -v
```
