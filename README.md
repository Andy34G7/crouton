# crouton

A scalable language model family (from ~15M prototypes up to ~300M parameters) designed to test **Merged-LoRA Dense MoE with $\text{ReLU}^2$** and **Hybrid Sparse Attention** against `shortcake`.

## Architecture
- **Flexible Scaling**: Continuous depth, width, head, and expert knobs (`CroutonConfig`).
- **Dense MoE ($\text{ReLU}^2$)**: 2-layer $\text{ReLU}^2$ MLP (`W_down(ReLU(W_up(x))²)`) with a shared base FFN plus input-dependent merged low-rank (LoRA) expert deltas via a learned softmax router (100% differentiable, zero token dropping).
- **Hybrid Sparse Attention**: Alternating Sliding Window Attention (`512` tokens) + attention sinks and Global Full Attention layers (default `3:1` SWA:Global), using `FlexAttention` on CUDA and SDPA fallback on CPU.
- **Context Length**: `2048` in pretraining, extendable to `4096` via YaRN/NTK RoPE scaling in mid-training.

## Tokenizer
- **`wp-bpe-ppig`** from [`../tokenisers`](../tokenisers) trained on our dataset (`32,768` vocab, tied embeddings).
- **Pretokenization**: Custom HF `Sequence` combining a Llama-3/cl100k-style regex `Split` and `Digits(individual_digits=True)`.

## Training & Post-Training Pipeline
1. **Tokenizer Training**: `python -m crouton.tokenizer.train_tokenizer`
2. **Pretraining**: Muon (2D linear weights) + AdamW (embeddings/norms/router) with WSD schedule (`bf16`, gradient accumulation, `torch.compile`) on FineWeb-Edu + Cosmopedia/SmolLM-corpus (`python -m crouton.train.pretrain`).
3. **Stage 1 — Annealing / Mid-training**: WSD decay phase + `4096` context extension on high-quality data (`python -m crouton.train.midtrain`).
4. **Stage 2 — Chat / Instruct SFT**: Assistant-masked SFT on SmolTalk / Tulu-3 (`python -m crouton.train.sft`).
5. **Stage 3 — Preference & Reasoning**: DPO/ORPO (`python -m crouton.train.dpo`) + optional GRPO (`python -m crouton.train.grpo`).
6. **Evaluation**: Validation BPB/PPL + `lm-eval-harness` (`python -m crouton.eval.eval_runner`).
