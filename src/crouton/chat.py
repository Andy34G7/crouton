"""Interactive and one-shot CLI for chatting with or completing from crouton checkpoints."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import sys

import torch
import torch.nn.functional as F
from tokenisers import WpBpePpigTokenizer

from crouton.model import CroutonForCausalLM
from crouton.train.common import get_device, instantiate_checkpoint_model, load_checkpoint_payload


@torch.no_grad()
def stream_generate(
    model: CroutonForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 40,
    eos_token_ids: list[int] | None = None,
    vocab_limit: int | None = None,
) -> Iterator[int]:
    """Yield generated token IDs one by one using KV-cached decoding."""
    if input_ids.ndim != 2 or input_ids.size(0) != 1 or input_ids.size(1) < 1:
        raise ValueError("stream_generate expects input_ids of shape [1, T] with T >= 1")
    stop_set = set(eos_token_ids or [])
    was_training = model.training
    model.eval()
    try:
        ids = input_ids
        past_key_values = None
        for _ in range(max(0, max_new_tokens)):
            if past_key_values is not None and ids.size(1) <= model.config.max_seq_len:
                out = model(ids[:, -1:], past_key_values=past_key_values, use_cache=True)
            else:
                ctx_ids = ids[:, -model.config.max_seq_len :]
                want_cache = ids.size(1) < model.config.max_seq_len
                out = model(ctx_ids, use_cache=want_cache)
            past_key_values = getattr(out, "past_key_values", None)
            logits = out.logits[:, -1, :].float()
            if vocab_limit is not None and 0 < vocab_limit < logits.size(-1):
                logits = logits[:, :vocab_limit]
            if temperature <= 0.0:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                scaled = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                    scaled = scaled.masked_fill(scaled < v[:, [-1]], float("-inf"))
                probs = F.softmax(scaled, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            tok_id = int(next_tok.item())
            if tok_id in stop_set:
                break
            yield tok_id
            ids = torch.cat([ids, next_tok], dim=-1)
    finally:
        if was_training:
            model.train()


def _generate_and_stream_text(
    model: CroutonForCausalLM,
    tokenizer: WpBpePpigTokenizer,
    prompt_text: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    stop_ids: list[int],
    stream: bool = True,
) -> str:
    prompt_ids = tokenizer.encode(prompt_text)
    if not prompt_ids:
        return ""
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    vocab_limit = int(tokenizer.vocab_size)
    gen_ids: list[int] = []
    printed = ""
    for tok_id in stream_generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_ids=stop_ids,
        vocab_limit=vocab_limit,
    ):
        gen_ids.append(tok_id)
        if stream:
            decoded = tokenizer.decode(gen_ids)
            if len(decoded) > len(printed):
                sys.stdout.write(decoded[len(printed) :])
                sys.stdout.flush()
                printed = decoded
    full_text = tokenizer.decode(gen_ids)
    if stream:
        if len(full_text) > len(printed):
            sys.stdout.write(full_text[len(printed) :])
        sys.stdout.write("\n")
        sys.stdout.flush()
    return full_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with or complete from a crouton checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--tokenizer", type=str, default="artifacts/tokenizer/wp_bpe_ppig_32k.json")
    parser.add_argument("--prompt", type=str, default=None, help="Optional one-shot prompt (non-interactive)")
    parser.add_argument("--raw", action="store_true", help="Raw completion mode (no ChatML tags; for base models)")
    parser.add_argument("--system", type=str, default=None, help="Optional system prompt for ChatML mode")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = get_device() if args.device == "auto" else torch.device(args.device)
    tokenizer = WpBpePpigTokenizer.from_pretrained(args.tokenizer)
    ckpt, cfg, step = load_checkpoint_payload(args.checkpoint)
    model = instantiate_checkpoint_model(ckpt, cfg, device).eval()

    im_end = tokenizer.token_to_id("<|im_end|>")
    eos = tokenizer.token_to_id("<|endoftext|>")
    stop_ids = [t for t in ((eos,) if args.raw else (im_end, eos)) if t is not None]

    if args.prompt is not None:
        if args.raw:
            prompt_text = args.prompt
        else:
            prefix = f"<|im_start|>system\n{args.system}<|im_end|>\n" if args.system else ""
            prompt_text = f"{prefix}<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"
        _generate_and_stream_text(
            model,
            tokenizer,
            prompt_text,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            stop_ids=stop_ids,
            stream=True,
        )
        return

    mode = "raw completion" if args.raw else "ChatML"
    print(
        f"Loaded {args.checkpoint} (step={step}, params={model.num_parameters():,}, "
        f"ctx={cfg.max_seq_len}, device={device}, mode={mode})"
    )
    print("Type /clear to reset history, or /exit to quit.")

    base_prefix = f"<|im_start|>system\n{args.system}<|im_end|>\n" if (args.system and not args.raw) else ""
    history = base_prefix

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in ("/exit", "/quit", "exit", "quit"):
            break
        if user_text.lower() == "/clear":
            history = base_prefix
            print("(history cleared)")
            continue

        if args.raw:
            sys.stdout.write("Crouton: ")
            sys.stdout.flush()
            _generate_and_stream_text(
                model,
                tokenizer,
                user_text,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                stop_ids=stop_ids,
                stream=True,
            )
        else:
            history += f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
            sys.stdout.write("Crouton: ")
            sys.stdout.flush()
            reply = _generate_and_stream_text(
                model,
                tokenizer,
                history,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                stop_ids=stop_ids,
                stream=True,
            )
            history += f"{reply.strip()}<|im_end|>\n"


if __name__ == "__main__":
    main()
