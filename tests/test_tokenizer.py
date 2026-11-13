"""Tests for wp-bpe-ppig pretokenizer rules and ChatML SFT masking."""

from tokenisers import _core

from crouton.data.posttrain_data import format_chatml_sft
from crouton.tokenizer import build_crouton_pretokenizer
from crouton.tokenizer.train_tokenizer import train_crouton_tokenizer


def test_pretokenizer_splits_digits_individually():
    pt = build_crouton_pretokenizer(individual_digits=True)
    segs = _core.pretokenize(b"Year 2026: 42+8=50", pt)
    assert b"2" in segs and b"0" in segs and b"6" in segs
    assert b"2026" not in segs


def test_train_wp_bpe_ppig_and_sft_masking(tmp_path):
    corpus = [
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>\n",
        "Dense MoE with ReLU^2 and sparse attention.",
    ] * 10
    tok_path = tmp_path / "tok.json"
    tok = train_crouton_tokenizer(corpus, output_path=tok_path, vocab_size=300)
    assert tok_path.exists()

    x, y = format_chatml_sft(
        [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}],
        tokenizer=tok,
        seq_len=64,
    )
    assert x.shape == (64,)
    assert y.shape == (64,)
    # User prompt tokens are masked (-100), assistant tokens are unmasked (>= 0)
    assert (y == -100).any()
    assert (y >= 0).any()
