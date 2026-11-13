"""Tests for wp-bpe-ppig pretokenizer rules, JSON/dir export, ChatML SFT masking, and multi-turn prefs."""

from tokenisers import WpBpePpigTokenizer, _core

from crouton.data import PreferenceDataset, format_chatml_sft_with_mask
from crouton.tokenizer import build_crouton_pretokenizer
from crouton.tokenizer.train_tokenizer import read_local_corpus, train_crouton_tokenizer


def test_pretokenizer_splits_digits_individually():
    pt = build_crouton_pretokenizer(individual_digits=True)
    segs = _core.pretokenize(b"Year 2026: 42+8=50", pt)
    assert b"2" in segs and b"0" in segs and b"6" in segs
    assert b"2026" not in segs


def test_train_wp_bpe_ppig_json_and_dir_and_sft_masking(tmp_path):
    corpus_file = tmp_path / "corpus.txt"
    lines = [
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>\n",
        "Dense MoE with ReLU^2 and sparse attention.",
    ] * 10
    corpus_file.write_text("\n".join(lines), encoding="utf-8")

    corpus = read_local_corpus(corpus_file)
    assert isinstance(corpus, list) and len(corpus) == 50

    # Single JSON file output
    tok_json = tmp_path / "tok.json"
    tok = train_crouton_tokenizer(corpus, output_path=tok_json, vocab_size=300)
    assert tok_json.is_file()
    tok_reloaded = WpBpePpigTokenizer.from_pretrained(str(tok_json))
    assert tok_reloaded.vocab_size == tok.vocab_size

    # Directory output
    tok_dir = tmp_path / "tok_dir"
    train_crouton_tokenizer(corpus, output_path=tok_dir, vocab_size=300)
    assert tok_dir.is_dir() and (tok_dir / "tokenizer.json").is_file()

    # SFT mask + attention_mask
    x, y, attn_mask = format_chatml_sft_with_mask(
        [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}],
        tokenizer=tok,
        seq_len=64,
    )
    assert x.shape == (64,)
    assert y.shape == (64,)
    assert attn_mask.shape == (64,)
    assert attn_mask.any() and (~attn_mask).any()
    assert (y == -100).any()
    assert (y >= 0).any()

    # Multi-turn preference preservation
    pref_ds = PreferenceDataset(
        pairs=[
            {
                "prompt": [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}, {"role": "user", "content": "Q2"}],
                "chosen": "A2 good",
                "rejected": "A2 bad",
            }
        ],
        tokenizer=tok,
        seq_len=64,
    )
    item = pref_ds[0]
    assert (item["chosen_labels"] != -100).any()
    assert (item["rejected_labels"] != -100).any()
