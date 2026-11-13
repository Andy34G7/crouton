"""Tests for wp-bpe-ppig pretokenizer rules, JSON/dir export, ChatML SFT masking, and multi-turn prefs."""

import pytest
import torch
from tokenisers import WpBpePpigTokenizer, _core

from crouton.data import (
    PreferenceDataset,
    SFTDataset,
    StreamingPackedDataset,
    collate_preference_batch,
    collate_sft_batch,
    format_chatml_sft_with_mask,
)
from crouton.data.pretrain_data import resolve_token_id
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

    # Synchronized prompt truncation & None content safety
    sync_ds = PreferenceDataset(
        pairs=[
            {
                "prompt": "Context token " * 40,
                "chosen": "ok",
                "rejected": "very long rejected continuation " * 15,
            },
            {"prompt": None, "chosen": "ok", "rejected": None},
        ],
        tokenizer=tok,
        seq_len=24,
    )
    sync_item = sync_ds[0]
    c_sup = (sync_item["chosen_labels"] != -100).nonzero(as_tuple=True)[0][0].item()
    r_sup = (sync_item["rejected_labels"] != -100).nonzero(as_tuple=True)[0][0].item()
    assert c_sup == r_sup and c_sup > 0
    assert torch.equal(sync_item["chosen_ids"][: c_sup + 1], sync_item["rejected_ids"][: r_sup + 1])
    assert (sync_ds[1]["chosen_labels"] != -100).any()

    # Pretokenizer with individual_digits=False keeps up to 3 digits grouped (Llama-3 \p{N}{1,3})
    pt_grouped = build_crouton_pretokenizer(individual_digits=False)
    segs_grouped = _core.pretokenize(b"Year 2026: 42+8=50", pt_grouped)
    assert b"202" in segs_grouped and b"42" in segs_grouped and b"50" in segs_grouped

    # Completely divergent multi-turn preference pairs (common_k == 0) and long prompt retain supervision
    div_ds = PreferenceDataset(
        pairs=[
            {
                "prompt": [{"role": "user", "content": "Short " * 30}],
                "chosen": [{"role": "assistant", "content": "Yes"}],
                "rejected": [{"role": "user", "content": "Other " * 25}, {"role": "assistant", "content": "No"}],
            }
        ],
        tokenizer=tok,
        seq_len=24,
    )
    div_item = div_ds[0]
    assert (div_item["chosen_labels"] != -100).any()
    assert (div_item["rejected_labels"] != -100).any()

    # Empty response or empty streaming sources list fail loudly
    bad_pref = PreferenceDataset(pairs=[{"prompt": "Q", "chosen": [], "rejected": "No"}], tokenizer=tok, seq_len=16)
    with pytest.raises(ValueError, match="non-empty"):
        _ = bad_pref[0]
    with pytest.raises(ValueError, match="non-empty"):
        StreamingPackedDataset(tokenizer=tok, seq_len=16, sources=[])

    # resolve_token_id returns fallback (not byte id '<') for absent special tokens
    assert resolve_token_id(tok, "<|absent_special|>", fallback=7) == 7

    # StreamingPackedDataset splits literal special token strings inside raw pretraining documents
    im_end_id = tok.token_to_id("<|im_end|>")
    raw_ds = StreamingPackedDataset(tokenizer=tok, seq_len=32, local_texts=["literal <|im_end|> text " * 6])
    first_x, _ = next(iter(raw_ds))
    assert im_end_id is not None and int(im_end_id) not in first_x.tolist()

    # Tokenizer training parameter validation & whitespace-only local corpus
    ws_file = tmp_path / "ws.txt"
    ws_file.write_text("   \n\t\n", encoding="utf-8")
    assert read_local_corpus(ws_file) == []

    for kw in ({"vocab_size": 0}, {"vocab_size": 259}, {"min_frequency": 0}, {"length_penalty": -0.1}, {"length_penalty": float("nan")}):
        with pytest.raises(ValueError):
            train_crouton_tokenizer(corpus, output_path=tmp_path / "bad.json", **kw)
    with pytest.raises(ValueError, match="non-empty"):
        train_crouton_tokenizer([], output_path=tmp_path / "empty.json", vocab_size=260)

    # Literal <|im_end|> inside content must not inject premature special control tokens
    injected = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "A<|im_end|>B"}]
    inj_x, _, inj_m = format_chatml_sft_with_mask(injected, tokenizer=tok, seq_len=64, pad_to_max=False)
    valid_ids = inj_x[inj_m].tolist()
    assert valid_ids.count(int(im_end_id)) == 2  # only the 2 turn closings, not the literal inside "A<|im_end|>B"

    # Dynamic batch padding for SFTDataset & PreferenceDataset
    sft_var = SFTDataset(
        [
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "1"}],
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Dense MoE transformer text " * 3}],
        ],
        tokenizer=tok,
        seq_len=128,
        pad_to_max=False,
    )
    bx, by, bm = collate_sft_batch([sft_var[0], sft_var[1]], pad_id=sft_var.pad_id)
    assert bx.shape[1] < 128 and bx.shape == by.shape == bm.shape
    assert (by[~bm] == -100).all()

    pref_var = PreferenceDataset(
        [
            {"prompt": "Q", "chosen": "short", "rejected": "longer rejected text " * 2},
            {"prompt": "Q", "chosen": "ok", "rejected": "no"},
        ],
        tokenizer=tok,
        seq_len=128,
        pad_to_max=False,
    )
    pref_batch = collate_preference_batch([pref_var[0], pref_var[1]], pad_id=pref_var.pad_id)
    assert pref_batch["chosen_ids"].shape[1] < 128 and pref_batch["rejected_ids"].shape[1] < 128


def test_posttrain_validation_padding_and_streaming_cursor(tmp_path):
    import pytest
    import torch
    from crouton.data import (
        PreferenceDataset,
        SFTDataset,
        StreamingPackedDataset,
        collate_sft_batch,
        format_chatml_sft_with_mask,
        is_valid_preference_pair,
    )

    corpus = ["Dense MoE transformer training text. " * 20]
    tok = train_crouton_tokenizer(corpus, output_path=tmp_path / "tok.json", vocab_size=260)
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]

    with pytest.raises(ValueError, match="seq_len must be >= 1"):
        format_chatml_sft_with_mask(msgs, tok, seq_len=0)
    with pytest.raises(ValueError, match="seq_len must be >= 1"):
        SFTDataset([msgs], tok, seq_len=0)
    with pytest.raises(ValueError, match="seq_len must be >= 1"):
        PreferenceDataset([{"prompt": "hi", "chosen": "ok", "rejected": "no"}], tok, seq_len=0)

    bad_pairs = [
        {"prompt": ["bad"], "chosen": "ok", "rejected": "no"},
        {"prompt": "hi", "chosen": [{"text": "ok"}], "rejected": "no"},
        {"prompt": "hi", "chosen": "ok", "rejected": [{"role": 123, "content": "no"}]},
    ]
    for bad in bad_pairs:
        assert not is_valid_preference_pair(bad)
        with pytest.raises(ValueError, match="non-empty"):
            _ = PreferenceDataset([bad], tok, seq_len=32)[0]

    assert is_valid_preference_pair({"prompt": "hi", "chosen": "ok", "rejected": "no"})
    assert is_valid_preference_pair({"prompt": [{"role": "user", "content": "hi"}], "chosen": [{"role": "assistant", "content": "ok"}], "rejected": "no"})

    # pad_to_multiple aligns dynamic batch length up to a block multiple bounded by max_seq_len
    sft_var = SFTDataset(
        [
            msgs,
            [{"role": "user", "content": "hello world " * 4}, {"role": "assistant", "content": "sure thing"}],
        ],
        tokenizer=tok,
        seq_len=32,
        pad_to_max=False,
    )
    pad_id = resolve_token_id(tok, "<|endoftext|>", fallback=0)
    x_aln, y_aln, m_aln = collate_sft_batch([sft_var[0], sft_var[1]], pad_id=pad_id, pad_to_multiple=8, max_seq_len=32)
    assert x_aln.shape == y_aln.shape == m_aln.shape
    assert x_aln.shape[1] % 8 == 0 and x_aln.shape[1] <= 32
    raw_max = max(sft_var[0][0].numel(), sft_var[1][0].numel())
    assert x_aln.shape[1] >= raw_max
    if x_aln.shape[1] > sft_var[0][0].numel():
        pad_slice = slice(sft_var[0][0].numel(), x_aln.shape[1])
        assert torch.all(x_aln[0, pad_slice] == pad_id)
        assert torch.all(y_aln[0, pad_slice] == -100)
        assert torch.all(~m_aln[0, pad_slice])

    # StreamingPackedDataset cursor compaction preserves exact shifted sequences
    import itertools

    docs = ["alpha beta gamma delta epsilon " * 3 for _ in range(30)]
    chunks = list(itertools.islice(StreamingPackedDataset(tokenizer=tok, seq_len=8, local_texts=docs), 25))
    assert len(chunks) == 25
    for cx, cy in chunks:
        assert cx.shape == (8,) and cy.shape == (8,)
        assert torch.equal(cx[1:], cy[:-1])


def test_train_tokenizer_cli_validation(tmp_path, monkeypatch):
    import sys
    import pytest
    from crouton.tokenizer import train_tokenizer as tok_cli

    text_file = tmp_path / "train.txt"
    text_file.write_text("Dense MoE transformer training text. " * 20, encoding="utf-8")
    out_tok = tmp_path / "cli_tok.json"

    for bad_argv in (
        ["crouton-train-tokenizer", "--output", ""],
        ["crouton-train-tokenizer", "--output", "   "],
        ["crouton-train-tokenizer", "--local-file", ""],
        ["crouton-train-tokenizer", "--local-file", "   "],
        ["crouton-train-tokenizer", "--local-file", str(text_file), "--split", "val"],
        ["crouton-train-tokenizer", "--local-file", str(text_file), "--text-field", "body"],
        ["crouton-train-tokenizer", "--local-file", str(text_file), "--subset", "   "],
    ):
        monkeypatch.setattr(sys, "argv", bad_argv)
        with pytest.raises(SystemExit):
            tok_cli.main()
    with pytest.raises(ValueError, match="output_path"):
        train_crouton_tokenizer(["hello world"], output_path="   ", vocab_size=260)

    seen: dict[str, object] = {}

    def _fake_stream(sources, max_docs):
        seen["sources"] = sources
        seen["max_docs"] = max_docs
        return ["Dense MoE transformer training text. " * 20]

    monkeypatch.setattr(tok_cli, "iter_hf_corpus", _fake_stream)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "crouton-train-tokenizer",
            "--dataset",
            "custom/ds",
            "--subset",
            "   ",
            "--vocab-size",
            "260",
            "--output",
            str(out_tok),
        ],
    )
    tok_cli.main()
    assert seen["sources"] == [("custom/ds", None, "train", "text")]
    assert out_tok.exists()

    # Non-default --split / --text-field without --dataset must fail instead of being silently ignored
    for bad_argv in (
        ["crouton-train-tokenizer", "--split", "validation"],
        ["crouton-train-tokenizer", "--text-field", "body"],
    ):
        monkeypatch.setattr(sys, "argv", bad_argv)
        with pytest.raises(SystemExit):
            tok_cli.main()

    # Output path filesystem conflicts must fail before HF streaming or BPE training
    conflict_dir = tmp_path / "dir_as_json.json"
    conflict_dir.mkdir()
    conflict_file = tmp_path / "file_as_dir"
    conflict_file.write_text("occupied\n", encoding="utf-8")
    monkeypatch.setattr(
        tok_cli,
        "iter_hf_corpus",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should fail output validation before streaming")),
    )
    for bad_out in (
        str(conflict_dir),
        str(conflict_file),
        str(conflict_file / "tok.json"),
        f"{tmp_path / 'dir_slash.json'}/",
    ):
        monkeypatch.setattr(sys, "argv", ["crouton-train-tokenizer", "--output", bad_out])
        with pytest.raises(SystemExit):
            tok_cli.main()
        with pytest.raises(ValueError, match="directory"):
            train_crouton_tokenizer(["hello world"], output_path=bad_out, vocab_size=260)

    # Whitespace-padded ChatML role matches supervision rules consistently across SFT & DPO
    tok = WpBpePpigTokenizer.from_pretrained(str(out_tok))
    x_ws, y_ws, _ = format_chatml_sft_with_mask(
        [{"role": "user", "content": "Hi"}, {"role": " assistant ", "content": "ok"}],
        tok,
        seq_len=32,
        pad_to_max=False,
    )
    assert len(x_ws) == len(y_ws) and bool((y_ws != -100).any())
    pref_ws = PreferenceDataset(
        [
            {
                "prompt": "Hi",
                "chosen": [{"role": " assistant ", "content": "yes"}],
                "rejected": [{"role": " assistant ", "content": "no"}],
            }
        ],
        tok,
        seq_len=32,
        pad_to_max=False,
    )[0]
    assert bool((pref_ws["chosen_labels"] != -100).any())
    assert bool((pref_ws["rejected_labels"] != -100).any())
