"""Tests for Muon + AdamW, WSD, checkpoint _orig_mod handling, and end-to-end training/eval stages."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from crouton.config import CroutonConfig, TrainConfig
from crouton.data import StreamingPackedDataset, format_chatml_sft_with_mask
from crouton.eval import CroutonLM, evaluate_bpb
from crouton.model import CroutonForCausalLM
from crouton.optim import build_optimizers, wsd_lr_multiplier
from crouton.tokenizer.train_tokenizer import train_crouton_tokenizer
from crouton.train.common import (
    load_checkpoint,
    next_or_restart,
    safe_decode,
    save_checkpoint,
    strip_orig_mod_prefix,
    validate_tokenizer_vocab,
)
from crouton.train.dpo import dpo_loss, orpo_loss, run_preference_tuning
from crouton.train.grpo import (
    _gather_completion_logps,
    build_builtin_math_tasks,
    extract_numeric_answer,
    grpo_step_loss,
    run_grpo,
)
from crouton.train.midtrain import run_midtrain
from crouton.train.pretrain import run_pretrain, run_validation_bpb
from crouton.train.sft import run_sft


def _tiny_cfg() -> CroutonConfig:
    return CroutonConfig(
        vocab_size=300,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        n_experts=2,
        expert_rank=4,
        max_seq_len=32,
        sliding_window=8,
        sink_tokens=2,
        global_layer_interval=2,
        rope_original_max_seq_len=32,
        use_flex_attention=False,
    )


def test_muon_adamw_step():
    model = CroutonForCausalLM(_tiny_cfg())
    muon_opt, adamw_opt = build_optimizers(model)
    assert len(muon_opt.param_groups[0]["params"]) > 0

    x = torch.randint(0, 300, (2, 8))
    y = torch.randint(0, 300, (2, 8))
    loss = model(x, targets=y).loss
    loss.backward()
    muon_opt.step()
    adamw_opt.step()


def test_wsd_schedule():
    assert wsd_lr_multiplier(0, total_steps=100, warmup_steps=10, decay_steps=20) == 0.1
    assert wsd_lr_multiplier(50, total_steps=100, warmup_steps=10, decay_steps=20) == 1.0
    assert abs(wsd_lr_multiplier(100, total_steps=100, warmup_steps=10, decay_steps=20, min_lr_ratio=0.1) - 0.1) < 1e-6


def test_dpo_and_grpo_losses():
    p_c = torch.tensor([-1.0, -2.0], requires_grad=True)
    p_r = torch.tensor([-3.0, -4.0], requires_grad=True)
    loss = dpo_loss(p_c, p_r, torch.tensor([-1.5, -2.5]), torch.tensor([-2.5, -3.5]), beta=0.1)
    loss.backward()
    assert torch.isfinite(loss)

    pol = torch.tensor([-1.0, -1.5, -2.0, -0.5], requires_grad=True)
    old = pol.detach()
    ref = torch.tensor([-1.1, -1.4, -2.1, -0.6])
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])
    g_loss = grpo_step_loss(pol, old, ref, rewards)
    g_loss.backward()
    assert torch.isfinite(g_loss)

    # Per-token [G, T=128] GRPO KL must not overflow fp32 even with large drift, and G=1 must not NaN
    pol_tok = torch.full((2, 128), -2.0, requires_grad=True)
    old_tok = torch.full((2, 128), -2.0)
    ref_tok = torch.full((2, 128), -0.5)
    mask_tok = torch.ones((2, 128), dtype=torch.bool)
    g_tok_loss = grpo_step_loss(pol_tok, old_tok, ref_tok, torch.tensor([1.0, 0.0]), completion_mask=mask_tok)
    g_tok_loss.backward()
    assert torch.isfinite(g_tok_loss) and torch.isfinite(pol_tok.grad).all()

    g1_loss = grpo_step_loss(pol_tok[:1], old_tok[:1], ref_tok[:1], torch.tensor([1.0]))
    assert torch.isfinite(g1_loss)

    # ORPO stability near P -> 1
    orpo_near_zero = orpo_loss(
        policy_chosen_logps=torch.tensor([-1e-6], requires_grad=True),
        policy_rejected_logps=torch.tensor([-5.0]),
        chosen_sft_loss=torch.tensor(0.01, requires_grad=True),
        chosen_lengths=torch.tensor([1]),
        rejected_lengths=torch.tensor([5]),
    )
    orpo_near_zero.backward()
    assert torch.isfinite(orpo_near_zero)

    # GSM8K numeric extraction edge cases
    assert extract_numeric_answer("Total is #### 1,200") == "1200"
    assert extract_numeric_answer("Final #### 42 (step 1)") == "42"
    assert extract_numeric_answer("Answer is 99.") == "99"


def test_build_optimizers_handles_orig_mod_wrapper():
    class _OrigModWrapper(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self._orig_mod = mod

    class _DDPWrapper(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.module = mod

    wrapped = _DDPWrapper(_OrigModWrapper(CroutonForCausalLM(_tiny_cfg())))
    muon_opt, _ = build_optimizers(wrapped)
    assert len(muon_opt.param_groups[0]["params"]) > 0


def test_next_or_restart_and_empty_dataset_guards(tmp_path):
    # Finite loader wraps cleanly on exhaustion
    dl = DataLoader([torch.tensor([1]), torch.tensor([2])], batch_size=1)
    it = iter(dl)
    vals = []
    for _ in range(5):
        v, it = next_or_restart(it, dl)
        vals.append(int(v.item()))
    assert vals == [1, 2, 1, 2, 1]

    empty_dl = DataLoader([], batch_size=1)
    with pytest.raises(RuntimeError, match="zero batches"):
        next_or_restart(iter(empty_dl), empty_dl)

    with pytest.raises(ValueError, match="vocab_size"):
        validate_tokenizer_vocab(SimpleNamespace(vocab_size=999), _tiny_cfg())

    # Finite short local_file runs without StopIteration even if steps > distinct packed chunks
    tok_path = tmp_path / "tok.json"
    train_crouton_tokenizer(["hello world foo bar baz"] * 3, output_path=tok_path, vocab_size=300)
    short_file = tmp_path / "short.txt"
    short_file.write_text("hello world foo bar baz " * 4, encoding="utf-8")
    cfg = _tiny_cfg()
    tcfg = TrainConfig(
        output_dir=str(tmp_path / "pre_short"),
        tokenizer_path=str(tok_path),
        seq_len=8,
        micro_batch_size=1,
        grad_accum_steps=1,
        total_steps=5,
        warmup_steps=1,
        decay_steps=1,
        compile_model=False,
        bf16=False,
    )
    run_pretrain(cfg, tcfg, local_file=str(short_file))
    assert (tmp_path / "pre_short" / "final.pt").exists()


def test_streaming_dataset_terminates_on_exhaustion(tmp_path):
    corpus = ["Hello world."] * 5
    tok_path = tmp_path / "tok.json"
    tok = train_crouton_tokenizer(corpus, output_path=tok_path, vocab_size=300)
    assert list(StreamingPackedDataset(tokenizer=tok, seq_len=16, local_texts=[])) == []
    assert list(StreamingPackedDataset(tokenizer=tok, seq_len=16, local_texts=["   ", "\n\t"])) == []

    # Long prompt left-truncation keeps assistant targets
    long_user = "word " * 100
    _, y, _ = format_chatml_sft_with_mask(
        [{"role": "user", "content": long_user}, {"role": "assistant", "content": "ok"}],
        tokenizer=tok,
        seq_len=16,
    )
    assert (y != -100).any()


def test_gather_completion_logps_clipped_prompt():
    model = CroutonForCausalLM(_tiny_cfg())
    seq = torch.randint(0, 300, (1, 14))
    logps = _gather_completion_logps(model, seq=seq, prompt_len=6, max_seq_len=5)
    assert logps.shape == (5,) and torch.isfinite(logps).all()


def test_strip_orig_mod_checkpoint_roundtrip(tmp_path):
    cfg = _tiny_cfg()
    model = CroutonForCausalLM(cfg)
    prefixed = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
    ckpt_path = tmp_path / "compiled.pt"
    torch.save({"model": prefixed, "config": cfg.to_dict(), "step": 5}, ckpt_path)

    loaded, loaded_cfg, step = load_checkpoint(ckpt_path, device=torch.device("cpu"))
    assert step == 5
    assert loaded_cfg.d_model == cfg.d_model
    assert set(strip_orig_mod_prefix(prefixed).keys()) == set(loaded.state_dict().keys())


def test_end_to_end_pipeline_and_eval_smoke(tmp_path, monkeypatch):
    corpus = [
        "Dense MoE with ReLU^2 and sparse attention scales cleanly from 15M to 300M.",
        "Compute 12 + 30. End with #### 42.",
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n",
    ] * 25
    tok_path = tmp_path / "tok.json"
    tok = train_crouton_tokenizer(corpus, output_path=tok_path, vocab_size=300)

    local_file = tmp_path / "train.txt"
    local_file.write_text("\n".join(corpus), encoding="utf-8")

    cfg = _tiny_cfg()
    cfg.vocab_size = tok.vocab_size

    pretrain_dir = tmp_path / "pretrain"
    train_cfg = TrainConfig(
        output_dir=str(pretrain_dir),
        tokenizer_path=str(tok_path),
        seq_len=16,
        micro_batch_size=1,
        grad_accum_steps=1,
        total_steps=2,
        warmup_steps=1,
        decay_steps=1,
        eval_interval=1,
        save_interval=10,
        compile_model=False,
        bf16=False,
    )
    run_pretrain(cfg, train_cfg, local_file=str(local_file), val_file=str(local_file))
    pre_ckpt = pretrain_dir / "final.pt"
    assert pre_ckpt.exists()

    midtrain_dir = tmp_path / "midtrain"
    run_midtrain(
        base_checkpoint=str(pre_ckpt),
        tokenizer_path=str(tok_path),
        output_dir=str(midtrain_dir),
        seq_len=32,
        rope_scale_factor=2.0,
        total_steps=1,
        micro_batch_size=1,
        grad_accum_steps=1,
        local_file=str(local_file),
    )
    mid_ckpt = midtrain_dir / "final.pt"
    assert mid_ckpt.exists()

    sft_dir = tmp_path / "sft"
    run_sft(
        base_checkpoint=str(mid_ckpt),
        tokenizer_path=str(tok_path),
        output_dir=str(sft_dir),
        seq_len=32,
        total_steps=1,
        micro_batch_size=2,
        grad_accum_steps=1,
        examples_override=[
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
            [{"role": "user", "content": "Explain Dense MoE briefly."}, {"role": "assistant", "content": "Dense MoE with ReLU^2!"}],
        ],
    )
    sft_ckpt = sft_dir / "final.pt"
    assert sft_ckpt.exists()

    orpo_dir = tmp_path / "orpo"
    run_preference_tuning(
        base_checkpoint=str(sft_ckpt),
        tokenizer_path=str(tok_path),
        output_dir=str(orpo_dir),
        seq_len=32,
        total_steps=1,
        micro_batch_size=1,
        grad_accum_steps=1,
        use_orpo=True,
        pairs_override=[{"prompt": "Hi", "chosen": "Hello!", "rejected": "Bye"}] * 2,
    )
    orpo_ckpt = orpo_dir / "final.pt"
    assert orpo_ckpt.exists()

    dpo_dir = tmp_path / "dpo"
    run_preference_tuning(
        base_checkpoint=str(sft_ckpt),
        tokenizer_path=str(tok_path),
        output_dir=str(dpo_dir),
        seq_len=32,
        total_steps=1,
        micro_batch_size=1,
        grad_accum_steps=1,
        use_orpo=False,
        pairs_override=[{"prompt": "Hi", "chosen": "Hello!", "rejected": "Bye"}] * 2,
    )
    assert (dpo_dir / "final.pt").exists()

    with pytest.raises(ValueError, match="zero valid"):
        run_sft(str(mid_ckpt), str(tok_path), str(tmp_path / "bad_sft"), examples_override=[])
    with pytest.raises(ValueError, match="zero valid"):
        run_preference_tuning(str(sft_ckpt), str(tok_path), str(tmp_path / "bad_dpo"), pairs_override=[])

    grpo_dir = tmp_path / "grpo"
    run_grpo(
        base_checkpoint=str(orpo_ckpt),
        tokenizer_path=str(tok_path),
        output_dir=str(grpo_dir),
        prompts_with_verifiers=build_builtin_math_tasks(n=2),
        group_size=2,
        steps=1,
        max_new_tokens=4,
    )
    grpo_ckpt = grpo_dir / "final.pt"
    assert grpo_ckpt.exists()

    bpb_metrics = evaluate_bpb(str(grpo_ckpt), str(tok_path), corpus[:3])
    assert bpb_metrics["bpb"] > 0.0

    lm = CroutonLM(str(grpo_ckpt), str(tok_path), device="cpu")
    ll_res = lm.loglikelihood([SimpleNamespace(args=("Hi", " Hello!"))])
    ll_nl = lm.loglikelihood([SimpleNamespace(args=("Q:\n", "42"))])
    ll_nl_in_cont = lm.loglikelihood([SimpleNamespace(args=("Q:", "\n42"))])
    assert len(ll_res) == 1 and isinstance(ll_res[0][0], float)
    # Preserving trailing '\n' in context avoids falsely scoring '\n' as part of digit continuation "42"
    assert ll_nl[0] != ll_nl_in_cont[0]

    # When leading-space tokens are merged in BPE, ("Dense", " MoE") and ("Dense ", "MoE") score identically
    full_tok_path = tmp_path / "full_tok.json"
    full_tok = train_crouton_tokenizer(corpus, output_path=full_tok_path, vocab_size=350)
    full_cfg = _tiny_cfg()
    full_cfg.vocab_size = full_tok.vocab_size
    full_ckpt = tmp_path / "full_tok.pt"
    save_checkpoint(full_ckpt, CroutonForCausalLM(full_cfg), full_cfg, step=1)
    full_lm = CroutonLM(str(full_ckpt), str(full_tok_path), device="cpu")
    assert full_lm.loglikelihood([SimpleNamespace(args=("Dense", " MoE"))]) == full_lm.loglikelihood(
        [SimpleNamespace(args=("Dense ", "MoE"))]
    )

    # Verify CroutonLM never triggers read-only @property device setters on LM bases
    assert isinstance(type(lm).device, property)
    assert lm.device == torch.device("cpu")
    with pytest.raises(AttributeError):
        lm.device = torch.device("cpu")  # type: ignore[misc]

    long_text = "Dense MoE with ReLU^2 and sparse attention. " * 20
    roll_res = lm.loglikelihood_rolling([SimpleNamespace(args=(long_text,))])
    assert len(roll_res) == 1 and isinstance(roll_res[0], float) and roll_res[0] < 0.0
    gen_res = lm.generate_until([SimpleNamespace(args=("Hi", {"max_gen_toks": 4, "until": "\n"}))])
    assert len(gen_res) == 1 and isinstance(gen_res[0], str)

    # Verify sliding-window BPB parity between pretrain.run_validation_bpb and eval.evaluate_bpb
    long_docs = [long_text, "Sparse attention and ReLU^2 MLPs scale cleanly across documents. " * 12]
    bpb_eval = evaluate_bpb(str(grpo_ckpt), str(tok_path), long_docs, seq_len=16)
    loaded_model, _, _ = load_checkpoint(str(grpo_ckpt), device=torch.device("cpu"))
    val_loss, val_bpb = run_validation_bpb(
        loaded_model, tok, long_docs, seq_len=16, device=torch.device("cpu"), amp_ctx=nullcontext()
    )
    assert pytest.approx(bpb_eval["loss_nats"], rel=1e-6) == val_loss
    assert pytest.approx(bpb_eval["bpb"], rel=1e-6) == val_bpb

    # Verify loglikelihood preserves newline / space before digits, and generate_until handles None kwargs
    ll_nl = lm.loglikelihood([SimpleNamespace(args=("Q:\n", "42")), SimpleNamespace(args=("A: ", "42"))])
    assert len(ll_nl) == 2 and all(isinstance(v[0], float) for v in ll_nl)
    gen_edge = lm.generate_until(
        [
            SimpleNamespace(args=("Hi", {"until": None, "max_gen_toks": None, "top_k": None})),
            SimpleNamespace(args=("Hi", {"max_gen_toks": 0})),
        ]
    )
    assert len(gen_edge) == 2 and gen_edge[1] == ""
    # Verify stop-string truncation picks the earliest occurrence regardless of `until` order
    suffix_ids = torch.tensor([tok.encode("alpha beta STOP1 gamma STOP2")], dtype=torch.long)
    monkeypatch.setattr(lm.model, "generate", lambda inp, *a, **k: torch.cat([inp, suffix_ids.to(inp.device)], dim=1))
    stop_out = lm.generate_until([SimpleNamespace(args=("Hi", {"max_gen_toks": 32, "until": ["STOP2", "STOP1"]}))])
    assert stop_out == ["alpha beta "]

    # Verify TokenizerCompat + apply_chat_template + safe_decode + padded-vocab generation
    assert lm.tokenizer.eos_token == "<|endoftext|>"
    assert lm.tokenizer.pad_token_id == lm.eot_token_id
    assert lm.tokenizer_name == "tok.json"
    chat_str = lm.apply_chat_template([{"role": "user", "content": "2+2?"}], add_generation_prompt=True)
    assert chat_str == "<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n"
    assert isinstance(safe_decode(tok, [0, tok.vocab_size + 50, 999999]), str)

    padded_cfg = _tiny_cfg()
    padded_cfg.vocab_size = tok.vocab_size + 32
    padded_model = CroutonForCausalLM(padded_cfg)
    padded_ckpt = tmp_path / "padded.pt"
    save_checkpoint(padded_ckpt, padded_model, padded_cfg, step=1)
    padded_lm = CroutonLM(str(padded_ckpt), str(tok_path), device="cpu")
    padded_out = padded_lm.generate_until([SimpleNamespace(args=("Hi", {"max_gen_toks": 4}))])
    assert len(padded_out) == 1 and isinstance(padded_out[0], str)


def test_utils_and_cli_edge_cases(tmp_path, monkeypatch):
    # Stacked _orig_mod / module prefix stripping and WSD edge schedules
    assert strip_orig_mod_prefix({"_orig_mod.module._orig_mod.w": 1}) == {"w": 1}
    assert wsd_lr_multiplier(0, total_steps=1, warmup_steps=0, decay_steps=0) == 1.0
    assert 0.0 < wsd_lr_multiplier(0, total_steps=10, warmup_steps=4, decay_steps=4) <= 1.0

    # pretrain --save-interval 0 runs without ZeroDivisionError
    from crouton.train import pretrain as pretrain_mod
    from crouton.train import sft as sft_mod

    data_file = tmp_path / "local.txt"
    data_file.write_text("Dense MoE transformer training text. " * 20, encoding="utf-8")
    tok_path = tmp_path / "tok.json"
    tok = train_crouton_tokenizer([data_file.read_text(encoding="utf-8")], output_path=tok_path, vocab_size=260)
    assert tok.vocab_size <= 260

    cfg_file = tmp_path / "tiny.yaml"
    cfg_file.write_text("vocab_size: 300\nd_model: 32\nn_layers: 2\nn_heads: 4\nn_kv_heads: 2\nd_ff: 64\nn_experts: 2\nexpert_rank: 4\nmax_seq_len: 16\nsliding_window: 4\nsink_tokens: 2\nglobal_layer_interval: 2\nuse_flex_attention: false\n", encoding="utf-8")
    out_dir = tmp_path / "pretrain_zero_save"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "crouton-pretrain",
            "--model-config",
            str(cfg_file),
            "--tokenizer",
            str(tok_path),
            "--local-file",
            str(data_file),
            "--output-dir",
            str(out_dir),
            "--steps",
            "1",
            "--micro-batch",
            "1",
            "--grad-accum",
            "1",
            "--save-interval",
            "0",
        ],
    )
    pretrain_mod.main()
    assert (out_dir / "final.pt").exists()

    # sft --dataset custom leaves --subset as None instead of forcing smoltalk "all"
    seen: dict[str, object] = {}

    def _fake_sft(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(sft_mod, "run_sft", _fake_sft)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "crouton-sft",
            "--base-checkpoint",
            str(out_dir / "final.pt"),
            "--dataset",
            "custom/chat-dataset",
            "--seed",
            "123",
        ],
    )
    sft_mod.main()
    assert seen["subset"] is None
    assert seen["seed"] == 123

    # sft --preset smoltalk respects --subset override and rejects conflicting --dataset
    seen.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "crouton-sft",
            "--base-checkpoint",
            str(out_dir / "final.pt"),
            "--preset",
            "smoltalk",
            "--subset",
            "everyday-conversations",
        ],
    )
    sft_mod.main()
    assert seen["dataset_name"] == "HuggingFaceTB/smoltalk"
    assert seen["subset"] == "everyday-conversations"

    monkeypatch.setattr(
        sys,
        "argv",
        ["crouton-sft", "--base-checkpoint", str(out_dir / "final.pt"), "--preset", "smoltalk", "--dataset", "x"],
    )
    with pytest.raises(SystemExit):
        sft_mod.main()

    # Empty val/local files and invalid CLI numeric bounds fail loudly
    empty_txt = tmp_path / "empty.txt"
    empty_txt.write_text("  \n\n\t", encoding="utf-8")
    with pytest.raises(ValueError, match="Evaluation texts"):
        evaluate_bpb(str(out_dir / "final.pt"), str(tok_path), [])
    with pytest.raises(ValueError, match="non-empty lines"):
        run_pretrain(
            CroutonConfig.from_yaml(cfg_file),
            TrainConfig(output_dir=str(tmp_path / "e1"), tokenizer_path=str(tok_path), seq_len=8, total_steps=1, micro_batch_size=1, grad_accum_steps=1, warmup_steps=0, decay_steps=0, bf16=False, compile_model=False),
            val_file=str(empty_txt),
        )
    with pytest.raises(ValueError, match="max_examples"):
        run_sft(
            str(out_dir / "final.pt"),
            str(tok_path),
            str(tmp_path / "e2"),
            max_examples=0,
            seq_len=8,
            total_steps=1,
            micro_batch_size=1,
            grad_accum_steps=1,
        )

    # Zero-supervision preference rows do not inject P=1.0 into DPO/ORPO
    run_preference_tuning(
        base_checkpoint=str(out_dir / "final.pt"),
        tokenizer_path=str(tok_path),
        output_dir=str(tmp_path / "orpo_zero_sup"),
        seq_len=16,
        total_steps=1,
        micro_batch_size=2,
        grad_accum_steps=1,
        use_orpo=True,
        pairs_override=[
            {"prompt": [{"role": "user", "content": "Only user"}], "chosen": [{"role": "user", "content": "Only user"}], "rejected": [{"role": "user", "content": "Only user"}]},
            {"prompt": "Q", "chosen": "Yes", "rejected": "No"},
        ],
    )
    assert (tmp_path / "orpo_zero_sup" / "final.pt").exists()

    # Batched CroutonLM.loglikelihood matches unbatched identically and supports long continuations
    ckpt_file = str(out_dir / "final.pt")
    lm1 = CroutonLM(ckpt_file, str(tok_path), batch_size=1, device="cpu", bf16=False)
    lm2 = CroutonLM(ckpt_file, str(tok_path), batch_size=2, device="cpu", bf16=False)
    reqs = [
        SimpleNamespace(args=("Dense", " MoE")),
        SimpleNamespace(args=("Transformer ", "training text.")),
        SimpleNamespace(args=("", "Dense MoE")),
    ]
    r1 = lm1.loglikelihood(reqs)
    r2 = lm2.loglikelihood(reqs)
    for (ll1, g1), (ll2, g2) in zip(r1, r2):
        assert ll1 == pytest.approx(ll2, abs=1e-5)
        assert g1 == g2
    assert lm1.tokenizer.encode("Dense", add_special_tokens=False) == lm1.tok_encode("Dense")
    long_cont = "Dense MoE transformer training text. " * 4
    ll_long = lm1.loglikelihood([SimpleNamespace(args=("Start:", long_cont))])
    assert isinstance(ll_long[0][0], float) and ll_long[0][0] < 0.0

    # End-to-end CLI main() smoke across tokenizer, midtrain, dpo, grpo, and eval_runner
    from crouton.eval import eval_runner as eval_mod
    from crouton.tokenizer import train_tokenizer as tok_mod
    from crouton.train import dpo as dpo_mod
    from crouton.train import grpo as grpo_mod
    from crouton.train import midtrain as mid_mod

    cli_tok = tmp_path / "cli_tok.json"
    monkeypatch.setattr(sys, "argv", ["crouton-tokenizer", "--local-file", str(data_file), "--output", str(cli_tok), "--vocab-size", "260"])
    tok_mod.main()
    assert cli_tok.exists()

    mid_out = tmp_path / "cli_mid"
    monkeypatch.setattr(sys, "argv", ["crouton-mid", "--base-checkpoint", ckpt_file, "--tokenizer", str(tok_path), "--output-dir", str(mid_out), "--local-file", str(data_file), "--seq-len", "16", "--steps", "1", "--micro-batch", "1", "--grad-accum", "1", "--no-bf16"])
    mid_mod.main()
    assert (mid_out / "final.pt").exists()

    grpo_out = tmp_path / "cli_grpo"
    monkeypatch.setattr(sys, "argv", ["crouton-grpo", "--base-checkpoint", str(mid_out / "final.pt"), "--tokenizer", str(tok_path), "--output-dir", str(grpo_out), "--builtin-math", "--steps", "1", "--group-size", "2", "--max-new-tokens", "4", "--no-bf16"])
    grpo_mod.main()
    assert (grpo_out / "final.pt").exists()

    eval_json = tmp_path / "eval.json"
    monkeypatch.setattr(sys, "argv", ["crouton-eval", "--checkpoint", str(grpo_out / "final.pt"), "--compare-checkpoint", ckpt_file, "--tokenizer", str(tok_path), "--val-file", str(data_file), "--batch-size", "2", "--no-bf16", "--output-json", str(eval_json)])
    eval_mod.main()
    assert eval_json.exists()

    monkeypatch.setattr(sys, "argv", ["crouton-eval", "--checkpoint", ckpt_file, "--tasks", ",,,"])
    with pytest.raises(SystemExit):
        eval_mod.main()
    monkeypatch.setattr(eval_mod, "HAS_LM_EVAL", False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["crouton-eval", "--checkpoint", str(tmp_path / "nonexistent.pt"), "--val-file", str(tmp_path / "nonexistent.txt"), "--tasks", "hellaswag"],
    )
    with pytest.raises(RuntimeError, match="lm-eval"):
        eval_mod.main()

    monkeypatch.setattr(sys, "argv", ["crouton-dpo", "--base-checkpoint", ckpt_file, "--beta", "0"])
    with pytest.raises(SystemExit):
        dpo_mod.main()
    monkeypatch.setattr(sys, "argv", ["crouton-dpo", "--base-checkpoint", ckpt_file, "--dataset", ""])
    with pytest.raises(SystemExit):
        dpo_mod.main()
    with pytest.raises(ValueError, match="beta"):
        run_preference_tuning(ckpt_file, str(tok_path), str(tmp_path / "bad_beta"), beta=0.0, pairs_override=[])

    monkeypatch.setattr(sys, "argv", ["crouton-grpo", "--base-checkpoint", ckpt_file, "--temperature", "0"])
    with pytest.raises(SystemExit):
        grpo_mod.main()
    monkeypatch.setattr(sys, "argv", ["crouton-grpo", "--base-checkpoint", ckpt_file, "--kl-beta", "nan"])
    with pytest.raises(SystemExit):
        grpo_mod.main()
    monkeypatch.setattr(grpo_mod, "load_gsm8k_tasks", lambda **kw: (_ for _ in ()).throw(AssertionError("should fail before GSM8K")))
    monkeypatch.setattr(
        sys,
        "argv",
        ["crouton-grpo", "--base-checkpoint", str(mid_out / "final.pt"), "--tokenizer", str(tok_path), "--max-new-tokens", "99999"],
    )
    with pytest.raises(ValueError, match="max_new_tokens"):
        grpo_mod.main()

    monkeypatch.setattr(
        sys,
        "argv",
        ["crouton-pretrain", "--model-config", str(tmp_path / "missing.yaml"), "--seq-len", "0"],
    )
    with pytest.raises(SystemExit):
        pretrain_mod.main()
    monkeypatch.setattr(
        sys,
        "argv",
        ["crouton-tokenizer", "--vocab-size", "259", "--dataset", "x"],
    )
    with pytest.raises(SystemExit):
        tok_mod.main()
    monkeypatch.setattr(sys, "argv", ["crouton-tokenizer", "--dataset", "  "])
    with pytest.raises(SystemExit):
        tok_mod.main()
    monkeypatch.setattr(sys, "argv", ["crouton-mid", "--base-checkpoint", ckpt_file, "--subset", "none"])
    with pytest.raises(SystemExit):
        mid_mod.main()
    monkeypatch.setattr(
        sys,
        "argv",
        ["crouton-mid", "--base-checkpoint", ckpt_file, "--local-file", str(data_file), "--dataset", "x"],
    )
    with pytest.raises(SystemExit):
        mid_mod.main()
    monkeypatch.setattr(sys, "argv", ["crouton-sft", "--base-checkpoint", ckpt_file, "--dataset", ""])
    with pytest.raises(SystemExit):
        sft_mod.main()

    # GRPO deterministic routing with router_jitter > 0 and vocab_limit normalization parity
    jit_cfg = _tiny_cfg()
    jit_cfg.vocab_size = tok.vocab_size + 16
    jit_cfg.router_jitter = 0.25
    jit_ckpt = tmp_path / "jit.pt"
    save_checkpoint(jit_ckpt, CroutonForCausalLM(jit_cfg), jit_cfg, step=1)
    run_grpo(
        base_checkpoint=str(jit_ckpt),
        tokenizer_path=str(tok_path),
        output_dir=str(tmp_path / "jit_grpo"),
        prompts_with_verifiers=build_builtin_math_tasks(n=2),
        group_size=2,
        steps=1,
        max_new_tokens=4,
    )
    assert (tmp_path / "jit_grpo" / "final.pt").exists()

    # BPB validation matches StreamingPackedDataset split_special_tokens=True on literal special-token text
    lit = "Dense <|im_end|> MoE <|endoftext|> test."
    bpb_lit = evaluate_bpb(ckpt_file, str(tok_path), [lit], seq_len=32, bf16=False)
    assert int(bpb_lit["tokens"]) == len(tok.encode(lit, split_special_tokens=True))


def test_sequence_logps_early_validation_cli_and_eval_guards(tmp_path, monkeypatch):
    import torch.nn.functional as F
    from crouton.eval import eval_runner as eval_mod
    from crouton.train import dpo as dpo_mod
    from crouton.train import grpo as grpo_mod
    from crouton.train import midtrain as midtrain_mod
    from crouton.train import pretrain as pretrain_mod
    from crouton.train import sft as sft_mod
    from crouton.train.common import save_checkpoint
    from crouton.train.dpo import sequence_logps

    # 1. sequence_logps matches naive fp32 log_softmax+gather across sparse & all-(-100) rows
    torch.manual_seed(0)
    logits = torch.randn(3, 10, 19, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.tensor(
        [
            [-100, -100, 1, 5, -100, 2, 0, -100, 18, -100],
            [-100] * 10,
            [3, 4, 7, -100, -100, -100, -100, -100, -100, -100],
        ],
        dtype=torch.long,
    )
    out = sequence_logps(logits, labels)
    ref_logits = logits.detach().clone().float().requires_grad_(True)
    ref_per_tok = F.log_softmax(ref_logits, dim=-1).gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    ref_out = (ref_per_tok * (labels != -100).float()).sum(dim=-1)
    assert out.dtype == torch.float32
    assert torch.allclose(out, ref_out, atol=1e-4)
    out.sum().backward()
    ref_out.sum().backward()
    assert logits.grad is not None and torch.allclose(logits.grad.float(), ref_logits.grad, atol=1e-2)

    all_masked_logits = torch.randn(2, 5, 11, dtype=torch.bfloat16, requires_grad=True)
    all_masked_labels = torch.full((2, 5), -100, dtype=torch.long)
    zero_out = sequence_logps(all_masked_logits, all_masked_labels)
    assert torch.equal(zero_out, torch.zeros(2, dtype=torch.float32))
    zero_out.sum().backward()
    assert all_masked_logits.grad is not None and torch.equal(all_masked_logits.grad, torch.zeros_like(all_masked_logits))

    # Build a tiny valid tokenizer and checkpoints for early-validation & ORPO smoke tests
    corpus = "Dense MoE transformer training text. " * 20
    tok_path = tmp_path / "tok.json"
    tok = train_crouton_tokenizer([corpus], output_path=tok_path, vocab_size=260)
    cfg = CroutonConfig(
        vocab_size=tok.vocab_size,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        n_experts=2,
        expert_rank=4,
        max_seq_len=32,
        sliding_window=8,
        sink_tokens=2,
        global_layer_interval=2,
        use_flex_attention=False,
    )
    model = CroutonForCausalLM(cfg)
    ckpt_ok = tmp_path / "ok.pt"
    save_checkpoint(str(ckpt_ok), model, cfg, step=0)

    cfg_bad = CroutonConfig(
        vocab_size=max(1, tok.vocab_size - 32),
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        n_experts=2,
        expert_rank=4,
        max_seq_len=32,
        sliding_window=8,
        sink_tokens=2,
        global_layer_interval=2,
        use_flex_attention=False,
    )
    ckpt_bad = tmp_path / "bad_vocab.pt"
    save_checkpoint(str(ckpt_bad), CroutonForCausalLM(cfg_bad), cfg_bad, step=0)

    # 2. run_sft and run_preference_tuning fail on malformed overrides BEFORE model allocation
    with monkeypatch.context() as mp:
        mp.setattr(
            sft_mod,
            "instantiate_checkpoint_model",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not allocate model")),
        )
        with pytest.raises(ValueError, match="ChatML"):
            run_sft(
                str(ckpt_ok),
                str(tok_path),
                str(tmp_path / "sft_bad"),
                seq_len=16,
                total_steps=1,
                micro_batch_size=1,
                grad_accum_steps=1,
                bf16=False,
                examples_override=[[{"bad": "shape"}]],
            )
    with monkeypatch.context() as mp:
        mp.setattr(
            dpo_mod,
            "instantiate_checkpoint_model",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not allocate model")),
        )
        with pytest.raises(ValueError, match="preference pair"):
            run_preference_tuning(
                str(ckpt_ok),
                str(tok_path),
                str(tmp_path / "dpo_bad"),
                seq_len=16,
                total_steps=1,
                micro_batch_size=1,
                grad_accum_steps=1,
                bf16=False,
                pairs_override=[{"prompt": ["bad"], "chosen": "ok", "rejected": "no"}],
            )

    # 3. Smoke-test both DPO (with reference model reordering) and ORPO end-to-end
    pref_pairs = [{"prompt": "What is 1+1?", "chosen": "2", "rejected": "9"}]
    for use_orpo, name in ((False, "dpo_run"), (True, "orpo_run")):
        out_dir = tmp_path / name
        run_preference_tuning(
            str(ckpt_ok),
            str(tok_path),
            str(out_dir),
            seq_len=24,
            total_steps=1,
            micro_batch_size=1,
            grad_accum_steps=1,
            use_orpo=use_orpo,
            bf16=False,
            pairs_override=pref_pairs,
        )
        assert (out_dir / "final.pt").exists()

    # 4. CLI whitespace --subset normalization and empty path fail-fast
    seen_mid: dict[str, object] = {}
    with monkeypatch.context() as mp:
        mp.setattr(midtrain_mod, "run_midtrain", lambda **kw: seen_mid.update(kw))
        mp.setattr(sys, "argv", ["crouton-midtrain", "--base-checkpoint", str(ckpt_ok), "--dataset", "custom/ds", "--subset", "   "])
        midtrain_mod.main()
        assert seen_mid["dataset_name"] == "custom/ds"
        assert seen_mid["subset"] is None

    seen_dpo: dict[str, object] = {}
    with monkeypatch.context() as mp:
        mp.setattr(dpo_mod, "run_preference_tuning", lambda **kw: seen_dpo.update(kw))
        mp.setattr(sys, "argv", ["crouton-dpo", "--base-checkpoint", str(ckpt_ok), "--subset", "   "])
        dpo_mod.main()
        assert seen_dpo["subset"] is None

    for fn, bad_argv in (
        (pretrain_mod.main, ["crouton-pretrain", "--model-config", ""]),
        (pretrain_mod.main, ["crouton-pretrain", "--model-config", "m.yaml", "--val-file", "   "]),
        (pretrain_mod.main, ["crouton-pretrain", "--model-config", "m.yaml", "--local-file", ""]),
        (midtrain_mod.main, ["crouton-midtrain", "--base-checkpoint", "   "]),
        (sft_mod.main, ["crouton-sft", "--base-checkpoint", "   "]),
        (dpo_mod.main, ["crouton-dpo", "--base-checkpoint", "   "]),
        (grpo_mod.main, ["crouton-grpo", "--base-checkpoint", "   "]),
        (grpo_mod.main, ["crouton-grpo", "--base-checkpoint", str(ckpt_ok), "--output-dir", "   "]),
    ):
        monkeypatch.setattr(sys, "argv", bad_argv)
        with pytest.raises(SystemExit):
            fn()

    # 5. eval_runner early validation and compare-checkpoint pre-validation ordering
    with pytest.raises(ValueError, match="seq_len must be >= 1"):
        evaluate_bpb(str(tmp_path / "nonexistent.pt"), str(tok_path), ["hello"], seq_len=0, bf16=False)
    with pytest.raises(ValueError, match="Evaluation texts"):
        evaluate_bpb(str(tmp_path / "nonexistent.pt"), str(tok_path), ["   \n"], seq_len=16, bf16=False)

    val_file = tmp_path / "val.txt"
    val_file.write_text("hello world test sentence.\n", encoding="utf-8")
    for bad_eval_argv in (
        ["crouton-eval", "--checkpoint", str(ckpt_ok), "--val-file", ""],
        ["crouton-eval", "--checkpoint", str(ckpt_ok), "--val-file", str(val_file), "--compare-checkpoint", ""],
    ):
        monkeypatch.setattr(sys, "argv", bad_eval_argv)
        with pytest.raises(SystemExit):
            eval_mod.main()

    with monkeypatch.context() as mp:
        mp.setattr(
            eval_mod,
            "evaluate_bpb",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should validate compare checkpoint before primary eval")),
        )
        mp.setattr(
            sys,
            "argv",
            [
                "crouton-eval",
                "--checkpoint",
                str(ckpt_ok),
                "--compare-checkpoint",
                str(ckpt_bad),
                "--tokenizer",
                str(tok_path),
                "--val-file",
                str(val_file),
                "--no-bf16",
            ],
        )
        with pytest.raises(ValueError, match="vocab"):
            eval_mod.main()
