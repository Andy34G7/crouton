"""Tests for Muon + AdamW, WSD, checkpoint _orig_mod handling, and end-to-end training/eval stages."""

from types import SimpleNamespace

import torch

from crouton.config import CroutonConfig, TrainConfig
from crouton.eval import CroutonLM, evaluate_bpb
from crouton.model import CroutonForCausalLM
from crouton.optim import build_optimizers, wsd_lr_multiplier
from crouton.tokenizer.train_tokenizer import train_crouton_tokenizer
from crouton.train.common import load_checkpoint, save_checkpoint, strip_orig_mod_prefix
from crouton.train.dpo import dpo_loss, run_preference_tuning
from crouton.train.grpo import build_builtin_math_tasks, grpo_step_loss, run_grpo
from crouton.train.midtrain import run_midtrain
from crouton.train.pretrain import run_pretrain
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


def test_end_to_end_pipeline_and_eval_smoke(tmp_path):
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
        micro_batch_size=1,
        grad_accum_steps=1,
        examples_override=[[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]] * 2,
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

    bpb_metrics = evaluate_bpb(str(grpo_ckpt), str(tok_path), corpus[:3], seq_len=32)
    assert bpb_metrics["bpb"] > 0.0

    lm = CroutonLM(str(grpo_ckpt), str(tok_path), device="cpu")
    ll_res = lm.loglikelihood([SimpleNamespace(args=("Hi", " Hello!"))])
    assert len(ll_res) == 1 and isinstance(ll_res[0][0], float)
    gen_res = lm.generate_until([SimpleNamespace(args=("Hi", {"max_gen_toks": 4, "until": []}))])
    assert len(gen_res) == 1 and isinstance(gen_res[0], str)
