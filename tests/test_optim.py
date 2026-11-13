"""Tests for Muon + AdamW optimizer partitioning, WSD schedule, and post-training losses."""

import torch

from crouton.config import CroutonConfig
from crouton.model import CroutonForCausalLM
from crouton.optim import build_optimizers, wsd_lr_multiplier
from crouton.train.dpo import dpo_loss
from crouton.train.grpo import grpo_step_loss


def test_muon_adamw_step():
    cfg = CroutonConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        n_experts=2,
        expert_rank=4,
        max_seq_len=16,
        use_flex_attention=False,
    )
    model = CroutonForCausalLM(cfg)
    muon_opt, adamw_opt = build_optimizers(model)
    assert len(muon_opt.param_groups[0]["params"]) > 0

    x = torch.randint(0, 256, (2, 8))
    y = torch.randint(0, 256, (2, 8))
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
