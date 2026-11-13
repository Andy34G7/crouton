"""Tests for Crouton model architecture, scaling configs, SWA+sink mask, and ReLU^2 MoE."""

import torch

from crouton.config import CroutonConfig
from crouton.model import CroutonForCausalLM, build_swa_sink_mask


def test_swa_sink_mask_properties():
    mask = build_swa_sink_mask(seq_len=16, window_size=4, sink_tokens=2, device=torch.device("cpu"))
    m = mask[0, 0]
    # Causal: upper triangle is False
    assert not m.triu(diagonal=1).any()
    # Sink tokens (cols 0, 1) are visible to all later queries
    assert m[10:, :2].all()
    # Token 10 cannot see token 5 (outside window=4 and not sink)
    assert not m[10, 5]
    # Token 10 can see token 7 (10 - 7 = 3 < 4)
    assert m[10, 7]


def test_forward_backward_and_context_extension():
    cfg = CroutonConfig(
        vocab_size=512,
        d_model=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        n_experts=4,
        expert_rank=8,
        max_seq_len=32,
        sliding_window=8,
        sink_tokens=2,
        global_layer_interval=2,
        use_flex_attention=False,
    )
    model = CroutonForCausalLM(cfg)
    x = torch.randint(0, 512, (2, 16))
    y = torch.randint(0, 512, (2, 16))
    out = model(x, targets=y)
    assert out.logits.shape == (2, 16, 512)
    assert out.loss is not None
    assert out.router_entropy is not None
    out.loss.backward()

    # Context extension beyond initial max_seq_len=32
    x_long = torch.randint(0, 512, (1, 48))
    out_long = model(x_long)
    assert out_long.logits.shape == (1, 48, 512)


def test_yaml_configs_instantiate():
    for name in ("configs/model_tiny.yaml", "configs/model_small.yaml", "configs/model_300m.yaml"):
        cfg = CroutonConfig.from_yaml(name)
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.is_global_layer(cfg.n_layers - 1)
