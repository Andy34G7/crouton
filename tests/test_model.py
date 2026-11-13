"""Tests for Crouton model architecture, YaRN/NTK RoPE, attention_mask, generate(), and configs."""

import pytest
import torch

from crouton.config import CroutonConfig
from crouton.model import CroutonForCausalLM, build_swa_sink_mask
from crouton.model.rope import apply_rope


def test_swa_sink_mask_properties():
    mask = build_swa_sink_mask(seq_len=16, window_size=4, sink_tokens=2, device=torch.device("cpu"))
    m = mask[0, 0]
    assert not m.triu(diagonal=1).any()
    assert m[10:, :2].all()
    assert not m[10, 5]
    assert m[10, 7]


def test_forward_backward_yarn_ntk_and_generate():
    for scaling_type in ("yarn", "ntk"):
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
            rope_scale_factor=2.0,
            rope_scaling_type=scaling_type,
            rope_original_max_seq_len=16,
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

    # Test attention_mask padding, all-(-100) NaN guard, and generate() with EOS stop
    prompt = torch.tensor([[1, 2, 3, 0, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    out_masked = model(prompt, attention_mask=mask)
    assert torch.isfinite(out_masked.logits).all()

    all_ignored = torch.full_like(prompt, -100)
    out_ignored = model(prompt, targets=all_ignored, attention_mask=mask)
    assert torch.isfinite(out_ignored.loss) and float(out_ignored.loss.item()) == 0.0
    out_ignored.loss.backward()

    gen = model.generate(prompt[:, :3], max_new_tokens=5, temperature=0.0, eos_token_ids=[999999])
    assert gen.shape == (1, 8)


def test_bf16_autocast_forward_backward():
    q = torch.randn(1, 4, 8, 16, dtype=torch.bfloat16)
    k = torch.randn(1, 2, 8, 16, dtype=torch.bfloat16)
    cos = torch.ones(8, 16, dtype=torch.float32)
    sin = torch.zeros(8, 16, dtype=torch.float32)
    q_rot, k_rot = apply_rope(q, k, cos, sin)
    assert q_rot.dtype == torch.bfloat16 and k_rot.dtype == torch.bfloat16

    cfg = CroutonConfig(
        vocab_size=256,
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
    x = torch.randint(0, 256, (2, 16))
    y = torch.randint(0, 256, (2, 16))
    attn_mask = torch.ones((2, 16), dtype=torch.bool)
    attn_mask[1, -3:] = False

    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        out = model(x, targets=y, attention_mask=attn_mask)
    assert out.loss is not None and torch.isfinite(out.loss)
    out.loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()

    model.zero_grad(set_to_none=True)
    all_ignored = torch.full_like(y, -100)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        out_ignored = model(x, targets=all_ignored)
    assert out_ignored.loss is not None and torch.isfinite(out_ignored.loss)
    assert float(out_ignored.loss.item()) == 0.0
    out_ignored.loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_yaml_configs_instantiate():
    for name in ("configs/model_tiny.yaml", "configs/model_small.yaml", "configs/model_300m.yaml"):
        cfg = CroutonConfig.from_yaml(name)
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.is_global_layer(cfg.n_layers - 1)


def test_model_config_validation_and_optional_flags():
    with pytest.raises(ValueError, match="max_seq_len"):
        CroutonConfig(max_seq_len=0)
    with pytest.raises(ValueError, match="d_model"):
        CroutonConfig(d_model=65, n_heads=4)
    with pytest.raises(ValueError, match="n_heads"):
        CroutonConfig(n_heads=4, n_kv_heads=3)

    cfg = CroutonConfig(
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        n_experts=4,
        expert_rank=8,
        max_seq_len=16,
        sliding_window=4,
        sink_tokens=2,
        global_layer_interval=2,
        tie_embeddings=False,
        qk_norm=False,
        logit_softcap=30.0,
        router_jitter=0.05,
        use_flex_attention=False,
    )
    model = CroutonForCausalLM(cfg)
    assert model.lm_head.weight is not model.tok_emb.weight
    model.train()
    x = torch.randint(0, 128, (2, 8))
    out = model(x, targets=x)
    assert out.logits.abs().max().item() <= 30.0 + 1e-4
    out.loss.backward()

    # Check generate() respects vocab_limit and restores prior training mode
    model.train()
    gen = model.generate(x[:, :3], max_new_tokens=4, temperature=0.8, eos_token_ids=[127], vocab_limit=64)
    assert gen.shape == (2, 7)
    assert (gen[:, 3:] < 64).all()
    assert model.training
