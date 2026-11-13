"""Tests for Crouton model architecture, YaRN/NTK RoPE, attention_mask, generate(), and configs."""

import torch

from crouton.config import CroutonConfig
from crouton.model import CroutonForCausalLM, build_swa_sink_mask


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

    # Test attention_mask padding and generate() with EOS stop
    prompt = torch.tensor([[1, 2, 3, 0, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    out_masked = model(prompt, attention_mask=mask)
    assert torch.isfinite(out_masked.logits).all()

    gen = model.generate(prompt[:, :3], max_new_tokens=5, temperature=0.0, eos_token_ids=[999999])
    assert gen.shape == (1, 8)


def test_yaml_configs_instantiate():
    for name in ("configs/model_tiny.yaml", "configs/model_small.yaml", "configs/model_300m.yaml"):
        cfg = CroutonConfig.from_yaml(name)
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.is_global_layer(cfg.n_layers - 1)
