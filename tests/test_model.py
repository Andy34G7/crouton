"""Tests for Crouton model architecture, YaRN/NTK RoPE, attention_mask, generate(), and configs."""

import math

import pytest
import torch

from crouton.config import CroutonConfig, TrainConfig
from crouton.model import CroutonForCausalLM, HybridSparseAttention, MergedLoRADenseMoE, build_swa_sink_mask
from crouton.model.rope import RotaryEmbedding, apply_rope


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

    # Test attention_mask padding, all-(-100) NaN guard, vocab_limit guard, and generate() EOS stop
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
    first_tok = int(gen[0, 3].item())
    gen_stop = model.generate(prompt[:, :3], max_new_tokens=5, temperature=0.0, eos_token_ids=[first_tok])
    assert gen_stop.shape == (1, 4) and int(gen_stop[0, -1].item()) == first_tok
    with pytest.raises(ValueError, match="vocab_limit"):
        model.generate(prompt[:, :3], max_new_tokens=1, vocab_limit=0)


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


def test_model_config_validation_and_optional_flags(tmp_path):
    for bad_kwargs in (
        {"max_seq_len": 0},
        {"d_model": 0},
        {"n_layers": 0},
        {"n_heads": 0},
        {"n_kv_heads": 0},
        {"d_ff": 0},
        {"rope_theta": 1.0},
        {"d_model": 65, "n_heads": 4},
        {"n_heads": 4, "n_kv_heads": 3},
    ):
        with pytest.raises(ValueError):
            CroutonConfig(**bad_kwargs)

    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("# comment only\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        CroutonConfig.from_yaml(empty_yaml)

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("unknown_field: 123\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid"):
        CroutonConfig.from_yaml(bad_yaml)

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
    yaml_roundtrip = tmp_path / "cfg.yaml"
    cfg.to_yaml(yaml_roundtrip)
    assert CroutonConfig.from_yaml(yaml_roundtrip) == cfg

    model = CroutonForCausalLM(cfg)
    assert model.lm_head.weight is not model.tok_emb.weight
    assert model.num_parameters(exclude_embeddings=False) == sum(p.numel() for p in model.parameters())
    assert model.num_parameters(exclude_embeddings=True) == model.num_parameters() - model.tok_emb.weight.numel()

    model.train()
    x = torch.randint(0, 128, (2, 9))
    raw_targets = x[:, 1:]
    assert not raw_targets.is_contiguous()
    out = model(x[:, :-1], targets=raw_targets)
    assert out.logits.abs().max().item() <= 30.0 + 1e-4
    assert out.router_entropy is not None and not out.router_entropy.requires_grad
    out.loss.backward()

    # Standalone HybridSparseAttention calls (precomputed_mask=None)
    cos, sin = model.rope(8, dtype=torch.float32)
    h = torch.randn(2, 8, 64)
    attn_mask = torch.ones((2, 8), dtype=torch.bool)
    attn_mask[1, -2:] = False
    for layer_idx in (0, 1):
        attn = HybridSparseAttention(cfg, layer_idx=layer_idx)
        assert attn(h, cos, sin, attention_mask=attn_mask).shape == (2, 8, 64)
        assert attn(h, cos, sin).shape == (2, 8, 64)

    # Check generate() respects vocab_limit, top_k, max_new_tokens=0, eos=None, and restores training mode
    model.train()
    assert model.generate(x[:, :3], max_new_tokens=0).shape == (2, 3)
    gen = model.generate(x[:, :3], max_new_tokens=4, temperature=0.8, top_k=4, eos_token_ids=None, vocab_limit=64)
    assert gen.shape == (2, 7)
    assert (gen[:, 3:] < 64).all()
    assert model.training

    # Right-padded attention_mask must match unpadded valid-prefix logits without 4D mask divergence,
    # while left-padded attention_mask still isolates leading padded keys.
    model.eval()
    prefix = torch.tensor([[5, 9, 12]], dtype=torch.long)
    pad_a = torch.tensor([[5, 9, 12, 0, 1]], dtype=torch.long)
    pad_b = torch.tensor([[5, 9, 12, 42, 99]], dtype=torch.long)
    right_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    with torch.no_grad():
        logits_ref = model(prefix).logits
        logits_a = model(pad_a, attention_mask=right_mask).logits[:, :3]
        logits_b = model(pad_b, attention_mask=right_mask).logits[:, :3]
    assert torch.allclose(logits_a, logits_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(logits_a, logits_b, atol=1e-6)

    left_a = torch.tensor([[7, 8, 5, 9, 12]], dtype=torch.long)
    left_b = torch.tensor([[99, 77, 5, 9, 12]], dtype=torch.long)
    left_mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool)
    with torch.no_grad():
        left_out_a = model(left_a, attention_mask=left_mask).logits[:, 2:]
        left_out_b = model(left_b, attention_mask=left_mask).logits[:, 2:]
    assert torch.allclose(left_out_a, left_out_b, atol=1e-6)

    for bad_float in (
        {"rope_theta": float("nan")},
        {"norm_eps": float("inf")},
        {"router_jitter": float("nan")},
        {"router_jitter": 1.0},
        {"router_jitter": 1.5},
    ):
        with pytest.raises(ValueError):
            CroutonConfig(**bad_float)
    with pytest.raises(ValueError):
        TrainConfig(muon_lr=float("nan"))

    # Eager sparse CE slicing matches dense ignore_index=-100 loss and handles all-masked batches
    cfg_ce = CroutonConfig(
        vocab_size=32,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        n_experts=2,
        expert_rank=4,
        max_seq_len=16,
        use_flex_attention=False,
    )
    m_ce = CroutonForCausalLM(cfg_ce)
    m_ce.eval()
    ce_x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    ce_tgt = torch.tensor([[2, -100, 4, -100], [-100, 7, -100, -100]], dtype=torch.long)
    m_ce.zero_grad()
    out_ce = m_ce(ce_x, targets=ce_tgt)
    ref_ce = torch.nn.functional.cross_entropy(
        out_ce.logits.reshape(-1, cfg_ce.vocab_size).float(),
        ce_tgt.reshape(-1),
        ignore_index=-100,
    )
    assert torch.allclose(out_ce.loss, ref_ce, atol=1e-6, rtol=1e-6)
    out_ce.loss.backward()
    assert m_ce.tok_emb.weight.grad is not None and torch.isfinite(m_ce.tok_emb.weight.grad).all()

    m_ce.zero_grad()
    all_masked = torch.full_like(ce_tgt, -100)
    out_zero = m_ce(ce_x, targets=all_masked)
    assert out_zero.loss.item() == 0.0
    out_zero.loss.backward()
    assert torch.isfinite(m_ce.tok_emb.weight.grad).all()

    # Preserve caller's first EOS token in eos_fill rather than sorting IDs
    eos_steps = [
        torch.tensor([[7], [2]], dtype=torch.long),
        torch.tensor([[9], [9]], dtype=torch.long),
    ]
    orig_forward = m_ce.forward

    def _mock_eos_forward(idx, *args, **kwargs):
        logits = torch.full((idx.size(0), idx.size(1), cfg_ce.vocab_size), -10.0)
        toks = eos_steps.pop(0) if eos_steps else torch.full((idx.size(0), 1), 9, dtype=torch.long)
        for row_idx in range(idx.size(0)):
            logits[row_idx, -1, int(toks[row_idx, 0].item())] = 10.0
        return type("Out", (), {"logits": logits, "loss": None, "router_entropy": None})()

    m_ce.forward = _mock_eos_forward  # type: ignore[method-assign]
    try:
        gen_eos = m_ce.generate(torch.tensor([[1], [1]], dtype=torch.long), max_new_tokens=2, eos_token_ids=[7, 3])
        assert gen_eos[0].tolist() == [1, 7, 7]
        assert gen_eos[1].tolist() == [1, 2, 9]
    finally:
        m_ce.forward = orig_forward  # type: ignore[method-assign]

    # Config type/YAML validation, generate() bounds, MoE scaling, YaRN formula, and compiled sparse-CE parity
    for bad_typed in (
        {"vocab_size": True},
        {"vocab_size": 32.5},
        {"tie_embeddings": "false"},
        {"norm_eps": None},
    ):
        with pytest.raises(ValueError):
            CroutonConfig(**bad_typed)  # type: ignore[arg-type]
    for bad_tcfg in ({"total_steps": True}, {"bf16": "true"}, {"muon_lr": None}):
        with pytest.raises(ValueError):
            TrainConfig(**bad_tcfg)  # type: ignore[arg-type]

    malformed_yaml = tmp_path / "malformed.yaml"
    malformed_yaml.write_text("d_model: [1, 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML"):
        CroutonConfig.from_yaml(malformed_yaml)

    partial_yaml = tmp_path / "partial.yaml"
    partial_yaml.write_text("d_model: 64\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required CroutonConfig keys"):
        CroutonConfig.from_yaml(partial_yaml)

    omitted_orig_yaml = tmp_path / "omitted_orig.yaml"
    omitted_orig_yaml.write_text(
        "vocab_size: 64\nd_model: 32\nn_layers: 2\nn_heads: 4\nn_kv_heads: 2\nd_ff: 64\nn_experts: 2\nexpert_rank: 4\nmax_seq_len: 1024\n",
        encoding="utf-8",
    )
    assert CroutonConfig.from_yaml(omitted_orig_yaml).rope_original_max_seq_len == 1024

    for bad_gen in (
        (torch.tensor([1, 2]), {}),
        (torch.zeros((0, 2), dtype=torch.long), {}),
        (torch.tensor([[1, 2]]), {"max_new_tokens": -1}),
        (torch.tensor([[1, 2]]), {"vocab_limit": 0}),
    ):
        with pytest.raises(ValueError):
            m_ce.generate(bad_gen[0], **bad_gen[1])

    # Direct MergedLoRADenseMoE scale=alpha/rank & gradient flow once down_B/up_B are non-zero
    moe = MergedLoRADenseMoE(CroutonConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=4, n_kv_heads=2, d_ff=32, n_experts=2, expert_rank=4, expert_alpha=8.0, use_flex_attention=False))
    assert moe.scale == 2.0
    with torch.no_grad():
        moe.up_B.fill_(0.05)
        moe.down_B.fill_(0.05)
    moe_x = torch.randn(2, 5, 16, requires_grad=True)
    moe_out, _ = moe(moe_x)
    moe_out.sum().backward()
    assert moe.router.weight.grad is not None and moe.router.weight.grad.abs().sum().item() > 0.0
    assert moe.up_A.grad is not None and moe.up_A.grad.abs().sum().item() > 0.0

    # Exact YaRN mscale and high/low-frequency inv_freq ramp check
    rope_yarn = RotaryEmbedding(
        head_dim=8,
        max_seq_len=32,
        theta=10000.0,
        scale_factor=2.0,
        scaling_type="yarn",
        original_max_seq_len=16,
    )
    expected_mscale = 0.1 * math.log(2.0) + 1.0
    assert abs(rope_yarn.mscale - expected_mscale) < 1e-6
    assert abs(float(rope_yarn.inv_freq[0]) - 1.0) < 1e-6
    expected_low = (1.0 / (10000.0 ** (6.0 / 8.0))) / 2.0
    assert abs(float(rope_yarn.inv_freq[-1]) - expected_low) < 1e-6

    # Runtime verification of torch.compiler.is_compiling() sparse-CE and mask branches on CPU
    m_ce.eval()
    compiled_ce = torch.compile(m_ce, backend="aot_eager")
    pad_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    with torch.no_grad():
        eager_out = m_ce(ce_x, attention_mask=pad_mask, targets=ce_tgt)
        comp_out = compiled_ce(ce_x, attention_mask=pad_mask, targets=ce_tgt)
    assert torch.allclose(comp_out.logits[:, :2], eager_out.logits[:, :2], atol=1e-5, rtol=1e-5)
    assert torch.allclose(comp_out.loss, eager_out.loss, atol=1e-5, rtol=1e-5)
    assert compiled_ce(ce_x, attention_mask=pad_mask, targets=all_masked).loss.item() == 0.0
    comp_rp = compiled_ce(ce_x, attention_mask=pad_mask, targets=ce_tgt, is_right_padded=True)
    assert torch.allclose(comp_rp.logits[:, :2], eager_out.logits[:, :2], atol=1e-5, rtol=1e-5)


def test_kv_cache_gemm_moe_zloss_and_yarn_clamp():
    cfg = CroutonConfig(
        vocab_size=64,
        d_model=32,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        n_experts=4,
        expert_rank=4,
        max_seq_len=12,
        sliding_window=4,
        sink_tokens=2,
        global_layer_interval=2,
        router_z_loss_coef=1e-2,
        expert_b_init_std=1e-2,
        use_flex_attention=False,
    )
    model = CroutonForCausalLM(cfg)
    model.train()

    # Non-zero expert_b_init_std + router_z_loss_coef gives step-0 router gradients and finite z-loss
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))
    out = model(x, targets=y)
    assert out.router_z_loss is not None and torch.isfinite(out.router_z_loss)
    out.loss.backward()
    for block in model.blocks:
        assert block.moe.router.weight.grad is not None
        assert block.moe.router.weight.grad.abs().sum().item() > 0.0

    # 2D GEMM MoE matches reference einsum exactly
    moe = model.blocks[0].moe
    moe.eval()
    h = torch.randn(2, 6, 32)
    with torch.no_grad():
        gemm_out, weights = moe(h)
        w = weights.to(h.dtype).unsqueeze(-1)
        up_ref = moe.w_up(h) + moe.scale * torch.einsum(
            "bter,erf->btf", torch.einsum("btd,edr->bter", h, moe.up_A) * w, moe.up_B
        )
        act_ref = torch.relu(up_ref).square()
        ref_out = moe.w_down(act_ref) + moe.scale * torch.einsum(
            "bter,erd->btd", torch.einsum("btf,efr->bter", act_ref, moe.down_A) * w, moe.down_B
        )
    assert torch.allclose(gemm_out, ref_out, atol=1e-5, rtol=1e-5)

    # KV-cached forward & generate() match uncached full-context decoding (including across max_seq_len)
    model.eval()
    prompt = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    with torch.no_grad():
        prefill = model(prompt, use_cache=True)
        step1 = model(torch.tensor([[6]], dtype=torch.long), past_key_values=prefill.past_key_values, use_cache=True)
        full_ref = model(torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long))
    assert torch.allclose(step1.logits[:, -1], full_ref.logits[:, -1], atol=1e-5, rtol=1e-5)

    # Generate past max_seq_len=12 (5 + 10 = 15 tokens) and verify exact parity with uncached greedy decode
    cached_gen = model.generate(prompt, max_new_tokens=10, temperature=0.0)
    uncached = prompt.clone()
    with torch.no_grad():
        for _ in range(10):
            nxt = model(uncached[:, -cfg.max_seq_len :]).logits[:, -1, :].argmax(dim=-1, keepdim=True)
            uncached = torch.cat([uncached, nxt], dim=-1)
    assert torch.equal(cached_gen, uncached)

    # Extreme YaRN original_max_seq_len does not produce invalid negative (high - low)
    rope_extreme = RotaryEmbedding(
        head_dim=8,
        max_seq_len=32,
        theta=100.0,
        scale_factor=2.0,
        scaling_type="yarn",
        original_max_seq_len=1_000_000,
    )
    assert torch.isfinite(rope_extreme.inv_freq).all()
