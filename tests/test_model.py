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
    for name in (
        "configs/model_tiny.yaml",
        "configs/model_small.yaml",
        "configs/model_base.yaml",
        "configs/model_medium.yaml",
        "configs/model_large.yaml",
        "configs/model_300m.yaml",
    ):
        cfg = CroutonConfig.from_yaml(name)
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.is_global_layer(cfg.n_layers - 1)

    for alias, expected_d in (
        ("tiny", 256),
        ("14m", 256),
        ("small", 384),
        ("30m", 384),
        ("crouton-base", 512),
        ("50m", 512),
        ("medium", 640),
        ("85m", 640),
        ("large", 1024),
        ("300m", 1024),
    ):
        pcfg = CroutonConfig.from_preset(alias)
        assert pcfg.d_model == expected_d
        assert pcfg.head_dim == 64
    with pytest.raises(ValueError, match="Unknown model preset"):
        CroutonConfig.from_preset("xxl")


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


def test_shard_writer_memmap_and_pretrain_resume(tmp_path):
    import numpy as np
    from crouton.data import ShardedMemmapDataset, discover_bin_shards
    from crouton.data.prepare_data import ShardWriter
    from crouton.train.pretrain import run_pretrain

    data_dir = tmp_path / "shards"
    writer = ShardWriter(data_dir, prefix="train", shard_size=64)
    tokens = np.arange(150, dtype=np.uint16) % 64
    writer.add(tokens)
    shards = writer.finalize()
    assert shards == ["train_0000.bin", "train_0001.bin", "train_0002.bin"]
    assert writer.total_tokens == 150

    # Write a val.bin split too
    (np.arange(40, dtype=np.uint16) % 64).tofile(data_dir / "val.bin")

    # Auto-discover via directory path AND via shortcake-style data/train.bin path
    assert len(discover_bin_shards(data_dir, split="train")) == 3
    assert len(discover_bin_shards(data_dir / "train.bin", split="train")) == 3

    ds = ShardedMemmapDataset(data_dir / "train.bin", seq_len=8, split="train", seed=7)
    assert ds.total_tokens == 150
    x, y = next(iter(ds))
    assert x.shape == (8,) and y.shape == (8,) and x.dtype == torch.long

    # Pretrain from uint16 shards + resume from saved checkpoint
    cfg = CroutonConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        n_experts=2,
        expert_rank=4,
        max_seq_len=8,
        sliding_window=4,
        sink_tokens=1,
        global_layer_interval=2,
        use_flex_attention=False,
    )
    out1 = tmp_path / "run1"
    tcfg1 = TrainConfig(
        output_dir=str(out1),
        tokenizer_path="unused.json",
        seq_len=8,
        micro_batch_size=2,
        grad_accum_steps=1,
        total_steps=3,
        warmup_steps=1,
        decay_steps=1,
        eval_interval=1,
        save_interval=2,
        bf16=False,
        compile_model=False,
    )
    run_pretrain(cfg, tcfg1, data_dir=str(data_dir))
    assert (out1 / "step_2.pt").is_file()
    assert (out1 / "final.pt").is_file()
    step_ckpt = torch.load(out1 / "step_2.pt", map_location="cpu", weights_only=False)
    assert "muon_opt" in step_ckpt and "adamw_opt" in step_ckpt

    out2 = tmp_path / "run2"
    tcfg2 = TrainConfig(
        output_dir=str(out2),
        tokenizer_path="unused.json",
        seq_len=8,
        micro_batch_size=2,
        grad_accum_steps=1,
        total_steps=4,
        warmup_steps=0,
        decay_steps=1,
        eval_interval=0,
        save_interval=0,
        bf16=False,
        compile_model=False,
    )
    run_pretrain(cfg, tcfg2, data_dir=str(data_dir), resume_checkpoint=str(out1 / "step_2.pt"))
    assert (out2 / "final.pt").is_file()

    # Verify crouton.chat.stream_generate parity with model.generate()
    from crouton.chat import stream_generate

    m_chat = CroutonForCausalLM(cfg).eval()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    streamed = list(stream_generate(m_chat, prompt, max_new_tokens=6, temperature=0.0))
    batch_gen = m_chat.generate(prompt, max_new_tokens=6, temperature=0.0)[0, 3:].tolist()
    assert streamed == batch_gen


def test_bidirectional_splade_head_losses_and_train_smoke(tmp_path):
    from crouton.model import (
        CroutonSPLADE,
        quad_reg_weight,
        splade_infonce_loss,
        splade_margin_mse_loss,
    )
    from crouton.train.common import save_checkpoint
    from crouton.train.splade import run_splade_train

    # Bidirectional SWA + sink mask is symmetric within window and allows future tokens
    bi_mask = build_swa_sink_mask(seq_len=12, window_size=4, sink_tokens=2, device=torch.device("cpu"), is_causal=False)
    bm = bi_mask[0, 0]
    assert bm[0, 2] and bm[2, 0]  # future token within window visible
    assert bm[10, 0] and bm[10, 1]  # sink tokens visible
    assert not bm[10, 4]  # outside window and not sink

    cfg = CroutonConfig(
        vocab_size=32768,
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
    backbone = CroutonForCausalLM(cfg)
    splade = CroutonSPLADE(backbone, tie_lm_head=True, pad_token_id=0, bidirectional=True)
    assert splade.splade_head.decoder.weight is backbone.tok_emb.weight

    # Pad isolation under bidirectional attention: trailing pad token values must not change sparse_vector
    splade.eval()
    a = torch.tensor([[5, 9, 12, 0, 0]], dtype=torch.long)
    b = torch.tensor([[5, 9, 12, 77, 99]], dtype=torch.long)
    m = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    with torch.no_grad():
        out_a = splade(a, attention_mask=m)
        out_b = splade(b, attention_mask=m)
    assert (out_a.sparse_vector >= 0.0).all()
    assert torch.allclose(out_a.sparse_vector, out_b.sparse_vector, atol=1e-5, rtol=1e-5)

    # Losses & quadratic warmup
    assert quad_reg_weight(0, 100, 0.05) == 0.0
    assert abs(quad_reg_weight(50, 100, 0.04) - 0.01) < 1e-9
    assert abs(quad_reg_weight(100, 100, 0.04) - 0.04) < 1e-9

    splade.train()
    q_o = splade(torch.tensor([[5, 9, 0], [6, 10, 11]]), attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool))
    p_o = splade(torch.tensor([[5, 12, 13], [6, 14, 0]]), attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool))
    n_o = splade(torch.tensor([[20, 21, 0], [22, 23, 24]]), attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool))
    loss = splade_infonce_loss(q_o.sparse_vector, p_o.sparse_vector, n_o.sparse_vector) + 0.5 * splade_margin_mse_loss(
        q_o.sparse_vector,
        p_o.sparse_vector,
        n_o.sparse_vector,
        torch.tensor([3.0, 2.5]),
        torch.tensor([0.5, 0.2]),
    )
    loss = loss + 0.01 * q_o.l1_loss + 0.01 * p_o.flops_loss
    assert torch.isfinite(loss)
    loss.backward()
    assert splade.splade_head.proj.weight.grad is not None

    # End-to-end run_splade_train smoke test (MNTP + SPLADE + save)
    ckpt_path = tmp_path / "base.pt"
    save_checkpoint(ckpt_path, backbone, cfg, step=10)
    out_dir = tmp_path / "splade_out"
    records = [
        {"query": "what is python", "pos": "python is a programming language", "neg": "bananas are yellow", "pos_score": 4.0, "neg_score": 0.1},
        {"query": "binary search tree", "pos": "a binary search tree stores sorted keys", "neg": "weather forecast rain", "pos_score": 3.5, "neg_score": 0.2},
    ]
    final_pt = run_splade_train(
        base_checkpoint=str(ckpt_path),
        tokenizer_path="artifacts/tokenizer/wp_bpe_ppig_32k.json",
        output_dir=str(out_dir),
        records=records,
        mntp_steps=1,
        total_steps=2,
        batch_size=2,
        grad_accum_steps=1,
        max_query_len=16,
        max_doc_len=24,
        bf16=False,
    )
    assert final_pt.is_file()
