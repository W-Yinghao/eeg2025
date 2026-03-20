#!/usr/bin/env python3
"""
Minimal tests for USBA adapter — covering bug fixes D1-D7.

Run:
    cd ~/eeg2025/NIPS_finetune
    python test_usba.py
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))


def test_d1_token_wise_gate():
    """D1. token_wise gate produces (B,T,1) shape and correct init."""
    from adapters.bottleneck import VariationalBottleneck
    B, T, D = 2, 10, 32
    vb = VariationalBottleneck(input_dim=D, latent_dim=8, gate_type='token_wise', gate_init=0.0)
    fused = torch.randn(B, T, D)
    h_orig = torch.randn(B, T, D)
    h_out, aux = vb(fused, h_orig)
    assert h_out.shape == (B, T, D), f"output shape {h_out.shape}"
    assert aux['mu'].shape == (B, T, 8), f"mu shape {aux['mu'].shape}"
    assert isinstance(aux['gate_value'], float)
    assert abs(aux['gate_value'] - 0.5) < 0.15, f"gate init {aux['gate_value']:.3f}"
    assert vb.residual_scale.item() == 0.0, "residual_scale should init to 0"
    print("[PASS] D1 token_wise gate")


def test_d2_channel_attention_4d():
    """D2. ChannelAttention 4D mode: cross-channel modulation."""
    from adapters.branches import ChannelAttention, build_spatial_branch
    # Eager init
    ca = ChannelAttention(d_model=32, reduction=4, n_channels=4)
    x = torch.randn(2, 20, 32)  # T=4*5
    out = ca(x, token_structure=(4, 5))
    assert out.shape == (2, 20, 32)
    assert ca._n_channels == 4
    assert ca.channel_mlp[0].in_features == 4
    # Lazy init
    ca2 = ChannelAttention(d_model=32, reduction=4)
    assert ca2._n_channels is None
    _ = ca2(x, token_structure=(4, 5))
    assert ca2._n_channels == 4
    # Factory
    ca3 = build_spatial_branch(32, 'channel_attention', n_channels=16)
    assert ca3._n_channels == 16
    print("[PASS] D2 ChannelAttention 4D mode")


def test_d3_gated_fusion_init():
    """D3. GatedFusion 3-way softmax init near [0.71, 0.14, 0.14]."""
    from adapters.branches import GatedFusion
    gf = GatedFusion(d_model=32)
    h = torch.randn(4, 10, 32)
    _, stats = gf(h, torch.randn(4, 10, 32), torch.randn(4, 10, 32))
    w_id = stats['gate_identity_mean']
    w_t = stats['gate_temporal_mean']
    w_s = stats['gate_spatial_mean']
    assert w_id > 0.6, f"identity should dominate: {w_id:.3f}"
    assert w_t < 0.25 and w_s < 0.25, f"t={w_t:.3f} s={w_s:.3f}"
    assert abs(w_id + w_t + w_s - 1.0) < 0.01, "weights must sum to 1"
    print("[PASS] D3 GatedFusion init")


def test_d4_backward_flow():
    """D4. Backward flow reaches encoder through residual_scale * decoder."""
    from adapters.bottleneck import VariationalBottleneck
    vb = VariationalBottleneck(input_dim=32, latent_dim=8, gate_type='layer_wise')
    vb.train()
    f = torch.randn(2, 5, 32, requires_grad=True)
    h = torch.randn(2, 5, 32)
    out, aux = vb(f, h)
    (out.sum() + aux['kl']).backward()
    assert vb.encoder[0].weight.grad is not None and vb.encoder[0].weight.grad.abs().sum() > 0
    assert vb.fc_mu.weight.grad is not None
    assert vb.decoder.weight.grad is not None
    assert vb.residual_scale.grad is not None
    print("[PASS] D4 backward flow")


def test_d5_categorical_kernel():
    """D5. Subject kernel is delta (categorical), not RBF."""
    from adapters.losses import _delta_kernel, class_conditional_hsic
    ids = torch.tensor([0, 0, 1, 1, 2])
    K = _delta_kernel(ids)
    assert (K.diag() == 1).all(), "diagonal should be 1"
    assert K[0, 1].item() == 1.0, "same subject"
    assert K[0, 2].item() == 0.0, "diff subject"
    assert (K == K.t()).all(), "symmetric"
    # HSIC smoke test
    z = torch.randn(10, 8)
    labels = torch.tensor([0]*5 + [1]*5)
    sids = torch.tensor([0, 0, 1, 1, 2, 3, 3, 4, 4, 5])
    val = class_conditional_hsic(z, labels, sids)
    assert val.dim() == 0 and val.item() >= 0
    print("[PASS] D5 categorical kernel")


def test_d6_inter_layer_aux():
    """D6. USBAAdapter preserves real _mu_last/_z_last from last layer."""
    from adapters.usba_config import USBAConfig
    from adapters.usba import USBAAdapter
    cfg = USBAConfig(latent_dim=8)
    adapter = USBAAdapter(d_model=32, config=cfg, num_layers=2)
    h = torch.randn(2, 10, 32)
    _, aux = adapter(h)
    assert '_mu_last' in aux and '_z_last' in aux
    assert aux['_mu_last'].shape == (2, 10, 8), f"got {aux['_mu_last'].shape}"
    assert aux['_z_last'].shape == (2, 10, 8)
    print("[PASS] D6 adapter aux preserves real mu/z")


def test_d7_full_forward():
    """D7. Full forward pass: shapes, frozen backbone, backward."""
    from adapters.usba_config import USBAConfig
    from adapters.injection import USBAInjectedModel, USBAInjector
    from adapters.losses import USBALoss

    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))
        def forward(self, x):
            B = x.shape[0]
            return torch.randn(B, 4, 5, 32)

    bb = DummyBackbone()
    for p in bb.parameters():
        p.requires_grad = False

    cfg = USBAConfig(latent_dim=8, factorized=True)
    model = USBAInjectedModel(bb, cfg, token_dim=32, num_classes=6, n_channels=4, seq_len=5)
    x = torch.randn(2, 4, 5, 32)
    out = model(x)
    assert out['logits'].shape == (2, 6)
    assert out['z_agg'].shape == (2, 32)
    assert '_mu_last' in out['adapter_aux']
    # Backbone frozen
    assert not any(p.requires_grad for p in model.backbone.parameters())
    # Adapter trainable
    assert sum(p.numel() for p in model.usba.parameters() if p.requires_grad) > 0
    # Backward
    criterion = USBALoss(beta=1e-4, kl_reduction='mean', budget_warmup_epochs=5)
    mu_pooled = out['adapter_aux']['_mu_last'].mean(dim=1)
    loss, ld = criterion(
        logits=out['logits'], labels=torch.tensor([0, 3]),
        adapter_aux=out['adapter_aux'], adapter=model.usba,
        subject_ids=torch.tensor([0, 1]), z_agg=mu_pooled,
    )
    loss.backward()
    assert model.head[0].weight.grad is not None
    print(f"[PASS] D7 full forward (loss={ld['total']:.4f})")


# ── Additional integration tests ──────────────────────────────────

def test_usba_layer_shapes():
    """Basic USBALayer shape test."""
    from adapters.usba_config import USBAConfig
    from adapters.usba import USBALayer
    B, T, D = 4, 80, 200
    config = USBAConfig(latent_dim=64, factorized=True, gate_type='layer_wise')
    layer = USBALayer(d_model=D, config=config, layer_idx=0)
    x = torch.randn(B, T, D)
    out, aux = layer(x)
    assert out.shape == (B, T, D)
    assert '_kl' in aux and aux['_kl'].dim() == 0
    print("[PASS] USBALayer shapes")


def test_all_gate_types():
    """Test all gate types work."""
    from adapters.usba_config import USBAConfig
    from adapters.usba import USBALayer
    B, T, D = 4, 80, 200
    for gt in ['layer_wise', 'token_wise', 'channel_wise']:
        config = USBAConfig(latent_dim=64, gate_type=gt)
        layer = USBALayer(d_model=D, config=config)
        out, _ = layer(torch.randn(B, T, D))
        assert out.shape == (B, T, D), f"Failed for {gt}"
    print("[PASS] all gate types")


def test_all_branch_types():
    """Test all temporal/spatial branch combos."""
    from adapters.usba_config import USBAConfig
    from adapters.usba import USBALayer
    B, T, D = 4, 80, 200
    for temp in ['depthwise_conv', 'low_rank_mix']:
        for spat in ['channel_attention', 'grouped_mlp']:
            config = USBAConfig(latent_dim=64, temporal_branch_type=temp, spatial_branch_type=spat)
            layer = USBALayer(d_model=D, config=config)
            out, _ = layer(torch.randn(B, T, D))
            assert out.shape == (B, T, D)
    print("[PASS] all branch types")


def test_4d_vs_3d_temporal():
    """4D-aware temporal conv produces different results than 3D."""
    from adapters.branches import DepthwiseTemporalConv
    B, C, S, D = 2, 8, 5, 200
    T = C * S
    branch = DepthwiseTemporalConv(d_model=D, kernel_size=5)
    branch.eval()
    x = torch.randn(B, T, D)
    out_4d = branch(x, token_structure=(C, S))
    out_3d = branch(x, token_structure=None)
    diff = (out_4d - out_3d).abs().mean().item()
    assert diff > 1e-6, f"4D vs 3D should differ: {diff}"
    print(f"[PASS] 4D vs 3D temporal (diff={diff:.6f})")


def test_training_step():
    """End-to-end: one training step with optimizer."""
    from adapters.usba_config import USBAConfig
    from adapters.injection import USBAInjector
    from adapters.losses import USBALoss

    B, C, S, P = 4, 16, 5, 200
    num_classes = 6

    class DummyBackbone(nn.Module):
        def forward(self, x):
            return torch.randn(x.shape) * 0.1

    backbone = DummyBackbone()
    config = USBAConfig(latent_dim=32, factorized=True)
    model = USBAInjector.inject(backbone, config, P, num_classes, C, S)
    criterion = USBALoss(beta=1e-4, kl_reduction='mean', budget_warmup_epochs=5)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    model.train()
    x = torch.randn(B, C, S, P)
    labels = torch.randint(0, num_classes, (B,))
    out = model(x)
    mu_pooled = out['adapter_aux']['_mu_last'].mean(dim=1)
    loss, ld = criterion(
        logits=out['logits'], labels=labels,
        adapter_aux=out['adapter_aux'], adapter=model.usba,
        z_agg=mu_pooled, current_epoch=1, beta_warmup_epochs=5,
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"[PASS] training step (loss={ld['total']:.4f})")


def test_config_new_fields():
    """Test new USBAConfig fields: kl_reduction, budget_warmup_epochs."""
    from adapters.usba_config import USBAConfig
    cfg = USBAConfig()
    assert cfg.kl_reduction == 'mean'
    assert cfg.budget_warmup_epochs == 10
    cfg2 = USBAConfig(kl_reduction='total', budget_warmup_epochs=20)
    assert cfg2.kl_reduction == 'total'
    assert cfg2.budget_warmup_epochs == 20
    d = cfg2.to_dict()
    assert d['kl_reduction'] == 'total'
    cfg3 = USBAConfig.from_dict(d)
    assert cfg3.kl_reduction == 'total'
    print("[PASS] config new fields")


if __name__ == '__main__':
    print("=" * 60)
    print("USBA Adapter Tests (bug fixes D1-D7 + integration)")
    print("=" * 60)

    tests = [
        test_d1_token_wise_gate,
        test_d2_channel_attention_4d,
        test_d3_gated_fusion_init,
        test_d4_backward_flow,
        test_d5_categorical_kernel,
        test_d6_inter_layer_aux,
        test_d7_full_forward,
        test_usba_layer_shapes,
        test_all_gate_types,
        test_all_branch_types,
        test_4d_vs_3d_temporal,
        test_training_step,
        test_config_new_fields,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'='*60}")
    sys.exit(1 if failed > 0 else 0)
