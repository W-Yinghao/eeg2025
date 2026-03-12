#!/usr/bin/env python3
"""
Minimal tests for USBA adapter.

Run:
    cd ~/eeg2025/NIPS_finetune
    python test_usba.py
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))


def test_usba_layer_shapes():
    """Test that USBALayer produces correct output shapes."""
    from adapters.usba_config import USBAConfig
    from adapters.usba import USBALayer

    B, T, D = 4, 80, 200
    config = USBAConfig(latent_dim=64, factorized=True, gate_type='layer_wise')
    layer = USBALayer(d_model=D, config=config, layer_idx=0)

    x = torch.randn(B, T, D)
    out, aux = layer(x)

    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"
    assert '_kl' in aux, "Missing KL in aux"
    assert isinstance(aux['_kl'], torch.Tensor), "KL should be tensor"
    assert aux['_kl'].dim() == 0, "KL should be scalar"
    print("[PASS] test_usba_layer_shapes")


def test_usba_adapter_multi_layer():
    """Test multi-layer USBAAdapter."""
    from adapters.usba_config import USBAConfig
    from adapters.usba import USBAAdapter

    B, T, D = 4, 80, 200
    config = USBAConfig(latent_dim=64, factorized=True)
    adapter = USBAAdapter(d_model=D, config=config, num_layers=3)

    x = torch.randn(B, T, D)
    out, aux = adapter(x)

    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"
    assert len(aux['_all_kls']) == 3, "Should have 3 KL values"
    assert len(aux['_all_gate_vals']) == 3, "Should have 3 gate values"
    print("[PASS] test_usba_adapter_multi_layer")


def test_usba_non_factorized():
    """Test non-factorized (ablation) mode."""
    from adapters.usba_config import USBAConfig
    from adapters.usba import USBALayer

    B, T, D = 4, 80, 200
    config = USBAConfig(latent_dim=64, factorized=False)
    layer = USBALayer(d_model=D, config=config)

    x = torch.randn(B, T, D)
    out, aux = layer(x)
    assert out.shape == (B, T, D)
    print("[PASS] test_usba_non_factorized")


def test_gate_types():
    """Test all gate types."""
    from adapters.usba_config import USBAConfig
    from adapters.usba import USBALayer

    B, T, D = 4, 80, 200
    for gate_type in ['layer_wise', 'token_wise', 'channel_wise']:
        config = USBAConfig(latent_dim=64, gate_type=gate_type)
        layer = USBALayer(d_model=D, config=config)
        x = torch.randn(B, T, D)
        out, aux = layer(x)
        assert out.shape == (B, T, D), f"Failed for gate_type={gate_type}"
    print("[PASS] test_gate_types")


def test_branch_types():
    """Test all temporal and spatial branch types."""
    from adapters.usba_config import USBAConfig
    from adapters.usba import USBALayer

    B, T, D = 4, 80, 200
    for temp in ['depthwise_conv', 'low_rank_mix']:
        for spat in ['channel_attention', 'grouped_mlp']:
            config = USBAConfig(
                latent_dim=64,
                temporal_branch_type=temp,
                spatial_branch_type=spat,
            )
            layer = USBALayer(d_model=D, config=config)
            x = torch.randn(B, T, D)
            out, aux = layer(x)
            assert out.shape == (B, T, D), f"Failed for {temp}/{spat}"
    print("[PASS] test_branch_types")


def test_usba_loss():
    """Test USBALoss forward."""
    from adapters.usba_config import USBAConfig
    from adapters.usba import USBAAdapter
    from adapters.losses import USBALoss

    B, T, D = 4, 80, 200
    num_classes = 6
    config = USBAConfig(latent_dim=64)
    adapter = USBAAdapter(d_model=D, config=config, num_layers=1)

    x = torch.randn(B, T, D)
    h_adapted, adapter_aux = adapter(x)
    z_agg = h_adapted.mean(dim=1)

    logits = torch.randn(B, num_classes)
    labels = torch.randint(0, num_classes, (B,))

    criterion = USBALoss(beta=1e-4, lambda_cc_inv=0.01, eta_budget=1e-3)

    # Without subject_ids
    loss, loss_dict = criterion(
        logits=logits, labels=labels,
        adapter_aux=adapter_aux, adapter=adapter,
    )
    assert loss.dim() == 0, "Loss should be scalar"
    assert 'total' in loss_dict
    assert loss_dict['cc_inv_active'] is False, "CC-inv should be inactive without subjects"
    print("[PASS] test_usba_loss (no subjects)")

    # With subject_ids
    subject_ids = torch.randint(0, 10, (B,))
    loss2, loss_dict2 = criterion(
        logits=logits, labels=labels,
        adapter_aux=adapter_aux, adapter=adapter,
        subject_ids=subject_ids, z_agg=z_agg,
    )
    assert loss2.dim() == 0
    print(f"[PASS] test_usba_loss (with subjects, cc_inv_active={loss_dict2['cc_inv_active']})")


def test_usba_loss_no_subject_graceful():
    """Test that loss works gracefully without subject_ids."""
    from adapters.usba_config import USBAConfig
    from adapters.usba import USBAAdapter
    from adapters.losses import USBALoss

    B, T, D = 4, 80, 200
    config = USBAConfig(latent_dim=64)
    adapter = USBAAdapter(d_model=D, config=config, num_layers=1)
    criterion = USBALoss(enable_cc_inv=True)

    x = torch.randn(B, T, D)
    h, aux = adapter(x)
    logits = torch.randn(B, 6)
    labels = torch.randint(0, 6, (B,))

    # Should not raise
    loss, ld = criterion(logits=logits, labels=labels, adapter_aux=aux,
                         adapter=adapter, subject_ids=None)
    assert ld['cc_inv'] == 0.0
    print("[PASS] test_usba_loss_no_subject_graceful")


def test_injected_model_with_dummy_backbone():
    """Integration test: full forward + loss with a dummy backbone."""
    from adapters.usba_config import USBAConfig
    from adapters.injection import USBAInjector
    from adapters.losses import USBALoss

    B, C, S, P = 4, 16, 5, 200
    D = P  # token_dim = 200
    num_classes = 6

    # Dummy frozen backbone that mimics CBraMod output shape
    class DummyBackbone(nn.Module):
        def forward(self, x):
            B, C, S, P = x.shape
            return torch.randn(B, C, S, P, device=x.device) * 0.1

    backbone = DummyBackbone()
    for p in backbone.parameters():
        p.requires_grad = False

    config = USBAConfig(
        latent_dim=64,
        factorized=True,
        gate_type='layer_wise',
        enable_cc_inv=True,
        enable_budget_reg=True,
    )

    model = USBAInjector.inject(
        backbone=backbone,
        config=config,
        token_dim=D,
        num_classes=num_classes,
        n_channels=C,
        seq_len=S,
    )

    criterion = USBALoss(beta=1e-4, lambda_cc_inv=0.01, eta_budget=1e-3)

    # Forward
    x = torch.randn(B, C, S, P)
    out = model(x)

    assert out['logits'].shape == (B, num_classes)
    assert out['z_agg'].shape == (B, D)

    # Loss without subjects
    loss, ld = criterion(
        logits=out['logits'],
        labels=torch.randint(0, num_classes, (B,)),
        adapter_aux=out['adapter_aux'],
        adapter=model.usba,
    )
    assert loss.requires_grad
    print("[PASS] test_injected_model_with_dummy_backbone (forward + loss)")

    # Backward
    loss.backward()
    # Check adapter params got gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.usba.parameters() if p.requires_grad)
    assert has_grad, "Adapter parameters should have gradients"
    print("[PASS] test_injected_model_with_dummy_backbone (backward)")


def test_training_step():
    """End-to-end: one training step with optimizer."""
    from adapters.usba_config import USBAConfig
    from adapters.injection import USBAInjector
    from adapters.losses import USBALoss

    B, C, S, P = 4, 16, 5, 200
    num_classes = 6

    class DummyBackbone(nn.Module):
        def forward(self, x):
            return torch.randn(x.shape, device=x.device) * 0.1

    backbone = DummyBackbone()
    config = USBAConfig(latent_dim=32, factorized=True)
    model = USBAInjector.inject(backbone, config, P, num_classes, C, S)
    criterion = USBALoss(beta=1e-4)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    # Training step
    model.train()
    x = torch.randn(B, C, S, P)
    labels = torch.randint(0, num_classes, (B,))

    out = model(x)
    loss, ld = criterion(
        logits=out['logits'], labels=labels,
        adapter_aux=out['adapter_aux'], adapter=model.usba,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"[PASS] test_training_step (loss={ld['total']:.4f}, kl={ld['kl_total']:.4f})")


def test_collect_metrics():
    """Test metrics collection."""
    from adapters.usba_config import USBAConfig
    from adapters.usba import USBAAdapter
    from adapters.losses import USBALoss, collect_usba_metrics

    B, T, D = 4, 80, 200
    config = USBAConfig(latent_dim=64)
    adapter = USBAAdapter(D, config, num_layers=2)

    x = torch.randn(B, T, D)
    h, aux = adapter(x)
    logits = torch.randn(B, 6)
    labels = torch.randint(0, 6, (B,))

    criterion = USBALoss()
    loss, ld = criterion(logits=logits, labels=labels, adapter_aux=aux, adapter=adapter)

    total_params = sum(p.numel() for p in adapter.parameters()) + 1000
    metrics = collect_usba_metrics(aux, adapter, ld, total_params)

    assert 'usba/trainable_params' in metrics
    assert 'usba/trainable_ratio' in metrics
    assert 'usba/kl_total' in metrics
    print(f"[PASS] test_collect_metrics (trainable_ratio={metrics['usba/trainable_ratio']:.1f}%)")


def test_4d_aware_branches():
    """Test that 4D-aware mode works correctly for CBraMod/CodeBrain-style tokens."""
    from adapters.usba_config import USBAConfig
    from adapters.usba import USBALayer, USBAAdapter

    B, C, S, D = 4, 16, 5, 200
    T = C * S  # = 80
    token_structure = (C, S)

    # Single layer with structure
    config = USBAConfig(latent_dim=64, factorized=True)
    layer = USBALayer(d_model=D, config=config)

    x = torch.randn(B, T, D)
    out, aux = layer(x, token_structure=token_structure)
    assert out.shape == (B, T, D), f"4D-aware output shape mismatch: {out.shape}"

    # Multi-layer adapter with structure
    adapter = USBAAdapter(d_model=D, config=config, num_layers=2)
    out2, aux2 = adapter(x, token_structure=token_structure)
    assert out2.shape == (B, T, D)
    print("[PASS] test_4d_aware_branches (CBraMod/CodeBrain 4D mode)")


def test_4d_vs_3d_different():
    """Verify that 4D-aware temporal branch produces different outputs than 3D."""
    from adapters.branches import DepthwiseTemporalConv

    B, C, S, D = 2, 8, 5, 200
    T = C * S

    branch = DepthwiseTemporalConv(d_model=D, kernel_size=5)
    branch.eval()

    x = torch.randn(B, T, D)

    # 4D-aware: conv applied per-channel along S=5 (never crosses channel boundary)
    out_4d = branch(x, token_structure=(C, S))
    # 3D flat: conv applied along T=40 (crosses channel boundaries)
    out_3d = branch(x, token_structure=None)

    # They should differ because 4D conv isolates channels
    diff = (out_4d - out_3d).abs().mean().item()
    assert diff > 1e-6, f"4D and 3D temporal branch should differ, diff={diff}"
    print(f"[PASS] test_4d_vs_3d_different (temporal branch mean diff={diff:.6f})")


def test_luna_style_3d():
    """Test LUNA-style tokens: (B, S, Q*D) with no channel structure."""
    from adapters.usba_config import USBAConfig
    from adapters.injection import USBAInjector

    B, S, QD = 4, 25, 256  # LUNA: 25 time patches, Q*D=4*64=256
    num_classes = 4

    class DummyLUNA(nn.Module):
        """Mimics LUNA encoder output: (B, S, Q*D) 3D."""
        def forward(self, x):
            B = x.shape[0]
            return torch.randn(B, S, QD, device=x.device) * 0.1

    backbone = DummyLUNA()
    config = USBAConfig(latent_dim=32, factorized=True)
    # n_channels=1 just for shape compat; injection will detect 'luna' as unknown → 3D
    model = USBAInjector.inject(backbone, config, QD, num_classes, n_channels=1, seq_len=S)

    assert model._token_structure is None, "LUNA should use 3D mode (no token_structure)"

    x = torch.randn(B, 1, S, QD)  # dummy input shape
    out = model(x)
    assert out['logits'].shape == (B, num_classes)
    print("[PASS] test_luna_style_3d (LUNA 3D mode, no channel structure)")


def test_cbramod_style_4d():
    """Test CBraMod-style backbone detection and 4D structure passing."""
    from adapters.usba_config import USBAConfig
    from adapters.injection import USBAInjector

    B, C, S, P = 4, 16, 5, 200
    num_classes = 6

    class CBraModDummy(nn.Module):
        """Named to trigger 'cbramod' detection."""
        def forward(self, x):
            return torch.randn(x.shape, device=x.device) * 0.1

    # Rename class to trigger detection
    CBraModDummy.__name__ = 'CBraModWithAdapters'

    backbone = CBraModDummy()
    config = USBAConfig(latent_dim=32, factorized=True)
    model = USBAInjector.inject(backbone, config, P, num_classes, C, S)

    assert model._token_structure == (C, S), f"CBraMod should have 4D structure, got {model._token_structure}"
    assert model._backbone_type == 'cbramod'

    x = torch.randn(B, C, S, P)
    out = model(x)
    assert out['logits'].shape == (B, num_classes)
    print(f"[PASS] test_cbramod_style_4d (token_structure={model._token_structure})")


if __name__ == '__main__':
    print("=" * 60)
    print("USBA Adapter Tests")
    print("=" * 60)

    tests = [
        test_usba_layer_shapes,
        test_usba_adapter_multi_layer,
        test_usba_non_factorized,
        test_gate_types,
        test_branch_types,
        test_usba_loss,
        test_usba_loss_no_subject_graceful,
        test_injected_model_with_dummy_backbone,
        test_training_step,
        test_collect_metrics,
        test_4d_aware_branches,
        test_4d_vs_3d_different,
        test_luna_style_3d,
        test_cbramod_style_4d,
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
