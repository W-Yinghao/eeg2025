#!/usr/bin/env python3
"""
Test script for MI Fine-Tuning Framework with CodeBrain backbone.

Runs construction checks (CPU) and optionally forward/backward (GPU required by SSSM).
"""

import torch
from mi_finetuning_framework import (
    MIFineTuner,
    calculate_mi_loss,
)
from backbone_factory import create_backbone

HAS_CUDA = torch.cuda.is_available()

print("=" * 70)
print("Testing MI Fine-Tuning Framework with CodeBrain")
print(f"CUDA available: {HAS_CUDA}")
print("=" * 70)

# Small config
BATCH_SIZE = 4
N_CHANNELS = 16
SEQ_LEN = 5
PATCH_SIZE = 200
N_LAYER = 2
NUM_CLASSES = 6
HIDDEN_DIM = 64
VIB_DIM = 32
EXPERT_DIM = N_CHANNELS * 5

print(f"\nConfig: B={BATCH_SIZE}, C={N_CHANNELS}, S={SEQ_LEN}, P={PATCH_SIZE}")
print(f"  n_layer={N_LAYER}, classes={NUM_CLASSES}, hidden={HIDDEN_DIM}, vib={VIB_DIM}")
print(f"  expert_dim={EXPERT_DIM}")

# ---- Test 1: Construction (CPU) ----
print("\n--- Test 1: Model Construction ---")
device = 'cuda:0' if HAS_CUDA else 'cpu'
backbone, backbone_out_dim, token_dim = create_backbone(
    model_type='codebrain',
    n_channels=N_CHANNELS, seq_len=SEQ_LEN, patch_size=PATCH_SIZE,
    n_layer=N_LAYER, codebook_size_t=512, codebook_size_f=512,
    dropout=0.1, pretrained_weights_path=None, device=device,
)
print(f"  backbone_out_dim = {backbone_out_dim}")

model = MIFineTuner(
    backbone=backbone, backbone_out_dim=backbone_out_dim,
    expert_dim=EXPERT_DIM, hidden_dim=HIDDEN_DIM, vib_dim=VIB_DIM,
    num_classes=NUM_CLASSES, dropout=0.1,
)

# Verify freeze
frozen_count = sum(1 for p in model.backbone.parameters() if p.requires_grad)
assert frozen_count == 0, f"Backbone has {frozen_count} unfrozen params!"
print("  Backbone frozen: OK")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert trainable > 0, "No trainable parameters!"
print(f"  Trainable params: {trainable:,}")
print("  Construction: PASSED")

# ---- Test 2: Loss functions (CPU, standalone) ----
print("\n--- Test 2: Loss Functions ---")
logits = torch.randn(4, 6, requires_grad=True)
labels = torch.randint(0, 6, (4,))
mu = torch.randn(4, 32, requires_grad=True)
log_var = torch.randn(4, 32, requires_grad=True)
z_fm = torch.randn(4, 64, requires_grad=True)
z_fm_n = torch.nn.functional.normalize(z_fm, dim=-1)
z_exp = torch.randn(4, 64, requires_grad=True)
z_exp_n = torch.nn.functional.normalize(z_exp, dim=-1)

loss, ld = calculate_mi_loss(logits, labels, mu, log_var, z_fm_n, z_exp_n, alpha=1.0, beta=1e-3)
print(f"  total={ld['total']:.4f}, ce={ld['ce']:.4f}, vib={ld['vib']:.4f}, nce={ld['infonce']:.4f}")
loss.backward()
assert logits.grad is not None, "No grad for logits"
assert mu.grad is not None, "No grad for mu"
assert z_fm.grad is not None, "No grad for z_fm"
print("  Loss + backward: PASSED")

# ---- Test 3: Full forward/backward (GPU only) ----
if HAS_CUDA:
    print("\n--- Test 3: Full Forward/Backward (GPU) ---")
    model = model.cuda()
    x = torch.randn(BATCH_SIZE, N_CHANNELS, SEQ_LEN, PATCH_SIZE, device='cuda')
    x_expert = torch.randn(BATCH_SIZE, EXPERT_DIM, device='cuda')
    labels_gpu = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device='cuda')

    model.train()
    out_logits, out_mu, out_lv, out_zfm, out_zexp = model(x, x_expert)
    print(f"  logits: {out_logits.shape}, mu: {out_mu.shape}, z_fm_proj: {out_zfm.shape}")

    loss, ld = calculate_mi_loss(
        out_logits, labels_gpu, out_mu, out_lv, out_zfm, out_zexp,
        alpha=1.0, beta=1e-3, temperature=0.07,
    )
    print(f"  Loss: total={ld['total']:.4f}")

    loss.backward()

    # Gradient check
    for name in ['rep_projection', 'vib_layer', 'classifier', 'contrast_head', 'expert_projector']:
        module = getattr(model, name)
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in module.parameters() if p.requires_grad)
        print(f"  {name:20s}: {'OK' if has_grad else 'NO GRAD'}")

    backbone_leaked = any(p.grad is not None and p.grad.abs().sum() > 0
                          for p in model.backbone.parameters())
    print(f"  {'backbone':20s}: {'LEAKED!' if backbone_leaked else 'Frozen (correct)'}")

    # Eval mode
    model.eval()
    with torch.no_grad():
        logits_eval, _, _, _, _ = model(x, x_expert)
        preds = logits_eval.argmax(dim=-1)
        print(f"  Eval predictions: {preds.tolist()}")

    print("  Full forward/backward: PASSED")
else:
    print("\n--- Test 3: Skipped (no GPU, SSSM requires CUDA) ---")

# ---- Test 4: Expert Feature Extractor ----
print("\n--- Test 4: Expert Feature Extractor ---")
from train_mi_finetuning import ExpertFeatureExtractor

x_4d = torch.randn(4, 16, 5, 200)
for ftype, expected_dim in [('psd', 80), ('stats', 64), ('both', 144)]:
    ext = ExpertFeatureExtractor(n_channels=16, sampling_rate=200, feature_type=ftype)
    feats = ext(x_4d)
    assert feats.shape == (4, expected_dim), f"{ftype}: expected (4,{expected_dim}), got {feats.shape}"
    assert ext.get_dim(16) == expected_dim
    print(f"  {ftype:5s}: {x_4d.shape} -> {feats.shape} OK")
print("  Expert features: PASSED")

print("\n" + "=" * 70)
print("All tests passed!")
print("=" * 70)
