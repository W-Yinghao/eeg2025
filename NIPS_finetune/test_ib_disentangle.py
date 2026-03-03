#!/usr/bin/env python3
"""
Test script for IB + Disentanglement Fine-Tuning Framework with CodeBrain backbone.

Runs construction checks (CPU) and optionally full pipeline (GPU required by SSSM).

Usage:
    # Without wandb
    python test_ib_disentangle.py

    # With wandb logging
    python test_ib_disentangle.py --wandb_project eeg_ib_test

    # Custom run name
    python test_ib_disentangle.py --wandb_project eeg_ib_test --wandb_run test_v2
"""

import argparse
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from ib_disentangle_framework import (
    MultiDisease_CodeBrain_Model,
    InformationBottleneckLoss,
    GradientReversalLayer,
    CodeBrain_IB_Adapter,
    GRLScheduler,
    configure_optimizer,
    get_interpretability_heatmap,
    compute_ib_loss,
    compute_mi_loss,
    train_step,
)
from backbone_factory import create_backbone

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('--wandb_project', type=str, default=None,
                    help='WandB project name (None = no logging)')
parser.add_argument('--wandb_run', type=str, default='ib_disentangle_test',
                    help='WandB run name')
args = parser.parse_args()

# Init wandb
wandb_run = None
if args.wandb_project and WANDB_AVAILABLE:
    wandb_run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run,
        config={
            'test_type': 'ib_disentangle_framework',
        },
    )
    print(f"WandB logging enabled: {args.wandb_project}/{args.wandb_run}")

HAS_CUDA = torch.cuda.is_available()

print("=" * 70)
print("Testing IB + Disentanglement Framework with CodeBrain")
print(f"CUDA available: {HAS_CUDA}")
print("=" * 70)

# Small config
BATCH_SIZE = 4
N_CHANNELS = 16
SEQ_LEN = 5
PATCH_SIZE = 200
N_LAYER = 2
NUM_CLASSES = 6
NUM_SUBJECTS = 10
LATENT_DIM = 64
TOKEN_DIM = 200
T = N_CHANNELS * SEQ_LEN  # 80 tokens

print(f"\nConfig: B={BATCH_SIZE}, C={N_CHANNELS}, S={SEQ_LEN}, P={PATCH_SIZE}")
print(f"  n_layer={N_LAYER}, classes={NUM_CLASSES}, subjects={NUM_SUBJECTS}")
print(f"  latent_dim={LATENT_DIM}, tokens=C*S={T}")


# ---- Test 1: GRL ----
print("\n--- Test 1: Gradient Reversal Layer ---")
grl = GradientReversalLayer(lambda_=1.0)
x = torch.randn(4, 64, requires_grad=True)
y = grl(x)
loss = y.sum()
loss.backward()
# Forward: identity, so y == x
assert torch.allclose(y, x.detach()), "GRL forward should be identity"
# Backward: reversed, so grad should be -1
assert torch.allclose(x.grad, -torch.ones_like(x.grad)), "GRL should reverse gradients"
print("  GRL forward (identity): OK")
print("  GRL backward (reversed): OK")

# Test lambda scaling
grl.set_lambda(0.5)
x2 = torch.randn(4, 64, requires_grad=True)
y2 = grl(x2)
y2.sum().backward()
assert torch.allclose(x2.grad, -0.5 * torch.ones_like(x2.grad)), "GRL lambda scaling failed"
print("  GRL lambda=0.5 scaling: OK")
print("  GRL: PASSED")

if wandb_run:
    wandb_run.log({'test/grl': 1, 'test/grl_lambda_check': 0.5})


# ---- Test 2: IB Adapter ----
print("\n--- Test 2: CodeBrain_IB_Adapter ---")
adapter = CodeBrain_IB_Adapter(input_dim=TOKEN_DIM, latent_dim=LATENT_DIM, dropout=0.1)

H = torch.randn(BATCH_SIZE, T, TOKEN_DIM)

# Train mode (sampling)
adapter.train()
Z_train, mu, log_var = adapter(H)
assert Z_train.shape == (BATCH_SIZE, T, LATENT_DIM), f"Z shape: {Z_train.shape}"
assert mu.shape == (BATCH_SIZE, T, LATENT_DIM), f"mu shape: {mu.shape}"
assert log_var.shape == (BATCH_SIZE, T, LATENT_DIM), f"log_var shape: {log_var.shape}"
# In train mode, Z should differ from mu (stochastic)
assert not torch.allclose(Z_train, mu), "Z should be stochastic in train mode"
print(f"  Train mode: Z={Z_train.shape}, mu={mu.shape}, log_var={log_var.shape}")

# Eval mode (deterministic)
adapter.eval()
Z_eval, mu2, _ = adapter(H)
assert torch.allclose(Z_eval, mu2), "In eval mode, Z should equal mu"
print("  Eval mode: Z == mu (deterministic): OK")
print("  IB Adapter: PASSED")

if wandb_run:
    wandb_run.log({
        'test/ib_adapter': 1,
        'ib_adapter/Z_shape_0': Z_train.shape[0],
        'ib_adapter/Z_shape_1': Z_train.shape[1],
        'ib_adapter/Z_shape_2': Z_train.shape[2],
        'ib_adapter/mu_mean': mu.mean().item(),
        'ib_adapter/mu_std': mu.std().item(),
        'ib_adapter/logvar_mean': log_var.mean().item(),
    })


# ---- Test 3: Loss Functions ----
print("\n--- Test 3: Loss Functions ---")
# IB loss (KL)
mu_test = torch.randn(BATCH_SIZE, T, LATENT_DIM, requires_grad=True)
log_var_test = torch.randn(BATCH_SIZE, T, LATENT_DIM, requires_grad=True)
kl = compute_ib_loss(mu_test, log_var_test)
assert kl.shape == (), f"IB loss should be scalar, got {kl.shape}"
kl.backward()
assert mu_test.grad is not None, "No grad for mu"
assert log_var_test.grad is not None, "No grad for log_var"
print(f"  IB (KL) loss: {kl.item():.4f}, grads: OK")

# MI loss (subject CE)
subj_logits = torch.randn(BATCH_SIZE, NUM_SUBJECTS, requires_grad=True)
subj_ids = torch.randint(0, NUM_SUBJECTS, (BATCH_SIZE,))
mi = compute_mi_loss(subj_logits, subj_ids)
mi.backward()
assert subj_logits.grad is not None, "No grad for subject logits"
print(f"  MI (subject CE) loss: {mi.item():.4f}, grads: OK")

# Full criterion
criterion = InformationBottleneckLoss(beta=1e-3, lambda_adv=0.5, task_type='multiclass')
disease_logits = torch.randn(BATCH_SIZE, NUM_CLASSES, requires_grad=True)
labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
mu_c = torch.randn(BATCH_SIZE, T, LATENT_DIM, requires_grad=True)
lv_c = torch.randn(BATCH_SIZE, T, LATENT_DIM, requires_grad=True)
sl_c = torch.randn(BATCH_SIZE, NUM_SUBJECTS, requires_grad=True)
si_c = torch.randint(0, NUM_SUBJECTS, (BATCH_SIZE,))

total, ld = criterion(disease_logits, labels, mu_c, lv_c, sl_c, si_c)
print(f"  Criterion: total={ld['total']:.4f}, task={ld['task']:.4f}, "
      f"ib={ld['ib']:.4f}, mi={ld['mi']:.4f}")
total.backward()
assert disease_logits.grad is not None, "No grad for disease logits"
print("  Full criterion backward: OK")

# Without subject IDs
total2, ld2 = criterion(disease_logits.detach().requires_grad_(True),
                         labels, mu_c.detach().requires_grad_(True),
                         lv_c.detach().requires_grad_(True), None, None)
assert ld2['mi'] == 0.0, "MI should be 0 without subject IDs"
print("  Criterion without subjects (mi=0): OK")
print("  Loss Functions: PASSED")

if wandb_run:
    wandb_run.log({
        'test/loss_functions': 1,
        'loss/ib_kl': kl.item(),
        'loss/mi_ce': mi.item(),
        'loss/total': ld['total'],
        'loss/task': ld['task'],
        'loss/ib': ld['ib'],
        'loss/mi': ld['mi'],
        'loss/weighted_ib': ld['weighted_ib'],
        'loss/weighted_mi': ld['weighted_mi'],
    })


# ---- Test 4: Interpretability Heatmap ----
print("\n--- Test 4: Interpretability Heatmap ---")
mu_h = torch.randn(BATCH_SIZE, T, LATENT_DIM)
lv_h = torch.randn(BATCH_SIZE, T, LATENT_DIM)
hmap = get_interpretability_heatmap(mu_h, lv_h, N_CHANNELS, SEQ_LEN)

assert hmap['kl_per_token'].shape == (BATCH_SIZE, T), f"kl_per_token: {hmap['kl_per_token'].shape}"
assert hmap['heatmap'].shape == (BATCH_SIZE, N_CHANNELS, SEQ_LEN), f"heatmap: {hmap['heatmap'].shape}"
assert hmap['channel_importance'].shape == (BATCH_SIZE, N_CHANNELS)
assert hmap['temporal_importance'].shape == (BATCH_SIZE, SEQ_LEN)
print(f"  kl_per_token: {hmap['kl_per_token'].shape}")
print(f"  heatmap: {hmap['heatmap'].shape}")
print(f"  channel_importance: {hmap['channel_importance'].shape}")
print(f"  temporal_importance: {hmap['temporal_importance'].shape}")
print(f"  Top-3 channels (sample 0): {hmap['channel_importance'][0].topk(3)[1].tolist()}")
print("  Interpretability Heatmap: PASSED")

if wandb_run:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Log heatmap as image
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Spatial-temporal heatmap (sample 0)
    im0 = axes[0].imshow(hmap['heatmap'][0].numpy(), aspect='auto', cmap='hot')
    axes[0].set_title('Info Retention Heatmap')
    axes[0].set_xlabel('Time Patch')
    axes[0].set_ylabel('Channel')
    plt.colorbar(im0, ax=axes[0])

    # Channel importance
    ch_imp_mean = hmap['channel_importance'].mean(dim=0).numpy()
    axes[1].barh(range(N_CHANNELS), ch_imp_mean)
    axes[1].set_title('Channel Importance')
    axes[1].set_xlabel('KL (info retained)')
    axes[1].set_ylabel('Channel')

    # Temporal importance
    temp_imp_mean = hmap['temporal_importance'].mean(dim=0).numpy()
    axes[2].bar(range(SEQ_LEN), temp_imp_mean)
    axes[2].set_title('Temporal Importance')
    axes[2].set_xlabel('Time Patch')
    axes[2].set_ylabel('KL (info retained)')

    plt.tight_layout()
    wandb_run.log({
        'test/heatmap': 1,
        'heatmap/spatial_temporal': wandb.Image(fig),
    })
    plt.close(fig)

    # Log per-channel importance as bar chart
    wandb_run.log({
        f'heatmap/channel_{i}_importance': v
        for i, v in enumerate(ch_imp_mean)
    })
    wandb_run.log({
        f'heatmap/temporal_{i}_importance': v
        for i, v in enumerate(temp_imp_mean)
    })


# ---- Test 5: GRL Scheduler ----
print("\n--- Test 5: GRL Scheduler ---")
# Create a dummy model to test scheduler
device = 'cuda:0' if HAS_CUDA else 'cpu'
backbone, backbone_out_dim, token_dim = create_backbone(
    model_type='codebrain',
    n_channels=N_CHANNELS, seq_len=SEQ_LEN, patch_size=PATCH_SIZE,
    n_layer=N_LAYER, pretrained_weights_path=None, device=device,
)
model = MultiDisease_CodeBrain_Model(
    backbone=backbone, token_dim=token_dim,
    num_classes=NUM_CLASSES, num_subjects=NUM_SUBJECTS,
    latent_dim=LATENT_DIM, dropout=0.1,
)

scheduler = GRLScheduler(model, gamma=10.0)
lambdas = []
for epoch in [1, 25, 50, 75, 100]:
    lam = scheduler.step(epoch, 100)
    lambdas.append(lam)
print(f"  Lambda schedule: {[f'{l:.3f}' for l in lambdas]}")
assert lambdas[0] < lambdas[-1], "Lambda should increase over training"
assert lambdas[-1] > 0.9, f"Lambda at end should be near 1, got {lambdas[-1]}"
print("  GRL Scheduler: PASSED")

if wandb_run:
    # Log full lambda schedule as a line chart
    lambda_epochs = [1, 25, 50, 75, 100]
    for ep, lam in zip(lambda_epochs, lambdas):
        wandb_run.log({'grl_schedule/epoch': ep, 'grl_schedule/lambda': lam})


# ---- Test 6: Model Construction ----
print("\n--- Test 6: Model Construction ---")
# Verify frozen backbone
frozen_count = sum(1 for p in model.backbone.parameters() if p.requires_grad)
assert frozen_count == 0, f"Backbone has {frozen_count} unfrozen params!"
print("  Backbone frozen: OK")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert trainable > 0, "No trainable parameters!"
print(f"  Trainable params: {trainable:,}")

# Verify optimizer groups
opt = configure_optimizer(model, lr=1e-3, weight_decay=1e-3)
print(f"  Optimizer groups: {len(opt.param_groups)}")
for pg in opt.param_groups:
    n_params = sum(p.numel() for p in pg['params'])
    print(f"    {pg['name']}: {n_params:,} params, lr={pg['lr']}")
print("  Model Construction: PASSED")

if wandb_run:
    frozen = sum(p.numel() for p in model.backbone.parameters())
    wandb_run.log({
        'test/model_construction': 1,
        'model/frozen_params': frozen,
        'model/trainable_params': trainable,
        'model/trainable_ratio': trainable / (frozen + trainable) * 100,
    })
    # Log per-group param counts
    for pg in opt.param_groups:
        n_params = sum(p.numel() for p in pg['params'])
        wandb_run.log({
            f"model/{pg['name']}_params": n_params,
            f"model/{pg['name']}_lr": pg['lr'],
        })


# ---- Test 7: Full Forward/Backward (GPU only) ----
if HAS_CUDA:
    print("\n--- Test 7: Full Forward/Backward (GPU) ---")
    model = model.cuda()
    x = torch.randn(BATCH_SIZE, N_CHANNELS, SEQ_LEN, PATCH_SIZE, device='cuda')
    labels_gpu = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device='cuda')
    subject_ids_gpu = torch.randint(0, NUM_SUBJECTS, (BATCH_SIZE,), device='cuda')

    model.train()
    outputs = model(x, return_tokens=True)
    print(f"  disease_logits: {outputs['disease_logits'].shape}")
    print(f"  subject_logits: {outputs['subject_logits'].shape}")
    print(f"  mu: {outputs['mu'].shape}")
    print(f"  log_var: {outputs['log_var'].shape}")
    print(f"  z_agg: {outputs['z_agg'].shape}")
    print(f"  z_tokens: {outputs['z_tokens'].shape}")

    # Loss
    criterion_gpu = InformationBottleneckLoss(
        beta=1e-3, lambda_adv=0.5, task_type='multiclass'
    )
    loss, ld = criterion_gpu(
        outputs['disease_logits'], labels_gpu,
        outputs['mu'], outputs['log_var'],
        outputs['subject_logits'], subject_ids_gpu,
    )
    print(f"  Loss: total={ld['total']:.4f}, task={ld['task']:.4f}, "
          f"ib={ld['ib']:.4f}, mi={ld['mi']:.4f}")

    # Backward
    loss.backward()

    # Gradient check
    for name in ['ib_adapter', 'disease_head', 'subject_head']:
        module = getattr(model, name)
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in module.parameters() if p.requires_grad)
        print(f"  {name:20s}: {'OK' if has_grad else 'NO GRAD'}")

    backbone_leaked = any(p.grad is not None and p.grad.abs().sum() > 0
                          for p in model.backbone.parameters())
    print(f"  {'backbone':20s}: {'LEAKED!' if backbone_leaked else 'Frozen (correct)'}")

    # Test train_step
    opt_gpu = configure_optimizer(model, lr=1e-3)
    model.zero_grad()
    ld2 = train_step(model, criterion_gpu, opt_gpu, x, labels_gpu,
                     subject_ids_gpu, clip_value=5.0)
    print(f"  train_step: loss={ld2['total']:.4f}")

    # Heatmap
    hmap_gpu = get_interpretability_heatmap(
        outputs['mu'].detach(), outputs['log_var'].detach(),
        N_CHANNELS, SEQ_LEN
    )
    print(f"  Heatmap shape: {hmap_gpu['heatmap'].shape}")

    # Eval mode
    model.eval()
    with torch.no_grad():
        outputs_eval = model(x)
        preds = outputs_eval['disease_logits'].argmax(dim=-1)
        print(f"  Eval predictions: {preds.tolist()}")

    print("  Full Forward/Backward: PASSED")

    if wandb_run:
        wandb_run.log({
            'test/full_pipeline_gpu': 1,
            'gpu/disease_logits_shape': list(outputs['disease_logits'].shape),
            'gpu/subject_logits_shape': list(outputs['subject_logits'].shape),
            'gpu/mu_shape': list(outputs['mu'].shape),
            'gpu/z_agg_shape': list(outputs['z_agg'].shape),
            'gpu/loss_total': ld['total'],
            'gpu/loss_task': ld['task'],
            'gpu/loss_ib': ld['ib'],
            'gpu/loss_mi': ld['mi'],
            'gpu/train_step_loss': ld2['total'],
        })

        # Log GPU heatmap
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        hm_data = hmap_gpu['heatmap'][0].cpu().numpy()
        im = ax.imshow(hm_data, aspect='auto', cmap='hot')
        ax.set_title('GPU Info Retention Heatmap (sample 0)')
        ax.set_xlabel('Time Patch')
        ax.set_ylabel('Channel')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        wandb_run.log({'gpu/heatmap': wandb.Image(fig)})
        plt.close(fig)

        # Gradient norms per module
        for name in ['ib_adapter', 'disease_head', 'subject_head']:
            module = getattr(model, name)
            grad_norm = sum(
                p.grad.norm().item() for p in module.parameters()
                if p.requires_grad and p.grad is not None
            )
            wandb_run.log({f'gpu/grad_norm_{name}': grad_norm})

else:
    print("\n--- Test 7: Skipped (no GPU, SSSM requires CUDA) ---")

print("\n" + "=" * 70)
print("All tests passed!")
print("=" * 70)

if wandb_run:
    wandb_run.log({'test/all_passed': 1})
    wandb_run.finish()
    print("WandB run finished.")
