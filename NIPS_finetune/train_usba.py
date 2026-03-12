#!/usr/bin/env python3
"""
Training Script for USBA (Universal Sufficient Bottleneck Adapter) Fine-Tuning.

Uses frozen backbone (CodeBrain / CBraMod / FEMBA / LUNA) + USBA adapter:
  - Factorized temporal + spatial branches with gated fusion
  - Variational bottleneck with configurable gate
  - Class-conditional HSIC invariance (auto-disabled without subject_id)
  - Budget regularization for gate sparsity

Usage:
    # Basic: CBraMod + USBA on TUEV
    python train_usba.py --dataset TUEV --model cbramod --usba --cuda 0

    # CodeBrain + USBA on AD_DIAGNOSIS
    python train_usba.py --dataset AD_DIAGNOSIS --model codebrain --usba --cuda 0

    # Ablation: no factorization
    python train_usba.py --dataset TUEV --model cbramod --usba --usba_no_factorize --cuda 0

    # Ablation: no cc-inv
    python train_usba.py --dataset TUEV --model cbramod --usba --usba_no_cc_inv --cuda 0

    # Inter-layer injection on CBraMod
    python train_usba.py --dataset TUEV --model cbramod --usba --usba_selected_layers all --cuda 0

    # With WandB
    python train_usba.py --dataset TUEV --model cbramod --usba --wandb_project eeg_usba --cuda 0
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)

# Existing infrastructure
sys.path.insert(0, os.path.dirname(__file__))
from finetune_tuev_lmdb import (
    DATASET_CONFIGS,
    EEGLMDBDataset,
    setup_seed,
)
from backbone_factory import create_backbone

# USBA
from adapters import USBAConfig, USBAInjector, USBALoss, collect_usba_metrics

# Subject-aware dataset (reuse from train_ib_disentangle.py)
from train_ib_disentangle import EEGDatasetWithSubjects, load_data_with_subjects

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════
# Training & Evaluation
# ══════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model, dataloader, criterion, optimizer, device,
    clip_value, use_subjects, current_epoch, beta_warmup_epochs,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    loss_accum = {}
    all_preds = []
    all_labels = []
    n = 0

    for batch in dataloader:
        eeg = batch[0].to(device)
        labels = batch[1].to(device)
        subject_ids = batch[2].to(device) if use_subjects and len(batch) > 2 else None

        # Forward
        out = model(eeg)
        logits = out['logits']
        z_agg = out['z_agg']
        adapter_aux = out['adapter_aux']

        # Loss
        loss, loss_dict = criterion(
            logits=logits,
            labels=labels,
            adapter_aux=adapter_aux,
            adapter=model.usba,
            subject_ids=subject_ids,
            z_agg=z_agg,
            current_epoch=current_epoch,
            beta_warmup_epochs=beta_warmup_epochs,
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        # Track
        total_loss += loss_dict['total']
        for k, v in loss_dict.items():
            if isinstance(v, (int, float)):
                loss_accum[k] = loss_accum.get(k, 0.0) + v
        n += 1

        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg = {k: v / max(n, 1) for k, v in loss_accum.items()}
    avg['bal_acc'] = balanced_accuracy_score(all_labels, all_preds)
    avg['preds'] = all_preds
    avg['labels'] = all_labels
    return avg


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate on val/test set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    n = 0

    for batch in dataloader:
        eeg = batch[0].to(device)
        labels = batch[1].to(device)

        out = model(eeg)
        logits = out['logits']
        loss = F.cross_entropy(logits, labels)

        total_loss += loss.item()
        n += 1
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(n, 1)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    return avg_loss, bal_acc, all_preds, all_labels


# ══════════════════════════════════════════════════════════════════════
# Argparse
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description='USBA EEG Fine-Tuning')

    # Dataset
    parser.add_argument('--dataset', type=str, default='TUEV',
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--cuda', type=int, default=0)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--clip_value', type=float, default=5.0)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--patience', type=int, default=15)

    # Backbone
    parser.add_argument('--model', type=str, default='cbramod',
                        choices=['codebrain', 'cbramod', 'femba', 'luna'])
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--codebook_size_t', type=int, default=4096)
    parser.add_argument('--codebook_size_f', type=int, default=4096)
    parser.add_argument('--n_layer_cbramod', type=int, default=12)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--dim_feedforward', type=int, default=800)
    parser.add_argument('--use_layer_adapters', action='store_true')
    parser.add_argument('--adapter_reduction', type=int, default=4)
    parser.add_argument('--luna_size', type=str, default='base')
    parser.add_argument('--pretrained_weights', type=str, default=None)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Subject handling
    parser.add_argument('--use_subjects', action='store_true', default=True)
    parser.add_argument('--no_subjects', action='store_true')

    # Logging
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='checkpoints_usba')
    parser.add_argument('--num_workers', type=int, default=0)

    # Label filtering
    parser.add_argument('--include_labels', type=int, nargs='+', default=None)
    parser.add_argument('--exclude_labels', type=int, nargs='+', default=None)
    parser.add_argument('--cross_subject', type=bool, default=None)

    # USBA-specific args
    USBAConfig.add_argparse_args(parser)

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    setup_seed(args.seed)

    if args.no_subjects:
        args.use_subjects = False

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_config = DATASET_CONFIGS[args.dataset]
    task_type = dataset_config['task_type']
    n_channels = dataset_config['n_channels']
    patch_size = dataset_config['patch_size']

    # Resolve pretrained weights
    if args.pretrained_weights is None:
        base_dir = os.path.dirname(__file__)
        luna_size = getattr(args, 'luna_size', 'base')
        weights_map = {
            'codebrain': os.path.join(base_dir, 'CodeBrain/Checkpoints/CodeBrain.pth'),
            'cbramod': os.path.join(base_dir, 'Cbramod_pretrained_weights.pth'),
            'luna': os.path.join(base_dir, f'BioFoundation/checkpoints/LUNA/LUNA_{luna_size}.safetensors'),
            'femba': None,
        }
        weights_path = weights_map.get(args.model)
    else:
        weights_path = args.pretrained_weights
        if not os.path.isabs(weights_path):
            weights_path = os.path.join(os.path.dirname(__file__), weights_path)

    # ── Load data ──────────────────────────────────────────────────────
    data_loaders, num_classes, seq_len, num_subjects = load_data_with_subjects(
        args, dataset_config
    )

    # ── Create backbone ────────────────────────────────────────────────
    backbone, backbone_out_dim, token_dim = create_backbone(
        model_type=args.model,
        n_channels=n_channels,
        seq_len=seq_len,
        patch_size=patch_size,
        n_layer=args.n_layer,
        codebook_size_t=args.codebook_size_t,
        codebook_size_f=args.codebook_size_f,
        n_layer_cbramod=args.n_layer_cbramod,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        use_layer_adapters=False,  # USBA handles its own injection
        dropout=args.dropout,
        pretrained_weights_path=weights_path,
        device=str(device),
    )

    # ── Build USBA config ──────────────────────────────────────────────
    usba_config = USBAConfig.from_args(args)
    usba_config.enabled = True  # force enable since this is the USBA training script
    usba_config.task_type = task_type
    usba_config.num_classes = num_classes

    # Print cc-inv status
    if usba_config.enable_cc_inv and args.use_subjects:
        print(f"[USBA] Class-conditional invariance: ENABLED (lambda={usba_config.lambda_cc_inv})")
    else:
        print(f"[USBA] Class-conditional invariance: DISABLED")
    print(f"[USBA] Budget regularization: {'ENABLED' if usba_config.enable_budget_reg else 'DISABLED'}")

    # ── Create model ───────────────────────────────────────────────────
    model = USBAInjector.inject(
        backbone=backbone,
        config=usba_config,
        token_dim=token_dim,
        num_classes=num_classes,
        n_channels=n_channels,
        seq_len=seq_len,
    ).to(device)

    # ── Loss ───────────────────────────────────────────────────────────
    criterion = USBALoss(
        beta=usba_config.beta,
        per_layer_beta=usba_config.per_layer_beta,
        lambda_cc_inv=usba_config.lambda_cc_inv,
        eta_budget=usba_config.eta_budget,
        enable_cc_inv=usba_config.enable_cc_inv and args.use_subjects,
        enable_budget_reg=usba_config.enable_budget_reg,
        task_type=task_type,
    )

    # ── Optimizer ──────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── WandB ──────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb_project and WANDB_AVAILABLE:
        run_name = args.wandb_run_name or (
            f"usba_{args.dataset}_{args.model}_d{usba_config.latent_dim}"
            f"_b{usba_config.beta}_s{args.seed}"
        )
        config_dict = {**vars(args), **usba_config.to_dict()}
        total_params = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in trainable_params)
        config_dict.update({
            'total_params': total_params,
            'trainable_params': trainable_count,
            'trainable_ratio': trainable_count / total_params * 100,
            'num_classes': num_classes,
            'num_subjects': num_subjects,
        })
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            group=args.wandb_group,
            config=config_dict,
        )

    # ── Save dir ───────────────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f'best_{args.dataset}_{args.model}_usba_d{usba_config.latent_dim}.pth'

    # ── Training loop ──────────────────────────────────────────────────
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"USBA Training: {args.epochs} epochs")
    print(f"  beta={usba_config.beta}, lambda_cc={usba_config.lambda_cc_inv}, "
          f"eta_budget={usba_config.eta_budget}")
    print(f"  gate_type={usba_config.gate_type}, factorized={usba_config.factorized}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, data_loaders['train'], criterion, optimizer, device,
            args.clip_value, args.use_subjects, epoch, usba_config.beta_warmup_epochs,
        )
        train_f1 = f1_score(train_metrics['labels'], train_metrics['preds'],
                            average='macro', zero_division=0)

        # Validate
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, data_loaders['val'], device
        )
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

        # Test
        test_loss, test_acc, test_preds, test_labels = evaluate(
            model, data_loaders['test'], device
        )
        test_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)

        scheduler.step()
        elapsed = time.time() - t0

        # Print
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train: loss={train_metrics.get('total', 0):.4f} acc={train_metrics['bal_acc']:.4f} | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"Test: acc={test_acc:.4f} f1={test_f1:.4f} | "
            f"kl={train_metrics.get('kl_total', 0):.4f} "
            f"gate={train_metrics.get('gate_mean', 0):.3f} "
            f"cc={train_metrics.get('cc_inv', 0):.4f} | "
            f"{elapsed:.1f}s"
        )

        # WandB
        if wandb_run:
            log_dict = {
                'epoch': epoch,
                'train/loss': train_metrics.get('total', 0),
                'train/task': train_metrics.get('task', 0),
                'train/kl': train_metrics.get('kl_total', 0),
                'train/cc_inv': train_metrics.get('cc_inv', 0),
                'train/budget': train_metrics.get('budget', 0),
                'train/gate_mean': train_metrics.get('gate_mean', 0),
                'train/bal_acc': train_metrics['bal_acc'],
                'train/f1_macro': train_f1,
                'val/loss': val_loss,
                'val/bal_acc': val_acc,
                'val/f1_macro': val_f1,
                'test/loss': test_loss,
                'test/bal_acc': test_acc,
                'test/f1_macro': test_f1,
                'lr': optimizer.param_groups[0]['lr'],
            }
            wandb_run.log(log_dict)

        # Best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'usba_config': usba_config.to_dict(),
                'args': vars(args),
            }, save_dir / ckpt_name)
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(best: epoch {best_epoch}, val_acc={best_val_acc:.4f})")
            break

    # ── Final evaluation ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Loading best model from epoch {best_epoch}")
    ckpt = torch.load(save_dir / ckpt_name, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, data_loaders['test'], device
    )
    test_f1_macro = f1_score(test_labels, test_preds, average='macro', zero_division=0)
    test_f1_weighted = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
    test_cm = confusion_matrix(test_labels, test_preds)

    label_names = dataset_config.get('label_names', {})
    target_names = [label_names.get(i, str(i)) for i in range(num_classes)]

    print(f"\nFinal Test Results:")
    print(f"  Best epoch:          {best_epoch}")
    print(f"  Best val acc:        {best_val_acc:.4f}")
    print(f"  Test balanced acc:   {test_acc:.4f}")
    print(f"  Test F1 (macro):     {test_f1_macro:.4f}")
    print(f"  Test F1 (weighted):  {test_f1_weighted:.4f}")
    print(f"\n  Per-class F1:")
    per_class_f1 = f1_score(test_labels, test_preds, average=None, zero_division=0)
    for i, name in enumerate(target_names):
        if i < len(per_class_f1):
            print(f"    {name:20s}: {per_class_f1[i]:.4f}")
    print(f"\n  Confusion Matrix:\n  {test_cm}")

    # USBA parameter stats
    param_info = USBAInjector.get_trainable_params(model)
    print(f"\n  USBA parameter budget:")
    print(f"    Total:     {param_info['total']:,}")
    print(f"    Trainable: {param_info['trainable']:,}")
    print(f"    Adapter:   {param_info['adapter']:,}")
    print(f"    Head:      {param_info['head']:,}")
    print(f"{'='*60}")

    if wandb_run:
        wandb_run.summary['test_bal_acc'] = test_acc
        wandb_run.summary['test_f1_macro'] = test_f1_macro
        wandb_run.summary['best_epoch'] = best_epoch
        wandb_run.finish()

    return test_acc


if __name__ == '__main__':
    main()
