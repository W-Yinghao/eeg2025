#!/usr/bin/env python3
"""
Training Script for Hybrid Hyperbolic Fine-Tuning (CodeBrain + HEEGNet-style head)

Usage:
    # TUEV dataset (6-class, 16ch, 5s)
    python train_hyperbolic_finetuning.py --dataset TUEV --cuda 0

    # TUAB dataset (binary, 16ch, 10s)
    python train_hyperbolic_finetuning.py --dataset TUAB --cuda 0

    # DIAGNOSIS dataset (4-class, 58ch, 1s)
    python train_hyperbolic_finetuning.py --dataset DIAGNOSIS --cuda 0

    # Custom hyperbolic dimension and HHSW weight
    python train_hyperbolic_finetuning.py --dataset TUEV --hyp_dim 64 --lambda_hhsw 0.1

    # Wandb logging
    python train_hyperbolic_finetuning.py --dataset TUEV --wandb_project eeg_hyperbolic
"""

import argparse
import os
import sys
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
try:
    from sklearn.metrics import (
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        classification_report,
    )
except (ImportError, AttributeError):
    # Fallback when sklearn unavailable (numpy version issue on login node)
    def balanced_accuracy_score(y_true, y_pred):
        classes = sorted(set(y_true))
        recalls = []
        for c in classes:
            mask = [yt == c for yt in y_true]
            if sum(mask) > 0:
                correct = sum(1 for m, yp in zip(mask, y_pred) if m and yp == c)
                recalls.append(correct / sum(mask))
        return sum(recalls) / max(len(recalls), 1)

    def f1_score(y_true, y_pred, average='macro', zero_division=0):
        classes = sorted(set(y_true) | set(y_pred))
        f1s, weights = [], []
        for c in classes:
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
            weights.append(sum(1 for yt in y_true if yt == c))
        if average == 'macro':
            return sum(f1s) / max(len(f1s), 1)
        total = sum(weights)
        return sum(f * w for f, w in zip(f1s, weights)) / total if total > 0 else 0.0

    def confusion_matrix(y_true, y_pred):
        classes = sorted(set(y_true) | set(y_pred))
        n = len(classes)
        cm = [[0] * n for _ in range(n)]
        c2i = {c: i for i, c in enumerate(classes)}
        for yt, yp in zip(y_true, y_pred):
            cm[c2i[yt]][c2i[yp]] += 1
        return cm

    def classification_report(y_true, y_pred, **kwargs):
        return "classification_report unavailable (sklearn not installed)"

# Import from the existing finetuning framework for data loading
sys.path.insert(0, os.path.dirname(__file__))
from finetune_tuev_lmdb import DATASET_CONFIGS, load_data

# Import our hybrid model
from hyperbolic_finetuning import (
    HybridCodeBrainFineTuner,
    LorentzManifold,
    compute_hybrid_loss,
    configure_optimizer,
    DSMDBNMomentumScheduler,
)
from backbone_factory import create_backbone


def setup_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='Hybrid Hyperbolic EEG Fine-Tuning')

    # Dataset
    parser.add_argument('--dataset', type=str, default='TUEV',
                        choices=list(DATASET_CONFIGS.keys()),
                        help='dataset to use')

    # Basic
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device')

    # Training
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
    parser.add_argument('--clip_value', type=float, default=1.0, help='gradient clipping (max norm)')
    parser.add_argument('--backbone_warmup_epochs', type=int, default=5,
                        help='freeze backbone for first N epochs during full_finetune (default: 5)')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='validation ratio')
    parser.add_argument('--patience', type=int, default=15, help='early stopping patience')

    # Hyperbolic architecture
    parser.add_argument('--hyp_dim', type=int, default=128,
                        help='intrinsic hyperbolic dimension (spatial components)')
    parser.add_argument('--curvature', type=float, default=-1.0,
                        help='negative curvature K of the Lorentz manifold')
    parser.add_argument('--lambda_hhsw', type=float, default=None,
                        help='HHSW loss weight (default: auto based on dataset type)')
    parser.add_argument('--num_projections', type=int, default=1000,
                        help='number of HHSW random slices')
    parser.add_argument('--eta_train_init', type=float, default=1.0,
                        help='initial DSMDBN training momentum')
    parser.add_argument('--eta_test', type=float, default=0.1,
                        help='DSMDBN test momentum')
    parser.add_argument('--eta_decay', type=float, default=0.99,
                        help='DSMDBN momentum decay rate')

    # Backbone model
    parser.add_argument('--model', type=str, default='codebrain',
                        choices=['codebrain', 'cbramod', 'femba', 'luna'],
                        help='backbone model type')
    # CodeBrain-specific
    parser.add_argument('--n_layer', type=int, default=8, help='SSSM layers (CodeBrain)')
    parser.add_argument('--codebook_size_t', type=int, default=4096, help='temporal codebook (CodeBrain)')
    parser.add_argument('--codebook_size_f', type=int, default=4096, help='frequency codebook (CodeBrain)')
    # CBraMod-specific
    parser.add_argument('--n_layer_cbramod', type=int, default=12, help='transformer layers (CBraMod)')
    parser.add_argument('--nhead', type=int, default=8, help='attention heads (CBraMod)')
    parser.add_argument('--dim_feedforward', type=int, default=800, help='FFN dim (CBraMod)')
    # Inter-layer adapters (CBraMod only)
    parser.add_argument('--use_layer_adapters', action='store_true',
                        help='insert trainable bottleneck adapters between CBraMod layers')
    parser.add_argument('--adapter_reduction', type=int, default=4,
                        help='adapter bottleneck reduction factor')
    # Pretrained weights
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help='path to pretrained weights (auto-resolved if None)')
    # Full fine-tuning
    parser.add_argument('--full_finetune', action='store_true',
                        help='unfreeze backbone for full fine-tuning (default: head-only)')
    parser.add_argument('--backbone_lr_ratio', type=float, default=0.01,
                        help='backbone lr = lr * ratio when full_finetune (default: 0.01)')

    # Logging
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='WandB project name (None = no logging)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='WandB run name')
    parser.add_argument('--save_dir', type=str, default='checkpoints_hyperbolic',
                        help='directory for saving checkpoints')
    parser.add_argument('--num_workers', type=int, default=4, help='dataloader workers')

    # Label filtering (for DIAGNOSIS dataset)
    parser.add_argument('--include_labels', type=int, nargs='+', default=None)
    parser.add_argument('--exclude_labels', type=int, nargs='+', default=None)
    parser.add_argument('--cross_subject', type=bool, default=None)

    return parser.parse_args()


def train_one_epoch(model, dataloader, optimizer, manifold, scheduler,
                    device, lambda_hhsw, num_projections, task_type, clip_value):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_hhsw = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    for batch in dataloader:
        eeg_data = batch[0].to(device)
        labels = batch[1].to(device)

        # Use subject IDs as domains if available, otherwise use batch index
        if len(batch) > 2 and batch[2] is not None:
            domains = batch[2].to(device)
        else:
            domains = torch.zeros(eeg_data.shape[0], dtype=torch.long, device=device)

        # Forward pass
        logits, features = model(eeg_data, domains)

        # Compute loss: L_CE + lambda * L_HHSW
        loss, loss_dict = compute_hybrid_loss(
            logits, labels, features, domains, manifold,
            lambda_hhsw=lambda_hhsw,
            num_projections=num_projections,
            task_type=task_type,
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Check for NaN/Inf in loss — skip this batch if so
        if not torch.isfinite(loss):
            optimizer.zero_grad()
            num_batches += 1
            continue

        # Separate gradient clipping for backbone vs head
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad and p.grad is not None]
        head_params = [p for n, p in model.named_parameters()
                       if not n.startswith('backbone.') and p.requires_grad and p.grad is not None]

        if backbone_params:
            # Tighter clipping for backbone to prevent gradient explosion
            bb_norm = torch.nn.utils.clip_grad_norm_(backbone_params, clip_value * 0.2)
            # Skip step if backbone gradients are still NaN after clipping
            if not torch.isfinite(bb_norm):
                optimizer.zero_grad()
                num_batches += 1
                continue
        if head_params:
            torch.nn.utils.clip_grad_norm_(head_params, clip_value)

        optimizer.step()

        # Update momentum scheduler
        if scheduler is not None:
            scheduler.step()

        # Track metrics
        total_loss += loss_dict['total']
        total_ce += loss_dict['ce']
        total_hhsw += loss_dict.get('hhsw', 0.0)
        num_batches += 1

        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(num_batches, 1)
    avg_ce = total_ce / max(num_batches, 1)
    avg_hhsw = total_hhsw / max(num_batches, 1)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    # Compute gradient norms for monitoring
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            module_name = name.split('.')[0]
            if module_name not in grad_norms:
                grad_norms[module_name] = 0.0
            grad_norms[module_name] += param.grad.norm().item() ** 2
    grad_norms = {k: v ** 0.5 for k, v in grad_norms.items()}

    return avg_loss, avg_ce, avg_hhsw, bal_acc, grad_norms, all_preds, all_labels


@torch.no_grad()
def evaluate(model, dataloader, manifold, device, lambda_hhsw, num_projections, task_type):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    for batch in dataloader:
        eeg_data = batch[0].to(device)
        labels = batch[1].to(device)

        if len(batch) > 2 and batch[2] is not None:
            domains = batch[2].to(device)
        else:
            domains = torch.zeros(eeg_data.shape[0], dtype=torch.long, device=device)

        logits, features = model(eeg_data, domains)

        # At eval time, only CE loss matters (no HHSW)
        loss_ce = F.cross_entropy(logits, labels)

        total_loss += loss_ce.item()
        total_ce += loss_ce.item()
        num_batches += 1

        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(num_batches, 1)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    return avg_loss, bal_acc, all_preds, all_labels


def main():
    args = parse_args()
    setup_seed(args.seed)

    # Device
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset config
    dataset_config = DATASET_CONFIGS[args.dataset]
    task_type = dataset_config['task_type']
    n_channels = dataset_config['n_channels']
    patch_size = dataset_config['patch_size']

    # Auto-set HHSW weight based on dataset type (following HEEGNet paper)
    if args.lambda_hhsw is None:
        if 'emotion' in args.dataset.lower() or args.dataset in ['SEED', 'FACED']:
            args.lambda_hhsw = 0.01
        else:
            args.lambda_hhsw = 0.5
    print(f"HHSW loss weight (lambda): {args.lambda_hhsw}")

    # Resolve pretrained weights path
    weights_path = args.pretrained_weights
    if weights_path is None:
        # Auto-resolve based on model type
        if args.model == 'codebrain':
            weights_path = os.path.join(os.path.dirname(__file__), 'CodeBrain/Checkpoints/CodeBrain.pth')
        else:
            weights_path = os.path.join(os.path.dirname(__file__), 'Cbramod_pretrained_weights.pth')
    elif not os.path.isabs(weights_path):
        weights_path = os.path.join(os.path.dirname(__file__), weights_path)

    # Load data
    data_loader, num_classes, seq_len = load_data(args, dataset_config)
    print(f"\nDataset: {args.dataset}")
    print(f"Classes: {num_classes}, Task: {task_type}")
    print(f"Channels: {n_channels}, Seq_len: {seq_len}, Patch_size: {patch_size}")
    print(f"Backbone: {args.model}")

    # Create backbone
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
        use_layer_adapters=args.use_layer_adapters,
        adapter_reduction=args.adapter_reduction,
        dropout=args.dropout,
        pretrained_weights_path=weights_path,
        device=str(device),
    )

    # Full fine-tuning: unfreeze backbone (with optional warmup)
    if args.full_finetune:
        if args.backbone_warmup_epochs > 0:
            # Keep backbone frozen during warmup; it will be unfrozen after warmup_epochs
            print(f"Full fine-tuning: backbone stays frozen for {args.backbone_warmup_epochs} warmup epochs")
            print(f"  Backbone lr = lr * {args.backbone_lr_ratio} after warmup")
        else:
            for param in backbone.parameters():
                param.requires_grad = True
            print(f"Full fine-tuning: backbone unfrozen (backbone_lr = lr * {args.backbone_lr_ratio})")

    # Create hybrid model
    model = HybridCodeBrainFineTuner(
        backbone=backbone,
        backbone_out_dim=backbone_out_dim,
        num_classes=num_classes,
        hyp_dim=args.hyp_dim,
        K=args.curvature,
        dropout=args.dropout,
        eta_train=args.eta_train_init,
        eta_test=args.eta_test,
    ).to(device)

    # Manifold for loss computation
    manifold = LorentzManifold(K=args.curvature)

    # Optimizer
    optimizer = configure_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay, use_riemannian=True,
        backbone_lr_ratio=args.backbone_lr_ratio if args.full_finetune else 0.1,
    )

    # Momentum scheduler for DSMDBN
    momentum_scheduler = DSMDBNMomentumScheduler(
        model, eta_init=args.eta_train_init,
        eta_min=0.1, decay=args.eta_decay
    )

    # WandB logging
    wandb_run = None
    wandb = None
    if args.wandb_project:
        try:
            import wandb as _wandb
            wandb = _wandb
            run_name = args.wandb_run_name or f"hyp_{args.dataset}_d{args.hyp_dim}_K{abs(args.curvature)}_s{args.seed}"

            # Comprehensive config
            config = vars(args).copy()
            frozen_params = sum(p.numel() for p in model.backbone.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            config.update({
                'model_type': 'HybridCodeBrainFineTuner',
                'frozen_params': frozen_params,
                'trainable_params': trainable_params,
                'trainable_ratio': trainable_params / (frozen_params + trainable_params) * 100,
                'num_classes': num_classes,
                'n_channels': n_channels,
                'seq_len': seq_len,
                'patch_size': patch_size,
                'backbone_out_dim': backbone_out_dim,
            })

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=config,
            )

            # Watch model for gradient/parameter histograms
            wandb.watch(model, log='gradients', log_freq=50)

            # Log model summary table
            summary_table = wandb.Table(
                columns=["Component", "Parameters", "Trainable"],
                data=[
                    ["Backbone (SSSM)", frozen_params, False],
                    ["Trainable Head", trainable_params, True],
                    ["Total", frozen_params + trainable_params, "-"],
                ]
            )
            wandb_run.log({"model/summary": summary_table})

        except ImportError:
            print("wandb not available, skipping logging")

    # Save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Starting training: {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()

        # Backbone warmup: freeze backbone for first N epochs during full fine-tuning
        if args.full_finetune and epoch == args.backbone_warmup_epochs + 1:
            for param in model.backbone.parameters():
                param.requires_grad = True
            # Rebuild optimizer to include backbone params
            optimizer = configure_optimizer(
                model, lr=args.lr, weight_decay=args.weight_decay, use_riemannian=True,
                backbone_lr_ratio=args.backbone_lr_ratio,
            )
            print(f"  >> Epoch {epoch}: backbone unfrozen (warmup complete)")

        # Train
        train_loss, train_ce, train_hhsw, train_acc, grad_norms, train_preds, train_labels = train_one_epoch(
            model, data_loader['train'], optimizer, manifold, momentum_scheduler,
            device, args.lambda_hhsw, args.num_projections, task_type, args.clip_value
        )
        train_f1_macro = f1_score(train_labels, train_preds, average='macro', zero_division=0)
        train_f1_weighted = f1_score(train_labels, train_preds, average='weighted', zero_division=0)

        # Validate
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, data_loader['val'], manifold, device,
            args.lambda_hhsw, args.num_projections, task_type
        )
        val_f1_macro = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        val_f1_weighted = f1_score(val_labels, val_preds, average='weighted', zero_division=0)

        # Test after every epoch
        test_loss_epoch, test_acc_epoch, test_preds_epoch, test_labels_epoch = evaluate(
            model, data_loader['test'], manifold, device,
            args.lambda_hhsw, args.num_projections, task_type
        )
        test_f1_macro_epoch = f1_score(test_labels_epoch, test_preds_epoch, average='macro', zero_division=0)
        test_f1_weighted_epoch = f1_score(test_labels_epoch, test_preds_epoch, average='weighted', zero_division=0)

        t_elapsed = time.time() - t_start

        # Logging
        eta_current = momentum_scheduler.get_eta()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train: loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1_macro:.4f} | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1_macro:.4f} | "
              f"Test: loss={test_loss_epoch:.4f} acc={test_acc_epoch:.4f} f1={test_f1_macro_epoch:.4f} | "
              f"eta={eta_current:.4f} | {t_elapsed:.1f}s")

        if wandb_run:
            log_dict = {
                'epoch': epoch,
                'train/loss': train_loss,
                'train/ce': train_ce,
                'train/hhsw': train_hhsw,
                'train/weighted_hhsw': args.lambda_hhsw * train_hhsw,
                'train/bal_acc': train_acc,
                'train/f1_macro': train_f1_macro,
                'train/f1_weighted': train_f1_weighted,
                'val/loss': val_loss,
                'val/bal_acc': val_acc,
                'val/f1_macro': val_f1_macro,
                'val/f1_weighted': val_f1_weighted,
                'test/loss': test_loss_epoch,
                'test/bal_acc': test_acc_epoch,
                'test/f1_macro': test_f1_macro_epoch,
                'test/f1_weighted': test_f1_weighted_epoch,
                'dsmdbn/eta': eta_current,
                'lr': current_lr,
                'time/epoch_seconds': t_elapsed,
            }
            # Gradient norms per module
            for module_name, gnorm in grad_norms.items():
                log_dict[f'grad_norm/{module_name}'] = gnorm
            wandb_run.log(log_dict)

        # Best model tracking
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            # Save checkpoint
            ckpt_path = save_dir / f'best_{args.dataset}_hyp.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': vars(args),
            }, ckpt_path)
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (best epoch: {best_epoch}, best val acc: {best_val_acc:.4f})")
            break

    # Final test evaluation
    print(f"\n{'='*60}")
    print(f"Loading best model from epoch {best_epoch}")
    ckpt = torch.load(save_dir / f'best_{args.dataset}_hyp.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, data_loader['test'], manifold, device,
        args.lambda_hhsw, args.num_projections, task_type
    )

    # Per-class metrics
    label_names = dataset_config.get('label_names', {})
    target_names = [label_names.get(i, str(i)) for i in range(num_classes)]
    test_f1_macro = f1_score(test_labels, test_preds, average='macro', zero_division=0)
    test_f1_weighted = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
    test_f1_per_class = f1_score(test_labels, test_preds, average=None, zero_division=0)
    test_cm = confusion_matrix(test_labels, test_preds)

    print(f"\nFinal Test Results:")
    print(f"  Best epoch:          {best_epoch}")
    print(f"  Best val acc:        {best_val_acc:.4f}")
    print(f"  Test loss:           {test_loss:.4f}")
    print(f"  Test balanced acc:   {test_acc:.4f}")
    print(f"  Test F1 (macro):     {test_f1_macro:.4f}")
    print(f"  Test F1 (weighted):  {test_f1_weighted:.4f}")
    print(f"\n  Per-class F1:")
    for i, name in enumerate(target_names):
        f1_val = test_f1_per_class[i] if i < len(test_f1_per_class) else 0.0
        print(f"    {name:20s}: {f1_val:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {test_cm}")
    print(f"{'='*60}")

    if wandb_run:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Basic test metrics
        wandb_run.log({
            'test/bal_acc': test_acc,
            'test/loss': test_loss,
            'test/f1_macro': test_f1_macro,
            'test/f1_weighted': test_f1_weighted,
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc,
        })

        # Per-class F1
        for i, name in enumerate(target_names):
            if i < len(test_f1_per_class):
                wandb_run.log({f'test/f1_{name}': test_f1_per_class[i]})

        # Confusion matrix as heatmap image
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.imshow(test_cm, interpolation='nearest', cmap='Blues')
        ax.set_title(f'Confusion Matrix - {args.dataset} (acc={test_acc:.4f})')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(target_names, rotation=45, ha='right')
        ax.set_yticklabels(target_names)
        plt.colorbar(im, ax=ax)
        # Annotate cells with counts
        for i in range(test_cm.shape[0]):
            for j in range(test_cm.shape[1]):
                ax.text(j, i, str(test_cm[i, j]),
                        ha='center', va='center',
                        color='white' if test_cm[i, j] > test_cm.max() / 2 else 'black')
        plt.tight_layout()
        wandb_run.log({'test/confusion_matrix': wandb.Image(fig)})
        plt.close(fig)

        # Confusion matrix as wandb table
        cm_table = wandb.Table(
            columns=['True\\Pred'] + target_names,
            data=[
                [target_names[i]] + [int(test_cm[i, j]) for j in range(num_classes)]
                for i in range(num_classes)
            ]
        )
        wandb_run.log({'test/confusion_matrix_table': cm_table})

        # Summary metrics for wandb run summary
        wandb_run.summary['test_bal_acc'] = test_acc
        wandb_run.summary['test_f1_macro'] = test_f1_macro
        wandb_run.summary['test_f1_weighted'] = test_f1_weighted
        wandb_run.summary['test_loss'] = test_loss
        wandb_run.summary['best_epoch'] = best_epoch
        wandb_run.summary['best_val_acc'] = best_val_acc

        wandb_run.finish()

    return test_acc


if __name__ == '__main__':
    main()
