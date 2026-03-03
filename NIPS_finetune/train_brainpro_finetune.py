#!/usr/bin/env python3
"""
BrainPro Downstream Fine-Tuning Script

Fine-tunes BrainPro on downstream EEG classification tasks using the existing
LMDB data infrastructure. Supports:
    - Flexible encoder selection (shared + state-specific)
    - Multiple token merge modes (mean, aggr, all)
    - Position embedding reset
    - Label smoothing
    - CosineAnnealingLR scheduler
    - Comprehensive ablation configurations

Ablation studies from the paper (Table 3):
    1. w/o masking        → pre-training ablation (different pretrained weights)
    2. w/o reconstruction → pre-training ablation
    3. w/o decoupling     → pre-training ablation
    4. w random retrieval → --random_retrieval flag
    5. w/o pre-training   → omit --pretrained_weights
    6. Full BrainPro      → default with pretrained weights

Additional ablations:
    - Token merge mode: --token_merge {mean, aggr, all} (Table 12)
    - Encoder config: --active_states {affect, motor, others, all} (Table 13)
    - Position embedding reset: --no_reset_pos_emb (Table 11)
    - Learning rate: --lr (Figure 6)

Usage:
    # Default: shared + affect encoder, mean pooling
    python train_brainpro_finetune.py --dataset TUEV --cuda 0

    # With pretrained weights
    python train_brainpro_finetune.py --dataset TUEV --pretrained_weights brainpro_epoch10.pth

    # Different encoder config
    python train_brainpro_finetune.py --dataset TUEV --active_states affect motor

    # Token merge ablation
    python train_brainpro_finetune.py --dataset TUEV --token_merge all --hidden_factor 8

    # Random retrieval ablation
    python train_brainpro_finetune.py --dataset TUEV --random_retrieval

    # From scratch (no pre-training)
    python train_brainpro_finetune.py --dataset TUEV
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

# Project imports
sys.path.insert(0, os.path.dirname(__file__))
from finetune_tuev_lmdb import DATASET_CONFIGS, load_data
from brainpro_model import (
    create_brainpro_finetune,
    DATASET_CHANNEL_NAMES,
)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description='BrainPro Fine-Tuning')

    # Dataset
    parser.add_argument('--dataset', type=str, default='TUEV',
                        choices=list(DATASET_CONFIGS.keys()),
                        help='dataset name')
    parser.add_argument('--datasets_dir', type=str, default=None,
                        help='override data directory')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='label smoothing for multi-class CE (0 = none)')
    parser.add_argument('--patience', type=int, default=15,
                        help='early stopping patience')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)

    # BrainPro Architecture
    parser.add_argument('--K_T', type=int, default=32, help='temporal feature dim')
    parser.add_argument('--K_C', type=int, default=32, help='channel-wise spatial filters')
    parser.add_argument('--K_R', type=int, default=32, help='region-wise spatial filters')
    parser.add_argument('--d_model', type=int, default=32, help='transformer hidden dim')
    parser.add_argument('--nhead', type=int, default=32, help='attention heads')
    parser.add_argument('--d_ff', type=int, default=64, help='transformer MLP dim')
    parser.add_argument('--n_layers', type=int, default=4, help='transformer layers')
    parser.add_argument('--patch_len', type=int, default=20, help='patch length')
    parser.add_argument('--patch_stride', type=int, default=20, help='patch stride')
    parser.add_argument('--dropout', type=float, default=0.1)

    # BrainPro Encoder Selection
    parser.add_argument('--active_states', nargs='+', default=['affect'],
                        choices=['affect', 'motor', 'others'],
                        help='brain states to activate (default: affect)')
    parser.add_argument('--token_merge', type=str, default='mean',
                        choices=['mean', 'aggr', 'all'],
                        help='token merge mode')
    parser.add_argument('--hidden_factor', type=int, default=1,
                        help='MLP hidden dimension multiplier')

    # Ablation flags
    parser.add_argument('--random_retrieval', action='store_true',
                        help='use random (fixed) spatial filters instead of learned')
    parser.add_argument('--no_reset_pos_emb', action='store_true',
                        help='do NOT reset positional embeddings after loading')
    parser.add_argument('--channel_names', nargs='+', default=None,
                        help='explicit channel names for spatial retrieval')

    # Pretrained weights
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help='path to pretrained BrainPro checkpoint')

    # Logging
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='WandB project name (None = no logging)')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='checkpoints_brainpro')

    return parser.parse_args()


def train_one_epoch(model, dataloader, optimizer, scheduler, device, task_type,
                    label_smoothing, clip_grad_norm):
    model.train()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        eeg_data = batch[0].to(device)
        labels = batch[1].to(device)

        optimizer.zero_grad()
        logits = model(eeg_data)

        # Compute loss
        if task_type == 'binary':
            logits_flat = logits.squeeze(-1) if logits.dim() > 1 and logits.shape[-1] == 1 else logits
            loss = F.binary_cross_entropy_with_logits(logits_flat, labels.float())
            preds = (logits_flat > 0).long()
        else:
            loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
            preds = logits.argmax(dim=-1)

        loss.backward()

        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / max(num_batches, 1)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    return avg_loss, bal_acc, all_preds, all_labels


@torch.no_grad()
def evaluate(model, dataloader, device, task_type):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []
    all_logits = []

    for batch in dataloader:
        eeg_data = batch[0].to(device)
        labels = batch[1].to(device)

        logits = model(eeg_data)

        if task_type == 'binary':
            logits_flat = logits.squeeze(-1) if logits.dim() > 1 and logits.shape[-1] == 1 else logits
            loss = F.binary_cross_entropy_with_logits(logits_flat, labels.float())
            preds = (logits_flat > 0).long()
            all_logits.extend(torch.sigmoid(logits_flat).cpu().numpy())
        else:
            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=-1)
            all_logits.extend(F.softmax(logits, dim=-1).cpu().numpy())

        total_loss += loss.item()
        num_batches += 1
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(num_batches, 1)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    metrics = {'loss': avg_loss, 'bal_acc': bal_acc}

    # Task-specific metrics
    if task_type == 'binary':
        try:
            metrics['auc_pr'] = average_precision_score(all_labels, all_logits)
            metrics['auroc'] = roc_auc_score(all_labels, all_logits)
        except ValueError:
            metrics['auc_pr'] = 0.0
            metrics['auroc'] = 0.0
    else:
        metrics['kappa'] = cohen_kappa_score(all_labels, all_preds)
        metrics['f1_weighted'] = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    metrics['f1_macro'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return metrics, all_preds, all_labels


def main():
    args = parse_args()
    setup_seed(args.seed)

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Dataset config
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    dataset_config = DATASET_CONFIGS[args.dataset].copy()
    if args.datasets_dir:
        dataset_config['data_dir'] = args.datasets_dir

    n_channels = dataset_config['n_channels']
    num_classes = dataset_config['num_classes']
    task_type = dataset_config['task_type']
    patch_size = dataset_config.get('patch_size', 200)
    seg_duration = dataset_config.get('segment_duration', 5)
    sampling_rate = 200  # All data resampled to 200Hz

    # Label smoothing only for multiclass
    label_smoothing = args.label_smoothing if task_type == 'multiclass' else 0.0

    # Load data
    data_loader, num_classes_loaded, seq_len = load_data(args, dataset_config)
    if num_classes_loaded != num_classes:
        num_classes = num_classes_loaded

    T_total = seq_len * patch_size  # Total time samples
    N_p = (T_total - args.patch_len) // args.patch_stride + 1

    print(f"\n{'='*60}")
    print(f"BrainPro Fine-Tuning - {args.dataset}")
    print(f"{'='*60}")
    print(f"Channels: {n_channels}, T={T_total}, N_patches={N_p}")
    print(f"Classes: {num_classes}, Task: {task_type}")
    print(f"Active encoders: shared + {args.active_states}")
    print(f"Token merge: {args.token_merge}, Hidden factor: {args.hidden_factor}")
    print(f"Label smoothing: {label_smoothing}")

    # Resolve channel names
    channel_names = args.channel_names
    if channel_names is None:
        channel_names = DATASET_CHANNEL_NAMES.get(args.dataset.upper())

    # Resolve pretrained weights
    weights_path = args.pretrained_weights
    if weights_path is not None and not os.path.isabs(weights_path):
        weights_path = os.path.join(os.path.dirname(__file__), weights_path)

    # Create model
    model = create_brainpro_finetune(
        n_channels=n_channels,
        num_classes=num_classes,
        task_type=task_type,
        channel_names=channel_names,
        dataset_name=args.dataset,
        active_states=args.active_states,
        token_merge=args.token_merge,
        hidden_factor=args.hidden_factor,
        random_retrieval=args.random_retrieval,
        K_T=args.K_T, K_C=args.K_C, K_R=args.K_R,
        d_model=args.d_model, nhead=args.nhead, d_ff=args.d_ff,
        n_layers=args.n_layers,
        patch_len=args.patch_len, patch_stride=args.patch_stride,
        max_patches=max(N_p * 3, 200),
        dropout=args.dropout,
        label_smoothing=label_smoothing,
        pretrained_path=weights_path,
        reset_pos_emb=not args.no_reset_pos_emb,
    ).to(device)

    # Optimizer: AdamW (Table 7)
    param_groups = model.get_param_groups(lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    # Scheduler: CosineAnnealingLR (Table 7)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )

    # WandB logging
    wandb_run = None
    wandb = None
    if args.wandb_project:
        try:
            import wandb as _wandb
            wandb = _wandb
            run_name = args.wandb_run_name or (
                f"brainpro_{args.dataset}_"
                f"{'_'.join(args.active_states)}_"
                f"{args.token_merge}_"
                f"hf{args.hidden_factor}_"
                f"s{args.seed}"
            )

            config = vars(args).copy()
            config.update({
                'model_type': 'BrainPro',
                'n_channels': n_channels,
                'num_classes': num_classes,
                'task_type': task_type,
                'T_total': T_total,
                'N_patches': N_p,
                'total_params': sum(p.numel() for p in model.parameters()),
                'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            })

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                group=args.wandb_group,
                config=config,
                tags=[args.dataset.lower(), 'brainpro', args.token_merge],
            )
            wandb.watch(model, log='gradients', log_freq=50)
        except ImportError:
            print("wandb not available, skipping logging")

    # Save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    print(f"\nStarting training: {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()

        # Train
        train_loss, train_acc, _, train_labels = train_one_epoch(
            model, data_loader['train'], optimizer, scheduler,
            device, task_type, label_smoothing, args.clip_grad_norm
        )

        # Validate
        val_metrics, val_preds, val_labels = evaluate(
            model, data_loader['val'], device, task_type
        )

        # Test
        test_metrics, test_preds, test_labels = evaluate(
            model, data_loader['test'], device, task_type
        )

        t_elapsed = time.time() - t_start
        current_lr = optimizer.param_groups[0]['lr']

        # Print progress
        if task_type == 'binary':
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
                  f"Val: acc={val_metrics['bal_acc']:.4f} auc={val_metrics.get('auroc', 0):.4f} | "
                  f"Test: acc={test_metrics['bal_acc']:.4f} auc={test_metrics.get('auroc', 0):.4f} | "
                  f"lr={current_lr:.6f} | {t_elapsed:.1f}s")
        else:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
                  f"Val: acc={val_metrics['bal_acc']:.4f} kappa={val_metrics.get('kappa', 0):.4f} | "
                  f"Test: acc={test_metrics['bal_acc']:.4f} kappa={test_metrics.get('kappa', 0):.4f} | "
                  f"lr={current_lr:.6f} | {t_elapsed:.1f}s")

        # WandB logging
        if wandb_run:
            log_dict = {
                'epoch': epoch,
                'train/loss': train_loss,
                'train/bal_acc': train_acc,
                'val/loss': val_metrics['loss'],
                'val/bal_acc': val_metrics['bal_acc'],
                'val/f1_macro': val_metrics['f1_macro'],
                'test/loss': test_metrics['loss'],
                'test/bal_acc': test_metrics['bal_acc'],
                'test/f1_macro': test_metrics['f1_macro'],
                'lr': current_lr,
                'time/epoch_seconds': t_elapsed,
            }
            if task_type == 'binary':
                log_dict.update({
                    'val/auc_pr': val_metrics.get('auc_pr', 0),
                    'val/auroc': val_metrics.get('auroc', 0),
                    'test/auc_pr': test_metrics.get('auc_pr', 0),
                    'test/auroc': test_metrics.get('auroc', 0),
                })
            else:
                log_dict.update({
                    'val/kappa': val_metrics.get('kappa', 0),
                    'val/f1_weighted': val_metrics.get('f1_weighted', 0),
                    'test/kappa': test_metrics.get('kappa', 0),
                    'test/f1_weighted': test_metrics.get('f1_weighted', 0),
                })
            wandb_run.log(log_dict)

        # Best model tracking
        if val_metrics['bal_acc'] > best_val_acc:
            best_val_acc = val_metrics['bal_acc']
            best_epoch = epoch
            patience_counter = 0

            ckpt_path = save_dir / f'best_brainpro_{args.dataset}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'args': vars(args),
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
                break

    # Final evaluation with best model
    print(f"\n{'='*60}")
    print(f"Loading best model from epoch {best_epoch}")
    ckpt = torch.load(save_dir / f'best_brainpro_{args.dataset}.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    test_metrics, test_preds, test_labels = evaluate(
        model, data_loader['test'], device, task_type
    )

    # Per-class metrics
    label_names = dataset_config.get('label_names', {})
    target_names = [label_names.get(i, str(i)) for i in range(num_classes)]
    test_f1_per_class = f1_score(test_labels, test_preds, average=None, zero_division=0)
    test_cm = confusion_matrix(test_labels, test_preds)

    print(f"\nFinal Test Results:")
    print(f"  Best epoch:        {best_epoch}")
    print(f"  Best val acc:      {best_val_acc:.4f}")
    print(f"  Test balanced acc: {test_metrics['bal_acc']:.4f}")
    print(f"  Test F1 (macro):   {test_metrics['f1_macro']:.4f}")
    if task_type == 'binary':
        print(f"  Test AUC-PR:       {test_metrics.get('auc_pr', 0):.4f}")
        print(f"  Test AUROC:        {test_metrics.get('auroc', 0):.4f}")
    else:
        print(f"  Test Kappa:        {test_metrics.get('kappa', 0):.4f}")
        print(f"  Test F1 (weighted):{test_metrics.get('f1_weighted', 0):.4f}")

    print(f"\n  Per-class F1:")
    for i, name in enumerate(target_names):
        f1_val = test_f1_per_class[i] if i < len(test_f1_per_class) else 0.0
        print(f"    {name:20s}: {f1_val:.4f}")
    print(f"\n  Confusion Matrix:\n  {test_cm}")
    print(f"{'='*60}")

    # WandB final logging
    if wandb_run:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        wandb_run.log({
            'test/bal_acc': test_metrics['bal_acc'],
            'test/loss': test_metrics['loss'],
            'test/f1_macro': test_metrics['f1_macro'],
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc,
        })

        if task_type == 'binary':
            wandb_run.log({
                'test/auc_pr': test_metrics.get('auc_pr', 0),
                'test/auroc': test_metrics.get('auroc', 0),
            })
        else:
            wandb_run.log({
                'test/kappa': test_metrics.get('kappa', 0),
                'test/f1_weighted': test_metrics.get('f1_weighted', 0),
            })

        # Per-class F1
        for i, name in enumerate(target_names):
            if i < len(test_f1_per_class):
                wandb_run.log({f'test/f1_{name}': test_f1_per_class[i]})

        # Confusion matrix heatmap
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.imshow(test_cm, interpolation='nearest', cmap='Blues')
        ax.set_title(f'BrainPro - {args.dataset} (acc={test_metrics["bal_acc"]:.4f})')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(target_names, rotation=45, ha='right')
        ax.set_yticklabels(target_names)
        plt.colorbar(im, ax=ax)
        for i in range(test_cm.shape[0]):
            for j in range(test_cm.shape[1]):
                ax.text(j, i, str(test_cm[i, j]),
                        ha='center', va='center',
                        color='white' if test_cm[i, j] > test_cm.max() / 2 else 'black')
        plt.tight_layout()
        wandb_run.log({'test/confusion_matrix': wandb.Image(fig)})
        plt.close(fig)

        # Summary
        wandb_run.summary['test_bal_acc'] = test_metrics['bal_acc']
        wandb_run.summary['test_f1_macro'] = test_metrics['f1_macro']
        wandb_run.summary['best_epoch'] = best_epoch
        wandb_run.summary['best_val_acc'] = best_val_acc

        wandb_run.finish()

    return test_metrics['bal_acc']


if __name__ == '__main__':
    main()
