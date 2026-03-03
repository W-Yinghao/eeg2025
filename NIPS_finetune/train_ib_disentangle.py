#!/usr/bin/env python3
"""
Training Script for IB + Disentanglement Fine-Tuning with CodeBrain Backbone

Uses frozen CodeBrain (SSSM) backbone + trainable IB adapter:
  - Token-level Information Bottleneck for fine-grained compression
  - GRL-based adversarial subject removal for disentanglement
  - Per-token interpretability heatmaps

Usage:
    # TUEV dataset (6-class, 16ch, 5s)
    python train_ib_disentangle.py --dataset TUEV --cuda 0

    # TUAB dataset (binary, 16ch, 10s)
    python train_ib_disentangle.py --dataset TUAB --cuda 0

    # With adversarial subject removal
    python train_ib_disentangle.py --dataset TUEV --lambda_adv 0.5 --cuda 0

    # Ablation: IB only (no adversarial)
    python train_ib_disentangle.py --dataset TUEV --lambda_adv 0 --cuda 0

    # Ablation: no IB (CE only)
    python train_ib_disentangle.py --dataset TUEV --beta 0 --lambda_adv 0 --cuda 0

    # WandB logging
    python train_ib_disentangle.py --dataset TUEV --wandb_project eeg_ib --cuda 0
"""

import argparse
import os
import sys
import pickle
import random
import time
from pathlib import Path

import lmdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report,
)

# Existing data infrastructure
sys.path.insert(0, os.path.dirname(__file__))
from finetune_tuev_lmdb import (
    DATASET_CONFIGS,
    EEGLMDBDataset,
    setup_seed,
)

# IB + Disentanglement framework
from ib_disentangle_framework import (
    MultiDisease_CodeBrain_Model,
    InformationBottleneckLoss,
    GRLScheduler,
    configure_optimizer,
    get_interpretability_heatmap,
    train_step,
)
from backbone_factory import create_backbone

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed")




# ==============================================================================
# Subject-aware Dataset Wrapper
# ==============================================================================

class EEGDatasetWithSubjects(Dataset):
    """
    Wraps EEGLMDBDataset to also return subject IDs for adversarial training.

    Subject IDs are extracted from the source_file field in LMDB samples.
    The subject ID is the first underscore-delimited token of source_file
    (e.g., 'aaaaamde' from 'aaaaamde_s001_t001.edf').
    """

    def __init__(self, base_dataset: EEGLMDBDataset):
        self.base_dataset = base_dataset
        self.subject_map = {}  # subject_str -> integer ID
        self._idx_to_subject = {}  # dataset_idx -> integer subject ID
        self._build_subject_map()

    def _build_subject_map(self):
        """Scan LMDB once to extract subject IDs from source_file."""
        if self.base_dataset.env is None:
            self.base_dataset._init_db()

        env = self.base_dataset.env
        indices = self.base_dataset.indices

        with env.begin(write=False) as txn:
            for i, actual_idx in enumerate(indices):
                key = f'{actual_idx:08d}'.encode()
                value = txn.get(key)
                if value is None:
                    self._idx_to_subject[i] = 0
                    continue

                sample = pickle.loads(value)

                # Extract subject from source_file (first token before '_')
                subj = sample.get('subject_id', None)
                if subj is None:
                    source_file = sample.get('source_file', '')
                    subj = source_file.split('_')[0] if source_file else f'unknown_{actual_idx}'

                if subj not in self.subject_map:
                    self.subject_map[subj] = len(self.subject_map)
                self._idx_to_subject[i] = self.subject_map[subj]

        print(f"  Found {len(self.subject_map)} subjects from source_file scan")

    @property
    def num_subjects(self):
        """Return number of unique subjects."""
        return max(len(self.subject_map), 1)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """Returns (signal, label, subject_id)."""
        signal, label = self.base_dataset[idx]
        subject_id = self._idx_to_subject.get(idx, 0)
        return signal, label, subject_id

    @staticmethod
    def collate_fn(batch):
        """Collate function returning (eeg, labels, subject_ids)."""
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        s_ids = np.array([x[2] for x in batch])
        return (
            torch.from_numpy(x_data).float(),
            torch.from_numpy(y_label).long(),
            torch.from_numpy(s_ids).long(),
        )


# ==============================================================================
# Data Loading
# ==============================================================================

def load_data_with_subjects(params, dataset_config):
    """
    Load data using existing infrastructure, wrap with subject ID support.

    Returns:
        data_loaders: dict with 'train', 'val', 'test' DataLoaders
        num_classes: number of disease classes
        seq_len: temporal sequence length
        num_subjects: number of unique subjects
    """
    data_dir = Path(dataset_config['data_dir'])
    splits = dataset_config['splits']
    single_lmdb = dataset_config.get('single_lmdb', False)
    cross_subject = dataset_config.get('cross_subject', False)

    if hasattr(params, 'cross_subject') and params.cross_subject is not None:
        cross_subject = params.cross_subject

    print(f"Loading {params.dataset} data from {data_dir}")

    # Handle label filtering
    include_labels = None
    exclude_labels = None
    if params.dataset == 'DIAGNOSIS':
        if hasattr(params, 'include_labels') and params.include_labels:
            include_labels = params.include_labels
        if hasattr(params, 'exclude_labels') and params.exclude_labels:
            exclude_labels = params.exclude_labels

    # Create base datasets
    if single_lmdb:
        lmdb_path = data_dir / splits['train']
        train_ratio = dataset_config.get('train_ratio', 0.7)
        val_ratio = dataset_config.get('val_ratio', 0.15)
        test_ratio = dataset_config.get('test_ratio', 0.15)

        base_train = EEGLMDBDataset(
            lmdb_path, dataset_config, split='train',
            val_ratio=val_ratio, train_ratio=train_ratio, test_ratio=test_ratio,
            include_labels=include_labels, exclude_labels=exclude_labels,
            cross_subject=cross_subject
        )
        base_val = EEGLMDBDataset(
            lmdb_path, dataset_config, split='val',
            val_ratio=val_ratio, train_ratio=train_ratio, test_ratio=test_ratio,
            include_labels=include_labels, exclude_labels=exclude_labels,
            cross_subject=cross_subject
        )
        base_test = EEGLMDBDataset(
            lmdb_path, dataset_config, split='test',
            val_ratio=val_ratio, train_ratio=train_ratio, test_ratio=test_ratio,
            include_labels=include_labels, exclude_labels=exclude_labels,
            cross_subject=cross_subject
        )
    else:
        val_from_train = (splits['train'] == splits['val'])
        train_lmdb = data_dir / splits['train']
        val_lmdb = data_dir / splits['val']
        test_lmdb = data_dir / splits['test']

        base_train = EEGLMDBDataset(
            train_lmdb, dataset_config, split='train',
            val_ratio=params.val_ratio, is_val_from_train=val_from_train,
            include_labels=include_labels, exclude_labels=exclude_labels
        )
        if val_from_train:
            base_val = EEGLMDBDataset(
                val_lmdb, dataset_config, split='val',
                val_ratio=params.val_ratio, is_val_from_train=True,
                include_labels=include_labels, exclude_labels=exclude_labels
            )
        else:
            base_val = EEGLMDBDataset(
                val_lmdb, dataset_config, split='val',
                include_labels=include_labels, exclude_labels=exclude_labels
            )
        base_test = EEGLMDBDataset(
            test_lmdb, dataset_config, split='test',
            include_labels=include_labels, exclude_labels=exclude_labels
        )

    # Wrap with subject ID support
    print("\nWrapping datasets with subject ID support:")
    train_dataset = EEGDatasetWithSubjects(base_train)
    val_dataset = EEGDatasetWithSubjects(base_val)
    test_dataset = EEGDatasetWithSubjects(base_test)

    # Determine number of classes
    if include_labels is not None:
        num_classes = len(include_labels)
    elif exclude_labels:
        num_classes = dataset_config['num_classes'] - len(exclude_labels)
    else:
        num_classes = dataset_config['num_classes']

    # Get number of subjects (from the largest split)
    num_subjects = max(
        train_dataset.num_subjects,
        val_dataset.num_subjects,
        test_dataset.num_subjects,
        2,  # minimum 2 for BatchNorm in subject head
    )

    print(f"\nDataset: {params.dataset}")
    print(f"Classes: {num_classes}, Task: {dataset_config['task_type']}")
    print(f"Subjects: {num_subjects}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create data loaders
    data_loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=params.batch_size,
            collate_fn=EEGDatasetWithSubjects.collate_fn,
            shuffle=True,
            num_workers=params.num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=params.batch_size,
            collate_fn=EEGDatasetWithSubjects.collate_fn,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=True,
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=params.batch_size,
            collate_fn=EEGDatasetWithSubjects.collate_fn,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=True,
        ),
    }

    # Get sequence length from a sample
    sample_batch = next(iter(data_loaders['train']))
    seq_len = sample_batch[0].shape[2]
    print(f"Input shape: {sample_batch[0].shape} (batch, channels, seq_len, patch_size)")

    return data_loaders, num_classes, seq_len, num_subjects


# ==============================================================================
# Training and Evaluation
# ==============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, clip_value,
                    use_subjects=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_task = 0.0
    total_ib = 0.0
    total_mi = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    for batch in dataloader:
        eeg_data = batch[0].to(device)
        labels = batch[1].to(device)
        subject_ids = batch[2].to(device) if use_subjects and len(batch) > 2 else None

        # Clamp subject IDs to valid range
        if subject_ids is not None:
            num_subj = model.num_subjects
            subject_ids = subject_ids % num_subj

        # Forward pass
        outputs = model(eeg_data)

        # Compute loss
        loss, loss_dict = criterion(
            disease_logits=outputs['disease_logits'],
            labels=labels,
            mu=outputs['mu'],
            log_var=outputs['log_var'],
            subject_logits=outputs['subject_logits'] if subject_ids is not None else None,
            subject_ids=subject_ids,
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        # Track metrics
        total_loss += loss_dict['total']
        total_task += loss_dict['task']
        total_ib += loss_dict['ib']
        total_mi += loss_dict['mi']
        num_batches += 1

        preds = outputs['disease_logits'].argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = max(num_batches, 1)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    return {
        'loss': total_loss / n,
        'task': total_task / n,
        'ib': total_ib / n,
        'mi': total_mi / n,
        'bal_acc': bal_acc,
        'preds': all_preds,
        'labels': all_labels,
    }


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    for batch in dataloader:
        eeg_data = batch[0].to(device)
        labels = batch[1].to(device)

        outputs = model(eeg_data)

        # CE loss for evaluation (works for both binary and multiclass
        # since disease_logits is always (B, num_classes))
        loss = F.cross_entropy(outputs['disease_logits'], labels)

        total_loss += loss.item()
        num_batches += 1

        preds = outputs['disease_logits'].argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(num_batches, 1)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    return avg_loss, bal_acc, all_preds, all_labels


@torch.no_grad()
def compute_heatmap(model, dataloader, device, n_channels, seq_len, max_batches=5):
    """Compute interpretability heatmap on a few batches."""
    model.eval()
    all_channel_imp = []
    all_temporal_imp = []

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        eeg_data = batch[0].to(device)
        outputs = model(eeg_data, return_tokens=True)

        hmap = get_interpretability_heatmap(
            outputs['mu'], outputs['log_var'], n_channels, seq_len
        )
        all_channel_imp.append(hmap['channel_importance'].cpu())
        all_temporal_imp.append(hmap['temporal_importance'].cpu())

    channel_imp = torch.cat(all_channel_imp, dim=0).mean(dim=0)  # (C,)
    temporal_imp = torch.cat(all_temporal_imp, dim=0).mean(dim=0)  # (S,)

    return channel_imp, temporal_imp


# ==============================================================================
# Main
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='IB + Disentanglement EEG Fine-Tuning')

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
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--patience', type=int, default=15)

    # IB + Disentanglement
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='IB bottleneck dimension per token')
    parser.add_argument('--beta', type=float, default=1e-3,
                        help='IB (KL) loss weight')
    parser.add_argument('--lambda_adv', type=float, default=0.5,
                        help='adversarial subject loss weight')
    parser.add_argument('--grl_gamma', type=float, default=10.0,
                        help='GRL lambda schedule steepness')
    parser.add_argument('--use_subjects', action='store_true', default=True,
                        help='use adversarial subject head')
    parser.add_argument('--no_subjects', action='store_true',
                        help='disable adversarial subject head')

    # Backbone model
    parser.add_argument('--model', type=str, default='codebrain',
                        choices=['codebrain', 'cbramod', 'femba', 'luna'], help='backbone model type')
    # CodeBrain-specific
    parser.add_argument('--n_layer', type=int, default=8, help='SSSM layers (CodeBrain)')
    parser.add_argument('--codebook_size_t', type=int, default=4096)
    parser.add_argument('--codebook_size_f', type=int, default=4096)
    # CBraMod-specific
    parser.add_argument('--n_layer_cbramod', type=int, default=12, help='transformer layers (CBraMod)')
    parser.add_argument('--nhead', type=int, default=8, help='attention heads (CBraMod)')
    parser.add_argument('--dim_feedforward', type=int, default=800, help='FFN dim (CBraMod)')
    # Inter-layer adapters (CBraMod only)
    parser.add_argument('--use_layer_adapters', action='store_true',
                        help='insert trainable bottleneck adapters between CBraMod layers')
    parser.add_argument('--adapter_reduction', type=int, default=4,
                        help='adapter bottleneck reduction factor')
    # LUNA-specific
    parser.add_argument('--luna_size', type=str, default='base', choices=['base', 'large', 'huge'])
    # Pretrained weights
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help='path to pretrained weights (auto-resolved if None)')

    # Logging
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None,
                        help='WandB group name to cluster related runs (e.g. one ablation sweep)')
    parser.add_argument('--save_dir', type=str, default='checkpoints_ib')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--heatmap_interval', type=int, default=10,
                        help='compute heatmap every N epochs (0=disabled)')

    # Label filtering
    parser.add_argument('--include_labels', type=int, nargs='+', default=None)
    parser.add_argument('--exclude_labels', type=int, nargs='+', default=None)
    parser.add_argument('--cross_subject', type=bool, default=None)

    return parser.parse_args()


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

    # Resolve pretrained weights path
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

    # Load data
    data_loaders, num_classes, seq_len, num_subjects = load_data_with_subjects(
        args, dataset_config
    )
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

    # Create model
    model = MultiDisease_CodeBrain_Model(
        backbone=backbone,
        token_dim=token_dim,
        num_classes=num_classes,
        num_subjects=num_subjects if args.use_subjects else 2,
        latent_dim=args.latent_dim,
        lambda_adv=args.lambda_adv if args.use_subjects else 0.0,
        dropout=args.dropout,
    ).to(device)

    # Loss
    criterion = InformationBottleneckLoss(
        beta=args.beta,
        lambda_adv=args.lambda_adv if args.use_subjects else 0.0,
        task_type=task_type,
    )

    # Optimizer
    optimizer = configure_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    # GRL scheduler
    grl_scheduler = GRLScheduler(model, gamma=args.grl_gamma) if args.use_subjects else None

    # WandB
    wandb_run = None
    if args.wandb_project and WANDB_AVAILABLE:
        run_name = args.wandb_run_name or (
            f"ib_{args.dataset}_d{args.latent_dim}_b{args.beta}_l{args.lambda_adv}_s{args.seed}"
        )

        config = vars(args).copy()
        frozen_params = sum(p.numel() for p in model.backbone.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        config.update({
            'model_type': 'MultiDisease_CodeBrain_Model',
            'frozen_params': frozen_params,
            'trainable_params': trainable_params,
            'trainable_ratio': trainable_params / (frozen_params + trainable_params) * 100,
            'num_classes': num_classes,
            'num_subjects': num_subjects,
            'n_channels': n_channels,
            'seq_len': seq_len,
            'patch_size': patch_size,
            'backbone_out_dim': backbone_out_dim,
        })

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            group=args.wandb_group,
            config=config,
        )
        wandb.watch(model, log='gradients', log_freq=50)

    # Save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Unique checkpoint name per experiment (prevents concurrent-job file collisions)
    # Encodes model, dataset, beta, lambda_adv to avoid cross-backbone and cross-config contamination
    beta_str = str(args.beta).replace('-', 'm')
    lam_str = str(args.lambda_adv).replace('-', 'm')
    ckpt_name = f'best_{args.dataset}_{args.model}_b{beta_str}_l{lam_str}_ib.pth'

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Starting training: {args.epochs} epochs")
    print(f"  IB beta={args.beta}, lambda_adv={args.lambda_adv}")
    print(f"  use_subjects={args.use_subjects}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()

        # Update GRL lambda
        grl_lambda = 0.0
        if grl_scheduler:
            grl_lambda = grl_scheduler.step(epoch, args.epochs)

        # Train
        train_metrics = train_one_epoch(
            model, data_loaders['train'], criterion, optimizer, device,
            args.clip_value, use_subjects=args.use_subjects,
        )
        train_f1_macro = f1_score(train_metrics['labels'], train_metrics['preds'], average='macro', zero_division=0)
        train_f1_weighted = f1_score(train_metrics['labels'], train_metrics['preds'], average='weighted', zero_division=0)

        # Validate
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, data_loaders['val'], criterion, device,
        )
        val_f1_macro = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        val_f1_weighted = f1_score(val_labels, val_preds, average='weighted', zero_division=0)

        # Test after every epoch
        test_loss, test_acc, test_preds, test_labels = evaluate(
            model, data_loaders['test'], criterion, device,
        )
        test_f1_macro = f1_score(test_labels, test_preds, average='macro', zero_division=0)
        test_f1_weighted = f1_score(test_labels, test_preds, average='weighted', zero_division=0)

        t_elapsed = time.time() - t_start

        # Logging
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train: loss={train_metrics['loss']:.4f} acc={train_metrics['bal_acc']:.4f} f1={train_f1_macro:.4f} | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1_macro:.4f} | "
            f"Test: loss={test_loss:.4f} acc={test_acc:.4f} f1={test_f1_macro:.4f} | "
            f"grl={grl_lambda:.3f} | {t_elapsed:.1f}s"
        )

        if wandb_run:
            wandb_run.log({
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/task': train_metrics['task'],
                'train/ib': train_metrics['ib'],
                'train/mi': train_metrics['mi'],
                'train/bal_acc': train_metrics['bal_acc'],
                'train/f1_macro': train_f1_macro,
                'train/f1_weighted': train_f1_weighted,
                'val/loss': val_loss,
                'val/bal_acc': val_acc,
                'val/f1_macro': val_f1_macro,
                'val/f1_weighted': val_f1_weighted,
                'test/loss': test_loss,
                'test/bal_acc': test_acc,
                'test/f1_macro': test_f1_macro,
                'test/f1_weighted': test_f1_weighted,
                'grl/lambda': grl_lambda,
                'lr': optimizer.param_groups[0]['lr'],
                'time/epoch_seconds': t_elapsed,
            })

        # Heatmap computation
        if (args.heatmap_interval > 0 and epoch % args.heatmap_interval == 0):
            ch_imp, temp_imp = compute_heatmap(
                model, data_loaders['val'], device, n_channels, seq_len
            )
            print(f"  Channel importance (top-5): {ch_imp.topk(min(5, len(ch_imp)))[0].tolist()}")
            print(f"  Temporal importance: {temp_imp.tolist()}")

            if wandb_run:
                wandb_run.log({
                    f'heatmap/channel_{i}': v.item()
                    for i, v in enumerate(ch_imp)
                })

        # Best model tracking
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            ckpt_path = save_dir / ckpt_name
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
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(best: epoch {best_epoch}, val_acc={best_val_acc:.4f})")
            break

    # Final test
    print(f"\n{'='*60}")
    print(f"Loading best model from epoch {best_epoch}")
    ckpt = torch.load(save_dir / ckpt_name, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    test_loss, test_acc, test_preds_final, test_labels_final = evaluate(
        model, data_loaders['test'], criterion, device,
    )

    # Final heatmap
    ch_imp, temp_imp = compute_heatmap(
        model, data_loaders['test'], device, n_channels, seq_len
    )

    # Per-class metrics
    label_names = dataset_config.get('label_names', {})
    target_names = [label_names.get(i, str(i)) for i in range(num_classes)]
    test_f1_macro_final = f1_score(test_labels_final, test_preds_final, average='macro', zero_division=0)
    test_f1_weighted_final = f1_score(test_labels_final, test_preds_final, average='weighted', zero_division=0)
    test_f1_per_class = f1_score(test_labels_final, test_preds_final, average=None, zero_division=0)
    test_cm = confusion_matrix(test_labels_final, test_preds_final)

    print(f"\nFinal Test Results:")
    print(f"  Best epoch:          {best_epoch}")
    print(f"  Best val acc:        {best_val_acc:.4f}")
    print(f"  Test loss:           {test_loss:.4f}")
    print(f"  Test balanced acc:   {test_acc:.4f}")
    print(f"  Test F1 (macro):     {test_f1_macro_final:.4f}")
    print(f"  Test F1 (weighted):  {test_f1_weighted_final:.4f}")
    print(f"\n  Per-class F1:")
    for i, name in enumerate(target_names):
        f1_val = test_f1_per_class[i] if i < len(test_f1_per_class) else 0.0
        print(f"    {name:20s}: {f1_val:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {test_cm}")
    print(f"\nInterpretability (test set):")
    print(f"  Channel importance (top-5): {ch_imp.topk(min(5, len(ch_imp)))[0].tolist()}")
    print(f"  Temporal importance: {temp_imp.tolist()}")
    print(f"{'='*60}")

    if wandb_run:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Basic test metrics
        wandb_run.log({
            'final_test/bal_acc': test_acc,
            'final_test/loss': test_loss,
            'final_test/f1_macro': test_f1_macro_final,
            'final_test/f1_weighted': test_f1_weighted_final,
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc,
        })

        # Per-class F1
        for i, name in enumerate(target_names):
            if i < len(test_f1_per_class):
                wandb_run.log({f'final_test/f1_{name}': test_f1_per_class[i]})

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
        for i in range(test_cm.shape[0]):
            for j in range(test_cm.shape[1]):
                ax.text(j, i, str(test_cm[i, j]),
                        ha='center', va='center',
                        color='white' if test_cm[i, j] > test_cm.max() / 2 else 'black')
        plt.tight_layout()
        wandb_run.log({'final_test/confusion_matrix': wandb.Image(fig)})
        plt.close(fig)

        # Summary metrics
        wandb_run.summary['test_bal_acc'] = test_acc
        wandb_run.summary['test_f1_macro'] = test_f1_macro_final
        wandb_run.summary['test_f1_weighted'] = test_f1_weighted_final
        wandb_run.summary['test_loss'] = test_loss
        wandb_run.summary['best_epoch'] = best_epoch
        wandb_run.summary['best_val_acc'] = best_val_acc

        wandb_run.finish()

    return test_acc


if __name__ == '__main__':
    main()
