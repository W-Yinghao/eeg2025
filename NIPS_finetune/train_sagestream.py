#!/usr/bin/env python3
"""
Training Script for SageStream Fine-Tuning (SA-MoE + IIB)

Uses frozen CodeBrain / CBraMod backbone + trainable SA-MoE layers + IIB:
  - Subject-Aware Mixture of Experts for adaptive token processing
  - Information Invariant Bottleneck for subject-invariant representations
  - GRL-based adversarial subject removal

Usage:
    # TUEV dataset (6-class, 16ch, 5s)
    python train_sagestream.py --dataset TUEV --cuda 0

    # TUAB dataset (binary, 16ch, 10s)
    python train_sagestream.py --dataset TUAB --cuda 0

    # With specific SA-MoE config
    python train_sagestream.py --dataset TUEV --num_experts 4 --top_k 2 --n_moe_layers 2

    # Ablation: MoE only (no IIB)
    python train_sagestream.py --dataset TUEV --alpha_kl 0 --beta_adv 0

    # Ablation: IIB only (no MoE, uses n_moe_layers=0)
    python train_sagestream.py --dataset TUEV --n_moe_layers 0

    # WandB logging with group
    python train_sagestream.py --dataset TUEV --wandb_project eeg_sagestream --wandb_group my_sweep
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

# Existing data infrastructure
sys.path.insert(0, os.path.dirname(__file__))
from finetune_tuev_lmdb import DATASET_CONFIGS, setup_seed
from train_ib_disentangle import (
    EEGDatasetWithSubjects,
    load_data_with_subjects,
)
from backbone_factory import create_backbone
from ib_disentangle_framework import GRLScheduler

# SageStream framework
from sagestream_framework import (
    SageStreamModel,
    SageStreamLoss,
    configure_optimizer,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed")


# ==============================================================================
# Training and Evaluation
# ==============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    clip_value, use_subjects=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_task = 0.0
    total_kl = 0.0
    total_adv = 0.0
    total_aux = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    for batch in dataloader:
        eeg_data = batch[0].to(device)
        labels = batch[1].to(device)
        subject_ids = batch[2].to(device) if use_subjects and len(batch) > 2 else None

        # Clamp subject IDs to valid range
        if subject_ids is not None:
            subject_ids = subject_ids % model.num_subjects

        # Clear MoE auxiliary state
        model.clear_aux_state()

        # Forward pass
        outputs = model(eeg_data, subject_ids=subject_ids)

        # Get MoE auxiliary loss
        aux_loss = model.get_total_aux_loss()

        # Compute loss
        loss, loss_dict = criterion(
            task_logits=outputs['task_logits'],
            labels=labels,
            mu=outputs['mu'],
            log_var=outputs['log_var'],
            subject_logits=outputs['subject_logits'] if subject_ids is not None else None,
            subject_ids=subject_ids,
            aux_loss=aux_loss,
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        # Track metrics
        total_loss += loss_dict['total']
        total_task += loss_dict['task']
        total_kl += loss_dict['kl']
        total_adv += loss_dict['adv']
        total_aux += loss_dict['aux']
        num_batches += 1

        if criterion.task_type == 'binary':
            preds = (torch.sigmoid(outputs['task_logits'].squeeze()) > 0.5).long()
        else:
            preds = outputs['task_logits'].argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = max(num_batches, 1)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    return {
        'loss': total_loss / n,
        'task': total_task / n,
        'kl': total_kl / n,
        'adv': total_adv / n,
        'aux': total_aux / n,
        'bal_acc': bal_acc,
        'preds': all_preds,
        'labels': all_labels,
    }


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation/test set (CE loss only)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    for batch in dataloader:
        eeg_data = batch[0].to(device)
        labels = batch[1].to(device)

        outputs = model(eeg_data, subject_ids=None)

        # Only task loss for evaluation
        if criterion.task_type == 'binary':
            loss = F.binary_cross_entropy_with_logits(
                outputs['task_logits'].squeeze(), labels.float()
            )
        else:
            loss = F.cross_entropy(outputs['task_logits'], labels)

        total_loss += loss.item()
        num_batches += 1

        if criterion.task_type == 'binary':
            preds = (torch.sigmoid(outputs['task_logits'].squeeze()) > 0.5).long()
        else:
            preds = outputs['task_logits'].argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(num_batches, 1)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    return avg_loss, bal_acc, all_preds, all_labels


# ==============================================================================
# Main
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='SageStream Fine-Tuning (SA-MoE + IIB)')

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

    # SA-MoE
    parser.add_argument('--num_experts', type=int, default=4,
                        help='number of experts in the shared pool')
    parser.add_argument('--top_k', type=int, default=2,
                        help='number of experts per token')
    parser.add_argument('--n_moe_layers', type=int, default=2,
                        help='number of SA-MoE layers (0 = skip MoE)')
    parser.add_argument('--d_ff', type=int, default=None,
                        help='expert FFN dim (default: 2 * token_dim)')
    parser.add_argument('--aux_weight', type=float, default=0.01,
                        help='MoE auxiliary load-balancing loss weight')
    parser.add_argument('--use_style', action='store_true', default=True,
                        help='enable subject style alignment in SA-MoE')
    parser.add_argument('--no_style', action='store_true',
                        help='disable subject style alignment')
    parser.add_argument('--subject_embed_dim', type=int, default=64)
    parser.add_argument('--style_hidden_dim', type=int, default=128)

    # IIB
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='IIB bottleneck dimension')
    parser.add_argument('--alpha_kl', type=float, default=1e-3,
                        help='KL divergence loss weight')
    parser.add_argument('--beta_adv', type=float, default=0.5,
                        help='adversarial subject loss weight')
    parser.add_argument('--grl_gamma', type=float, default=10.0,
                        help='GRL lambda schedule steepness')
    parser.add_argument('--use_subjects', action='store_true', default=True,
                        help='use adversarial subject head')
    parser.add_argument('--no_subjects', action='store_true',
                        help='disable adversarial subject head')

    # Backbone
    parser.add_argument('--model', type=str, default='codebrain',
                        choices=['codebrain', 'cbramod', 'femba', 'luna'])
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--codebook_size_t', type=int, default=4096)
    parser.add_argument('--codebook_size_f', type=int, default=4096)
    parser.add_argument('--n_layer_cbramod', type=int, default=12)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--dim_feedforward', type=int, default=800)
    parser.add_argument('--use_layer_adapters', action='store_true')
    parser.add_argument('--adapter_reduction', type=int, default=4)
    parser.add_argument('--pretrained_weights', type=str, default=None)
    # LUNA-specific
    parser.add_argument('--luna_size', type=str, default='base', choices=['base', 'large', 'huge'])

    # Logging
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='checkpoints_sagestream')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Unique run name for checkpoint file (avoids overwrite across ablations)')
    parser.add_argument('--num_workers', type=int, default=0)

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
    if args.no_style:
        args.use_style = False

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_config = DATASET_CONFIGS[args.dataset]
    task_type = dataset_config['task_type']
    n_channels = dataset_config['n_channels']
    patch_size = dataset_config['patch_size']

    # Resolve pretrained weights
    if args.pretrained_weights is None:
        base_dir = os.path.dirname(__file__)
        weights_map = {
            'codebrain': os.path.join(base_dir, 'CodeBrain/Checkpoints/CodeBrain.pth'),
            'cbramod': os.path.join(base_dir, 'Cbramod_pretrained_weights.pth'),
            'luna': os.path.join(base_dir, f'BioFoundation/checkpoints/LUNA/LUNA_{args.luna_size}.safetensors'),
            'femba': None,  # FEMBA has no pure pretrained backbone
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

    # For binary classification with BCE loss, model outputs 1 logit (not 2)
    model_num_classes = 1 if task_type == 'binary' else num_classes

    # Create SageStream model
    model = SageStreamModel(
        backbone=backbone,
        token_dim=token_dim,
        num_classes=model_num_classes,
        num_subjects=num_subjects if args.use_subjects else 2,
        latent_dim=args.latent_dim,
        n_moe_layers=args.n_moe_layers,
        num_experts=args.num_experts,
        top_k=args.top_k,
        d_ff=args.d_ff,
        aux_loss_weight=args.aux_weight,
        use_style=args.use_style and args.use_subjects,
        subject_embed_dim=args.subject_embed_dim,
        style_hidden_dim=args.style_hidden_dim,
        dropout=args.dropout,
        lambda_adv=args.beta_adv if args.use_subjects else 0.0,
    ).to(device)

    # Print parameter summary
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    moe_params = (sum(p.numel() for p in model.expert_pool.parameters())
                  + sum(p.numel() for p in model.router.parameters()))
    style_params = sum(
        p.numel() for layer in model.moe_layers
        for name, p in layer.named_parameters()
        if 'style_align' in name
    )
    ib_params = sum(p.numel() for p in model.ib_encoder.parameters())

    print(f"\nSageStream parameter summary:")
    print(f"  Frozen params:       {frozen_params:,}")
    print(f"  Trainable params:    {trainable_params:,}")
    print(f"  Trainable ratio:     {trainable_params / (frozen_params + trainable_params) * 100:.2f}%")
    print(f"  SA-MoE (experts+router): {moe_params:,}")
    print(f"  Style alignment:     {style_params:,}")
    print(f"  IIB encoder:         {ib_params:,}")
    print(f"  Latent dim:          {args.latent_dim}")
    print(f"  MoE layers:          {args.n_moe_layers}")
    print(f"  Experts: {args.num_experts}, top_k: {args.top_k}")
    print(f"  Output classes:      {num_classes}")

    # Loss
    criterion = SageStreamLoss(
        alpha_kl=args.alpha_kl,
        beta_adv=args.beta_adv if args.use_subjects else 0.0,
        aux_weight=args.aux_weight,
        task_type=task_type,
    )

    # Optimizer
    optimizer = configure_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    # GRL scheduler
    grl_scheduler = None
    if args.use_subjects and args.beta_adv > 0:
        grl_scheduler = GRLScheduler(model, gamma=args.grl_gamma)

    # WandB
    wandb_run = None
    if args.wandb_project and WANDB_AVAILABLE:
        run_name = args.wandb_run_name or (
            f"sage_{args.dataset}_e{args.num_experts}k{args.top_k}_"
            f"L{args.n_moe_layers}_a{args.alpha_kl}_b{args.beta_adv}_s{args.seed}"
        )

        config = vars(args).copy()
        config.update({
            'model_type': 'SageStreamModel',
            'frozen_params': frozen_params,
            'trainable_params': trainable_params,
            'moe_params': moe_params,
            'style_params': style_params,
            'ib_params': ib_params,
            'num_classes': num_classes,
            'num_subjects': num_subjects,
            'n_channels': n_channels,
            'seq_len': seq_len,
            'patch_size': patch_size,
            'token_dim': token_dim,
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

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Starting training: {args.epochs} epochs")
    print(f"  SA-MoE: {args.n_moe_layers} layers, {args.num_experts} experts, top_k={args.top_k}")
    print(f"  IIB: alpha_kl={args.alpha_kl}, beta_adv={args.beta_adv}")
    print(f"  use_subjects={args.use_subjects}, use_style={args.use_style}")
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
        train_f1_macro = f1_score(
            train_metrics['labels'], train_metrics['preds'],
            average='macro', zero_division=0
        )
        train_f1_weighted = f1_score(
            train_metrics['labels'], train_metrics['preds'],
            average='weighted', zero_division=0
        )

        # Validate
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, data_loaders['val'], criterion, device,
        )
        val_f1_macro = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        val_f1_weighted = f1_score(val_labels, val_preds, average='weighted', zero_division=0)

        # Test
        test_loss, test_acc, test_preds, test_labels = evaluate(
            model, data_loaders['test'], criterion, device,
        )
        test_f1_macro = f1_score(test_labels, test_preds, average='macro', zero_division=0)
        test_f1_weighted = f1_score(test_labels, test_preds, average='weighted', zero_division=0)

        t_elapsed = time.time() - t_start

        # Console logging
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
                'train/kl': train_metrics['kl'],
                'train/adv': train_metrics['adv'],
                'train/aux': train_metrics['aux'],
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

        # Best model tracking
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            ckpt_name = args.run_name or f'{args.dataset}_sagestream'
            ckpt_path = save_dir / f'best_{ckpt_name}.pth'
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

    # =========================================================================
    # Final test with best model
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Loading best model from epoch {best_epoch}")
    ckpt_name = args.run_name or f'{args.dataset}_sagestream'
    ckpt_path = save_dir / f'best_{ckpt_name}.pth'
    ckpt = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if missing:
        print(f"  WARNING: Missing keys in checkpoint: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  WARNING: Unexpected keys in checkpoint: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    test_loss, test_acc, test_preds_final, test_labels_final = evaluate(
        model, data_loaders['test'], criterion, device,
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
    print(f"{'='*60}")

    if wandb_run:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        wandb_run.log({
            'final_test/bal_acc': test_acc,
            'final_test/loss': test_loss,
            'final_test/f1_macro': test_f1_macro_final,
            'final_test/f1_weighted': test_f1_weighted_final,
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc,
        })

        for i, name in enumerate(target_names):
            if i < len(test_f1_per_class):
                wandb_run.log({f'final_test/f1_{name}': test_f1_per_class[i]})

        # Confusion matrix
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
