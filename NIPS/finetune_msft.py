#!/usr/bin/env python3
"""
MSFT Fine-tuning Script for CodeBrain/CBraMod EEG Classification.

Multi-Scale FineTuning (MSFT) wraps a frozen backbone (CodeBrain SSSM or CBraMod) with
trainable scale-specific adapters, cross-scale aggregators, and per-scale
classifiers to capture multi-scale temporal patterns in EEG data.

Based on: "Multi-Scale Finetuning for Encoder-based Time Series Foundation Models"

Usage:
    # TUEV with MSFT using CodeBrain (default)
    python finetune_msft.py --dataset TUEV --cuda 0 --epochs 50

    # TUEV with MSFT using CBraMod
    python finetune_msft.py --dataset TUEV --cuda 0 --model cbramod

    # TUAB with 4 scales
    python finetune_msft.py --dataset TUAB --cuda 0 --num_scales 4

    # AD_DIAGNOSIS with custom learning rate
    python finetune_msft.py --dataset AD_DIAGNOSIS --cuda 0 --lr 5e-4

    # Disable wandb
    python finetune_msft.py --dataset TUEV --cuda 0 --no_wandb
"""

import argparse
import copy
import math
import os
import random
import sys
from datetime import datetime
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse data-loading infrastructure from existing script
from finetune_tuev_lmdb import (
    EEGLMDBDataset,
    DATASET_CONFIGS,
    load_data,
    setup_seed,
    plot_tsne,
)

# MSFT modules
from msft_modules import (
    MSFTModel, create_msft_model,
    MSFTCBraModModel, create_msft_cbramod_model,
)

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Logging will be disabled.")

DEFAULT_CODEBRAIN_WEIGHTS = "/home/infres/yinwang/eeg2025/NIPS/CodeBrain/Checkpoints/CodeBrain.pth"
DEFAULT_CBRAMOD_WEIGHTS = "/home/infres/yinwang/eeg2025/NIPS/Cbramod_pretrained_weights.pth"
DEFAULT_OUTPUT_DIR = "/home/infres/yinwang/eeg2025/NIPS/checkpoints"


# ============================================================================
# Trainer
# ============================================================================
class MSFTTrainer:
    """Trainer for MSFT models with frozen backbone and trainable adapters."""

    def __init__(self, params, data_loader, model, dataset_config, use_wandb=True):
        self.params = params
        self.data_loader = data_loader
        self.model = model.cuda()
        self.dataset_config = dataset_config
        self.task_type = dataset_config['task_type']
        self.label_names = dataset_config['label_names']
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.best_model_states = None

        # Loss
        if self.task_type == 'binary':
            self.criterion = BCEWithLogitsLoss().cuda()
        else:
            self.criterion = CrossEntropyLoss(
                label_smoothing=params.label_smoothing
            ).cuda()

        # Freeze backbone, collect trainable params
        trainable_params = []
        frozen_count = 0
        trainable_count = 0
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
                frozen_count += param.numel()
            else:
                trainable_params.append(param)
                trainable_count += param.numel()

        print(f"MSFT parameter summary:")
        print(f"  Frozen backbone parameters:  {frozen_count:,}")
        print(f"  Trainable MSFT parameters:   {trainable_count:,}")
        print(f"  Trainable ratio:             {trainable_count / (frozen_count + trainable_count) * 100:.2f}%")

        # Optimizer (only trainable parameters)
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=params.lr,
            weight_decay=params.weight_decay,
        )

        # Scheduler
        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=params.epochs * self.data_length,
            eta_min=1e-6,
        )

    def evaluate(self, data_loader):
        """Evaluate model and return metrics."""
        self.model.eval()
        truths, preds, probs = [], [], []

        for x, y in tqdm(data_loader, mininterval=1, desc="Evaluating", leave=False):
            x, y = x.cuda(), y.cuda()
            with torch.no_grad():
                logits = self.model(x)
                if self.task_type == 'binary':
                    prob = torch.sigmoid(logits).squeeze()
                    pred_y = (prob > 0.5).long()
                    prob_np = prob.cpu().numpy()
                    if prob_np.ndim == 0:
                        probs.append(prob_np.item())
                    else:
                        probs.extend(prob_np.tolist())
                else:
                    pred_y = torch.max(logits, dim=-1)[1]

            y_np = y.cpu().numpy()
            pred_np = pred_y.cpu().numpy()
            if y_np.ndim == 0:
                truths.append(y_np.item())
                preds.append(pred_np.item())
            else:
                truths.extend(y_np.tolist())
                preds.extend(pred_np.tolist())

        truths = np.array(truths)
        preds = np.array(preds)

        acc = balanced_accuracy_score(truths, preds)
        f1 = f1_score(truths, preds,
                       average='weighted' if self.task_type == 'multiclass' else 'binary')
        kappa = cohen_kappa_score(truths, preds)
        cm = confusion_matrix(truths, preds)

        if self.task_type == 'binary':
            probs = np.array(probs)
            try:
                roc_auc = roc_auc_score(truths, probs)
                pr_auc = average_precision_score(truths, probs)
            except Exception:
                roc_auc, pr_auc = 0.0, 0.0
            return acc, kappa, f1, cm, roc_auc, pr_auc

        return acc, kappa, f1, cm, None, None

    def extract_features(self, data_loader, max_samples=2000):
        """Extract backbone features for t-SNE visualization."""
        self.model.eval()
        all_features, all_labels = [], []
        count = 0

        for x, y in tqdm(data_loader, mininterval=1, desc="Extracting features", leave=False):
            if count >= max_samples:
                break
            x = x.cuda()
            with torch.no_grad():
                # Use backbone directly for feature extraction
                feats = self.model.backbone(x)
                feats_flat = feats.reshape(feats.size(0), -1)
            all_features.append(feats_flat.cpu().numpy())
            all_labels.append(y.numpy())
            count += x.size(0)

        features = np.concatenate(all_features, axis=0)[:max_samples]
        labels = np.concatenate(all_labels, axis=0)[:max_samples]
        return features, labels

    def generate_tsne_plots(self, epoch):
        """Generate t-SNE plots and log to wandb."""
        if not self.use_wandb:
            return

        max_samples = getattr(self.params, 'tsne_samples', 2000)
        print(f"  Generating t-SNE visualizations (max {max_samples} samples)...")
        tsne_images = {}

        for split_name, loader in [('train', self.data_loader['train']),
                                    ('val', self.data_loader['val']),
                                    ('test', self.data_loader['test'])]:
            try:
                features, labels = self.extract_features(loader, max_samples)
                fig = plot_tsne(
                    features, labels, self.label_names,
                    title=f"t-SNE - {split_name} (Epoch {epoch})",
                    max_samples=max_samples,
                )
                tsne_images[f"tsne/{split_name}"] = wandb.Image(fig)
                plt.close(fig)
            except Exception as e:
                print(f"  Warning: t-SNE failed for {split_name}: {e}")

        if tsne_images:
            wandb.log(tsne_images, step=epoch)

    def train(self):
        """Main training loop."""
        acc_best = -1
        best_epoch = 0

        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            correct, total = 0, 0

            pbar = tqdm(self.data_loader['train'], mininterval=10,
                        desc=f"Epoch {epoch + 1}/{self.params.epochs}")
            for x, y in pbar:
                self.optimizer.zero_grad()
                x, y = x.cuda(), y.cuda()

                logits = self.model(x)

                if self.task_type == 'binary':
                    loss = self.criterion(logits.squeeze(), y.float())
                    pred_y = (torch.sigmoid(logits.squeeze()) > 0.5).long()
                else:
                    loss = self.criterion(logits, y)
                    pred_y = torch.max(logits, dim=-1)[1]

                loss.backward()
                losses.append(loss.item())
                correct += (pred_y == y).sum().item()
                total += y.size(0)

                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.params.clip_value,
                    )

                self.optimizer.step()
                self.optimizer_scheduler.step()
                pbar.set_postfix({'loss': f'{np.mean(losses[-100:]):.4f}'})

            train_loss = np.mean(losses)
            train_acc = correct / total
            current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            epoch_time = (timer() - start_time) / 60

            # Eval
            with torch.no_grad():
                val_acc, val_kappa, val_f1, val_cm, val_roc, val_pr = self.evaluate(
                    self.data_loader['val'])
                test_acc, test_kappa, test_f1, test_cm, test_roc, test_pr = self.evaluate(
                    self.data_loader['test'])

            # Scale mixing weights
            scale_weights = self.model.get_scale_weights()

            # Logging
            log_dict = {
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/acc': train_acc,
                'val/balanced_acc': val_acc,
                'val/kappa': val_kappa,
                'val/f1': val_f1,
                'test/balanced_acc': test_acc,
                'test/kappa': test_kappa,
                'test/f1': test_f1,
                'learning_rate': current_lr,
                'epoch_time_min': epoch_time,
            }
            for i, w in enumerate(scale_weights):
                log_dict[f'scale_weight/{i}'] = w

            if self.task_type == 'binary':
                log_dict.update({
                    'val/roc_auc': val_roc, 'val/pr_auc': val_pr,
                    'test/roc_auc': test_roc, 'test/pr_auc': test_pr,
                })

            if self.use_wandb:
                wandb.log(log_dict)

            # t-SNE
            if self.params.tsne_interval > 0 and (epoch + 1) % self.params.tsne_interval == 0:
                self.generate_tsne_plots(epoch + 1)

            print(f"\nEpoch {epoch + 1}/{self.params.epochs}:")
            print(f"  Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.5f}")
            print(f"  Val Balanced Acc: {val_acc:.5f}, Kappa: {val_kappa:.5f}, F1: {val_f1:.5f}",
                  end='')
            if self.task_type == 'binary':
                print(f", ROC-AUC: {val_roc:.5f}, PR-AUC: {val_pr:.5f}")
            else:
                print()
            print(f"  Test Balanced Acc: {test_acc:.5f}, Kappa: {test_kappa:.5f}, F1: {test_f1:.5f}",
                  end='')
            if self.task_type == 'binary':
                print(f", ROC-AUC: {test_roc:.5f}, PR-AUC: {test_pr:.5f}")
            else:
                print()
            print(f"  Scale weights: {[f'{w:.3f}' for w in scale_weights]}")
            print(f"  LR: {current_lr:.6f}, Time: {epoch_time:.2f} mins")
            print(f"  Val Confusion Matrix:\n{val_cm}")

            # Save best
            if val_acc > acc_best:
                print(f"  Balanced Acc improved {acc_best:.5f} -> {val_acc:.5f}, saving...")
                acc_best = val_acc
                best_epoch = epoch + 1
                self.best_model_states = copy.deepcopy(self.model.state_dict())

        # Load best model
        if self.best_model_states is not None:
            self.model.load_state_dict(self.best_model_states)

        # Final test
        print("\n" + "=" * 60)
        print("Final Test Evaluation (Best Model)")
        print("=" * 60)

        with torch.no_grad():
            test_acc, test_kappa, test_f1, test_cm, test_roc, test_pr = self.evaluate(
                self.data_loader['test'])

        print(f"Test Results:")
        print(f"  Balanced Accuracy: {test_acc:.5f}")
        print(f"  Kappa: {test_kappa:.5f}")
        print(f"  F1: {test_f1:.5f}")
        if self.task_type == 'binary':
            print(f"  ROC-AUC: {test_roc:.5f}")
            print(f"  PR-AUC: {test_pr:.5f}")
        print(f"Confusion Matrix:\n{test_cm}")
        print(f"Scale weights: {self.model.get_scale_weights()}")

        if self.use_wandb:
            final_log = {
                'final_test/balanced_acc': test_acc,
                'final_test/kappa': test_kappa,
                'final_test/f1': test_f1,
                'best_epoch': best_epoch,
            }
            if self.task_type == 'binary':
                final_log['final_test/roc_auc'] = test_roc
                final_log['final_test/pr_auc'] = test_pr
            wandb.log(final_log)

            cm_table = wandb.Table(
                columns=[f"Pred_{self.label_names.get(i, str(i))}"
                         for i in range(test_cm.shape[1])],
                data=[[int(v) for v in row] for row in test_cm],
            )
            wandb.log({"final_test/confusion_matrix": cm_table})

        # Save
        if not os.path.isdir(self.params.model_dir):
            os.makedirs(self.params.model_dir)

        dataset_name = self.params.dataset.lower()
        model_path = os.path.join(
            self.params.model_dir,
            f"msft_{dataset_name}_scales{self.params.num_scales}"
            f"_epoch{best_epoch}_bal_acc_{test_acc:.5f}"
            f"_kappa_{test_kappa:.5f}_f1_{test_f1:.5f}.pth",
        )
        torch.save(self.model.state_dict(), model_path)
        print(f"\nModel saved to: {model_path}")

        return test_acc, test_kappa, test_f1


# ============================================================================
# Args
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='MSFT Fine-tuning with CodeBrain/CBraMod backbone')

    # Model selection
    parser.add_argument('--model', type=str, default='codebrain',
                        choices=['codebrain', 'cbramod'],
                        help='backbone model: codebrain (SSSM) or cbramod (CBraMod)')

    # Dataset
    parser.add_argument('--dataset', type=str, default='TUEV',
                        choices=list(DATASET_CONFIGS.keys()),
                        help='dataset to use')

    # Basic
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--cuda', type=int, default=0)

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate for trainable MSFT params (default higher than full FT)')
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--clip_value', type=float, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--val_ratio', type=float, default=0.15)

    # MSFT-specific
    parser.add_argument('--num_scales', type=int, default=3,
                        help='number of scales including original (e.g., 3 = [1x, 2x, 4x])')

    # Backbone architecture
    parser.add_argument('--n_layer', type=int, default=8,
                        help='number of layers: residual layers for SSSM (default 8), transformer layers for CBraMod (default 12)')
    parser.add_argument('--codebook_size_t', type=int, default=4096,
                        help='(SSSM only) temporal codebook size')
    parser.add_argument('--codebook_size_f', type=int, default=4096,
                        help='(SSSM only) frequency codebook size')
    parser.add_argument('--dim_feedforward', type=int, default=800,
                        help='(CBraMod only) feedforward dimension in transformer')
    parser.add_argument('--nhead', type=int, default=8,
                        help='(CBraMod only) number of attention heads')

    # Data
    parser.add_argument('--datasets_dir', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)

    # Label filtering (DIAGNOSIS dataset)
    parser.add_argument('--include_labels', type=int, nargs='+', default=None)
    parser.add_argument('--exclude_labels', type=int, nargs='+', default=None)

    # Cross-subject split
    parser.add_argument('--cross_subject', action='store_true', default=None)
    parser.add_argument('--no_cross_subject', action='store_false', dest='cross_subject')

    # t-SNE
    parser.add_argument('--tsne_interval', type=int, default=1)
    parser.add_argument('--tsne_samples', type=int, default=2000)

    # Weights
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help='path to pretrained weights (auto-selected if not specified)')
    parser.add_argument('--no_pretrained', action='store_true', default=False)

    # Output
    parser.add_argument('--model_dir', type=str, default=DEFAULT_OUTPUT_DIR)

    # WandB
    parser.add_argument('--no_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='eeg-msft')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================
def main():
    params = parse_args()

    if params.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {params.dataset}")

    dataset_config = DATASET_CONFIGS[params.dataset].copy()

    if params.datasets_dir:
        dataset_config['data_dir'] = params.datasets_dir

    # Handle DIAGNOSIS label filtering
    if params.dataset == 'DIAGNOSIS' and (params.include_labels or params.exclude_labels):
        original_label_names = dataset_config['label_names'].copy()
        if params.include_labels:
            kept_labels = set(params.include_labels)
        else:
            all_labels = set(original_label_names.keys())
            kept_labels = all_labels - set(params.exclude_labels or [])
        new_num_classes = len(kept_labels)
        dataset_config['num_classes'] = new_num_classes
        dataset_config['task_type'] = 'binary' if new_num_classes == 2 else 'multiclass'
        sorted_labels = sorted(kept_labels)
        dataset_config['label_names'] = {
            new_idx: original_label_names[orig_idx]
            for new_idx, orig_idx in enumerate(sorted_labels)
        }

    # Setup
    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)
    device = f'cuda:{params.cuda}'

    # Auto-select pretrained weights and default n_layer based on model type
    if params.model == 'cbramod':
        default_weights = DEFAULT_CBRAMOD_WEIGHTS
        default_n_layer = 12
        model_name = 'CBraMod'
    else:
        default_weights = DEFAULT_CODEBRAIN_WEIGHTS
        default_n_layer = 8
        model_name = 'CodeBrain (SSSM)'

    # Use default n_layer if not explicitly changed
    if params.n_layer == 8 and params.model == 'cbramod':
        params.n_layer = default_n_layer
    elif params.n_layer == 8 and params.model == 'codebrain':
        pass  # keep 8 as default for codebrain

    # Determine pretrained weights path
    pretrained_path = None
    if not params.no_pretrained:
        pretrained_path = params.pretrained_weights or default_weights

    print("=" * 60)
    print(f"MSFT Fine-tuning with {model_name} - {params.dataset}")
    print(f"  Scales: {params.num_scales} (downsample factors: "
          f"{[2**k for k in range(params.num_scales)]})")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: {model_name}, n_layer: {params.n_layer}")
    print(f"Pretrained weights: {pretrained_path}")
    print(f"Parameters: {params}")

    # WandB
    use_wandb = not params.no_wandb and WANDB_AVAILABLE
    if use_wandb:
        run_name = params.wandb_run_name or (
            f"MSFT_{params.model}_{params.dataset}_s{params.num_scales}"
            f"_bs{params.batch_size}_lr{params.lr}"
            f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        wandb.init(
            project=params.wandb_project,
            entity=params.wandb_entity,
            name=run_name,
            config=vars(params),
            tags=[params.dataset.lower(), 'msft', params.model, 'eeg'],
        )
        print(f"WandB initialized: {wandb.run.url}")
    else:
        print("WandB logging disabled")

    # Load data
    print("\nLoading data...")
    data_loader, num_classes, seq_len = load_data(params, dataset_config)

    # Create MSFT model
    print(f"\nCreating MSFT model with {model_name} backbone (num_scales={params.num_scales})...")
    n_channels = dataset_config.get('n_channels', 16)

    if params.model == 'cbramod':
        model = create_msft_cbramod_model(
            num_classes=num_classes,
            task_type=dataset_config['task_type'],
            n_channels=n_channels,
            seq_len=seq_len,
            patch_size=200,
            dropout=params.dropout,
            n_layer=params.n_layer,
            num_scales=params.num_scales,
            dim_feedforward=params.dim_feedforward,
            nhead=params.nhead,
            pretrained_weights_path=pretrained_path,
            device=device,
        )
    else:  # codebrain
        model = create_msft_model(
            num_classes=num_classes,
            task_type=dataset_config['task_type'],
            n_channels=n_channels,
            seq_len=seq_len,
            patch_size=200,
            dropout=params.dropout,
            n_layer=params.n_layer,
            num_scales=params.num_scales,
            codebook_size_t=params.codebook_size_t,
            codebook_size_f=params.codebook_size_f,
            pretrained_weights_path=pretrained_path,
            device=device,
        )
    print(f"Model created: {n_channels} channels, seq_len={seq_len}, "
          f"{params.num_scales} scales")

    # Log architecture
    if use_wandb:
        wandb.watch(model, log='gradients', log_freq=100)

    # Train
    print("\nStarting MSFT training...")
    trainer = MSFTTrainer(params, data_loader, model, dataset_config, use_wandb)
    test_acc, test_kappa, test_f1 = trainer.train()

    if use_wandb:
        wandb.finish()

    print("\n" + "=" * 60)
    print("MSFT Training Complete!")
    print(f"Final: Balanced Acc={test_acc:.5f}, Kappa={test_kappa:.5f}, F1={test_f1:.5f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
