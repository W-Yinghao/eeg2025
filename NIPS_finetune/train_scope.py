#!/usr/bin/env python3
"""
SCOPE Training Script: Structured COnfidence-aware Prototype-guided Adaptation

Two-stage training for EEG Foundation Model fine-tuning under limited labels.

Stage 1: External Structured Supervision Construction
    1a. Train Task-Prior Network (TPN) on labeled data with ETF regularization
    1b. Initialize & refine prototype bank via k-means + Sinkhorn-Knopp
    1c. Generate confidence-aware pseudo-labels for unlabeled data

Stage 2: Prototype-Conditioned Adaptation
    - Freeze backbone, train ProAdapters + classifier
    - Warm-up: first N epochs with labeled data only
    - Semi-supervised: combine labeled + pseudo-labeled data

Ablation studies (Table 2):
    Supervision:   --no_etf, --no_prototype, --no_supervision_construction
    ProAdapter:    --no_proadapter, --no_confidence_weights, --no_prototype_conditioning
    Training:      --no_warmup, --sequential_training, --two_stage_training

Usage:
    # Full SCOPE with CodeBrain
    python train_scope.py --dataset TUEV --model codebrain --label_ratio 0.3

    # Ablation: no ETF
    python train_scope.py --dataset TUEV --model codebrain --label_ratio 0.3 --no_etf

    # Ablation: no ProAdapter (self-training on frozen backbone)
    python train_scope.py --dataset TUEV --model codebrain --label_ratio 0.3 --no_proadapter
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
from torch.utils.data import DataLoader, Subset, ConcatDataset

try:
    from sklearn.metrics import (
        balanced_accuracy_score, f1_score, cohen_kappa_score,
        roc_auc_score, average_precision_score, confusion_matrix,
    )
    SKLEARN_METRICS = True
except (ImportError, AttributeError):
    SKLEARN_METRICS = False

    def balanced_accuracy_score(y_true, y_pred):
        """Fallback balanced accuracy."""
        classes = sorted(set(y_true))
        recalls = []
        for c in classes:
            mask = [yt == c for yt in y_true]
            if sum(mask) > 0:
                correct = sum(1 for m, yp in zip(mask, y_pred) if m and yp == c)
                recalls.append(correct / sum(mask))
        return sum(recalls) / max(len(recalls), 1)

    def f1_score(y_true, y_pred, average='weighted'):
        """Fallback weighted F1."""
        classes = sorted(set(y_true) | set(y_pred))
        n = len(y_true)
        f1s, weights = [], []
        for c in classes:
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            support = sum(1 for yt in y_true if yt == c)
            f1s.append(f1)
            weights.append(support)
        total = sum(weights)
        return sum(f * w for f, w in zip(f1s, weights)) / total if total > 0 else 0.0

    def cohen_kappa_score(y_true, y_pred):
        """Fallback Cohen's Kappa."""
        n = len(y_true)
        classes = sorted(set(y_true) | set(y_pred))
        po = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / n
        pe = sum(
            (sum(1 for yt in y_true if yt == c) / n) *
            (sum(1 for yp in y_pred if yp == c) / n)
            for c in classes
        )
        return (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

    def roc_auc_score(y_true, y_score):
        """Fallback AUC-ROC (binary only)."""
        pairs = sorted(zip(y_score, y_true), reverse=True)
        tp, fp, auc = 0, 0, 0.0
        pos = sum(1 for _, y in pairs if y == 1)
        neg = len(pairs) - pos
        for score, label in pairs:
            if label == 1:
                tp += 1
            else:
                fp += 1
                auc += tp
        return auc / (pos * neg) if (pos * neg) > 0 else 0.5

    def average_precision_score(y_true, y_score):
        """Fallback average precision (binary only)."""
        pairs = sorted(zip(y_score, y_true), reverse=True)
        tp, fp, ap = 0, 0, 0.0
        for score, label in pairs:
            if label == 1:
                tp += 1
                ap += tp / (tp + fp)
            else:
                fp += 1
        pos = sum(1 for _, y in pairs if y == 1)
        return ap / pos if pos > 0 else 0.0

    def confusion_matrix(y_true, y_pred):
        """Fallback confusion matrix."""
        classes = sorted(set(y_true) | set(y_pred))
        n = len(classes)
        cm = [[0] * n for _ in range(n)]
        c2i = {c: i for i, c in enumerate(classes)}
        for yt, yp in zip(y_true, y_pred):
            cm[c2i[yt]][c2i[yp]] += 1
        return cm

sys.path.insert(0, os.path.dirname(__file__))
from finetune_tuev_lmdb import DATASET_CONFIGS, load_data
from backbone_factory import create_backbone
from scope_framework import (
    TaskPriorNetwork, PrototypeBank, ConfidenceAwareFusion,
    ProAdapter, SCOPEModel,
    create_tpn, create_scope_model,
    get_tpn_config,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Data Splitting: Labeled / Unlabeled
# =============================================================================

def split_labeled_unlabeled(dataset, label_ratio: float = 0.3, seed: int = 42):
    """Split a dataset into labeled and unlabeled subsets.

    Args:
        dataset: PyTorch Dataset
        label_ratio: Fraction of samples to label (default 0.3)
        seed: Random seed for reproducibility

    Returns:
        labeled_indices: list of indices for labeled subset
        unlabeled_indices: list of indices for unlabeled subset
    """
    n = len(dataset)
    n_labeled = max(1, int(n * label_ratio))

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    labeled_indices = indices[:n_labeled].tolist()
    unlabeled_indices = indices[n_labeled:].tolist()

    print(f"  Split: {n_labeled} labeled ({label_ratio*100:.0f}%) + "
          f"{n - n_labeled} unlabeled ({(1-label_ratio)*100:.0f}%)")

    return labeled_indices, unlabeled_indices


# =============================================================================
# Stage 1: Task-Prior Network Training
# =============================================================================

def train_tpn(
    tpn: TaskPriorNetwork,
    train_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 5e-4,
    min_lr: float = 1e-6,
    weight_decay: float = 5e-3,
    lambda_etf: float = 0.1,
    label_smoothing: float = 0.05,
    use_etf: bool = True,
    device: str = 'cuda:0',
):
    """Train TPN on labeled data with optional ETF regularization.

    Args:
        tpn: Task-Prior Network
        train_loader: DataLoader for labeled data
        num_epochs: Training epochs
        lr: Learning rate
        lambda_etf: ETF loss weight (0.1 default)
        use_etf: Whether to use ETF (ablation toggle)
        device: Device string
    """
    tpn = tpn.to(device)
    optimizer = torch.optim.AdamW(tpn.parameters(), lr=lr, betas=(0.9, 0.999),
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=min_lr)

    num_classes = tpn.num_classes
    # TPN always outputs [B, num_classes] (including num_classes=2 for binary),
    # so use CrossEntropyLoss uniformly. BCE is only for SCOPEModel which outputs [B, 1].
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # For binary classification (K=2), ETF degenerates and provides no useful
    # geometric constraint. Use supervised contrastive loss instead to enforce
    # inter-class separation in the embedding space.
    use_contrastive_for_binary = (num_classes == 2) and use_etf
    if use_contrastive_for_binary:
        reg_label = "SupCon"
    elif use_etf and num_classes >= 3:
        reg_label = "ETF"
    else:
        reg_label = "OFF"

    print(f"\n{'='*60}")
    print(f"Stage 1a: Training Task-Prior Network")
    print(f"{'='*60}")
    print(f"  Epochs: {num_epochs}, LR: {lr}, Regularizer: {reg_label}")
    print(f"  λ_reg: {lambda_etf}")

    best_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        tpn.train()
        total_loss = 0.0
        total_ce = 0.0
        total_reg = 0.0
        n_batches = 0

        for batch in train_loader:
            data, labels = batch[0].to(device), batch[1].to(device)

            logits, embeddings = tpn(data)
            ce_loss = criterion(logits, labels)

            reg_loss = torch.tensor(0.0, device=device)
            if use_contrastive_for_binary:
                reg_loss = tpn.compute_supervised_contrastive_loss(embeddings, labels)
            elif use_etf and num_classes >= 3:
                reg_loss = tpn.compute_etf_loss()

            loss = ce_loss + lambda_etf * reg_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tpn.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_reg += reg_loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{num_epochs}: loss={avg_loss:.4f} "
                  f"ce={total_ce/max(n_batches,1):.4f} "
                  f"{reg_label.lower()}={total_reg/max(n_batches,1):.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    tpn.eval()
    print(f"  TPN training complete. Best loss: {best_loss:.4f}")
    return tpn


# =============================================================================
# Stage 1b: Prototype Learning
# =============================================================================

def train_prototypes(
    tpn: TaskPriorNetwork,
    prototype_bank: PrototypeBank,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-3,
    device: str = 'cuda:0',
):
    """Initialize and refine prototype bank.

    1. Extract TPN embeddings from unlabeled data
    2. Initialize prototypes via k-means
    3. Refine with cross-entropy on labeled data

    Args:
        tpn: Frozen Task-Prior Network
        prototype_bank: PrototypeBank module
        labeled_loader: DataLoader for labeled data
        unlabeled_loader: DataLoader for unlabeled data (or None)
    """
    print(f"\n{'='*60}")
    print(f"Stage 1b: Prototype Learning")
    print(f"{'='*60}")

    tpn.eval()
    prototype_bank = prototype_bank.to(device)

    # Extract embeddings from unlabeled data (or labeled if no unlabeled)
    all_embeddings = []
    all_labels = []
    source_loader = unlabeled_loader if unlabeled_loader is not None else labeled_loader

    with torch.no_grad():
        for batch in source_loader:
            data = batch[0].to(device)
            logits, z = tpn(data)
            pred = logits.argmax(dim=1)
            all_embeddings.append(z.cpu().numpy())
            all_labels.append(pred.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(f"  Extracted {len(all_embeddings)} embeddings for prototype init")

    # Initialize prototypes via k-means
    prototype_bank.initialize_from_embeddings(all_embeddings, all_labels)

    # Refine prototypes on labeled data
    optimizer = torch.optim.Adam([prototype_bank.prototypes], lr=lr)

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        n_batches = 0

        for batch in labeled_loader:
            data, labels = batch[0].to(device), batch[1].to(device)

            with torch.no_grad():
                _, z = tpn(data)

            loss = prototype_bank.compute_prototype_loss(z, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{num_epochs}: proto_loss={total_loss/max(n_batches,1):.4f}")

    print(f"  Prototype learning complete")
    return prototype_bank


# =============================================================================
# Stage 1c: Generate Pseudo-Labels
# =============================================================================

@torch.no_grad()
def generate_pseudo_labels(
    tpn: TaskPriorNetwork,
    prototype_bank: PrototypeBank,
    fusion: ConfidenceAwareFusion,
    dataloader: DataLoader,
    device: str = 'cuda:0',
):
    """Generate confidence-aware pseudo-labels for unlabeled data.

    Returns:
        pseudo_labels: (N,) tensor, -1 for rejected samples
        confidence: (N,) tensor
        proto_sims: (N, K) prototype similarity vectors
        acceptance_rate: fraction of accepted samples
    """
    tpn.eval()

    all_pseudo_labels = []
    all_confidence = []
    all_proto_sims = []
    n_total = 0
    n_accepted = 0

    for batch in dataloader:
        data = batch[0].to(device)

        # TPN prediction
        logits, z = tpn(data)

        # Prototype prediction
        _, class_sim = prototype_bank.predict(z)

        # Fuse predictions
        pseudo_labels, confidence, agreement = fusion.fuse(logits, class_sim)

        all_pseudo_labels.append(pseudo_labels.cpu())
        all_confidence.append(confidence.cpu())
        all_proto_sims.append(class_sim.cpu())

        n_total += len(pseudo_labels)
        n_accepted += (pseudo_labels >= 0).sum().item()

    pseudo_labels = torch.cat(all_pseudo_labels)
    confidence = torch.cat(all_confidence)
    proto_sims = torch.cat(all_proto_sims)

    acceptance_rate = n_accepted / max(n_total, 1)
    print(f"  Pseudo-labels: {n_accepted}/{n_total} accepted ({acceptance_rate*100:.1f}%)")
    print(f"  Confidence: mean={confidence.mean():.3f}, std={confidence.std():.3f}")

    return pseudo_labels, confidence, proto_sims, acceptance_rate


# =============================================================================
# Stage 2: ProAdapter Training
# =============================================================================

def train_stage2(
    model: SCOPEModel,
    labeled_loader: DataLoader,
    unlabeled_dataset,
    pseudo_labels: torch.Tensor,
    confidence: torch.Tensor,
    proto_sims: torch.Tensor,
    val_loader: DataLoader,
    num_classes: int,
    task_type: str = 'multiclass',
    epochs: int = 60,
    warmup_epochs: int = 10,
    lr: float = 1e-4,
    min_lr: float = 1e-6,
    weight_decay: float = 0.01,
    pseudo_ratio: float = 2.0,
    batch_size: int = 64,
    patience: int = 15,
    device: str = 'cuda:0',
    use_confidence_weights: bool = True,
    use_warmup: bool = True,
    sequential_training: bool = False,
    two_stage_training: bool = False,
    wandb_run=None,
):
    """Train ProAdapter + classifier (Stage 2).

    Args:
        model: SCOPEModel with frozen backbone
        labeled_loader: DataLoader for labeled data
        unlabeled_dataset: Dataset for unlabeled data
        pseudo_labels: (N_u,) pseudo-labels
        confidence: (N_u,) confidence scores
        proto_sims: (N_u, K) prototype similarity vectors
        val_loader: DataLoader for validation
        ... (training hyperparams)
    """
    model = model.to(device)
    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.999),
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=min_lr)

    if task_type == 'multiclass':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    print(f"\n{'='*60}")
    print(f"Stage 2: ProAdapter Training")
    print(f"{'='*60}")
    print(f"  Epochs: {epochs}, Warmup: {warmup_epochs if use_warmup else 0}")
    print(f"  LR: {lr}, Weight decay: {weight_decay}")
    print(f"  Pseudo ratio: {pseudo_ratio}")
    print(f"  Confidence weights: {'ON' if use_confidence_weights else 'OFF'}")

    # Build pseudo-labeled loader
    n_pseudo_per_batch = int(batch_size * pseudo_ratio)
    accepted_mask = pseudo_labels >= 0

    if accepted_mask.sum() > 0 and unlabeled_dataset is not None:
        accepted_indices = torch.where(accepted_mask)[0].tolist()
        accepted_pseudo = pseudo_labels[accepted_mask]
        accepted_conf = confidence[accepted_mask]
        accepted_sims = proto_sims[accepted_mask]
    else:
        accepted_indices = []

    best_metric = -float('inf')
    best_epoch = 0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t_start = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_sup = 0.0
        epoch_pseudo = 0.0
        n_batches = 0

        # Determine if we use unlabeled data this epoch
        use_unlabeled = not two_stage_training or epoch > epochs // 2
        if use_warmup and epoch <= warmup_epochs:
            use_unlabeled = False

        for batch in labeled_loader:
            data, labels = batch[0].to(device), batch[1].to(device)

            # Supervised loss
            logits = model(data, proto_sim=None)
            if task_type == 'binary':
                loss_sup = criterion(logits.squeeze(-1), labels.float())
            else:
                loss_sup = criterion(logits, labels)

            # Pseudo-supervised loss
            loss_pseudo = torch.tensor(0.0, device=device)
            if use_unlabeled and len(accepted_indices) > 0 and not sequential_training:
                # Sample pseudo-labeled data
                n_sample = min(n_pseudo_per_batch, len(accepted_indices))
                sample_idx = np.random.choice(len(accepted_indices), n_sample, replace=True)
                pseudo_data_list = []
                pseudo_label_list = []
                pseudo_conf_list = []
                pseudo_sim_list = []

                for idx in sample_idx:
                    real_idx = accepted_indices[idx]
                    sample = unlabeled_dataset[real_idx]
                    pseudo_data_list.append(torch.tensor(sample[0]) if not isinstance(sample[0], torch.Tensor) else sample[0])
                    pseudo_label_list.append(accepted_pseudo[idx])
                    pseudo_conf_list.append(accepted_conf[idx])
                    pseudo_sim_list.append(accepted_sims[idx])

                if pseudo_data_list:
                    pseudo_data = torch.stack(pseudo_data_list).to(device)
                    pseudo_lbl = torch.stack(pseudo_label_list).to(device)
                    pseudo_conf_batch = torch.stack(pseudo_conf_list).to(device)
                    pseudo_sim_batch = torch.stack(pseudo_sim_list).to(device)

                    pseudo_logits = model(pseudo_data, proto_sim=pseudo_sim_batch)

                    if task_type == 'binary':
                        per_sample_loss = F.binary_cross_entropy_with_logits(
                            pseudo_logits.squeeze(-1), pseudo_lbl.float(), reduction='none')
                    else:
                        per_sample_loss = F.cross_entropy(pseudo_logits, pseudo_lbl, reduction='none')

                    if use_confidence_weights:
                        loss_pseudo = (pseudo_conf_batch * per_sample_loss).mean()
                    else:
                        loss_pseudo = per_sample_loss.mean()

            loss = loss_sup + loss_pseudo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_sup += loss_sup.item()
            epoch_pseudo += loss_pseudo.item()
            n_batches += 1

        # Sequential training: process unlabeled data after labeled
        if sequential_training and use_unlabeled and len(accepted_indices) > 0:
            n_pseudo_batches = max(1, len(accepted_indices) // batch_size)
            for _ in range(n_pseudo_batches):
                n_sample = min(batch_size, len(accepted_indices))
                sample_idx = np.random.choice(len(accepted_indices), n_sample, replace=True)
                pseudo_data_list = []
                pseudo_label_list = []
                pseudo_conf_list = []
                pseudo_sim_list = []

                for idx in sample_idx:
                    real_idx = accepted_indices[idx]
                    sample = unlabeled_dataset[real_idx]
                    pseudo_data_list.append(torch.tensor(sample[0]) if not isinstance(sample[0], torch.Tensor) else sample[0])
                    pseudo_label_list.append(accepted_pseudo[idx])
                    pseudo_conf_list.append(accepted_conf[idx])
                    pseudo_sim_list.append(accepted_sims[idx])

                pseudo_data = torch.stack(pseudo_data_list).to(device)
                pseudo_lbl = torch.stack(pseudo_label_list).to(device)
                pseudo_conf_batch = torch.stack(pseudo_conf_list).to(device)
                pseudo_sim_batch = torch.stack(pseudo_sim_list).to(device)

                pseudo_logits = model(pseudo_data, proto_sim=pseudo_sim_batch)

                if task_type == 'binary':
                    per_sample_loss = F.binary_cross_entropy_with_logits(
                        pseudo_logits.squeeze(-1), pseudo_lbl.float(), reduction='none')
                else:
                    per_sample_loss = F.cross_entropy(pseudo_logits, pseudo_lbl, reduction='none')

                if use_confidence_weights:
                    loss_pseudo = (pseudo_conf_batch * per_sample_loss).mean()
                else:
                    loss_pseudo = per_sample_loss.mean()

                optimizer.zero_grad()
                loss_pseudo.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

                epoch_pseudo += loss_pseudo.item()
                n_batches += 1

        scheduler.step()
        t_elapsed = time.time() - t_start

        # Evaluate
        metrics = evaluate(model, val_loader, num_classes, task_type, device)

        if task_type == 'multiclass':
            primary_metric = metrics['kappa']
            metric_name = 'kappa'
        else:
            primary_metric = metrics['auroc']
            metric_name = 'auroc'

        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} "
                  f"sup={epoch_sup/max(n_batches,1):.4f} "
                  f"pseudo={epoch_pseudo/max(n_batches,1):.4f} | "
                  f"{metric_name}={primary_metric:.4f} | {t_elapsed:.1f}s")

        if wandb_run:
            wandb_run.log({
                'epoch': epoch,
                'stage2/loss': avg_loss,
                'stage2/loss_sup': epoch_sup / max(n_batches, 1),
                'stage2/loss_pseudo': epoch_pseudo / max(n_batches, 1),
                f'stage2/val_{metric_name}': primary_metric,
                'lr': optimizer.param_groups[0]['lr'],
            })

        # Early stopping
        if primary_metric > best_metric:
            best_metric = primary_metric
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()
                         if v.requires_grad or 'adapter' in k or 'classifier' in k}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
            break

    # Load best state
    if best_state is not None:
        current_state = model.state_dict()
        current_state.update(best_state)
        model.load_state_dict(current_state, strict=False)

    print(f"  Stage 2 complete. Best {metric_name}: {best_metric:.4f} at epoch {best_epoch}")
    return model, best_metric


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(model, dataloader, num_classes, task_type, device):
    """Evaluate model on a dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in dataloader:
        data, labels = batch[0].to(device), batch[1].to(device)
        logits = model(data, proto_sim=None)

        if task_type == 'binary':
            probs = torch.sigmoid(logits.squeeze(-1))
            preds = (probs > 0.5).long()
        else:
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Safety: truncate to actual dataset size in case of batch padding mismatch
    n_samples = len(dataloader.dataset)
    if len(all_preds) > n_samples:
        all_preds = all_preds[:n_samples]
        all_labels = all_labels[:n_samples]
        all_probs = all_probs[:n_samples]

    metrics = {}
    if task_type == 'multiclass':
        metrics['kappa'] = cohen_kappa_score(all_labels, all_preds)
        metrics['weighted_f1'] = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        metrics['balanced_acc'] = balanced_accuracy_score(all_labels, all_preds)
    else:
        metrics['auroc'] = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
        metrics['auprc'] = average_precision_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
        metrics['balanced_acc'] = balanced_accuracy_score(all_labels, all_preds)

    return metrics


# =============================================================================
# Main
# =============================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='SCOPE: Prototype-Guided EFM Adaptation')

    # Data
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--label_ratio', type=float, default=0.3,
                        help='Fraction of training data used as labeled (default 0.3)')

    # Backbone
    parser.add_argument('--model', type=str, default='codebrain',
                        choices=['codebrain', 'cbramod', 'femba', 'luna'])
    parser.add_argument('--pretrained_weights', type=str, default=None)
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--n_layer_cbramod', type=int, default=12)
    parser.add_argument('--dim_feedforward', type=int, default=800)
    parser.add_argument('--nhead', type=int, default=8)

    # TPN (Stage 1a)
    parser.add_argument('--tpn_epochs', type=int, default=50)
    parser.add_argument('--tpn_lr', type=float, default=5e-4)
    parser.add_argument('--lambda_etf', type=float, default=0.1,
                        help='ETF loss weight (default 0.1)')

    # Prototype (Stage 1b)
    parser.add_argument('--num_prototypes', type=int, default=3,
                        help='Prototypes per class M (default 3)')
    parser.add_argument('--proto_epochs', type=int, default=50)
    parser.add_argument('--proto_lr', type=float, default=1e-3)
    parser.add_argument('--proto_temperature', type=float, default=10.0)
    parser.add_argument('--sinkhorn_iters', type=int, default=3)
    parser.add_argument('--sinkhorn_epsilon', type=float, default=0.05)

    # Fusion (Stage 1c)
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Confidence threshold ρ (default 0.5)')

    # ProAdapter (Stage 2)
    parser.add_argument('--adapter_layers', type=int, default=3,
                        help='Number of ProAdapter layers L (default 3)')
    parser.add_argument('--lambda_proto', type=float, default=0.1,
                        help='Prototype modulation scaling')
    parser.add_argument('--lambda_scale', type=float, default=0.1,
                        help='Alpha scaling factor')
    parser.add_argument('--flatten_classifier', action='store_true', default=False,
                        help='Use flatten-based 3-layer MLP classifier instead of pooling-based compact classifier')

    # Training (Stage 2)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--pseudo_ratio', type=float, default=2.0,
                        help='Ratio of pseudo-labeled to labeled samples per batch')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=int, default=0)

    # Ablation flags (Table 2)
    # Supervision construction
    parser.add_argument('--no_etf', action='store_true',
                        help='Ablation: disable ETF guidance')
    parser.add_argument('--no_prototype', action='store_true',
                        help='Ablation: disable prototype clustering')
    parser.add_argument('--no_supervision_construction', action='store_true',
                        help='Ablation: disable entire Stage 1 (no pseudo-labels)')
    # ProAdapter design
    parser.add_argument('--no_proadapter', action='store_true',
                        help='Ablation: disable ProAdapter (self-training on frozen backbone)')
    parser.add_argument('--no_confidence_weights', action='store_true',
                        help='Ablation: disable confidence weighting')
    parser.add_argument('--no_prototype_conditioning', action='store_true',
                        help='Ablation: disable prototype conditioning in ProAdapter')
    # Training strategy
    parser.add_argument('--no_warmup', action='store_true',
                        help='Ablation: no warm-up phase')
    parser.add_argument('--sequential_training', action='store_true',
                        help='Ablation: sequential labeled→unlabeled per epoch')
    parser.add_argument('--two_stage_training', action='store_true',
                        help='Ablation: two-stage (supervised then unsupervised)')

    # Sensitivity analysis
    # (adapter_layers, confidence_threshold, num_prototypes, lambda_etf, pseudo_ratio
    #  are already parameterized above)

    # Logging
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_group', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='checkpoints_scope')

    return parser.parse_args(argv)


def main():
    args = parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'

    # Dataset config
    dataset_config = DATASET_CONFIGS[args.dataset].copy()
    num_classes = dataset_config['num_classes']
    task_type = dataset_config['task_type']
    n_channels = dataset_config['n_channels']
    sampling_rate = dataset_config.get('sampling_rate', 200)
    segment_duration = dataset_config.get('segment_duration', 5)
    patch_size = dataset_config.get('patch_size', 200)
    seq_len = int(sampling_rate * segment_duration / patch_size)
    chunk_size = sampling_rate * segment_duration  # T

    print(f"\n{'='*60}")
    print(f"SCOPE: Structured Prototype-Guided Adaptation")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset} ({task_type}, {num_classes} classes)")
    print(f"Channels: {n_channels}, Segment: {segment_duration}s, SeqLen: {seq_len}")
    print(f"Label ratio: {args.label_ratio}")
    print(f"Backbone: {args.model}")
    print(f"Device: {device}")

    # Ablation summary
    ablation_tags = []
    if args.no_etf: ablation_tags.append('no_etf')
    if args.no_prototype: ablation_tags.append('no_proto')
    if args.no_supervision_construction: ablation_tags.append('no_supcon')
    if args.no_proadapter: ablation_tags.append('no_adapter')
    if args.no_confidence_weights: ablation_tags.append('no_conf')
    if args.no_prototype_conditioning: ablation_tags.append('no_pcond')
    if args.no_warmup: ablation_tags.append('no_warmup')
    if args.sequential_training: ablation_tags.append('sequential')
    if args.two_stage_training: ablation_tags.append('twostage')
    if not ablation_tags:
        ablation_tags.append('full')
    print(f"Ablation: {', '.join(ablation_tags)}")

    # WandB
    wandb_run = None
    if args.wandb_project and WANDB_AVAILABLE:
        run_name = args.wandb_run_name or \
                   f"SCOPE_{args.dataset}_{args.model}_{'_'.join(ablation_tags)}"
        wandb_run = wandb.init(
            project=args.wandb_project, name=run_name,
            group=args.wandb_group,
            config=vars(args),
            tags=['scope', args.dataset.lower(), args.model] + ablation_tags,
        )

    # =========================================================================
    # Load Data
    # =========================================================================
    print(f"\nLoading data...")

    # Create a simple params-like object for load_data
    class Params:
        pass
    params = Params()
    params.dataset = args.dataset
    params.batch_size = args.batch_size
    params.seed = args.seed
    params.cuda = args.cuda
    params.datasets_dir = None
    params.include_labels = None
    params.exclude_labels = None
    params.val_ratio = 0.15
    params.num_workers = 0

    # When val is split from train (e.g. TUEV), the val set shares the same
    # patient sessions as training data, causing an optimistic validation signal.
    # Fix: if val comes from train, use train fully and split eval into val+test.
    val_from_train = (dataset_config['splits']['train'] == dataset_config['splits']['val'])
    if val_from_train:
        # Load train without carving out val (use val_ratio=0 → all data is "train")
        params.val_ratio = 0.0
    data_loader, _, _ = load_data(params, dataset_config)

    train_dataset = data_loader['train'].dataset

    if val_from_train:
        # Split the eval/test set into val (30%) + test (70%) for independent validation
        test_full_dataset = data_loader['test'].dataset
        n_total = len(test_full_dataset)
        n_val = int(n_total * 0.3)
        indices = list(range(n_total))
        np.random.seed(args.seed)
        np.random.shuffle(indices)
        val_indices = indices[:n_val]
        test_indices = indices[n_val:]
        val_dataset = Subset(test_full_dataset, val_indices)
        test_dataset = Subset(test_full_dataset, test_indices)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0, pin_memory=True)
        print(f"  Val/test split from eval: {n_val} val + {n_total - n_val} test (independent)")
    else:
        val_loader = data_loader['val']
        test_loader = data_loader['test']

    # Split training data into labeled / unlabeled
    labeled_indices, unlabeled_indices = split_labeled_unlabeled(
        train_dataset, args.label_ratio, args.seed)

    labeled_dataset = Subset(train_dataset, labeled_indices)
    unlabeled_dataset = Subset(train_dataset, unlabeled_indices) if unlabeled_indices else None

    labeled_loader = DataLoader(labeled_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=0, pin_memory=True,
                                drop_last=len(labeled_dataset) > args.batch_size)
    unlabeled_loader = None
    if unlabeled_dataset is not None and len(unlabeled_dataset) > 0:
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=0, pin_memory=True)

    print(f"  Labeled: {len(labeled_dataset)} samples")
    print(f"  Unlabeled: {len(unlabeled_dataset) if unlabeled_dataset else 0} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")

    # =========================================================================
    # Stage 1: External Structured Supervision Construction
    # =========================================================================
    pseudo_labels = None
    confidence = None
    proto_sims = None

    if not args.no_supervision_construction:
        # Stage 1a: Train TPN
        tpn = create_tpn(
            n_channels=n_channels,
            chunk_size=chunk_size,
            num_classes=num_classes,
            dataset_name=args.dataset,
            dropout=args.dropout,
        )
        tpn = train_tpn(
            tpn, labeled_loader,
            num_epochs=args.tpn_epochs,
            lr=args.tpn_lr,
            lambda_etf=args.lambda_etf if not args.no_etf else 0.0,
            use_etf=not args.no_etf,
            device=device,
        )

        # Stage 1b: Prototype Learning
        if not args.no_prototype:
            prototype_bank = PrototypeBank(
                feature_dim=tpn.feature_dim,
                num_classes=num_classes,
                num_prototypes_per_class=args.num_prototypes,
                temperature=args.proto_temperature,
                sinkhorn_iters=args.sinkhorn_iters,
                sinkhorn_epsilon=args.sinkhorn_epsilon,
            )
            prototype_bank = train_prototypes(
                tpn, prototype_bank, labeled_loader, unlabeled_loader,
                num_epochs=args.proto_epochs, lr=args.proto_lr, device=device,
            )

            # Stage 1c: Generate pseudo-labels
            if unlabeled_loader is not None:
                fusion = ConfidenceAwareFusion(
                    num_classes=num_classes,
                    confidence_threshold=args.confidence_threshold,
                )
                pseudo_labels, confidence, proto_sims, acc_rate = generate_pseudo_labels(
                    tpn, prototype_bank, fusion, unlabeled_loader, device=device,
                )

                if wandb_run:
                    wandb_run.log({
                        'stage1/pseudo_acceptance_rate': acc_rate,
                        'stage1/confidence_mean': confidence.mean().item(),
                    })
        else:
            # w/o Prototype: use only TPN predictions as pseudo-labels
            if unlabeled_loader is not None:
                print("\n  w/o Prototype: using TPN predictions only")
                tpn.eval()
                all_pseudo = []
                all_conf = []
                all_sims = []
                with torch.no_grad():
                    for batch in unlabeled_loader:
                        data = batch[0].to(device)
                        logits, _ = tpn(data)
                        probs = F.softmax(logits, dim=1)
                        preds = probs.argmax(dim=1)
                        max_prob = probs.max(dim=1).values
                        # Use max prob as confidence
                        all_pseudo.append(preds.cpu())
                        all_conf.append(max_prob.cpu())
                        all_sims.append(probs.cpu())

                pseudo_labels = torch.cat(all_pseudo)
                confidence = torch.cat(all_conf)
                proto_sims = torch.cat(all_sims)

                # Filter by threshold
                low_conf = confidence < args.confidence_threshold
                pseudo_labels[low_conf] = -1
                n_accepted = (pseudo_labels >= 0).sum().item()
                print(f"  TPN pseudo-labels: {n_accepted}/{len(pseudo_labels)} accepted")

    # =========================================================================
    # Stage 2: Prototype-Conditioned Adaptation
    # =========================================================================

    # Create backbone
    pretrained_path = args.pretrained_weights
    if pretrained_path is None:
        if args.model == 'codebrain':
            default_path = os.path.join(os.path.dirname(__file__),
                                        'CodeBrain', 'Checkpoints', 'CodeBrain.pth')
            if os.path.exists(default_path):
                pretrained_path = default_path
        elif args.model == 'cbramod':
            default_path = os.path.join(os.path.dirname(__file__),
                                        'Cbramod_pretrained_weights.pth')
            if os.path.exists(default_path):
                pretrained_path = default_path

    backbone, backbone_out_dim, token_dim = create_backbone(
        model_type=args.model,
        n_channels=n_channels,
        seq_len=seq_len,
        patch_size=patch_size,
        n_layer=args.n_layer,
        n_layer_cbramod=args.n_layer_cbramod,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        pretrained_weights_path=pretrained_path,
        device=device,
    )

    # Create SCOPE model
    if args.no_proadapter:
        # w/o ProAdapter: just frozen backbone + classifier (no adapters)
        adapter_layers = 0
    else:
        adapter_layers = args.adapter_layers

    model = create_scope_model(
        backbone=backbone,
        num_classes=num_classes,
        backbone_out_dim=backbone_out_dim,
        token_dim=token_dim,
        n_channels=n_channels,
        seq_len=seq_len,
        adapter_layers=adapter_layers,
        dropout=args.dropout,
        use_prototype_conditioning=not args.no_prototype_conditioning,
        lambda_proto=args.lambda_proto,
        lambda_scale=args.lambda_scale,
        pooling_classifier=not args.flatten_classifier,
    )

    # Train Stage 2
    model, best_val_metric = train_stage2(
        model=model,
        labeled_loader=labeled_loader,
        unlabeled_dataset=unlabeled_dataset,
        pseudo_labels=pseudo_labels if pseudo_labels is not None else torch.tensor([]),
        confidence=confidence if confidence is not None else torch.tensor([]),
        proto_sims=proto_sims if proto_sims is not None else torch.tensor([]),
        val_loader=val_loader,
        num_classes=num_classes,
        task_type=task_type,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        pseudo_ratio=args.pseudo_ratio,
        batch_size=args.batch_size,
        patience=args.patience,
        device=device,
        use_confidence_weights=not args.no_confidence_weights,
        use_warmup=not args.no_warmup,
        sequential_training=args.sequential_training,
        two_stage_training=args.two_stage_training,
        wandb_run=wandb_run,
    )

    # =========================================================================
    # Test Evaluation
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Test Evaluation")
    print(f"{'='*60}")

    test_metrics = evaluate(model, test_loader, num_classes, task_type, device)

    if task_type == 'multiclass':
        print(f"  Kappa:       {test_metrics['kappa']:.4f}")
        print(f"  Weighted F1: {test_metrics['weighted_f1']:.4f}")
        print(f"  Balanced Acc: {test_metrics['balanced_acc']:.4f}")
    else:
        print(f"  AUROC: {test_metrics['auroc']:.4f}")
        print(f"  AUPRC: {test_metrics['auprc']:.4f}")
        print(f"  Balanced Acc: {test_metrics['balanced_acc']:.4f}")

    if wandb_run:
        for k, v in test_metrics.items():
            wandb_run.log({f'test/{k}': v})

    # Save checkpoint
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ablation_str = '_'.join(ablation_tags)
    ckpt_path = save_dir / f'scope_{args.dataset}_{args.model}_{ablation_str}.pth'
    torch.save({
        'model_state_dict': {k: v for k, v in model.state_dict().items()
                             if 'backbone' not in k},
        'test_metrics': test_metrics,
        'args': vars(args),
    }, ckpt_path)
    print(f"  Saved: {ckpt_path}")

    print(f"\n{'='*60}")
    print(f"SCOPE Complete!")
    print(f"{'='*60}")

    if wandb_run:
        wandb_run.finish()


if __name__ == '__main__':
    main()
