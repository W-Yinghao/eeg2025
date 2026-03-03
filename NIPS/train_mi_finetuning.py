#!/usr/bin/env python3
"""
Training Script for Information-Theoretic Fine-Tuning with CodeBrain Backbone

Uses frozen CodeBrain (SSSM) backbone + trainable MI fine-tuning head:
  - VIB for noise suppression
  - InfoNCE for expert feature alignment (PSD / statistical)

Usage:
    python train_mi_finetuning.py --dataset TUEV --cuda 0
    python train_mi_finetuning.py --dataset TUAB --alpha 1.0 --beta 1e-3 --expert_feature psd
    python train_mi_finetuning.py --dataset TUEV --alpha 0 --beta 0  # baseline (CE only)
"""

import argparse
import copy
import os
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)

# Existing data infrastructure
from finetune_tuev_lmdb import (
    DATASET_CONFIGS,
    load_data,
    setup_seed,
)

# MI framework
from mi_finetuning_framework import (
    create_codebrain_backbone,
    MIFineTuner,
    calculate_mi_loss,
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed")

DEFAULT_CODEBRAIN_WEIGHTS = (
    "/home/infres/yinwang/eeg2025/NIPS/CodeBrain/Checkpoints/CodeBrain.pth"
)


# ==============================================================================
# Expert Feature Extraction (operates on 4D EEG data)
# ==============================================================================

class ExpertFeatureExtractor:
    """
    Extract expert features from 4D EEG data (B, C, S, P).

    The 4D patch format is first reshaped to a raw time series (B, C, S*P)
    before computing spectral or statistical features.

    Supported:
      - 'psd': Power Spectral Density in 5 standard EEG bands (dim = C * 5)
      - 'stats': Mean / std / min / max per channel (dim = C * 4)
      - 'both': Concatenation of PSD + stats (dim = C * 9)
    """

    def __init__(self, n_channels: int, sampling_rate: int, feature_type: str = 'psd'):
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.feature_type = feature_type

    def _to_time_series(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, C, S, P) -> (B, C, S*P) raw time series."""
        if x.ndim == 4:
            B, C, S, P = x.shape
            return x.reshape(B, C, S * P)
        return x  # already 3D

    def extract_psd_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        PSD in standard EEG frequency bands via FFT.

        Bands: delta(0.5-4), theta(4-8), alpha(8-13), beta(13-30), gamma(30-50)
        Output: (B, C * 5), log1p-transformed.
        """
        x = self._to_time_series(x)
        B, C, T = x.shape

        fft = torch.fft.rfft(x, dim=-1)
        power = torch.abs(fft) ** 2
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.sampling_rate)

        bands = [
            (0.5, 4),   # delta
            (4, 8),     # theta
            (8, 13),    # alpha
            (13, 30),   # beta
            (30, 50),   # gamma
        ]

        parts = []
        for low, high in bands:
            mask = (freqs >= low) & (freqs < high)
            if mask.sum() == 0:
                # Not enough freq resolution for this band
                parts.append(torch.zeros(B, C, device=x.device))
            else:
                parts.append(power[:, :, mask].mean(dim=-1))

        psd = torch.cat(parts, dim=-1)  # (B, C*5)
        return torch.log1p(psd)

    def extract_stat_features(self, x: torch.Tensor) -> torch.Tensor:
        """Mean, std, min, max per channel. Output: (B, C * 4)."""
        x = self._to_time_series(x)
        return torch.cat([
            x.mean(dim=-1),
            x.std(dim=-1),
            x.min(dim=-1)[0],
            x.max(dim=-1)[0],
        ], dim=-1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_type == 'psd':
            return self.extract_psd_features(x)
        elif self.feature_type == 'stats':
            return self.extract_stat_features(x)
        elif self.feature_type == 'both':
            return torch.cat([
                self.extract_psd_features(x),
                self.extract_stat_features(x),
            ], dim=-1)
        raise ValueError(f"Unknown feature type: {self.feature_type}")

    def get_dim(self, n_channels: int) -> int:
        """Return output dimension given number of channels."""
        n_bands = 5
        if self.feature_type == 'psd':
            return n_channels * n_bands
        elif self.feature_type == 'stats':
            return n_channels * 4
        elif self.feature_type == 'both':
            return n_channels * (n_bands + 4)
        raise ValueError(f"Unknown feature type: {self.feature_type}")


# ==============================================================================
# Trainer
# ==============================================================================

class MITrainer:
    """Trainer for MI Fine-Tuning Framework with CodeBrain backbone."""

    def __init__(
        self,
        model: MIFineTuner,
        data_loader: Dict[str, DataLoader],
        expert_extractor: ExpertFeatureExtractor,
        dataset_config: Dict,
        params: argparse.Namespace,
        use_wandb: bool = True,
    ):
        self.model = model.cuda()
        self.data_loader = data_loader
        self.expert_extractor = expert_extractor
        self.dataset_config = dataset_config
        self.params = params
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.task_type = dataset_config['task_type']
        self.best_model_state = None

        # Optimizer: only trainable params
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=params.lr,
            weight_decay=params.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=params.epochs * len(data_loader['train']),
            eta_min=1e-6,
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        losses, ce_l, vib_l, nce_l = [], [], [], []
        correct, total = 0, 0

        pbar = tqdm(
            self.data_loader['train'],
            desc=f"Epoch {epoch + 1}/{self.params.epochs}",
            mininterval=10,
        )
        for x, y in pbar:
            x, y = x.cuda(), y.cuda()

            # Expert features (deterministic, no grad needed)
            with torch.no_grad():
                x_expert = self.expert_extractor(x).cuda()

            self.optimizer.zero_grad()
            logits, mu, log_var, z_fm_proj, z_expert_proj = self.model(x, x_expert)

            loss, ld = calculate_mi_loss(
                logits, y, mu, log_var, z_fm_proj, z_expert_proj,
                alpha=self.params.alpha,
                beta=self.params.beta,
                temperature=self.params.temperature,
                task_type=self.task_type,
            )

            loss.backward()
            if self.params.clip_value > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.params.clip_value,
                )
            self.optimizer.step()
            self.scheduler.step()

            losses.append(ld['total'])
            ce_l.append(ld['ce'])
            vib_l.append(ld['vib'])
            nce_l.append(ld['infonce'])

            if self.task_type == 'binary':
                pred = (torch.sigmoid(logits.squeeze()) > 0.5).long()
            else:
                pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            pbar.set_postfix(loss=f"{np.mean(losses[-100:]):.4f}", acc=f"{correct / total:.4f}")

        return {
            'train/loss': np.mean(losses),
            'train/acc': correct / total,
            'train/ce': np.mean(ce_l),
            'train/vib': np.mean(vib_l),
            'train/infonce': np.mean(nce_l),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }

    @torch.no_grad()
    def evaluate(self, split: str = 'val') -> Dict[str, float]:
        self.model.eval()
        truths, preds, probs = [], [], []

        for x, y in tqdm(self.data_loader[split], desc=f"Eval {split}", leave=False):
            x, y = x.cuda(), y.cuda()
            x_expert = self.expert_extractor(x).cuda()

            logits, _, _, _, _ = self.model(x, x_expert)

            if self.task_type == 'binary':
                prob = torch.sigmoid(logits.squeeze())
                pred = (prob > 0.5).long()
                if prob.ndim == 0:
                    probs.append(prob.cpu().item())
                else:
                    probs.extend(prob.cpu().numpy().tolist())
            else:
                pred = logits.argmax(dim=-1)

            if y.ndim == 0:
                truths.append(y.cpu().item())
                preds.append(pred.cpu().item())
            else:
                truths.extend(y.cpu().numpy().tolist())
                preds.extend(pred.cpu().numpy().tolist())

        truths = np.array(truths)
        preds = np.array(preds)

        acc = balanced_accuracy_score(truths, preds)
        kappa = cohen_kappa_score(truths, preds)
        f1 = f1_score(truths, preds, average='binary' if self.task_type == 'binary' else 'weighted')

        metrics = {
            f'{split}/balanced_acc': acc,
            f'{split}/kappa': kappa,
            f'{split}/f1': f1,
        }

        if self.task_type == 'binary' and probs:
            try:
                metrics[f'{split}/roc_auc'] = roc_auc_score(truths, np.array(probs))
            except Exception:
                pass

        return metrics

    def train(self) -> Tuple[float, float, float]:
        best_val_acc = -1
        best_epoch = 0

        for epoch in range(self.params.epochs):
            train_m = self.train_epoch(epoch)
            val_m = self.evaluate('val')
            test_m = self.evaluate('test')

            all_m = {**train_m, **val_m, **test_m, 'epoch': epoch + 1}
            if self.use_wandb:
                wandb.log(all_m)

            print(f"\nEpoch {epoch + 1}/{self.params.epochs}:")
            print(f"  Train Loss: {train_m['train/loss']:.5f}  "
                  f"CE: {train_m['train/ce']:.5f}  "
                  f"VIB: {train_m['train/vib']:.5f}  "
                  f"NCE: {train_m['train/infonce']:.5f}")
            print(f"  Val  BalAcc: {val_m['val/balanced_acc']:.5f}  "
                  f"Kappa: {val_m['val/kappa']:.5f}  "
                  f"F1: {val_m['val/f1']:.5f}")
            print(f"  Test BalAcc: {test_m['test/balanced_acc']:.5f}  "
                  f"Kappa: {test_m['test/kappa']:.5f}  "
                  f"F1: {test_m['test/f1']:.5f}")

            if val_m['val/balanced_acc'] > best_val_acc:
                best_val_acc = val_m['val/balanced_acc']
                best_epoch = epoch + 1
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                print(f"  -> New best model (val_acc={best_val_acc:.5f})")

        # Final eval with best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        final = self.evaluate('test')
        print(f"\nFinal Test (best epoch {best_epoch}):")
        print(f"  BalAcc: {final['test/balanced_acc']:.5f}  "
              f"Kappa: {final['test/kappa']:.5f}  "
              f"F1: {final['test/f1']:.5f}")

        self._save_model(final, best_epoch)

        if self.use_wandb:
            wandb.log({
                'final_test/balanced_acc': final['test/balanced_acc'],
                'final_test/kappa': final['test/kappa'],
                'final_test/f1': final['test/f1'],
                'best_epoch': best_epoch,
            })

        return final['test/balanced_acc'], final['test/kappa'], final['test/f1']

    def _save_model(self, metrics: Dict, epoch: int):
        os.makedirs(self.params.model_dir, exist_ok=True)
        path = os.path.join(
            self.params.model_dir,
            f"mi_codebrain_{self.params.dataset.lower()}_"
            f"a{self.params.alpha}_b{self.params.beta}_"
            f"epoch{epoch}_acc{metrics['test/balanced_acc']:.4f}.pth",
        )
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'params': vars(self.params),
            'metrics': metrics,
        }, path)
        print(f"Model saved: {path}")


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='MI Fine-Tuning with CodeBrain')

    # Dataset
    p.add_argument('--dataset', type=str, default='TUEV', choices=list(DATASET_CONFIGS.keys()))
    p.add_argument('--datasets_dir', type=str, default=None)

    # MI hyperparams
    p.add_argument('--alpha', type=float, default=1.0, help='InfoNCE weight')
    p.add_argument('--beta', type=float, default=1e-3, help='VIB weight')
    p.add_argument('--temperature', type=float, default=0.07, help='InfoNCE temperature')
    p.add_argument('--vib_dim', type=int, default=128, help='VIB bottleneck dim')
    p.add_argument('--hidden_dim', type=int, default=256, help='Internal representation dim')
    p.add_argument('--expert_feature', type=str, default='psd', choices=['psd', 'stats', 'both'])

    # CodeBrain backbone
    p.add_argument('--pretrained_weights', type=str, default=DEFAULT_CODEBRAIN_WEIGHTS)
    p.add_argument('--n_layer', type=int, default=8, help='SSSM residual layers')
    p.add_argument('--codebook_size_t', type=int, default=4096)
    p.add_argument('--codebook_size_f', type=int, default=4096)

    # Training
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--clip_value', type=float, default=1.0)
    p.add_argument('--val_ratio', type=float, default=0.15)
    p.add_argument('--label_smoothing', type=float, default=0.0)

    # System
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--cuda', type=int, default=0)
    p.add_argument('--num_workers', type=int, default=4)

    # Output
    p.add_argument('--model_dir', type=str, default='./checkpoints_mi')
    p.add_argument('--no_wandb', action='store_true', default=False)
    p.add_argument('--wandb_project', type=str, default='eeg-mi-finetuning')
    p.add_argument('--wandb_run_name', type=str, default=None)
    p.add_argument('--wandb_entity', type=str, default=None)

    return p.parse_args()


def main():
    params = parse_args()

    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)
    device = f'cuda:{params.cuda}'

    print("=" * 70)
    print("MI Fine-Tuning with CodeBrain (SSSM) Backbone")
    print("=" * 70)
    print(f"  Dataset:    {params.dataset}")
    print(f"  alpha(NCE): {params.alpha}  beta(VIB): {params.beta}  tau: {params.temperature}")
    print(f"  Expert:     {params.expert_feature}")
    print(f"  VIB dim:    {params.vib_dim}  Hidden dim: {params.hidden_dim}")
    print("=" * 70)

    # Dataset config
    dataset_config = DATASET_CONFIGS[params.dataset].copy()
    if params.datasets_dir:
        dataset_config['data_dir'] = params.datasets_dir

    n_channels = dataset_config.get('n_channels', 16)
    sampling_rate = dataset_config.get('sampling_rate', 200)

    # WandB
    use_wandb = not params.no_wandb and WANDB_AVAILABLE
    if use_wandb:
        run_name = params.wandb_run_name or (
            f"MI_codebrain_{params.dataset}_"
            f"a{params.alpha}_b{params.beta}_{params.expert_feature}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        wandb.init(
            project=params.wandb_project,
            entity=params.wandb_entity,
            name=run_name,
            config=vars(params),
            tags=[params.dataset.lower(), 'codebrain', 'mi-finetuning', params.expert_feature],
        )

    # Load data
    print("\nLoading data...")
    data_loader, num_classes, seq_len = load_data(params, dataset_config)
    print(f"  Classes: {num_classes}, seq_len: {seq_len}, n_channels: {n_channels}")

    # Expert feature extractor
    expert_extractor = ExpertFeatureExtractor(n_channels, sampling_rate, params.expert_feature)
    expert_dim = expert_extractor.get_dim(n_channels)
    print(f"  Expert features: {params.expert_feature}, dim={expert_dim}")

    # Create CodeBrain backbone
    print("\nCreating CodeBrain backbone...")
    backbone, backbone_out_dim = create_codebrain_backbone(
        n_channels=n_channels,
        seq_len=seq_len,
        patch_size=200,
        n_layer=params.n_layer,
        codebook_size_t=params.codebook_size_t,
        codebook_size_f=params.codebook_size_f,
        dropout=0.1,
        pretrained_weights_path=params.pretrained_weights,
        device=device,
    )
    print(f"  Backbone output dim (flat): {backbone_out_dim}")

    # Create MI fine-tuner
    print("\nCreating MIFineTuner...")
    output_classes = 1 if dataset_config['task_type'] == 'binary' else num_classes
    model = MIFineTuner(
        backbone=backbone,
        backbone_out_dim=backbone_out_dim,
        expert_dim=expert_dim,
        hidden_dim=params.hidden_dim,
        vib_dim=params.vib_dim,
        num_classes=output_classes,
        dropout=0.1,
    )

    # Train
    print("\nStarting MI fine-tuning...")
    trainer = MITrainer(model, data_loader, expert_extractor, dataset_config, params, use_wandb)
    test_acc, test_kappa, test_f1 = trainer.train()

    if use_wandb:
        wandb.finish()

    print("\n" + "=" * 70)
    print(f"Done! BalAcc={test_acc:.5f}, Kappa={test_kappa:.5f}, F1={test_f1:.5f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
