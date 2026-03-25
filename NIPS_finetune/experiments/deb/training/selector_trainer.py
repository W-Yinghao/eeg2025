"""
Training loop for selector experiments (Exp 6/7).

Extends the original DEBTrainer with:
  - True partial FT regime logging
  - Gate sparsity tracking
  - Consistency training with augmented views
  - Class-wise metrics (abnormal recall/F1)
  - JSON summary export
"""

import os
import sys
import json
import time
import copy
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, recall_score,
    precision_score, confusion_matrix, classification_report,
)

from ..evaluation.evaluator import Evaluator
from .selector_loss import SelectorLoss
from .augmentations import EEGAugmentor

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SelectorTrainer:
    """
    Trainer for selector experiments (Exp 6/7).

    Supports:
      - baseline and selector modes
      - sparse and consistency regularization
      - per-class metrics with abnormal focus
      - structured JSON result export
    """

    def __init__(self, model, cfg: dict, data_loaders: dict,
                 dataset_config: dict, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.data_loaders = data_loaders
        self.dataset_config = dataset_config
        self.device = device
        self.mode = model.mode
        self.label_names = dataset_config.get('label_names', {})
        self.num_classes = cfg.get('num_classes', len(self.label_names))

        # Loss
        cw = dataset_config.get('class_weights')
        if cw is None:
            cw = cfg.get('class_weights')
        class_weights = torch.tensor(cw, dtype=torch.float32, device=device) if cw else None

        self.criterion = SelectorLoss(
            enable_sparse=cfg.get('enable_sparse', False),
            sparse_lambda=cfg.get('sparse_lambda', 1e-3),
            sparse_type=cfg.get('sparse_type', 'l1'),
            enable_consistency=cfg.get('enable_consistency', False),
            consistency_lambda=cfg.get('consistency_lambda', 1e-2),
            consistency_type=cfg.get('consistency_type', 'l2'),
            enable_vib=cfg.get('enable_vib', False),
            vib_beta=cfg.get('vib_beta', 1e-4),
            vib_warmup_epochs=cfg.get('vib_warmup_epochs', 5),
            class_weights=class_weights,
            label_smoothing=cfg.get('label_smoothing', 0.0),
        )

        # Augmentor for consistency training
        if cfg.get('enable_consistency', False):
            self.augmentor = EEGAugmentor(
                enable_time_shift=cfg.get('aug_time_shift', True),
                enable_amplitude_jitter=cfg.get('aug_amplitude_jitter', True),
                enable_time_mask=cfg.get('aug_time_mask', True),
                time_shift_max=cfg.get('aug_time_shift_max', 1),
                jitter_std=cfg.get('aug_jitter_std', 0.05),
                mask_ratio=cfg.get('aug_mask_ratio', 0.1),
            )
        else:
            self.augmentor = None

        # Optimizer
        param_groups = model.get_param_groups(cfg)
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=cfg.get('weight_decay', 1e-3),
        )

        # Scheduler
        if cfg.get('scheduler', 'cosine') == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg['epochs']
            )
        else:
            self.scheduler = None

        # Tracking
        self.best_val_metric = 0.0
        self.best_epoch = 0
        self.best_model_state = None
        self.train_history = []

        # WandB
        self.wandb_run = None
        if cfg.get('wandb_project') and WANDB_AVAILABLE:
            run_name = cfg.get('wandb_run_name') or (
                f"{cfg['mode']}_{cfg.get('regime','frozen')}"
                f"_{cfg['dataset']}_{cfg.get('model','codebrain')}"
                f"_s{cfg['seed']}"
            )
            self.wandb_run = wandb.init(
                project=cfg['wandb_project'],
                name=run_name,
                config=cfg,
            )

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[SelectorTrainer] mode={self.mode} regime={cfg.get('regime','frozen')} "
              f"trainable={n_trainable:,}")

    def train(self):
        """Run full training loop."""
        cfg = self.cfg
        epochs = cfg['epochs']
        patience = cfg.get('patience', 15)
        patience_counter = 0

        print(f"\n{'='*70}")
        print(f"  Selector Training: {self.mode} | regime={cfg.get('regime','frozen')}")
        print(f"  Dataset: {cfg['dataset']} | Model: {cfg.get('model','codebrain')}")
        print(f"  Epochs: {epochs} | Patience: {patience}")
        if cfg.get('enable_sparse'):
            print(f"  Sparse: {cfg.get('sparse_type','l1')} "
                  f"lambda={cfg.get('sparse_lambda',0)}")
        if cfg.get('enable_consistency'):
            print(f"  Consistency: {cfg.get('consistency_type','l2')} "
                  f"lambda={cfg.get('consistency_lambda',0)}")
        print(f"{'='*70}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_metrics = self._train_epoch(epoch)

            val_results = self._evaluate(self.data_loaders['val'])

            test_results = None
            if cfg.get('eval_test_every_epoch', False):
                test_results = self._evaluate(self.data_loaders['test'])

            if self.scheduler:
                self.scheduler.step()

            elapsed = time.time() - t0

            # Primary metric
            primary = cfg.get('primary_metric', 'balanced_accuracy')
            val_metric = val_results['bal_acc'] if primary == 'balanced_accuracy' \
                else val_results['f1_macro']

            # Print epoch summary
            line = (
                f"Ep {epoch:3d}/{epochs} | "
                f"Train: loss={train_metrics.get('total',0):.4f} "
                f"acc={train_metrics.get('bal_acc',0):.4f} | "
                f"Val: acc={val_results['bal_acc']:.4f} "
                f"f1={val_results['f1_macro']:.4f}"
            )
            if cfg.get('enable_sparse') and self.mode == 'selector':
                line += f" | sp={train_metrics.get('sparse',0):.4f}"
            if cfg.get('enable_consistency') and self.mode == 'selector':
                line += f" | cons={train_metrics.get('consistency',0):.4f}"
            if self.mode == 'selector':
                gate_mean = train_metrics.get('temporal_gate_mean', -1)
                if gate_mean >= 0:
                    line += f" | g_t={gate_mean:.3f}"
            if test_results:
                line += f" | Test: acc={test_results['bal_acc']:.4f}"
            line += f" | {elapsed:.1f}s"
            print(line)

            # WandB logging
            if self.wandb_run:
                log = {
                    'epoch': epoch,
                    'train/total_loss': train_metrics.get('total', 0),
                    'train/ce_loss': train_metrics.get('ce', 0),
                    'train/bal_acc': train_metrics.get('bal_acc', 0),
                    'val/bal_acc': val_results['bal_acc'],
                    'val/f1_macro': val_results['f1_macro'],
                    'lr': self.optimizer.param_groups[-1]['lr'],
                }
                if self.mode == 'selector':
                    for k in ('sparse', 'consistency', 'vib_kl',
                              'temporal_gate_mean', 'frequency_gate_mean',
                              'temporal_active_ratio', 'frequency_active_ratio'):
                        if k in train_metrics:
                            log[f'train/{k}'] = train_metrics[k]
                if test_results:
                    log['test/bal_acc'] = test_results['bal_acc']
                    log['test/f1_macro'] = test_results['f1_macro']
                self.wandb_run.log(log)

            # Track epoch results
            self.train_history.append({
                'epoch': epoch,
                'train': train_metrics,
                'val': {k: v for k, v in val_results.items()
                        if k not in ('preds', 'labels', 'confusion')},
            })

            # Best model
            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                self.best_epoch = epoch
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best: epoch {self.best_epoch}, metric={self.best_val_metric:.4f})")
                break

        return self._final_evaluation()

    def _train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()
        loss_accum = {}
        gate_stats_accum = {}
        all_preds = []
        all_labels = []
        n = 0

        for batch in self.data_loaders['train']:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)

            # Forward
            if self.mode == 'selector':
                out = self.model(x, return_gates=True)
            else:
                out = self.model(x)

            logits = out['logits']
            model_out = out if self.mode == 'selector' else None

            # Consistency: forward augmented view
            aug_out = None
            if (self.augmentor is not None and self.mode == 'selector'
                    and self.cfg.get('enable_consistency', False)):
                x_aug = self.augmentor(x)
                aug_out = self.model(x_aug, return_gates=True)

            # Loss
            loss, loss_dict = self.criterion(
                logits=logits, labels=y,
                model_out=model_out,
                aug_model_out=aug_out,
                current_epoch=epoch,
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            clip = self.cfg.get('clip_value', 5.0)
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()

            # Accumulate
            n += 1
            for k, v in loss_dict.items():
                if isinstance(v, (int, float)):
                    loss_accum[k] = loss_accum.get(k, 0.0) + v

            # Gate stats
            if self.mode == 'selector' and hasattr(self.model.head, 'get_gate_sparsity_stats'):
                g_stats = self.model.head.get_gate_sparsity_stats(
                    out.get('temporal_gate'), out.get('frequency_gate')
                )
                for k, v in g_stats.items():
                    gate_stats_accum[k] = gate_stats_accum.get(k, 0.0) + v

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        avg = {k: v / max(n, 1) for k, v in loss_accum.items()}
        for k, v in gate_stats_accum.items():
            avg[k] = v / max(n, 1)
        avg['bal_acc'] = balanced_accuracy_score(all_labels, all_preds)
        return avg

    @torch.no_grad()
    def _evaluate(self, dataloader) -> Dict:
        """Evaluate with extended metrics."""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)

            out = self.model(x)
            logits = out['logits']

            loss = torch.nn.functional.cross_entropy(logits, y)
            total_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        result = {
            'bal_acc': balanced_accuracy_score(all_labels, all_preds),
            'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1_per_class': f1_score(all_labels, all_preds, average=None, zero_division=0),
            'recall_per_class': recall_score(all_labels, all_preds, average=None, zero_division=0),
            'precision_per_class': precision_score(all_labels, all_preds, average=None, zero_division=0),
            'ce_loss': total_loss / max(n_batches, 1),
            'confusion': confusion_matrix(all_labels, all_preds),
            'preds': all_preds,
            'labels': all_labels,
        }

        # Abnormal-focused metrics (for binary tasks: class 0 = abnormal typically)
        # For TUAB: 0=abnormal, 1=normal
        # For TUSZ: 0=seizure, 1=non-seizure
        num_classes = len(set(all_labels))
        if num_classes == 2:
            # Assume class 0 is the "abnormal" / positive class
            result['abnormal_recall'] = result['recall_per_class'][0]
            result['abnormal_f1'] = result['f1_per_class'][0]
            result['abnormal_precision'] = result['precision_per_class'][0]

        return result

    def _final_evaluation(self) -> Dict:
        """Load best model and run final evaluation."""
        print(f"\n{'='*70}")
        if self.best_model_state:
            print(f"Loading best model from epoch {self.best_epoch}")
            self.model.load_state_dict(self.best_model_state)
        else:
            print("No improvement — using last model")

        test_results = self._evaluate(self.data_loaders['test'])

        # Print detailed results
        self._print_detailed_results(test_results)

        # Save checkpoint
        save_dir = Path(self.cfg.get('save_dir', 'checkpoints_selector'))
        save_dir.mkdir(parents=True, exist_ok=True)

        regime = self.cfg.get('regime', 'frozen')
        ckpt_name = (
            f"best_{self.cfg['dataset']}_{self.cfg.get('model','codebrain')}"
            f"_{self.mode}_{regime}"
            f"_acc{test_results['bal_acc']:.4f}"
            f"_s{self.cfg['seed']}.pth"
        )
        torch.save({
            'epoch': self.best_epoch,
            'model_state_dict': self.model.state_dict(),
            'val_metric': self.best_val_metric,
            'test_results': {k: v for k, v in test_results.items()
                             if k not in ('preds', 'labels')},
            'cfg': self.cfg,
            'regime_info': self.model.get_regime_info(),
            'param_summary': self.model.get_summary(),
        }, save_dir / ckpt_name)
        print(f"  Saved checkpoint: {save_dir / ckpt_name}")

        # Save JSON summary
        summary = self._build_json_summary(test_results)
        json_name = ckpt_name.replace('.pth', '.json')
        with open(save_dir / json_name, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  Saved summary: {save_dir / json_name}")

        if self.wandb_run:
            self.wandb_run.summary['test_bal_acc'] = test_results['bal_acc']
            self.wandb_run.summary['test_f1_macro'] = test_results['f1_macro']
            self.wandb_run.summary['best_epoch'] = self.best_epoch
            if 'abnormal_recall' in test_results:
                self.wandb_run.summary['test_abnormal_recall'] = test_results['abnormal_recall']
                self.wandb_run.summary['test_abnormal_f1'] = test_results['abnormal_f1']
            self.wandb_run.finish()

        print(f"{'='*70}")
        return test_results

    def _print_detailed_results(self, results: Dict):
        """Print detailed per-class results."""
        print(f"\n  TEST Results:")
        print(f"    Balanced Accuracy:  {results['bal_acc']:.4f}")
        print(f"    Macro F1:           {results['f1_macro']:.4f}")
        print(f"    Weighted F1:        {results['f1_weighted']:.4f}")
        print(f"    CE Loss:            {results['ce_loss']:.4f}")

        if 'abnormal_recall' in results:
            print(f"\n  Abnormal-Focused:")
            print(f"    Abnormal Recall:    {results['abnormal_recall']:.4f}")
            print(f"    Abnormal F1:        {results['abnormal_f1']:.4f}")
            print(f"    Abnormal Precision: {results['abnormal_precision']:.4f}")

        print(f"\n  Per-class metrics:")
        for i, (f1, rec, prec) in enumerate(zip(
                results['f1_per_class'],
                results['recall_per_class'],
                results['precision_per_class'])):
            name = self.label_names.get(i, str(i))
            print(f"    {name:20s}: F1={f1:.4f}  Rec={rec:.4f}  Prec={prec:.4f}")

        print(f"\n  Confusion Matrix:\n    {results['confusion']}")

    def _build_json_summary(self, test_results: Dict) -> Dict:
        """Build JSON-serializable summary."""
        cfg = self.cfg
        summary = {
            'experiment': 'selector',
            'mode': self.mode,
            'regime': cfg.get('regime', 'frozen'),
            'dataset': cfg['dataset'],
            'model': cfg.get('model', 'codebrain'),
            'seed': cfg['seed'],
            'best_epoch': self.best_epoch,
            'best_val_metric': float(self.best_val_metric),
            'test': {
                'bal_acc': float(test_results['bal_acc']),
                'f1_macro': float(test_results['f1_macro']),
                'f1_weighted': float(test_results['f1_weighted']),
                'f1_per_class': [float(x) for x in test_results['f1_per_class']],
                'recall_per_class': [float(x) for x in test_results['recall_per_class']],
                'precision_per_class': [float(x) for x in test_results['precision_per_class']],
                'ce_loss': float(test_results['ce_loss']),
                'confusion_matrix': test_results['confusion'].tolist(),
            },
            'config': {
                'regime': cfg.get('regime', 'frozen'),
                'enable_sparse': cfg.get('enable_sparse', False),
                'sparse_lambda': cfg.get('sparse_lambda', 0),
                'sparse_type': cfg.get('sparse_type', 'l1'),
                'enable_consistency': cfg.get('enable_consistency', False),
                'consistency_lambda': cfg.get('consistency_lambda', 0),
                'consistency_type': cfg.get('consistency_type', 'l2'),
                'lr_head': cfg.get('lr_head'),
                'lr_ratio': cfg.get('lr_ratio'),
                'epochs': cfg.get('epochs'),
                'patience': cfg.get('patience'),
            },
            'param_summary': {
                'total_params': self.model.get_summary()['total_params'],
                'trainable_params': self.model.get_summary()['trainable_params'],
                'trainable_ratio': self.model.get_summary()['trainable_ratio'],
            },
            'regime_info': {
                'n_blocks_unfrozen': self.model.get_regime_info()['n_blocks_unfrozen'],
                'total_blocks': self.model.get_regime_info()['total_blocks'],
            },
        }

        if 'abnormal_recall' in test_results:
            summary['test']['abnormal_recall'] = float(test_results['abnormal_recall'])
            summary['test']['abnormal_f1'] = float(test_results['abnormal_f1'])
            summary['test']['abnormal_precision'] = float(test_results['abnormal_precision'])

        return summary
