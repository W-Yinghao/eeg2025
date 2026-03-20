"""
Training loop for DEB experiments.

Supports:
  - baseline training (frozen / partial / full fine-tune)
  - DEB training with KL annealing and sparse gate regularization
  - Early stopping on balanced accuracy (configurable)
  - WandB logging (optional)
  - Differential learning rates for backbone vs head
"""

import os
import sys
import time
import copy
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score

from ..evaluation.evaluator import Evaluator
from .losses import DEBLoss

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class DEBTrainer:
    """
    Trainer for DEB experiments.

    Handles both baseline and DEB modes. The mode is determined by
    the model's mode attribute (set during construction).
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

        # Loss
        cw = dataset_config.get('class_weights')
        if cw is None:
            cw = cfg.get('class_weights')
        class_weights = torch.tensor(cw, dtype=torch.float32, device=device) if cw else None

        self.criterion = DEBLoss(
            beta=cfg.get('beta', 1e-4),
            beta_warmup_epochs=cfg.get('beta_warmup_epochs', 5),
            sparse_lambda=cfg.get('sparse_lambda', 1e-3),
            enable_sparse_reg=cfg.get('enable_sparse_reg', True),
            enable_consistency=cfg.get('enable_consistency', False),
            consistency_lambda=cfg.get('consistency_lambda', 0.0),
            class_weights=class_weights,
            label_smoothing=cfg.get('label_smoothing', 0.0),
        )

        # Optimizer with param groups
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

        # WandB
        self.wandb_run = None
        if cfg.get('wandb_project') and WANDB_AVAILABLE:
            run_name = cfg.get('wandb_run_name') or (
                f"deb_{cfg['mode']}_{cfg['dataset']}_{cfg['model']}_s{cfg['seed']}"
            )
            self.wandb_run = wandb.init(
                project=cfg['wandb_project'],
                name=run_name,
                config=cfg,
            )

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[DEBTrainer] Trainable params: {n_trainable:,}")

    def train(self):
        """Run full training loop."""
        cfg = self.cfg
        epochs = cfg['epochs']
        patience = cfg.get('patience', 15)
        patience_counter = 0

        print(f"\n{'='*60}")
        print(f"DEB Training ({self.mode}): {epochs} epochs")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # Train one epoch
            train_metrics = self._train_epoch(epoch)

            # Validate
            val_results = Evaluator.evaluate(
                self.model, self.data_loaders['val'], self.device
            )

            # Optional test eval
            test_results = None
            if cfg.get('eval_test_every_epoch', False):
                test_results = Evaluator.evaluate(
                    self.model, self.data_loaders['test'], self.device
                )

            if self.scheduler:
                self.scheduler.step()

            elapsed = time.time() - t0

            # Determine primary metric
            primary = cfg.get('primary_metric', 'balanced_accuracy')
            if primary == 'balanced_accuracy':
                val_metric = val_results['bal_acc']
            else:
                val_metric = val_results['f1_macro']

            # Print
            line = (
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train: total={train_metrics.get('total', 0):.4f} "
                f"ce={train_metrics.get('ce', 0):.4f} "
                f"acc={train_metrics.get('bal_acc', 0):.4f} | "
                f"Val: ce={val_results['ce_loss']:.4f} "
                f"bal_acc={val_results['bal_acc']:.4f} "
                f"f1={val_results['f1_macro']:.4f}"
            )
            if self.mode == 'deb':
                line += (
                    f" | kl={train_metrics.get('kl', 0):.4f} "
                    f"sp={train_metrics.get('sparse', 0):.4f}"
                )
            if test_results:
                line += f" | Test: acc={test_results['bal_acc']:.4f}"
            line += f" | {elapsed:.1f}s"
            print(line)

            # WandB
            if self.wandb_run:
                log = {
                    'epoch': epoch,
                    'train/total_loss': train_metrics.get('total', 0),
                    'train/ce_loss': train_metrics.get('ce', 0),
                    'train/bal_acc': train_metrics.get('bal_acc', 0),
                    'val/ce_loss': val_results['ce_loss'],
                    'val/bal_acc': val_results['bal_acc'],
                    'val/f1_macro': val_results['f1_macro'],
                    'lr': self.optimizer.param_groups[-1]['lr'],
                }
                if self.mode == 'deb':
                    log['train/kl'] = train_metrics.get('kl', 0)
                    log['train/sparse'] = train_metrics.get('sparse', 0)
                if test_results:
                    log['test/bal_acc'] = test_results['bal_acc']
                    log['test/f1_macro'] = test_results['f1_macro']
                self.wandb_run.log(log)

            # Best model tracking
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

        # Load best model and final evaluation
        self._final_evaluation()

    def _train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        loss_accum = {}
        all_preds = []
        all_labels = []
        n = 0

        for batch in self.data_loaders['train']:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)

            # Forward
            if self.mode == 'deb':
                out = self.model(x, return_gates=True)
            else:
                out = self.model(x)

            logits = out['logits']

            # Loss
            adapter_out = out if self.mode == 'deb' else None
            loss, loss_dict = self.criterion(
                logits=logits, labels=y,
                adapter_out=adapter_out,
                current_epoch=epoch,
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            clip = self.cfg.get('clip_value', 5.0)
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()

            # Track
            n += 1
            for k, v in loss_dict.items():
                if isinstance(v, (int, float)):
                    loss_accum[k] = loss_accum.get(k, 0.0) + v

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        avg = {k: v / max(n, 1) for k, v in loss_accum.items()}
        avg['bal_acc'] = balanced_accuracy_score(all_labels, all_preds)
        return avg

    def _final_evaluation(self):
        """Load best model and run final test evaluation."""
        print(f"\n{'='*60}")
        if self.best_model_state:
            print(f"Loading best model from epoch {self.best_epoch}")
            self.model.load_state_dict(self.best_model_state)
        else:
            print("No improvement during training — using last model")

        test_results = Evaluator.evaluate(
            self.model, self.data_loaders['test'], self.device
        )
        Evaluator.print_results(test_results, 'test', self.label_names)

        # Save checkpoint
        save_dir = Path(self.cfg.get('save_dir', 'checkpoints_deb'))
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_name = (
            f"best_{self.cfg['dataset']}_{self.cfg['model']}_{self.mode}"
            f"_acc{test_results['bal_acc']:.4f}.pth"
        )
        torch.save({
            'epoch': self.best_epoch,
            'model_state_dict': self.model.state_dict(),
            'val_metric': self.best_val_metric,
            'test_bal_acc': test_results['bal_acc'],
            'test_f1_macro': test_results['f1_macro'],
            'cfg': self.cfg,
        }, save_dir / ckpt_name)
        print(f"  Saved checkpoint to {save_dir / ckpt_name}")

        if self.wandb_run:
            self.wandb_run.summary['test_bal_acc'] = test_results['bal_acc']
            self.wandb_run.summary['test_f1_macro'] = test_results['f1_macro']
            self.wandb_run.summary['best_epoch'] = self.best_epoch
            self.wandb_run.finish()

        print(f"{'='*60}")
        return test_results
