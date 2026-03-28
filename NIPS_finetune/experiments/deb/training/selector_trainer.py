"""
Training loop for selector experiments (Exp 6/7).

Extends the original DEBTrainer with:
  - True partial FT regime logging
  - Gate sparsity tracking
  - Consistency training with augmented views
  - Class-wise metrics (abnormal recall/F1)
  - JSON summary export
"""

import csv
import os
import signal
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
            sparse_lambda_temporal=cfg.get('sparse_lambda_temporal'),
            sparse_lambda_frequency=cfg.get('sparse_lambda_frequency'),
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
                p_time_shift=cfg.get('aug_p_time_shift'),
                p_jitter=cfg.get('aug_p_jitter'),
                p_mask=cfg.get('aug_p_mask'),
            )
        else:
            self.augmentor = None

        # Optimizer
        param_groups = model.get_param_groups(cfg)
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=cfg.get('weight_decay', 1e-3),
        )

        # Scheduler (with optional warmup)
        self._build_scheduler()

        # Tracking
        self.best_val_metric = 0.0
        self.best_epoch = 0
        self.best_model_state = None
        self.train_history = []
        self.epoch_gate_rows = []

        # AMP (mixed precision)
        self.use_amp = cfg.get('amp', False)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

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

        # Preemption / resume support
        self._preempted = False
        self._setup_signal_handler()

        # Resume from checkpoint if requested
        self.start_epoch = 1
        resume_path = cfg.get('resume')
        if resume_path:
            self._load_resume_checkpoint(resume_path)

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[SelectorTrainer] mode={self.mode} regime={cfg.get('regime','frozen')} "
              f"trainable={n_trainable:,}")

    # ── Scheduler / Staged Training ──────────────────────────────────────

    def _build_scheduler(self):
        """Build LR scheduler based on self.cfg."""
        cfg = self.cfg
        warmup_epochs = cfg.get('warmup_epochs', 0)
        if cfg.get('scheduler', 'cosine') == 'cosine':
            if warmup_epochs > 0:
                warmup_sched = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.01, end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=max(cfg['epochs'] - warmup_epochs, 1),
                )
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_sched, cosine_sched],
                    milestones=[warmup_epochs],
                )
            else:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=cfg['epochs']
                )
        else:
            self.scheduler = None

    def train_stage(self, n_epochs, stage_name='Stage1'):
        """Train for n_epochs without early stopping or final evaluation.

        Used as the first stage of staged partial fine-tuning.
        Model weights are updated in place; no best-model tracking.
        """
        print(f"\n{'='*70}")
        print(f"  {stage_name}: {n_epochs} epochs")
        for i, pg in enumerate(self.optimizer.param_groups):
            n_params = sum(p.numel() for p in pg['params'])
            print(f"  param_group[{i}]: lr={pg['lr']:.1e} ({n_params:,} params)")
        print(f"{'='*70}\n")

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_metrics = self._train_epoch(epoch)
            _collect = (self.mode == 'selector')
            val_results = self._evaluate(self.data_loaders['val'],
                                         collect_gates=_collect)
            if self.scheduler:
                self.scheduler.step()
            elapsed = time.time() - t0

            line = (
                f"{stage_name} Ep {epoch}/{n_epochs} | "
                f"Train: loss={train_metrics.get('total',0):.4f} "
                f"acc={train_metrics.get('bal_acc',0):.4f} | "
                f"Val: acc={val_results['bal_acc']:.4f} "
                f"f1={val_results['f1_macro']:.4f}"
            )
            if self.mode == 'selector':
                gate_mean = train_metrics.get('temporal_gate_mean', -1)
                if gate_mean >= 0:
                    line += f" | g_t={gate_mean:.3f}"
                gate_f_mean = train_metrics.get('frequency_gate_mean', -1)
                if gate_f_mean >= 0:
                    line += f" g_f={gate_f_mean:.3f}"
            line += f" | {elapsed:.1f}s"
            print(line)

            if self.wandb_run:
                log = {
                    'epoch': epoch,
                    'stage': stage_name,
                    'train/total_loss': train_metrics.get('total', 0),
                    'train/bal_acc': train_metrics.get('bal_acc', 0),
                    'val/bal_acc': val_results['bal_acc'],
                    'val/f1_macro': val_results['f1_macro'],
                }
                if self.mode == 'selector':
                    for k in ('temporal_gate_mean', 'frequency_gate_mean',
                              'temporal_active_ratio', 'frequency_active_ratio'):
                        if k in train_metrics:
                            log[f'train/{k}'] = train_metrics[k]
                    for k, v in val_results.get('gate_stats', {}).items():
                        log[f'val/gate_{k}'] = v
                self.wandb_run.log(log)

            # Preemption during stage: exit immediately (short stages are cheap to redo)
            if self._preempted:
                print(f"\n[PREEMPT] Signal received during {stage_name}. "
                      f"Stage is short — will re-run on next submission.")
                sys.exit(124)

        print(f"\n{stage_name} complete.")

    def reset_for_stage(self, stage_cfg):
        """Reset optimizer and scheduler for a new training stage.

        Call after modifying model freeze state (e.g., freezing backbone
        for stage2 of staged partial FT).
        """
        self.cfg.update(stage_cfg)
        param_groups = self.model.get_param_groups(self.cfg)
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.cfg.get('weight_decay', 1e-3),
        )
        self._build_scheduler()

        # Reset tracking for new stage
        self.best_val_metric = 0.0
        self.best_epoch = 0
        self.best_model_state = None
        self.train_history = []
        self.epoch_gate_rows = []
        self.start_epoch = 1

        # Rebuild AMP scaler if needed
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n[SelectorTrainer] Stage reset: trainable={n_trainable:,}")

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
        if self.start_epoch > 1:
            print(f"  Resuming from epoch {self.start_epoch}")
        if cfg.get('enable_sparse'):
            if cfg.get('sparse_lambda_temporal') is not None or cfg.get('sparse_lambda_frequency') is not None:
                print(f"  Sparse (branch-aware): {cfg.get('sparse_type','l1')} "
                      f"lambda_t={cfg.get('sparse_lambda_temporal', 0)} "
                      f"lambda_f={cfg.get('sparse_lambda_frequency', 0)}")
            else:
                print(f"  Sparse: {cfg.get('sparse_type','l1')} "
                      f"lambda={cfg.get('sparse_lambda',0)}")
        if cfg.get('enable_consistency'):
            print(f"  Consistency: {cfg.get('consistency_type','l2')} "
                  f"lambda={cfg.get('consistency_lambda',0)}")
        print(f"{'='*70}\n")

        for epoch in range(self.start_epoch, epochs + 1):
            t0 = time.time()

            train_metrics = self._train_epoch(epoch)

            _collect = (self.mode == 'selector')
            val_results = self._evaluate(self.data_loaders['val'], collect_gates=_collect)

            test_results = None
            if cfg.get('eval_test_every_epoch', False):
                test_results = self._evaluate(self.data_loaders['test'], collect_gates=_collect)

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
                gate_f_mean = train_metrics.get('frequency_gate_mean', -1)
                if gate_f_mean >= 0:
                    line += f" g_f={gate_f_mean:.3f}"
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
                    # Val gate stats
                    for k, v in val_results.get('gate_stats', {}).items():
                        log[f'val/gate_{k}'] = v
                if test_results:
                    log['test/bal_acc'] = test_results['bal_acc']
                    log['test/f1_macro'] = test_results['f1_macro']
                    for k, v in test_results.get('gate_stats', {}).items():
                        log[f'test/gate_{k}'] = v
                self.wandb_run.log(log)

            # Track epoch results
            self.train_history.append({
                'epoch': epoch,
                'train': train_metrics,
                'val': {k: v for k, v in val_results.items()
                        if k not in ('preds', 'labels', 'confusion')},
            })

            # Track gate stats per epoch for CSV
            if self.mode == 'selector':
                gate_row = {'epoch': epoch}
                gate_row['train_g_t_mean'] = train_metrics.get('temporal_gate_mean')
                gate_row['train_g_t_std'] = train_metrics.get('temporal_gate_std')
                gate_row['train_g_f_mean'] = train_metrics.get('frequency_gate_mean')
                gate_row['train_g_f_std'] = train_metrics.get('frequency_gate_std')
                gate_row['train_gate_entropy'] = train_metrics.get('temporal_entropy')
                gate_row['train_gate_coverage_0.5'] = train_metrics.get('temporal_active_ratio')
                for k, v in val_results.get('gate_stats', {}).items():
                    gate_row[f'val_{k}'] = v
                if test_results:
                    for k, v in test_results.get('gate_stats', {}).items():
                        gate_row[f'test_{k}'] = v
                self.epoch_gate_rows.append(gate_row)

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

            # Check preemption signal
            if self._preempted:
                print(f"\n[PREEMPT] Signal received after epoch {epoch}. "
                      f"Saving resume checkpoint...")
                self._save_resume_checkpoint(epoch, patience_counter)
                sys.exit(124)  # special exit code for requeue

        else:
            # Loop finished without break — not preempted, not early-stopped
            pass

        return self._final_evaluation()

    # ── Preemption / Resume ──────────────────────────────────────────────

    def _setup_signal_handler(self):
        """Register SIGUSR1 handler for SLURM preemption.

        SLURM sends SIGUSR1 via ``#SBATCH --signal=B:USR1@<seconds>``.
        The handler sets a flag; the training loop checks it after each
        epoch, saves a resume checkpoint, and exits with code 124.
        """
        def _handler(signum, frame):
            print(f"\n[PREEMPT] Received signal {signum} — "
                  f"will save checkpoint after current epoch.")
            self._preempted = True

        try:
            signal.signal(signal.SIGUSR1, _handler)
        except (OSError, ValueError):
            # SIGUSR1 not available (e.g. Windows) — silently skip
            pass

    def _resume_checkpoint_path(self) -> Path:
        """Deterministic path for the resume checkpoint.

        Note: regime is NOT included in the filename because:
          - Each config already has its own save_dir (no collision)
          - Staged training changes regime mid-run (top1 → staged_partial),
            which would cause save/load path mismatch on resume
        """
        cfg = self.cfg
        save_dir = Path(cfg.get('save_dir', 'checkpoints_selector'))
        save_dir.mkdir(parents=True, exist_ok=True)
        name = (
            f"resume_{cfg['dataset']}_{cfg.get('model','codebrain')}"
            f"_{self.mode}"
            f"_s{cfg['seed']}.pth"
        )
        return save_dir / name

    def _save_resume_checkpoint(self, epoch: int, patience_counter: int):
        """Save everything needed to resume training."""
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': (self.scheduler.state_dict()
                                     if self.scheduler else None),
            'best_val_metric': self.best_val_metric,
            'best_epoch': self.best_epoch,
            'best_model_state': self.best_model_state,
            'patience_counter': patience_counter,
            'train_history': self.train_history,
            'cfg': self.cfg,
        }
        path = self._resume_checkpoint_path()
        torch.save(ckpt, path)
        print(f"[PREEMPT] Resume checkpoint saved: {path}")

    def _load_resume_checkpoint(self, path: str):
        """Restore training state from a resume checkpoint."""
        path = Path(path)
        if path.is_dir():
            # auto-detect: look for resume_*.pth in the directory
            path = self._resume_checkpoint_path()
        if not path.exists():
            print(f"[Resume] No checkpoint found at {path} — starting fresh.")
            return

        print(f"[Resume] Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if self.scheduler and ckpt.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        self.best_val_metric = ckpt['best_val_metric']
        self.best_epoch = ckpt['best_epoch']
        self.best_model_state = ckpt['best_model_state']
        self.train_history = ckpt.get('train_history', [])
        self.start_epoch = ckpt['epoch'] + 1

        print(f"[Resume] Resuming from epoch {self.start_epoch} "
              f"(best so far: epoch {self.best_epoch}, "
              f"metric={self.best_val_metric:.4f})")

        # Delete the resume checkpoint so a completed run won't re-resume
        path.unlink()
        print(f"[Resume] Deleted consumed checkpoint: {path}")

    # ── Training ──────────────────────────────────────────────────────────

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

            # Forward + Loss (with optional AMP)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
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

            # Backward (with optional AMP scaler)
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                clip = self.cfg.get('clip_value', 5.0)
                if clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
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
    def _evaluate(self, dataloader, collect_gates: bool = False) -> Dict:
        """Evaluate with extended metrics and optional gate collection."""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        n_batches = 0
        all_gate_t = []
        all_gate_f = []

        for batch in dataloader:
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                if collect_gates and self.mode == 'selector':
                    out = self.model(x, return_gates=True)
                    if out.get('temporal_gate') is not None:
                        all_gate_t.append(out['temporal_gate'].cpu())
                    if out.get('frequency_gate') is not None:
                        all_gate_f.append(out['frequency_gate'].cpu())
                else:
                    out = self.model(x)
                logits = out['logits']

            loss = torch.nn.functional.cross_entropy(logits.float(), y)
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
        num_classes = len(set(all_labels))
        if num_classes == 2:
            result['abnormal_recall'] = result['recall_per_class'][0]
            result['abnormal_f1'] = result['f1_per_class'][0]
            result['abnormal_precision'] = result['precision_per_class'][0]

        # Gate stats (when collected)
        if collect_gates and (all_gate_t or all_gate_f):
            result['gate_stats'] = self._compute_detailed_gate_stats(
                all_gate_t, all_gate_f, all_labels
            )
            # Per-class consistency eval (if augmentor available)
            if self.augmentor is not None and all_gate_t:
                cons_stats = self._eval_consistency_per_class(
                    dataloader, all_labels
                )
                result['gate_stats'].update(cons_stats)

        return result

    def _compute_detailed_gate_stats(self, all_gate_t, all_gate_f, all_labels) -> Dict:
        """Compute detailed gate statistics including per-class breakdown."""
        stats = {}
        labels = np.array(all_labels)
        abnormal_mask = (labels == 0)
        normal_mask = (labels == 1)

        if all_gate_t:
            gt = torch.cat(all_gate_t, dim=0).float()  # (N, S, 1)
            gt_vals = gt.squeeze(-1)  # (N, S)
            stats['g_t_mean'] = gt_vals.mean().item()
            stats['g_t_std'] = gt_vals.std().item()
            stats['gate_coverage_temporal_0.5'] = (gt_vals > 0.5).float().mean().item()
            g = gt_vals.clamp(1e-7, 1 - 1e-7)
            ent = -(g * g.log() + (1 - g) * (1 - g).log())
            stats['gate_entropy_temporal'] = ent.mean().item()
            # Top-K ratio: fraction of total gate mass in top K% of positions
            stats.update(self._compute_topk_ratios(gt_vals, prefix='temporal'))
            # Abnormal-only
            if abnormal_mask.sum() > 0:
                gt_abn = gt_vals[abnormal_mask]
                stats['abnormal_only_g_t_mean'] = gt_abn.mean().item()
                stats['abnormal_only_g_t_std'] = gt_abn.std().item()
                stats['abnormal_only_gate_coverage_temporal'] = (gt_abn > 0.5).float().mean().item()
                g_a = gt_abn.clamp(1e-7, 1 - 1e-7)
                ent_a = -(g_a * g_a.log() + (1 - g_a) * (1 - g_a).log())
                stats['abnormal_only_gate_entropy_temporal'] = ent_a.mean().item()
                stats.update(self._compute_topk_ratios(gt_abn, prefix='abnormal_only_temporal'))
            # Normal-only
            if normal_mask.sum() > 0:
                gt_norm = gt_vals[normal_mask]
                stats['normal_only_g_t_mean'] = gt_norm.mean().item()
                stats['normal_only_g_t_std'] = gt_norm.std().item()
                stats['normal_only_gate_coverage_temporal'] = (gt_norm > 0.5).float().mean().item()
                g_n = gt_norm.clamp(1e-7, 1 - 1e-7)
                ent_n = -(g_n * g_n.log() + (1 - g_n) * (1 - g_n).log())
                stats['normal_only_gate_entropy_temporal'] = ent_n.mean().item()
                stats.update(self._compute_topk_ratios(gt_norm, prefix='normal_only_temporal'))

        if all_gate_f:
            gf = torch.cat(all_gate_f, dim=0).float()  # (N, C, 1)
            gf_vals = gf.squeeze(-1)  # (N, C)
            stats['g_f_mean'] = gf_vals.mean().item()
            stats['g_f_std'] = gf_vals.std().item()
            stats['gate_coverage_frequency_0.5'] = (gf_vals > 0.5).float().mean().item()
            g = gf_vals.clamp(1e-7, 1 - 1e-7)
            ent = -(g * g.log() + (1 - g) * (1 - g).log())
            stats['gate_entropy_frequency'] = ent.mean().item()
            stats.update(self._compute_topk_ratios(gf_vals, prefix='frequency'))
            # Abnormal-only
            if abnormal_mask.sum() > 0:
                gf_abn = gf_vals[abnormal_mask]
                stats['abnormal_only_g_f_mean'] = gf_abn.mean().item()
                stats['abnormal_only_g_f_std'] = gf_abn.std().item()
                stats['abnormal_only_gate_coverage_frequency'] = (gf_abn > 0.5).float().mean().item()
                g_a = gf_abn.clamp(1e-7, 1 - 1e-7)
                ent_a = -(g_a * g_a.log() + (1 - g_a) * (1 - g_a).log())
                stats['abnormal_only_gate_entropy_frequency'] = ent_a.mean().item()
                stats.update(self._compute_topk_ratios(gf_abn, prefix='abnormal_only_frequency'))
            # Normal-only
            if normal_mask.sum() > 0:
                gf_norm = gf_vals[normal_mask]
                stats['normal_only_g_f_mean'] = gf_norm.mean().item()
                stats['normal_only_g_f_std'] = gf_norm.std().item()
                stats['normal_only_gate_coverage_frequency'] = (gf_norm > 0.5).float().mean().item()
                g_n = gf_norm.clamp(1e-7, 1 - 1e-7)
                ent_n = -(g_n * g_n.log() + (1 - g_n) * (1 - g_n).log())
                stats['normal_only_gate_entropy_frequency'] = ent_n.mean().item()
                stats.update(self._compute_topk_ratios(gf_norm, prefix='normal_only_frequency'))

        # Combined gate stats (average of temporal + frequency where both exist)
        t_ent = stats.get('gate_entropy_temporal')
        f_ent = stats.get('gate_entropy_frequency')
        if t_ent is not None and f_ent is not None:
            stats['gate_entropy'] = (t_ent + f_ent) / 2
        elif t_ent is not None:
            stats['gate_entropy'] = t_ent

        t_cov = stats.get('gate_coverage_temporal_0.5')
        f_cov = stats.get('gate_coverage_frequency_0.5')
        if t_cov is not None and f_cov is not None:
            stats['gate_coverage_0.5'] = (t_cov + f_cov) / 2
        elif t_cov is not None:
            stats['gate_coverage_0.5'] = t_cov

        at_ent = stats.get('abnormal_only_gate_entropy_temporal')
        af_ent = stats.get('abnormal_only_gate_entropy_frequency')
        if at_ent is not None and af_ent is not None:
            stats['abnormal_only_gate_entropy'] = (at_ent + af_ent) / 2
        elif at_ent is not None:
            stats['abnormal_only_gate_entropy'] = at_ent

        at_cov = stats.get('abnormal_only_gate_coverage_temporal')
        af_cov = stats.get('abnormal_only_gate_coverage_frequency')
        if at_cov is not None and af_cov is not None:
            stats['abnormal_only_gate_coverage'] = (at_cov + af_cov) / 2
        elif at_cov is not None:
            stats['abnormal_only_gate_coverage'] = at_cov

        # Normal-only combined stats
        nt_ent = stats.get('normal_only_gate_entropy_temporal')
        nf_ent = stats.get('normal_only_gate_entropy_frequency')
        if nt_ent is not None and nf_ent is not None:
            stats['normal_only_gate_entropy'] = (nt_ent + nf_ent) / 2
        elif nt_ent is not None:
            stats['normal_only_gate_entropy'] = nt_ent

        nt_cov = stats.get('normal_only_gate_coverage_temporal')
        nf_cov = stats.get('normal_only_gate_coverage_frequency')
        if nt_cov is not None and nf_cov is not None:
            stats['normal_only_gate_coverage'] = (nt_cov + nf_cov) / 2
        elif nt_cov is not None:
            stats['normal_only_gate_coverage'] = nt_cov

        # Combined top-K ratios (average temporal + frequency)
        for k_pct in (10, 20):
            t_val = stats.get(f'gate_top{k_pct}_ratio_temporal')
            f_val = stats.get(f'gate_top{k_pct}_ratio_frequency')
            if t_val is not None and f_val is not None:
                stats[f'gate_top{k_pct}_ratio'] = (t_val + f_val) / 2
            elif t_val is not None:
                stats[f'gate_top{k_pct}_ratio'] = t_val

        # Cross-class delta/ratio for frequency gate (class bias indicator)
        abn_gf = stats.get('abnormal_only_g_f_mean')
        norm_gf = stats.get('normal_only_g_f_mean')
        if abn_gf is not None and norm_gf is not None:
            stats['delta_gf_abn_minus_norm'] = abn_gf - norm_gf
            stats['ratio_gf_abn_over_norm'] = abn_gf / max(norm_gf, 1e-8)

        return stats

    @torch.no_grad()
    def _eval_consistency_per_class(self, dataloader, all_labels) -> Dict[str, float]:
        """Compute per-sample gate consistency under augmentation, grouped by class.

        For each batch: run original and augmented through the model,
        compute L2 similarity between temporal gates, group by label.
        """
        self.model.eval()
        per_sample_cons = []  # (similarity, label) pairs
        labels_np = np.array(all_labels)

        sample_idx = 0
        for batch in dataloader:
            x = batch['x'].to(self.device)
            B = x.shape[0]
            batch_labels = labels_np[sample_idx:sample_idx + B]
            sample_idx += B

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                out_orig = self.model(x, return_gates=True)
                x_aug = self.augmentor(x)
                out_aug = self.model(x_aug, return_gates=True)

            gt_orig = out_orig.get('temporal_gate')
            gt_aug = out_aug.get('temporal_gate')
            if gt_orig is None or gt_aug is None:
                continue

            # Per-sample L2 distance: (B,)
            min_len = min(gt_orig.shape[1], gt_aug.shape[1])
            diff = (gt_orig[:, :min_len] - gt_aug[:, :min_len]).squeeze(-1)
            l2_per_sample = (diff ** 2).mean(dim=1).cpu().numpy()  # (B,)

            for i in range(B):
                if i < len(batch_labels):
                    per_sample_cons.append((l2_per_sample[i], batch_labels[i]))

        if not per_sample_cons:
            return {}

        cons_vals = np.array([c[0] for c in per_sample_cons])
        cons_labels = np.array([c[1] for c in per_sample_cons])

        stats = {
            'gate_consistency_l2': float(cons_vals.mean()),
        }

        # Per-class
        for cls_id, cls_name in [(0, 'abnormal'), (1, 'normal')]:
            mask = (cons_labels == cls_id)
            if mask.sum() > 0:
                stats[f'{cls_name}_only_consistency_l2'] = float(cons_vals[mask].mean())

        return stats

    @staticmethod
    def _compute_topk_ratios(gate_vals: torch.Tensor,
                             prefix: str) -> Dict[str, float]:
        """Compute fraction of total gate mass in top-K% positions.

        Args:
            gate_vals: (N, S) gate activations
            prefix: e.g. 'temporal', 'frequency', 'abnormal_only_temporal'

        Returns:
            dict with gate_top10_ratio_{prefix}, gate_top20_ratio_{prefix}
        """
        result = {}
        N, S = gate_vals.shape
        total_mass = gate_vals.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (N, 1)
        sorted_vals, _ = gate_vals.sort(dim=1, descending=True)  # (N, S)
        for k_pct in (10, 20):
            k = max(1, int(S * k_pct / 100))
            topk_mass = sorted_vals[:, :k].sum(dim=1, keepdim=True)  # (N, 1)
            ratio = (topk_mass / total_mass).mean().item()
            result[f'gate_top{k_pct}_ratio_{prefix}'] = ratio
        return result

    def _save_epoch_gate_csv(self, save_dir: Path, ckpt_name: str):
        """Save per-epoch gate statistics as CSV."""
        if not self.epoch_gate_rows:
            return
        csv_name = ckpt_name.replace('.pth', '_epoch_gate_stats.csv')
        csv_path = save_dir / csv_name
        # Collect all possible keys
        all_keys = set()
        for row in self.epoch_gate_rows:
            all_keys.update(row.keys())
        fieldnames = ['epoch'] + sorted(k for k in all_keys if k != 'epoch')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.epoch_gate_rows:
                clean = {}
                for k, v in row.items():
                    if isinstance(v, float):
                        clean[k] = f'{v:.6f}'
                    elif v is not None:
                        clean[k] = v
                writer.writerow(clean)
        print(f"  Saved epoch gate stats: {csv_path}")

    def _final_evaluation(self) -> Dict:
        """Load best model and run final evaluation."""
        print(f"\n{'='*70}")
        if self.best_model_state:
            print(f"Loading best model from epoch {self.best_epoch}")
            self.model.load_state_dict(self.best_model_state)
        else:
            print("No improvement — using last model")

        test_results = self._evaluate(
            self.data_loaders['test'],
            collect_gates=(self.mode == 'selector'),
        )

        # Print detailed results
        self._print_detailed_results(test_results)

        # Print gate stats if available
        gate_stats = test_results.get('gate_stats', {})
        if gate_stats:
            print(f"\n  Gate Statistics (test):")
            for k, v in sorted(gate_stats.items()):
                print(f"    {k:40s}: {v:.4f}")

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

        # Save training curve CSV
        self._save_training_curve(save_dir, ckpt_name)

        # Save config snapshot
        config_name = ckpt_name.replace('.pth', '_config.json')
        with open(save_dir / config_name, 'w') as f:
            json.dump(self.cfg, f, indent=2, default=str)
        print(f"  Saved config: {save_dir / config_name}")

        # Save class-wise metrics JSON
        classwise_name = ckpt_name.replace('.pth', '_classwise.json')
        classwise = {}
        for i, name in self.label_names.items():
            classwise[name] = {
                'f1': float(test_results['f1_per_class'][i]),
                'recall': float(test_results['recall_per_class'][i]),
                'precision': float(test_results['precision_per_class'][i]),
            }
        with open(save_dir / classwise_name, 'w') as f:
            json.dump(classwise, f, indent=2)
        print(f"  Saved class-wise metrics: {save_dir / classwise_name}")

        # Save markdown summary
        self._save_markdown_summary(save_dir, ckpt_name, test_results, summary)

        # Save gate stats JSON and epoch gate CSV (selector mode)
        if self.mode == 'selector':
            if gate_stats:
                gate_name = ckpt_name.replace('.pth', '_gate_stats.json')
                with open(save_dir / gate_name, 'w') as f:
                    json.dump(gate_stats, f, indent=2)
                print(f"  Saved gate stats: {save_dir / gate_name}")
            if self.epoch_gate_rows:
                self._save_epoch_gate_csv(save_dir, ckpt_name)

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

    def _save_training_curve(self, save_dir: Path, ckpt_name: str):
        """Save training history as CSV."""
        if not self.train_history:
            return
        csv_name = ckpt_name.replace('.pth', '_curve.csv')
        csv_path = save_dir / csv_name

        # Determine all keys across epochs
        fieldnames = ['epoch']
        # Add train keys
        train_keys = set()
        val_keys = set()
        for entry in self.train_history:
            train_keys.update(entry.get('train', {}).keys())
            val_keys.update(entry.get('val', {}).keys())
        train_keys = sorted(train_keys)
        val_keys = sorted(val_keys)
        fieldnames += [f'train_{k}' for k in train_keys]
        fieldnames += [f'val_{k}' for k in val_keys]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.train_history:
                row = {'epoch': entry['epoch']}
                for k in train_keys:
                    v = entry.get('train', {}).get(k)
                    if isinstance(v, (int, float)):
                        row[f'train_{k}'] = f'{v:.6f}'
                    elif v is not None:
                        row[f'train_{k}'] = str(v)
                for k in val_keys:
                    v = entry.get('val', {}).get(k)
                    if isinstance(v, (int, float)):
                        row[f'val_{k}'] = f'{v:.6f}'
                    elif isinstance(v, np.ndarray):
                        row[f'val_{k}'] = str(v.tolist())
                    elif v is not None:
                        row[f'val_{k}'] = str(v)
                writer.writerow(row)
        print(f"  Saved training curve: {csv_path}")

    def _save_markdown_summary(self, save_dir: Path, ckpt_name: str,
                               test_results: Dict, summary: Dict):
        """Save a brief markdown summary."""
        md_name = ckpt_name.replace('.pth', '_summary.md')
        md_path = save_dir / md_name
        cfg = self.cfg

        lines = [
            f"# Experiment Summary",
            f"",
            f"| Item | Value |",
            f"|------|-------|",
            f"| Experiment | {cfg.get('wandb_project', 'N/A')} |",
            f"| Mode | {self.mode} |",
            f"| Regime | {cfg.get('regime', 'frozen')} |",
            f"| Dataset | {cfg['dataset']} |",
            f"| Model | {cfg.get('model', 'codebrain')} |",
            f"| Seed | {cfg['seed']} |",
            f"| Best Epoch | {self.best_epoch} |",
            f"| Best Val Metric | {self.best_val_metric:.4f} |",
            f"",
            f"## Test Results",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Balanced Accuracy | {test_results['bal_acc']:.4f} |",
            f"| Macro F1 | {test_results['f1_macro']:.4f} |",
            f"| Weighted F1 | {test_results['f1_weighted']:.4f} |",
            f"| CE Loss | {test_results['ce_loss']:.4f} |",
        ]

        if 'abnormal_recall' in test_results:
            lines += [
                f"| Abnormal Recall | {test_results['abnormal_recall']:.4f} |",
                f"| Abnormal F1 | {test_results['abnormal_f1']:.4f} |",
                f"| Abnormal Precision | {test_results['abnormal_precision']:.4f} |",
            ]

        lines += [
            f"",
            f"## Per-Class Metrics",
            f"",
            f"| Class | F1 | Recall | Precision |",
            f"|-------|-----|--------|-----------|",
        ]
        for i, (f1, rec, prec) in enumerate(zip(
                test_results['f1_per_class'],
                test_results['recall_per_class'],
                test_results['precision_per_class'])):
            name = self.label_names.get(i, str(i))
            lines.append(f"| {name} | {f1:.4f} | {rec:.4f} | {prec:.4f} |")

        lines += [
            f"",
            f"## Training Config",
            f"",
            f"| Param | Value |",
            f"|-------|-------|",
            f"| Epochs (max) | {cfg.get('epochs')} |",
            f"| Patience | {cfg.get('patience')} |",
            f"| Batch size | {cfg.get('batch_size')} |",
            f"| LR head | {cfg.get('lr_head')} |",
            f"| LR backbone | {cfg.get('lr_backbone')} |",
            f"| Warmup epochs | {cfg.get('warmup_epochs', 0)} |",
            f"| Scheduler | {cfg.get('scheduler')} |",
            f"| Grad clip | {cfg.get('clip_value')} |",
            f"| Trainable params | {summary['param_summary']['trainable_params']:,} |",
            f"| Total params | {summary['param_summary']['total_params']:,} |",
            f"",
        ]

        # Gate stats section (selector mode)
        gs = test_results.get('gate_stats', {})
        if gs:
            lines += [
                f"## Gate Statistics",
                f"",
                f"| Metric | Value |",
                f"|--------|-------|",
            ]
            for k, v in sorted(gs.items()):
                lines.append(f"| {k} | {v:.4f} |")
            lines.append(f"")

        with open(md_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"  Saved markdown summary: {md_path}")

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
                'lr_backbone': cfg.get('lr_backbone'),
                'lr_ratio': cfg.get('lr_ratio'),
                'warmup_epochs': cfg.get('warmup_epochs', 0),
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
                'unfrozen_layer_names': self.model.get_regime_info().get('unfrozen_layer_names', []),
            },
        }

        if 'abnormal_recall' in test_results:
            summary['test']['abnormal_recall'] = float(test_results['abnormal_recall'])
            summary['test']['abnormal_f1'] = float(test_results['abnormal_f1'])
            summary['test']['abnormal_precision'] = float(test_results['abnormal_precision'])

        if 'gate_stats' in test_results:
            summary['gate_stats'] = test_results['gate_stats']

        return summary
