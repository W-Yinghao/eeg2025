"""
SageStream trainer implementation.
"""

from typing import Dict, Any, Optional, List
import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

# Add parent paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from .base_trainer import BaseTrainer
from ..utils.metrics import compute_metrics
from ..utils.wandb_logger import WandbLogger


class SageStreamTrainer(BaseTrainer):
    """
    Trainer for SageStream model.

    Handles the two-stage training process:
    1. Source domain training with subject-aware components
    2. Test-time adaptation (handled separately in TTA module)

    Supports IIB (Information Invariant Bottleneck) training with combined loss:
    L_total = L_task + alpha * L_KL + beta * L_adv
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 5e-5,
        weight_decay: float = 1e-5,
        aux_loss_weight: float = 0.001,
        scheduler_patience: int = 3,
        scheduler_factor: float = 0.5,
        num_classes: int = 2,
        output_dir: str = "./outputs",
        wandb_logger: Optional[WandbLogger] = None,
        # IIB loss weights (can override model defaults)
        iib_kl_loss_weight: Optional[float] = None,
        iib_adv_loss_weight: Optional[float] = None
    ):
        super().__init__(model, device, output_dir)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.aux_loss_weight = aux_loss_weight
        self.num_classes = num_classes
        self.wandb_logger = wandb_logger

        # IIB loss weights
        self.iib_kl_loss_weight = iib_kl_loss_weight
        self.iib_adv_loss_weight = iib_adv_loss_weight

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=scheduler_factor,
            patience=scheduler_patience
        )

        # Set training stage
        if hasattr(model, 'set_training_stage'):
            model.set_training_stage("source_domain")

    def _process_batch(
        self,
        batch_data: tuple
    ) -> tuple:
        """Process a batch of data."""
        if len(batch_data) == 3:
            eeg_data, labels, subject_ids = batch_data
            subject_ids = subject_ids.to(self.device)
        else:
            eeg_data, labels = batch_data
            subject_ids = None

        eeg_data = eeg_data.to(self.device)
        labels = labels.to(self.device)

        return eeg_data, labels, subject_ids

    def _get_model_output(
        self,
        eeg_data: torch.Tensor,
        subject_ids: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None
    ) -> tuple:
        """Get model output and extract logits, aux_loss, and IIB losses."""
        outputs = self.model.classify(x_enc=eeg_data, subject_ids=subject_ids, labels=labels)

        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            aux_loss = getattr(outputs, 'aux_loss', torch.tensor(0.0))
            iib_loss = getattr(outputs, 'iib_loss', None)
            iib_losses = getattr(outputs, 'iib_losses', None)
        else:
            logits = outputs
            aux_loss = torch.tensor(0.0)
            iib_loss = None
            iib_losses = None

        return logits, aux_loss, iib_loss, iib_losses

    def _collect_predictions(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        preds_list: list,
        labels_list: list,
        probs_list: list
    ):
        """Collect predictions for metric computation."""
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        preds_list.extend(preds.detach().cpu().numpy())
        labels_list.extend(labels.detach().cpu().numpy())

        if self.num_classes == 2:
            probs_list.extend(probs[:, 1].detach().cpu().numpy())
        else:
            probs_list.append(probs.detach().cpu().numpy())

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0
        total_task_loss = 0.0
        total_aux_loss = 0.0
        total_iib_loss = 0.0
        total_kl_loss = 0.0
        total_adv_loss = 0.0

        for batch_data in train_loader:
            eeg_data, labels, subject_ids = self._process_batch(batch_data)

            self.optimizer.zero_grad()

            # Forward pass (pass labels for ICML IIB variant which needs them for CI loss)
            logits, aux_loss, iib_loss, iib_losses = self._get_model_output(eeg_data, subject_ids, labels)

            # Compute task loss (classification)
            task_loss = self.criterion(logits, labels)

            # Compute total loss
            total_loss_batch = task_loss

            # Add auxiliary loss (MoE load balancing)
            if isinstance(aux_loss, torch.Tensor) and aux_loss.numel() > 0 and aux_loss.item() > 0:
                total_loss_batch = total_loss_batch + self.aux_loss_weight * aux_loss
                total_aux_loss += aux_loss.item()

            # Add IIB losses if available
            if iib_loss is not None and isinstance(iib_loss, torch.Tensor) and iib_loss.numel() > 0:
                # Check if this is ICML variant (has 'total_iib_loss' key in iib_losses)
                is_icml = iib_losses is not None and 'total_iib_loss' in iib_losses

                if is_icml:
                    # ICML variant: total_iib_loss already includes all components
                    # It replaces the task CE loss with its own inv_loss + env_loss
                    # So we add the full iib_loss to the task_loss
                    total_loss_batch = total_loss_batch + iib_loss
                elif self.iib_kl_loss_weight is not None or self.iib_adv_loss_weight is not None:
                    # NIPS variant: recompute with trainer weights
                    kl_weight = self.iib_kl_loss_weight if self.iib_kl_loss_weight is not None else 0.1
                    adv_weight = self.iib_adv_loss_weight if self.iib_adv_loss_weight is not None else 0.1
                    if iib_losses is not None:
                        weighted_iib_loss = kl_weight * iib_losses['kl_loss'] + adv_weight * iib_losses['adv_loss']
                        total_loss_batch = total_loss_batch + weighted_iib_loss
                else:
                    # Use model's pre-computed iib_loss (already weighted)
                    total_loss_batch = total_loss_batch + iib_loss

                total_iib_loss += iib_loss.item() if iib_loss is not None else 0.0

                # Track individual IIB losses (handle both variants)
                if iib_losses is not None:
                    total_kl_loss += iib_losses.get('kl_loss', iib_losses.get('ib_loss', torch.tensor(0.0))).item()
                    total_adv_loss += iib_losses.get('adv_loss', iib_losses.get('ci_loss', torch.tensor(0.0))).item()

            # Backward pass
            total_loss_batch.backward()
            self.optimizer.step()

            # Collect predictions
            self._collect_predictions(logits, labels, all_preds, all_labels, all_probs)
            total_loss += total_loss_batch.item()
            total_task_loss += task_loss.item()

        # Compute metrics
        num_batches = len(train_loader)
        probs_array = np.array(all_probs) if self.num_classes == 2 else np.concatenate(all_probs, axis=0)
        metrics = compute_metrics(all_labels, all_preds, probs_array, self.num_classes)
        metrics['loss'] = total_loss / num_batches
        metrics['task_loss'] = total_task_loss / num_batches
        metrics['aux_loss'] = total_aux_loss / num_batches
        metrics['iib_loss'] = total_iib_loss / num_batches
        metrics['kl_loss'] = total_kl_loss / num_batches
        metrics['adv_loss'] = total_adv_loss / num_batches

        return metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_data in val_loader:
                eeg_data, labels, subject_ids = self._process_batch(batch_data)

                logits, _, _, _ = self._get_model_output(eeg_data, subject_ids)

                self._collect_predictions(logits, labels, all_preds, all_labels, all_probs)

        # Compute metrics
        probs_array = np.array(all_probs) if self.num_classes == 2 else np.concatenate(all_probs, axis=0)
        metrics = compute_metrics(all_labels, all_preds, probs_array, self.num_classes)

        # Update scheduler
        self.scheduler.step(metrics['balanced_accuracy'])

        return metrics

    def _save_best_model(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Save the best model state including optimizer."""
        self.best_model_state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_val_balanced_acc': val_metrics.get('balanced_accuracy', 0)
        }

    def _print_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Print epoch summary with IIB loss information."""
        train_acc = train_metrics.get('accuracy', 0)
        train_f1 = train_metrics.get('f1_macro', 0)
        val_acc = val_metrics.get('accuracy', 0)
        val_f1 = val_metrics.get('f1_macro', 0)

        # Base summary
        summary = (f"Epoch {epoch}/{total_epochs}: "
                   f"Train Acc={train_acc:.4f}, F1={train_f1:.4f} | "
                   f"Val Acc={val_acc:.4f}, F1={val_f1:.4f}")

        # Add IIB loss info if available
        iib_loss = train_metrics.get('iib_loss', 0)
        kl_loss = train_metrics.get('kl_loss', 0)
        adv_loss = train_metrics.get('adv_loss', 0)

        if iib_loss > 0 or kl_loss > 0 or adv_loss > 0:
            summary += f" | IIB={iib_loss:.4f} (KL/IB={kl_loss:.4f}, Adv/CI={adv_loss:.4f})"

        print(summary)


class KFoldTrainer:
    """
    Wrapper for k-fold cross-validation training.

    Handles the complete k-fold training pipeline including
    model creation, training, evaluation, and result aggregation.
    """

    def __init__(
        self,
        config,  # SageStreamConfig
        model_class,
        output_dir: str = "./outputs"
    ):
        self.config = config
        self.model_class = model_class
        self.output_dir = output_dir

        self.device = torch.device(config.training.device)

    def _create_model(self):
        """Create a new model instance."""
        decoupling_config = self.config.get_decoupling_config()
        model_kwargs = self.config.get_model_kwargs()

        model = self.model_class.from_pretrained(
            model_path=self.config.model.model_path,
            decoupling_config=decoupling_config,
            model_kwargs=model_kwargs
        ).to(self.device)

        model.task_name = "classification"
        model.set_training_stage("source_domain")

        return model

    def _create_trainer(self, model, wandb_logger=None) -> SageStreamTrainer:
        """Create a trainer for the model."""
        return SageStreamTrainer(
            model=model,
            device=self.device,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            aux_loss_weight=self.config.training.aux_loss_weight,
            scheduler_patience=self.config.training.scheduler_patience,
            scheduler_factor=self.config.training.scheduler_factor,
            num_classes=self.config.model.num_classes,
            output_dir=self.output_dir,
            wandb_logger=wandb_logger,
            # IIB loss weights from config
            iib_kl_loss_weight=self.config.iib.kl_loss_weight if hasattr(self.config, 'iib') else None,
            iib_adv_loss_weight=self.config.iib.adv_loss_weight if hasattr(self.config, 'iib') else None
        )

    def train_single_fold(
        self,
        fold_idx: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader
    ) -> Dict[str, Any]:
        """Train and evaluate a single fold."""
        print(f"\n{'='*50}")
        print(f"Training Fold {fold_idx + 1}")
        print(f"{'='*50}")

        # Create wandb logger for this fold
        wandb_logger = WandbLogger(self.config, fold=fold_idx + 1)

        # Create model and trainer
        model = self._create_model()
        trainer = self._create_trainer(model, wandb_logger)

        # Watch model if enabled
        if wandb_logger.enabled:
            wandb_logger.watch_model(model, log_freq=self.config.wandb.log_freq)

        try:
            # Train
            result = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=self.config.training.epochs,
                early_stop_patience=self.config.training.early_stop_patience,
                early_stop_metric=self.config.training.early_stop_metric
            )

            # Load best model for evaluation
            if result['best_model_state'] is not None:
                model.load_state_dict(result['best_model_state']['model_state_dict'])

                # Save model
                model_path = os.path.join(self.output_dir, f'best_model_fold_{fold_idx + 1}.pth')
                torch.save(result['best_model_state'], model_path)

                # Evaluate on test set
                test_metrics = self._evaluate_on_test(model, test_loader)

                # Log test metrics to wandb
                if wandb_logger.enabled:
                    wandb_logger.log_test_metrics(test_metrics)

                result['test_metrics'] = test_metrics
                result['fold'] = fold_idx + 1
                result['model_path'] = model_path

                print(f"\nFold {fold_idx + 1} Test Results:")
                self._print_metrics(test_metrics)

        finally:
            # Finish wandb run
            wandb_logger.finish()

        return result

    def _evaluate_on_test(
        self,
        model: nn.Module,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate model on test set."""
        model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_data in test_loader:
                if len(batch_data) == 3:
                    eeg_data, labels, subject_ids = batch_data
                    subject_ids = subject_ids.to(self.device)
                else:
                    eeg_data, labels = batch_data
                    subject_ids = None

                eeg_data = eeg_data.to(self.device)
                labels = labels.to(self.device)

                outputs = model.classify(x_enc=eeg_data, subject_ids=subject_ids)

                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs

                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())

                if self.config.model.num_classes == 2:
                    all_probs.extend(probs[:, 1].detach().cpu().numpy())
                else:
                    all_probs.append(probs.detach().cpu().numpy())

        probs_array = np.array(all_probs) if self.config.model.num_classes == 2 else np.concatenate(all_probs, axis=0)
        metrics = compute_metrics(all_labels, all_preds, probs_array, self.config.model.num_classes)

        return metrics

    def _print_metrics(self, metrics: Dict[str, float]):
        """Print metrics in a formatted way."""
        print(f"  Accuracy:          {metrics.get('accuracy', 0):.4f}")
        print(f"  Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.4f}")
        print(f"  F1 Macro:          {metrics.get('f1_macro', 0):.4f}")
        print(f"  Precision Macro:   {metrics.get('precision_macro', 0):.4f}")
        print(f"  Recall Macro:      {metrics.get('recall_macro', 0):.4f}")

    def run_k_fold(
        self,
        fold_loaders: List[tuple[DataLoader, DataLoader, DataLoader]]
    ) -> Dict[str, Any]:
        """
        Run complete k-fold cross-validation.

        Args:
            fold_loaders: List of (train_loader, val_loader, test_loader) tuples

        Returns:
            Dictionary with aggregated results
        """
        all_fold_results = []

        for fold_idx, (train_loader, val_loader, test_loader) in enumerate(fold_loaders):
            fold_result = self.train_single_fold(
                fold_idx, train_loader, val_loader, test_loader
            )
            all_fold_results.append(fold_result)

        # Aggregate results
        return self._aggregate_results(all_fold_results)

    def _aggregate_results(self, all_fold_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results from all folds."""
        successful_folds = [r for r in all_fold_results if r.get('best_model_state') is not None]

        if not successful_folds:
            return {
                'status': 'failed',
                'fold_results': all_fold_results
            }

        # Compute mean metrics
        metric_keys = ['accuracy', 'balanced_accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        mean_metrics = {}

        for key in metric_keys:
            values = [r['test_metrics'].get(key, 0) for r in successful_folds]
            mean_metrics[key] = np.mean(values)
            mean_metrics[f'{key}_std'] = np.std(values)

        print(f"\n{'='*50}")
        print("K-Fold Cross Validation Results")
        print(f"{'='*50}")
        print(f"Completed folds: {len(successful_folds)}/{len(all_fold_results)}")
        print(f"\nMean Test Metrics:")
        for key in metric_keys:
            mean = mean_metrics[key]
            std = mean_metrics.get(f'{key}_std', 0)
            print(f"  {key}: {mean:.4f} (+/- {std:.4f})")

        return {
            'status': 'success',
            'mean_metrics': mean_metrics,
            'fold_results': all_fold_results,
            'num_successful_folds': len(successful_folds)
        }
