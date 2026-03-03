"""
Base trainer class for SageStream.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path


class BaseTrainer(ABC):
    """
    Abstract base class for trainers.

    Provides common functionality for training, validation, and model management.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: str = "./outputs"
    ):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.best_model_state = None
        self.training_history = []

    @abstractmethod
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        pass

    @abstractmethod
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate on validation set."""
        pass

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stop_patience: int = 5,
        early_stop_metric: str = "balanced_accuracy"
    ) -> Dict[str, Any]:
        """
        Full training loop with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            early_stop_patience: Patience for early stopping
            early_stop_metric: Metric to monitor for early stopping

        Returns:
            Dictionary containing best model state and training history
        """
        best_metric = -float('inf')
        epochs_without_improvement = 0

        import sys
        print(f"\nStarting training for {epochs} epochs...")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Early stopping patience: {early_stop_patience}")
        print()
        sys.stdout.flush()

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs} starting...")
            sys.stdout.flush()

            # Update epoch for GRL schedule (if model supports it)
            if hasattr(self.model, 'set_epoch'):
                self.model.set_epoch(epoch)
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'set_epoch'):
                self.model.model.set_epoch(epoch)

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Record history
            epoch_record = {
                'epoch': epoch + 1,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            self.training_history.append(epoch_record)

            # Print progress
            self._print_epoch_summary(epoch + 1, epochs, train_metrics, val_metrics)

            # Log to wandb if enabled
            if hasattr(self, 'wandb_logger') and self.wandb_logger is not None:
                lr = self.optimizer.param_groups[0]['lr']
                self.wandb_logger.log_epoch(epoch + 1, train_metrics, val_metrics, lr)

            # Check for improvement
            current_metric = val_metrics.get(early_stop_metric, 0)
            if current_metric > best_metric:
                best_metric = current_metric
                epochs_without_improvement = 0
                self._save_best_model(epoch + 1, train_metrics, val_metrics)
            else:
                epochs_without_improvement += 1

            # Early stopping
            if epochs_without_improvement >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        return {
            'best_model_state': self.best_model_state,
            'training_history': self.training_history,
            'best_val_metric': best_metric
        }

    def _save_best_model(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Save the best model state."""
        self.best_model_state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }

    def _print_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Print epoch summary."""
        train_acc = train_metrics.get('accuracy', 0)
        train_f1 = train_metrics.get('f1_macro', 0)
        val_acc = val_metrics.get('accuracy', 0)
        val_f1 = val_metrics.get('f1_macro', 0)

        print(f"Epoch {epoch}/{total_epochs}: "
              f"Train Acc={train_acc:.4f}, F1={train_f1:.4f} | "
              f"Val Acc={val_acc:.4f}, F1={val_f1:.4f}")

    def load_best_model(self):
        """Load the best model state."""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state['model_state_dict'])
        return self.model

    def save_checkpoint(self, filepath: str, **extra_info):
        """Save a checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            **extra_info
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str):
        """Load a checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint
