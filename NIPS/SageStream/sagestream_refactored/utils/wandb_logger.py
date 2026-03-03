"""
Wandb logging utilities for SageStream.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import asdict

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandbLogger:
    """
    Weights & Biases logger for experiment tracking.

    Handles initialization, logging, and cleanup of wandb runs.
    """

    def __init__(
        self,
        config: Any,  # SageStreamConfig
        enabled: bool = True,
        fold: Optional[int] = None
    ):
        """
        Initialize wandb logger.

        Args:
            config: SageStreamConfig object
            enabled: Whether logging is enabled
            fold: Optional fold number for cross-validation
        """
        self.enabled = enabled and WANDB_AVAILABLE and config.wandb.enable
        self.config = config
        self.fold = fold
        self._run = None

        if self.enabled:
            self._init_wandb()

    def _init_wandb(self):
        """Initialize wandb run."""
        # Prefer user-specified wandb run_name over default experiment_name
        if hasattr(self.config, 'wandb') and self.config.wandb.run_name:
            run_name = self.config.wandb.run_name
        else:
            run_name = self.config.experiment_name
        if self.fold is not None:
            run_name = f"{run_name}_fold{self.fold}"

        # Create config dict for logging
        config_dict = self._create_config_dict()

        self._run = wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            name=run_name,
            config=config_dict,
            reinit=True,
            settings=wandb.Settings(start_method="thread")
        )

        print(f"Wandb initialized: {self._run.url}")

    def _create_config_dict(self) -> Dict[str, Any]:
        """Create a flat config dict for wandb."""
        from sagestream_refactored.configs import SageStreamConfig

        config_dict = {
            "dataset": self.config.data.dataset_name,
            "experiment_name": self.config.experiment_name,
        }

        # Add model config
        config_dict.update({
            f"model/{k}": v for k, v in asdict(self.config.model).items()
        })

        # Add training config
        config_dict.update({
            f"training/{k}": v for k, v in asdict(self.config.training).items()
        })

        # Add moe config
        config_dict.update({
            f"moe/{k}": v for k, v in asdict(self.config.moe).items()
        })

        # Add ablation config
        config_dict.update({
            f"ablation/{k}": v for k, v in asdict(self.config.ablation).items()
        })

        return config_dict

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to wandb.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if self.enabled and self._run is not None:
            wandb.log(metrics, step=step)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: Optional[float] = None
    ):
        """
        Log epoch metrics.

        Args:
            epoch: Current epoch number
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
            lr: Current learning rate
        """
        if not self.enabled:
            return

        log_dict = {"epoch": epoch}

        # Add training metrics
        for k, v in train_metrics.items():
            log_dict[f"train/{k}"] = v

        # Add validation metrics
        for k, v in val_metrics.items():
            log_dict[f"val/{k}"] = v

        # Add learning rate
        if lr is not None:
            log_dict["train/learning_rate"] = lr

        self.log(log_dict, step=epoch)

    def log_test_metrics(self, metrics: Dict[str, float]):
        """
        Log final test metrics.

        Args:
            metrics: Test metrics dictionary
        """
        if not self.enabled:
            return

        log_dict = {f"test/{k}": v for k, v in metrics.items()}
        self.log(log_dict)

    def log_ablation_comparison(self, results: Dict[str, Dict[str, float]]):
        """
        Log ablation study comparison results.

        Args:
            results: Dictionary mapping experiment names to their metrics
        """
        if not self.enabled:
            return

        for exp_name, metrics in results.items():
            log_dict = {f"ablation/{exp_name}/{k}": v for k, v in metrics.items()}
            self.log(log_dict)

    def watch_model(self, model, log_freq: int = 100):
        """
        Watch model for gradient and parameter logging.

        Args:
            model: PyTorch model
            log_freq: Logging frequency in batches
        """
        if self.enabled and self.config.wandb.log_model:
            wandb.watch(model, log_freq=log_freq)

    def finish(self):
        """Finish wandb run."""
        if self.enabled and self._run is not None:
            wandb.finish()
            self._run = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    @staticmethod
    def is_available() -> bool:
        """Check if wandb is available."""
        return WANDB_AVAILABLE


def create_wandb_logger(
    config: Any,
    fold: Optional[int] = None
) -> WandbLogger:
    """
    Create a wandb logger.

    Args:
        config: SageStreamConfig object
        fold: Optional fold number

    Returns:
        WandbLogger instance
    """
    return WandbLogger(config, fold=fold)
