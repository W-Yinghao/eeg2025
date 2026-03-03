"""
Test-Time Adaptation (TTA) module for SageStream.

This module implements the STSA (Style-Transfer Style Alignment) method
for test-time adaptation to new subjects.
"""

from typing import Dict, Any, Optional, Tuple
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

# Add parent paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from MoE_moment.momentfm.models.layers.SA_MoE import StyleAdaptor
from .utils.metrics import compute_metrics


class STSAAdapter:
    """
    Style-Transfer Style Alignment (STSA) for Test-Time Adaptation.

    This method adapts the model to new subjects at test time by:
    1. Using a StyleAdaptor to learn subject-specific style parameters
    2. Computing confidence weights based on style discrepancy
    3. Updating the adaptor using pseudo-labels with confidence weighting
    """

    def __init__(
        self,
        model: nn.Module,
        num_channels: int = 16,
        feature_dim: int = 512,
        learning_rate: float = 5e-4,
        device: torch.device = None
    ):
        """
        Initialize STSA adapter.

        Args:
            model: The SageStream model to adapt
            num_channels: Number of input channels
            feature_dim: Feature dimension (d_model)
            learning_rate: Learning rate for adaptor optimization
            device: Device to run on
        """
        self.model = model
        self.num_channels = num_channels
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Will be created when adapt() is called
        self.adaptor = None
        self.optimizer = None

    def _create_adaptor(self) -> StyleAdaptor:
        """Create the style adaptor."""
        return StyleAdaptor(
            num_channels=self.num_channels,
            feature_dim=self.feature_dim
        ).to(self.device)

    def _setup_model_for_tta(self):
        """Setup model for test-time adaptation."""
        self.model.eval()

        # Freeze all model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Create adaptor and setup in model blocks
        self.adaptor = self._create_adaptor()
        self.adaptor.train()

        # Find layers with STSA support and switch them
        for block in self.model.model.encoder.block:
            if hasattr(block, 'shared_knowledge'):
                if hasattr(block.shared_knowledge, 'switch_to_STSA'):
                    block.shared_knowledge.switch_to_STSA(self.adaptor)

        # Create optimizer for adaptor only
        self.optimizer = optim.Adam(self.adaptor.parameters(), lr=self.learning_rate)

    def _restore_model(self):
        """Restore model to pre-training mode."""
        for block in self.model.model.encoder.block:
            if hasattr(block, 'shared_knowledge'):
                if hasattr(block.shared_knowledge, 'switch_to_pretrain_mode'):
                    block.shared_knowledge.switch_to_pretrain_mode()

    def _compute_confidence_weights(
        self,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute confidence weights based on style discrepancy.

        Args:
            logits: Model output logits

        Returns:
            Confidence weights for each sample
        """
        confidence_weights = []

        for block in self.model.model.encoder.block:
            if hasattr(block, 'shared_knowledge'):
                if hasattr(block.shared_knowledge, 'get_STSA_tta_features'):
                    features = block.shared_knowledge.get_STSA_tta_features()

                    if features[0] is not None:
                        raw_features, norm_features, gamma, beta = features

                        B_times_C, S, feat_dim = raw_features.shape
                        B, C = gamma.shape[0], gamma.shape[1]

                        raw_features_4d = raw_features.view(B, C, S, feat_dim)

                        # Compute temporal statistics
                        true_mean_temporal = raw_features_4d.mean(dim=2)
                        true_std_temporal = raw_features_4d.std(dim=2)

                        # Compute spatial statistics
                        true_mean_spatial = raw_features_4d.mean(dim=1)
                        true_std_spatial = raw_features_4d.std(dim=1)

                        # Prior statistics from gamma/beta
                        prior_gamma_temporal = gamma
                        prior_beta_temporal = beta
                        prior_gamma_spatial = gamma.mean(dim=1, keepdim=True)
                        prior_beta_spatial = beta.mean(dim=1, keepdim=True)

                        epsilon = 1e-8

                        # Temporal discrepancy
                        temporal_err_mean = torch.abs(true_mean_temporal - prior_beta_temporal) / \
                            (torch.abs(true_mean_temporal) + epsilon)
                        temporal_err_std = torch.abs(true_std_temporal - prior_gamma_temporal) / \
                            (torch.abs(true_std_temporal) + epsilon)
                        temporal_discrepancy = (temporal_err_mean + temporal_err_std).mean(dim=-1)

                        # Spatial discrepancy
                        spatial_err_mean = torch.abs(true_mean_spatial - prior_beta_spatial) / \
                            (torch.abs(true_mean_spatial) + epsilon)
                        spatial_err_std = torch.abs(true_std_spatial - prior_gamma_spatial) / \
                            (torch.abs(true_std_spatial) + epsilon)
                        spatial_discrepancy = (spatial_err_mean + spatial_err_std).mean(dim=-1)

                        # Combined confidence
                        temporal_confidence = temporal_discrepancy.mean(dim=1)
                        spatial_confidence = spatial_discrepancy.mean(dim=1)
                        combined_confidence = (temporal_confidence + spatial_confidence) / 2

                        confidence_weights.append(combined_confidence)

        if confidence_weights:
            return torch.stack(confidence_weights).mean(dim=0)

        # Default: uniform weights
        return torch.ones(logits.shape[0], device=self.device)

    def adapt(
        self,
        test_loader: DataLoader,
        steps_per_batch: int = 1,
        num_classes: int = 2
    ) -> Dict[str, Any]:
        """
        Perform test-time adaptation.

        Args:
            test_loader: Test data loader
            steps_per_batch: Number of adaptation steps per batch
            num_classes: Number of classes

        Returns:
            Dictionary containing adaptation results and metrics
        """
        self._setup_model_for_tta()

        all_preds = []
        all_labels = []
        all_probs = []

        total_batches = len(test_loader)
        print(f"  Total batches to process: {total_batches}")
        sys.stdout.flush()

        try:
            for batch_idx, batch_data in enumerate(test_loader):
                # Progress output every 10 batches
                if batch_idx % 10 == 0:
                    print(f"  Processing batch {batch_idx + 1}/{total_batches}...")
                    sys.stdout.flush()

                if len(batch_data) == 3:
                    inputs, _, subject_ids = batch_data
                    subject_ids = subject_ids.to(self.device)
                    labels = batch_data[1]
                else:
                    inputs, labels = batch_data
                    subject_ids = None

                inputs = inputs.to(self.device)

                # Adaptation steps
                for step in range(steps_per_batch):
                    with torch.enable_grad():
                        self.optimizer.zero_grad()

                        outputs = self.model.classify(x_enc=inputs, subject_ids=subject_ids)

                        if hasattr(outputs, 'logits'):
                            logits = outputs.logits
                        else:
                            logits = outputs

                        # Compute confidence weights
                        confidence_weights = self._compute_confidence_weights(logits)

                        # Pseudo-labels
                        with torch.no_grad():
                            pseudo_labels = torch.argmax(logits, dim=1)

                        # Weighted cross-entropy loss
                        ce_loss_per_sample = F.cross_entropy(logits, pseudo_labels, reduction='none')
                        weighted_loss = (confidence_weights * ce_loss_per_sample).mean()

                        weighted_loss.backward()
                        self.optimizer.step()

                # Evaluation after adaptation
                with torch.no_grad():
                    eval_outputs = self.model.classify(x_enc=inputs, subject_ids=subject_ids)

                    if hasattr(eval_outputs, 'logits'):
                        eval_logits = eval_outputs.logits
                    else:
                        eval_logits = eval_outputs

                    eval_probs = torch.softmax(eval_logits, dim=1)
                    eval_preds = torch.argmax(eval_logits, dim=1)

                    all_preds.append(eval_preds.cpu())
                    all_labels.append(labels.cpu())

                    if num_classes == 2:
                        all_probs.append(eval_probs[:, 1].cpu())
                    else:
                        all_probs.append(eval_probs.cpu())

        finally:
            self._restore_model()

        # Compute final metrics
        final_preds = torch.cat(all_preds).numpy()
        final_labels = torch.cat(all_labels).numpy()

        if num_classes == 2:
            final_probs = torch.cat(all_probs).numpy()
        else:
            final_probs = torch.cat(all_probs, dim=0).numpy()

        metrics = compute_metrics(final_labels, final_preds, final_probs, num_classes)

        return {
            'metrics': metrics,
            'predictions': final_preds,
            'true_labels': final_labels,
            'probabilities': final_probs
        }


class TTAManager:
    """
    Manager for Test-Time Adaptation experiments.

    Provides a unified interface for running TTA experiments with
    different methods and configurations.
    """

    AVAILABLE_METHODS = ['STSA']

    def __init__(
        self,
        config,  # TTAConfig
        device: torch.device = None
    ):
        """
        Initialize TTA manager.

        Args:
            config: TTAConfig object
            device: Device to run on
        """
        self.config = config
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def run_tta(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        num_classes: int = 2,
        num_channels: int = 16,
        feature_dim: int = 512
    ) -> Optional[Dict[str, Any]]:
        """
        Run test-time adaptation.

        Args:
            model: Model to adapt
            test_loader: Test data loader
            num_classes: Number of classes
            num_channels: Number of input channels
            feature_dim: Feature dimension

        Returns:
            Dictionary with TTA results or None if TTA is disabled
        """
        if not self.config.enable_tta:
            return None

        if self.config.tta_method not in self.AVAILABLE_METHODS:
            print(f"Warning: Unknown TTA method {self.config.tta_method}. Available: {self.AVAILABLE_METHODS}")
            return None

        print(f"\nRunning TTA with method: {self.config.tta_method}")
        sys.stdout.flush()

        if self.config.tta_method == "STSA":
            # STSA requires SA-MoE blocks; skip if MoE is disabled
            use_moe = getattr(model.model, 'use_moe', True) if hasattr(model, 'model') else True
            if not use_moe:
                print("  STSA TTA skipped: SA-MoE is disabled (no style adaptation layers available)")
                return None

            adapter = STSAAdapter(
                model=model,
                num_channels=num_channels,
                feature_dim=feature_dim,
                learning_rate=self.config.tta_learning_rate,
                device=self.device
            )

            # Adjust batch size if needed
            if self.config.tta_batch_size != test_loader.batch_size:
                test_loader = self._adjust_batch_size(test_loader)

            result = adapter.adapt(
                test_loader=test_loader,
                steps_per_batch=self.config.tta_steps_per_batch,
                num_classes=num_classes
            )

            print(f"TTA Results:")
            print(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
            print(f"  Balanced Accuracy: {result['metrics']['balanced_accuracy']:.4f}")
            print(f"  F1 Macro: {result['metrics']['f1_macro']:.4f}")

            return result

        return None

    def _adjust_batch_size(self, loader: DataLoader) -> DataLoader:
        """Create a new data loader with adjusted batch size."""
        return DataLoader(
            loader.dataset,
            batch_size=self.config.tta_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory
        )


def initialize_unknown_subject_embeddings(
    model: nn.Module,
    train_subject_ids: list,
    test_subject_ids: list
):
    """
    Initialize embeddings for unknown test subjects.

    For subjects not seen during training, initialize their embeddings
    as the mean of training subject embeddings.

    Args:
        model: SageStream model
        train_subject_ids: List of training subject IDs
        test_subject_ids: List of test subject IDs
    """
    # Skip if SA-MoE is disabled (no subject embeddings to initialize)
    use_moe = getattr(model.model, 'use_moe', True) if hasattr(model, 'model') else True
    if not use_moe:
        return

    train_subjects = set(train_subject_ids)
    test_subjects = set(test_subject_ids)
    unknown_subjects = test_subjects - train_subjects

    if len(unknown_subjects) == 0:
        return

    # Find subject embedding modules
    subject_embedding_modules = []
    for block in model.model.encoder.block:
        if hasattr(block, 'shared_knowledge') and hasattr(block.shared_knowledge, 'subject_embedding'):
            if hasattr(block.shared_knowledge.subject_embedding, 'weight'):
                subject_embedding_modules.append(block.shared_knowledge.subject_embedding)

    # Initialize unknown subject embeddings
    for subject_embedding in subject_embedding_modules:
        num_embeddings = subject_embedding.weight.shape[0]

        with torch.no_grad():
            # Map subject IDs to valid embedding indices using modulo
            train_subject_indices = [sid % num_embeddings for sid in train_subjects]
            # Filter to valid unique indices
            train_subject_indices = list(set(idx for idx in train_subject_indices if 0 <= idx < num_embeddings))

            if not train_subject_indices:
                continue

            train_embeddings = subject_embedding.weight[train_subject_indices]
            mean_embedding = train_embeddings.mean(dim=0)

            for unknown_subject in unknown_subjects:
                unknown_index = unknown_subject % num_embeddings
                if 0 <= unknown_index < num_embeddings:
                    subject_embedding.weight[unknown_index].copy_(mean_embedding)

    print(f"Initialized embeddings for {len(unknown_subjects)} unknown subjects")
