"""
Loss functions for selector experiments (Exp 6/7/7C).

Supports:
  - CE only (Exp 6: baseline and selector)
  - CE + sparse regularization (Exp 7A)
  - CE + consistency regularization (Exp 7B)
  - CE + sparse + consistency (Exp 7)
  - CE + branch-aware sparse + consistency (Exp 7C)
  - CE + VIB KL (extension point, disabled by default)

Sparse regularization options:
  - L1 on gate activations (encourage sparse selection)
  - Gate entropy penalty (encourage binary decisions)
  - Coverage penalty (penalize selecting too many tokens)
  - Branch-aware: separate lambda per gate branch (temporal vs frequency)

Consistency regularization:
  - Applied between gate maps of original and augmented views
  - Supports L2, cosine, and KL divergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SelectorLoss(nn.Module):
    """Combined loss for selector experiments."""

    def __init__(
        self,
        # Sparse regularization
        enable_sparse: bool = False,
        sparse_lambda: float = 1e-3,
        sparse_type: str = 'l1',  # 'l1' | 'entropy' | 'coverage'
        # Branch-aware sparse (Exp 7C): per-gate lambdas override sparse_lambda
        sparse_lambda_temporal: Optional[float] = None,
        sparse_lambda_frequency: Optional[float] = None,
        # Consistency regularization
        enable_consistency: bool = False,
        consistency_lambda: float = 1e-2,
        consistency_type: str = 'l2',  # 'l2' | 'cosine' | 'kl'
        # VIB extension (disabled by default)
        enable_vib: bool = False,
        vib_beta: float = 1e-4,
        vib_warmup_epochs: int = 5,
        # CE options
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.enable_sparse = enable_sparse
        self.sparse_lambda = sparse_lambda
        self.sparse_type = sparse_type
        self.sparse_lambda_temporal = sparse_lambda_temporal
        self.sparse_lambda_frequency = sparse_lambda_frequency
        self._branch_aware = (sparse_lambda_temporal is not None
                              or sparse_lambda_frequency is not None)
        self.enable_consistency = enable_consistency
        self.consistency_lambda = consistency_lambda
        self.consistency_type = consistency_type
        self.enable_vib = enable_vib
        self.vib_beta = vib_beta
        self.vib_warmup_epochs = vib_warmup_epochs
        self.label_smoothing = label_smoothing

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        model_out: Optional[Dict] = None,
        aug_model_out: Optional[Dict] = None,
        current_epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss.

        Args:
            logits: (B, num_classes)
            labels: (B,)
            model_out: selector output dict (with gates, evidence)
            aug_model_out: selector output from augmented view (for consistency)
            current_epoch: for VIB warmup

        Returns:
            total_loss: scalar
            loss_dict: breakdown for logging
        """
        # CE loss
        loss_ce = F.cross_entropy(
            logits, labels,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

        loss_dict = {
            'ce': loss_ce.item(),
            'sparse': 0.0,
            'consistency': 0.0,
            'vib_kl': 0.0,
        }

        total = loss_ce

        if model_out is not None:
            # Sparse regularization
            if self.enable_sparse and self._branch_aware:
                # Branch-aware sparse (Exp 7C): separate lambda per gate
                sp_t, sp_f = self._compute_sparse_per_branch(model_out)
                lam_t = self.sparse_lambda_temporal or 0.0
                lam_f = self.sparse_lambda_frequency or 0.0
                sparse_weighted = lam_t * sp_t + lam_f * sp_f
                total = total + sparse_weighted
                # Log individual branch losses + combined for compatibility
                n = int(sp_t > 0) + int(sp_f > 0)
                loss_dict['sparse'] = ((sp_t.item() + sp_f.item()) / max(n, 1))
                loss_dict['sparse_temporal'] = sp_t.item()
                loss_dict['sparse_frequency'] = sp_f.item()
            elif self.enable_sparse and self.sparse_lambda > 0:
                sparse_loss = self._compute_sparse_loss(model_out)
                total = total + self.sparse_lambda * sparse_loss
                loss_dict['sparse'] = sparse_loss.item()
                # Per-branch breakdown for analysis (always available)
                loss_dict['sparse_temporal'] = self._last_sparse_per_branch.get('temporal', 0.0)
                loss_dict['sparse_frequency'] = self._last_sparse_per_branch.get('frequency', 0.0)

            # Consistency regularization
            if (self.enable_consistency and self.consistency_lambda > 0
                    and aug_model_out is not None):
                consistency_loss = self._compute_consistency_loss(
                    model_out, aug_model_out
                )
                total = total + self.consistency_lambda * consistency_loss
                loss_dict['consistency'] = consistency_loss.item()

            # VIB KL (extension point)
            if self.enable_vib:
                kl = model_out.get('kl', torch.tensor(0.0, device=logits.device))
                warmup = min(current_epoch / max(self.vib_warmup_epochs, 1), 1.0)
                kl_weighted = self.vib_beta * warmup * kl
                total = total + kl_weighted
                loss_dict['vib_kl'] = kl.item()

        loss_dict['total'] = total.item()
        return total, loss_dict

    def _compute_sparse_loss(self, model_out: Dict) -> torch.Tensor:
        """Compute sparse regularization on gate activations.

        Also stores per-branch values in self._last_sparse_per_branch
        for logging even in uniform mode.
        """
        device = model_out['logits'].device
        sparse_loss = torch.tensor(0.0, device=device)
        n_gates = 0
        self._last_sparse_per_branch = {}

        for key in ('temporal_gate', 'frequency_gate'):
            gate = model_out.get(key)
            if gate is None:
                continue

            val = self._sparse_single_gate(gate)
            sparse_loss = sparse_loss + val
            branch = 'temporal' if key == 'temporal_gate' else 'frequency'
            self._last_sparse_per_branch[branch] = val.item()
            n_gates += 1

        if n_gates > 0:
            sparse_loss = sparse_loss / n_gates

        return sparse_loss

    def _compute_sparse_per_branch(
        self, model_out: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute sparse loss separately for temporal and frequency gates.

        Returns (sparse_temporal, sparse_frequency) as individual scalars.
        Used by branch-aware sparse mode (Exp 7C) where each branch has
        its own lambda.
        """
        device = model_out['logits'].device
        sparse_t = torch.tensor(0.0, device=device)
        sparse_f = torch.tensor(0.0, device=device)

        for key, target in [('temporal_gate', 'temporal'),
                            ('frequency_gate', 'frequency')]:
            gate = model_out.get(key)
            if gate is None:
                continue
            val = self._sparse_single_gate(gate)
            if target == 'temporal':
                sparse_t = val
            else:
                sparse_f = val

        return sparse_t, sparse_f

    def _sparse_single_gate(self, gate: torch.Tensor) -> torch.Tensor:
        """Compute sparse penalty for a single gate tensor."""
        if self.sparse_type == 'l1':
            return gate.mean()
        elif self.sparse_type == 'entropy':
            g = gate.squeeze(-1).clamp(1e-7, 1 - 1e-7)
            entropy = -(g * g.log() + (1 - g) * (1 - g).log())
            return entropy.mean()
        elif self.sparse_type == 'coverage':
            return (gate > 0.5).float().mean()
        return torch.tensor(0.0, device=gate.device)

    def _compute_consistency_loss(
        self,
        orig_out: Dict,
        aug_out: Dict,
    ) -> torch.Tensor:
        """Compute consistency between original and augmented gate maps."""
        device = orig_out['logits'].device
        consistency_loss = torch.tensor(0.0, device=device)
        n_terms = 0

        for key in ('temporal_gate', 'frequency_gate'):
            g_orig = orig_out.get(key)
            g_aug = aug_out.get(key)
            if g_orig is None or g_aug is None:
                continue

            # Handle potential size mismatch from augmentation
            min_len = min(g_orig.shape[1], g_aug.shape[1])
            g_orig = g_orig[:, :min_len]
            g_aug = g_aug[:, :min_len]

            if self.consistency_type == 'l2':
                consistency_loss = consistency_loss + F.mse_loss(g_orig, g_aug)
            elif self.consistency_type == 'cosine':
                g_o = g_orig.squeeze(-1)
                g_a = g_aug.squeeze(-1)
                cos_sim = F.cosine_similarity(g_o, g_a, dim=1)
                consistency_loss = consistency_loss + (1.0 - cos_sim).mean()
            elif self.consistency_type == 'kl':
                # Normalize to probability distributions
                g_o = F.softmax(g_orig.squeeze(-1), dim=1)
                g_a = F.softmax(g_aug.squeeze(-1), dim=1)
                kl = F.kl_div(g_a.log(), g_o, reduction='batchmean')
                consistency_loss = consistency_loss + kl
            n_terms += 1

        if n_terms > 0:
            consistency_loss = consistency_loss / n_terms

        return consistency_loss
