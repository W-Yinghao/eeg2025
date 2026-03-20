"""
DEB loss functions.

Minimal loss:
    L = L_ce + beta * L_kl + sparse_lambda * L_sparse

Components:
  - L_ce: weighted cross-entropy
  - L_kl: KL divergence from the variational bottleneck (with warmup)
  - L_sparse: L1 on gate activations (encourage sparse evidence selection)

Consistency loss interface is reserved but disabled by default.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class DEBLoss(nn.Module):
    """
    Combined loss for DEB training.

    For baseline mode, only L_ce is active (pass adapter_out=None).
    For DEB mode, L_kl and L_sparse are added.
    """

    def __init__(
        self,
        beta: float = 1e-4,
        beta_warmup_epochs: int = 5,
        sparse_lambda: float = 1e-3,
        enable_sparse_reg: bool = True,
        enable_consistency: bool = False,
        consistency_lambda: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.beta = beta
        self.beta_warmup_epochs = beta_warmup_epochs
        self.sparse_lambda = sparse_lambda
        self.enable_sparse_reg = enable_sparse_reg
        self.enable_consistency = enable_consistency
        self.consistency_lambda = consistency_lambda
        self.label_smoothing = label_smoothing

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        adapter_out: Optional[Dict] = None,
        current_epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            logits: (B, num_classes)
            labels: (B,)
            adapter_out: DEB head output dict (with 'kl', 'temporal_gate', etc.)
                         None for baseline mode.
            current_epoch: for KL warmup

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
            'kl': 0.0,
            'sparse': 0.0,
            'consistency': 0.0,
        }

        total = loss_ce

        if adapter_out is not None:
            # KL loss with warmup
            kl = adapter_out.get('kl', torch.tensor(0.0, device=logits.device))
            warmup = min(current_epoch / max(self.beta_warmup_epochs, 1), 1.0)
            kl_weighted = self.beta * warmup * kl
            total = total + kl_weighted
            loss_dict['kl'] = kl.item()
            loss_dict['kl_weighted'] = kl_weighted.item()

            # Sparse gate regularization
            if self.enable_sparse_reg and self.sparse_lambda > 0:
                sparse_loss = torch.tensor(0.0, device=logits.device)
                n_gates = 0
                for key in ('temporal_gate', 'frequency_gate'):
                    gate = adapter_out.get(key)
                    if gate is not None:
                        sparse_loss = sparse_loss + gate.mean()
                        n_gates += 1
                if n_gates > 0:
                    sparse_loss = sparse_loss / n_gates
                    total = total + self.sparse_lambda * sparse_loss
                    loss_dict['sparse'] = sparse_loss.item()

            # Consistency loss placeholder
            if self.enable_consistency and self.consistency_lambda > 0:
                # TODO: implement cross-view consistency
                pass

        loss_dict['total'] = total.item()
        return total, loss_dict
