"""
Selector-only head: gates + fusion + classifier, NO variational bottleneck.

Architecture:
    backbone features (B, C, S, D)
      ├─ temporal gate: attention over S → gated H_t (B, S, D)
      ├─ frequency gate: attention over C → gated H_f (B, C, D)
      ├─ fuse → (B, D) or (B, 2*D)
      └─ classifier(fused) → logits (B, num_classes)

Removed vs full DEB:
  - No mu / logvar / reparameterize / KL
  - No sparse regularization (gates are trained purely via CE gradient)
  - No latent_dim bottleneck — classifier operates on fused evidence directly

Purpose: ablation to test whether the gating/selection mechanism alone
(without the information bottleneck) improves over a plain head.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from .evidence_bottleneck import TemporalGate, FrequencyGate


class SelectorHead(nn.Module):
    """
    Gate-select + fuse + classify.  No variational bottleneck.
    """

    def __init__(
        self,
        token_dim: int,
        num_classes: int,
        gate_hidden: int = 64,
        enable_temporal_gate: bool = True,
        enable_frequency_gate: bool = True,
        fusion: str = 'concat',
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.fusion = fusion
        self.enable_temporal_gate = enable_temporal_gate
        self.enable_frequency_gate = enable_frequency_gate

        # Reuse the same gate modules from DEB
        if enable_temporal_gate:
            self.temporal_gate = TemporalGate(token_dim, gate_hidden)
        if enable_frequency_gate:
            self.frequency_gate = FrequencyGate(token_dim, gate_hidden)

        # Fusion dimension
        if fusion == 'concat' and enable_temporal_gate and enable_frequency_gate:
            fuse_dim = token_dim * 2
        else:
            fuse_dim = token_dim

        # Classifier directly on fused evidence (no latent bottleneck)
        self.classifier = nn.Sequential(
            nn.Linear(fuse_dim, fuse_dim),
            nn.BatchNorm1d(fuse_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fuse_dim, num_classes),
        )

    def forward(self, features: torch.Tensor,
                return_gates: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, C, S, D) backbone output
            return_gates: if True, include gate activations in output

        Returns:
            dict with:
              'logits':  (B, num_classes)
            optionally:
              'temporal_gate':  (B, S, 1)
              'frequency_gate': (B, C, 1)
        """
        H_t, H_f = self._extract_views(features)

        # Apply gates
        gate_t = gate_f = None
        if self.enable_temporal_gate:
            H_t_gated, gate_t = self.temporal_gate(H_t)
            e_t = H_t_gated.mean(dim=1)  # (B, D)
        else:
            e_t = H_t.mean(dim=1)

        if self.enable_frequency_gate and H_f is not None:
            H_f_gated, gate_f = self.frequency_gate(H_f)
            e_f = H_f_gated.mean(dim=1)  # (B, D)
        else:
            e_f = None

        # Fuse
        if e_f is not None:
            if self.fusion == 'concat':
                fused = torch.cat([e_t, e_f], dim=-1)  # (B, 2*D)
            else:
                fused = e_t + e_f  # (B, D)
        else:
            fused = e_t

        # Classify directly — no VIB
        logits = self.classifier(fused)

        out = {'logits': logits}
        if return_gates:
            out['temporal_gate'] = gate_t
            out['frequency_gate'] = gate_f

        return out

    @staticmethod
    def _extract_views(features: torch.Tensor):
        if features.dim() == 4:
            H_t = features.mean(dim=1)  # (B, S, D)
            H_f = features.mean(dim=2)  # (B, C, D)
        elif features.dim() == 3:
            H_t = features
            H_f = None
        else:
            H_t = features.unsqueeze(1)
            H_f = None
        return H_t, H_f
