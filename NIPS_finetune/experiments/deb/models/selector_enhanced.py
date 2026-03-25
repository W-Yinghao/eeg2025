"""
Enhanced Selector Head with sparse/consistency regularization support.

Extends the original SelectorHead with:
  1. Gate export interfaces for explainability evaluation
  2. Sparse regularization support (L1, entropy, coverage)
  3. Consistency regularization support (gate map consistency under augmentation)
  4. Extension point for VIB (disabled by default)

Architecture:
    backbone features (B, C, S, D)
      |-- temporal gate: attention over S -> gated H_t (B, S, D)
      |-- frequency gate: attention over C -> gated H_f (B, C, D)
      |-- fuse -> (B, D) or (B, 2*D)
      |-- classifier(fused) -> logits (B, num_classes)

Gate outputs are always available via return_gates=True for explainability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .evidence_bottleneck import TemporalGate, FrequencyGate


class EnhancedSelectorHead(nn.Module):
    """
    Selector head with full gate export and regularization support.

    Compared to the original SelectorHead:
      - Always exports gate maps when return_gates=True
      - Exports fused evidence representation for analysis
      - Supports sparse regularization computation
      - Extension point for VIB (disabled by default, can be enabled later)
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
        # VIB extension point (disabled by default)
        enable_vib: bool = False,
        vib_latent_dim: int = 64,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.fusion = fusion
        self.enable_temporal_gate = enable_temporal_gate
        self.enable_frequency_gate = enable_frequency_gate
        self.enable_vib = enable_vib

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

        # VIB extension point (disabled by default)
        if enable_vib:
            self.vib_encoder = nn.Sequential(
                nn.Linear(fuse_dim, fuse_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.fc_mu = nn.Linear(fuse_dim, vib_latent_dim)
            self.fc_logvar = nn.Linear(fuse_dim, vib_latent_dim)
            classifier_in = vib_latent_dim
        else:
            classifier_in = fuse_dim

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, classifier_in),
            nn.BatchNorm1d(classifier_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_in, num_classes),
        )

    def forward(self, features: torch.Tensor,
                return_gates: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, C, S, D) backbone output
            return_gates: if True, include all gate/evidence info in output

        Returns:
            dict with:
              'logits':  (B, num_classes)
            if return_gates:
              'temporal_gate':    (B, S, 1) sigmoid gate activations
              'frequency_gate':   (B, C, 1) sigmoid gate activations
              'fused_evidence':   (B, fuse_dim) fused evidence before classifier
              'temporal_evidence': (B, S, D) gated temporal features
              'frequency_evidence': (B, C, D) gated frequency features
            if enable_vib and return_gates:
              'mu':     (B, latent_dim)
              'logvar': (B, latent_dim)
              'kl':     scalar
        """
        H_t, H_f = self._extract_views(features)

        # Apply gates
        gate_t = gate_f = None
        H_t_gated = H_f_gated = None

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

        # VIB path (extension point)
        if self.enable_vib:
            a = self.vib_encoder(fused)
            mu = self.fc_mu(a)
            logvar = torch.clamp(self.fc_logvar(a), -10, 10)
            if self.training:
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
            else:
                z = mu
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            logits = self.classifier(z)
        else:
            logits = self.classifier(fused)

        out = {'logits': logits}

        if return_gates:
            out['temporal_gate'] = gate_t
            out['frequency_gate'] = gate_f
            out['fused_evidence'] = fused
            if H_t_gated is not None:
                out['temporal_evidence'] = H_t_gated
            if H_f_gated is not None:
                out['frequency_evidence'] = H_f_gated
            if self.enable_vib:
                out['mu'] = mu
                out['logvar'] = logvar
                out['kl'] = kl

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

    def get_gate_sparsity_stats(self, gate_t, gate_f) -> Dict[str, float]:
        """Compute gate sparsity statistics for logging."""
        stats = {}
        if gate_t is not None:
            stats['temporal_gate_mean'] = gate_t.mean().item()
            stats['temporal_gate_std'] = gate_t.std().item()
            stats['temporal_active_ratio'] = (gate_t > 0.5).float().mean().item()
            stats['temporal_entropy'] = self._gate_entropy(gate_t).item()
        if gate_f is not None:
            stats['frequency_gate_mean'] = gate_f.mean().item()
            stats['frequency_gate_std'] = gate_f.std().item()
            stats['frequency_active_ratio'] = (gate_f > 0.5).float().mean().item()
            stats['frequency_entropy'] = self._gate_entropy(gate_f).item()
        return stats

    @staticmethod
    def _gate_entropy(gate: torch.Tensor) -> torch.Tensor:
        """Compute binary entropy of gate activations."""
        g = gate.squeeze(-1).clamp(1e-7, 1 - 1e-7)
        entropy = -(g * g.log() + (1 - g) * (1 - g).log())
        return entropy.mean()
