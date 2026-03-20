"""
Minimal Disease Evidence Bottleneck (DEB) head.

Architecture:
    backbone features (B, C, S, D)
      ├─ temporal gate: attention over S → gated H_t (B, S, D)
      ├─ frequency gate: attention over C → gated H_f (B, C, D)
      ├─ fuse → (B, D) or (B, 2*D)
      ├─ mu, logvar
      ├─ z = reparameterize(mu, logvar)
      └─ classifier(z) → logits (B, num_classes)

NOT implemented in this minimal version:
  - channel gate (per-electrode attention conditioned on montage)
  - shared-private disentanglement
  - HSIC / MMD invariance
  - montage consistency loss
  - multi-view raw/spec/spatial parallel branches
  - query scorer

Extension points are marked with comments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class TemporalGate(nn.Module):
    """
    Attention gate over the temporal (S) axis.

    Input:  H_t (B, S, D)   — backbone features pooled over channels
    Output: gated (B, S, D) — element-wise gated features
            gate  (B, S, 1) — gate activations for interpretability

    The gate scores each time-step's relevance to the classification task.
    """

    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, S, D)
        Returns:
            gated: (B, S, D)
            gate:  (B, S, 1) sigmoid activations
        """
        gate = torch.sigmoid(self.attn(h))  # (B, S, 1)
        return h * gate, gate


class FrequencyGate(nn.Module):
    """
    Attention gate over the channel/frequency (C) axis.

    For CodeBrain/CBraMod, the C axis corresponds to EEG channels.
    The gate scores each channel's diagnostic relevance.

    Input:  H_f (B, C, D)   — backbone features pooled over time
    Output: gated (B, C, D) — gated features
            gate  (B, C, 1) — gate activations
    """

    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.attn(h))  # (B, C, 1)
        return h * gate, gate


class EvidenceBottleneck(nn.Module):
    """
    Minimal Disease Evidence Bottleneck.

    Pipeline:
      1. Extract H_t (B, S, D) and H_f (B, C, D) from backbone output
      2. Apply temporal gate and frequency gate
      3. Pool and fuse gated evidence
      4. Variational bottleneck: mu, logvar, reparameterize → z_e
      5. Classify from z_e

    The gates provide interpretability: which time windows and which
    channels contribute most to the disease prediction.

    # ── Extension point: channel gate ──────────────────────────────
    # A future ChannelGate conditioned on montage_id / channel_coordinates
    # would replace or augment the FrequencyGate.
    #
    # ── Extension point: shared-private split ──────────────────────
    # z_e could be split into z_shared (disease) and z_private (subject),
    # with an adversarial loss on z_private.
    #
    # ── Extension point: consistency loss ──────────────────────────
    # A cross-view consistency term between H_t-derived and H_f-derived
    # evidence could be added as an auxiliary loss.
    """

    def __init__(
        self,
        token_dim: int,
        num_classes: int,
        latent_dim: int = 64,
        gate_hidden: int = 64,
        enable_temporal_gate: bool = True,
        enable_frequency_gate: bool = True,
        fusion: str = 'concat',  # 'concat' | 'add'
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.latent_dim = latent_dim
        self.fusion = fusion
        self.enable_temporal_gate = enable_temporal_gate
        self.enable_frequency_gate = enable_frequency_gate

        # Gates
        if enable_temporal_gate:
            self.temporal_gate = TemporalGate(token_dim, gate_hidden)
        if enable_frequency_gate:
            self.frequency_gate = FrequencyGate(token_dim, gate_hidden)

        # Fusion dimension
        if fusion == 'concat' and enable_temporal_gate and enable_frequency_gate:
            fuse_dim = token_dim * 2
        else:
            fuse_dim = token_dim

        # Variational bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(fuse_dim, fuse_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fc_mu = nn.Linear(fuse_dim, latent_dim)
        self.fc_logvar = nn.Linear(fuse_dim, latent_dim)

        # Classifier from latent
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, num_classes),
        )

    def _extract_views(self, features: torch.Tensor):
        """
        Extract H_t and H_f from backbone output.

        Args:
            features: (B, C, S, D) or (B, T, D)

        Returns:
            H_t: (B, S, D) or (B, T, D)
            H_f: (B, C, D) or None
        """
        if features.dim() == 4:
            H_t = features.mean(dim=1)  # (B, S, D)
            H_f = features.mean(dim=2)  # (B, C, D)
        elif features.dim() == 3:
            H_t = features              # (B, T, D)
            H_f = None
        else:
            H_t = features.unsqueeze(1)  # (B, 1, D)
            H_f = None
        return H_t, H_f

    def forward(self, features: torch.Tensor,
                return_gates: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, C, S, D) backbone output
            return_gates: if True, include gate activations in output

        Returns:
            dict with:
              'logits':  (B, num_classes)
              'mu':      (B, latent_dim)
              'logvar':  (B, latent_dim)
              'z_e':     (B, latent_dim) evidence embedding
              'kl':      scalar KL divergence
            optionally:
              'temporal_gate':  (B, S, 1)
              'frequency_gate': (B, C, 1)
        """
        B = features.shape[0]
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
            else:  # add
                fused = e_t + e_f  # (B, D)
        else:
            fused = e_t  # (B, D)

        # Variational bottleneck
        a = self.encoder(fused)
        mu = self.fc_mu(a)                               # (B, latent_dim)
        logvar = torch.clamp(self.fc_logvar(a), -10, 10) # (B, latent_dim)

        if self.training:
            std = torch.exp(0.5 * logvar)
            z_e = mu + std * torch.randn_like(std)
        else:
            z_e = mu

        # KL divergence
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Classify
        logits = self.classifier(z_e)

        out = {
            'logits': logits,
            'mu': mu,
            'logvar': logvar,
            'z_e': z_e,
            'kl': kl,
        }
        if return_gates:
            out['temporal_gate'] = gate_t
            out['frequency_gate'] = gate_f

        return out
