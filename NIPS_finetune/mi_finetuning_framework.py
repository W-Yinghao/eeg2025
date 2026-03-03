#!/usr/bin/env python3
"""
Information-Theoretic Fine-Tuning Framework for EEG Foundation Models

Combines:
1. InfoMax (MI Maximization): Align foundation model representations with expert features
2. Variational Information Bottleneck (VIB): Suppress subject-specific noise

Backbone: CodeBrain (SSSM) - frozen pretrained encoder

Architecture:
    Raw EEG (B, C, S, P) -> Frozen SSSM Backbone -> (B, C, S, 200)
                                                        |
                                               RepProjection (trainable)
                                                        |
                                                   Z_FM (B, H)
                                                    /        \\
                                              VIB Layer    ContrastHead
                                                 |              |
                                            z (B, V)      z_proj (B, H)
                                                 |              |
                                            Classifier     InfoNCE loss
                                                 |        with Z_expert
                                            CE loss

    Expert Features -> ExpertProjector -> Z_expert (B, H)

    Loss = CE(logits, labels) + beta*KL(mu, log_var) + alpha*InfoNCE(z_proj, Z_expert)
"""

import math
import os
import sys
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Backbone factory (shared across all frameworks)
from backbone_factory import create_backbone


# ==============================================================================
# 1. Expert Feature Projector
# ==============================================================================

class ExpertProjector(nn.Module):
    """
    Project expert features (PSD, stats, etc.) into the same embedding space
    as the foundation model representation for contrastive learning.
    """

    def __init__(self, expert_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(expert_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x_expert: torch.Tensor) -> torch.Tensor:
        """(B, expert_dim) -> (B, hidden_dim), L2-normalized."""
        out = self.projector(x_expert)
        return F.normalize(out, p=2, dim=-1)


# ==============================================================================
# 3. Variational Information Bottleneck (VIB)
# ==============================================================================

class VIBLayer(nn.Module):
    """
    Encode Z_FM as Gaussian N(mu, sigma^2), sample via reparameterization.
    KL regularization pushes toward N(0, I), compressing subject-specific noise.
    """

    def __init__(self, input_dim: int, vib_dim: int, dropout: float = 0.1):
        super().__init__()
        self.vib_dim = vib_dim

        self.fc_mu = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, vib_dim),
        )
        self.fc_log_var = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, vib_dim),
        )

    def forward(self, Z_FM: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(B, input_dim) -> z (B, vib_dim), mu, log_var."""
        mu = self.fc_mu(Z_FM)
        log_var = torch.clamp(self.fc_log_var(Z_FM), min=-10, max=10)

        if self.training:
            std = torch.exp(0.5 * log_var)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu

        return z, mu, log_var


# ==============================================================================
# 4. Classification Head
# ==============================================================================

class ClassifierHead(nn.Module):
    """MLP classifier on top of VIB bottleneck."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.BatchNorm1d(input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier(z)


# ==============================================================================
# 5. MIFineTuner - Main Wrapper
# ==============================================================================

class MIFineTuner(nn.Module):
    """
    Information-Theoretic Fine-Tuning wrapper for a frozen backbone.

    Trainable components:
      - rep_projection: Flatten + project backbone output to hidden_dim
      - vib_layer: Variational Information Bottleneck
      - classifier: Classification head on VIB output
      - contrast_head: Projection head for InfoNCE (on Z_FM)
      - expert_projector: Project expert features for InfoNCE

    Args:
        backbone: Pre-trained model (will be frozen)
        backbone_out_dim: Flattened size of backbone output (e.g. 16*5*200)
        expert_dim: Dimension of expert features (e.g. n_channels * 5 for PSD)
        hidden_dim: Internal representation dimension
        vib_dim: VIB bottleneck dimension (compression target)
        num_classes: Number of output classes
        dropout: Dropout probability
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_out_dim: int,
        expert_dim: int,
        hidden_dim: int = 256,
        vib_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.backbone = backbone
        self.backbone_out_dim = backbone_out_dim

        # Check if backbone has trainable adapter params (CBraMod with adapters)
        self.has_backbone_adapters = any(
            p.requires_grad for p in self.backbone.parameters()
        )

        # Trainable: flatten backbone output -> hidden_dim
        self.rep_projection = nn.Sequential(
            nn.Linear(backbone_out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Trainable: VIB
        self.vib_layer = VIBLayer(hidden_dim, vib_dim, dropout)

        # Trainable: classifier
        self.classifier = ClassifierHead(vib_dim, num_classes, dropout)

        # Trainable: contrastive projection head for InfoNCE
        # Applied on Z_FM so InfoNCE trains rep_projection through this path
        self.contrast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Trainable: expert feature projector
        self.expert_projector = ExpertProjector(expert_dim, hidden_dim, dropout)

        # Print parameter summary
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = frozen + trainable
        print(f"MIFineTuner parameter summary:")
        print(f"  Frozen params:     {frozen:,}")
        print(f"  Trainable params:  {trainable:,}")
        print(f"  Trainable ratio:   {trainable / total * 100:.2f}%")
        if self.has_backbone_adapters:
            adapter_params = sum(
                p.numel() for name, p in self.backbone.named_parameters()
                if p.requires_grad
            )
            print(f"  Backbone adapters: {adapter_params:,}")

    def forward(
        self, x: torch.Tensor, x_expert: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: EEG input (B, n_channels, seq_len, patch_size) for CodeBrain
            x_expert: Pre-computed expert features (B, expert_dim)

        Returns:
            logits: (B, num_classes)
            mu: VIB mean (B, vib_dim)
            log_var: VIB log variance (B, vib_dim)
            z_fm_proj: Normalized FM projection for InfoNCE (B, hidden_dim)
            z_expert_proj: Normalized expert projection for InfoNCE (B, hidden_dim)
        """
        B = x.shape[0]

        # Backbone forward
        # When adapters have gradients, skip no_grad to allow backprop through them
        if self.has_backbone_adapters:
            backbone_out = self.backbone(x)  # (B, C, S, 200)
        else:
            with torch.no_grad():
                backbone_out = self.backbone(x)  # (B, C, S, 200)

        # Flatten backbone output
        backbone_flat = backbone_out.reshape(B, -1)  # (B, backbone_out_dim)

        # Project to hidden representation (TRAINABLE - key gradient entry point)
        Z_FM = self.rep_projection(backbone_flat)  # (B, hidden_dim)

        # VIB path: compress and classify
        z, mu, log_var = self.vib_layer(Z_FM)  # z: (B, vib_dim)
        logits = self.classifier(z)  # (B, num_classes)

        # InfoNCE path: project Z_FM for contrastive learning
        # Gradients from InfoNCE flow back through contrast_head -> Z_FM -> rep_projection
        z_fm_proj = F.normalize(self.contrast_head(Z_FM), p=2, dim=-1)

        # Project expert features
        z_expert_proj = self.expert_projector(x_expert)  # (B, hidden_dim), normalized

        return logits, mu, log_var, z_fm_proj, z_expert_proj


# ==============================================================================
# 6. Loss Functions
# ==============================================================================

def compute_vib_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    KL divergence: D_KL(q(z|x) || N(0,I)).
    = -0.5 * sum(1 + log_var - mu^2 - exp(log_var)), averaged over batch.
    """
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return kl / mu.size(0)


def compute_infonce_loss(
    z_fm: torch.Tensor,
    z_expert: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Symmetric InfoNCE loss between FM projections and expert projections.

    Positive pairs: (z_fm[i], z_expert[i]) from the same sample.
    Negative pairs: all other combinations in the batch.

    Both inputs must be L2-normalized.
    """
    B = z_fm.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=z_fm.device)

    # Cosine similarity matrix: (B, B)
    sim = torch.matmul(z_fm, z_expert.T) / temperature

    labels = torch.arange(B, device=z_fm.device)

    # Symmetric InfoNCE
    loss_fm2exp = F.cross_entropy(sim, labels)
    loss_exp2fm = F.cross_entropy(sim.T, labels)

    return (loss_fm2exp + loss_exp2fm) / 2


def calculate_mi_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    z_fm_proj: torch.Tensor,
    z_expert_proj: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1e-3,
    temperature: float = 0.07,
    task_type: str = 'multiclass',
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Total loss = CE + beta * VIB_KL + alpha * InfoNCE.

    Args:
        logits, labels: Classification outputs and targets
        mu, log_var: VIB parameters
        z_fm_proj, z_expert_proj: L2-normalized projections for InfoNCE
        alpha: InfoNCE weight
        beta: VIB weight
        temperature: InfoNCE temperature
        task_type: 'multiclass' or 'binary'
        label_smoothing: Label smoothing factor (0 = no smoothing)

    Returns:
        total_loss, loss_dict (for logging)
    """
    # Classification (with optional label smoothing)
    if task_type == 'binary':
        targets = labels.float()
        if label_smoothing > 0:
            targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing
        loss_ce = F.binary_cross_entropy_with_logits(logits.squeeze(), targets)
    else:
        loss_ce = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)

    # VIB
    loss_vib = compute_vib_loss(mu, log_var)

    # InfoNCE
    loss_infonce = compute_infonce_loss(z_fm_proj, z_expert_proj, temperature)

    total = loss_ce + beta * loss_vib + alpha * loss_infonce

    loss_dict = {
        'total': total.item(),
        'ce': loss_ce.item(),
        'vib': loss_vib.item(),
        'infonce': loss_infonce.item(),
        'weighted_vib': (beta * loss_vib).item(),
        'weighted_infonce': (alpha * loss_infonce).item(),
    }
    return total, loss_dict
