#!/usr/bin/env python3
"""
Information Bottleneck + Disentanglement Fine-Tuning Framework for EEG Foundation Models

Compared to the previous MI fine-tuning framework (mi_finetuning_framework.py):
  - Token-level IB: preserves spatial-temporal token structure from backbone (B,T,D)
    instead of flattening everything to a single vector
  - Explicit subject disentanglement via Gradient Reversal Layer (GRL) instead of
    indirect InfoNCE + VIB approach
  - Clinical interpretability via per-token information retention heatmaps

Architecture:
    Raw EEG (B, C, S, P) -> Frozen SSSM Backbone -> H (B, C, S, 200)
                                                       |
                                              Reshape to (B, T, D)  [T=C*S, D=200]
                                                       |
                                              IB Adapter (token-level)
                                           mu (B,T,D'), logvar (B,T,D')
                                                       |
                                             Reparameterize -> Z (B,T,D')
                                                       |
                                              Aggregate -> z_agg (B, D')
                                                    /            \\
                                          Disease Head        Subject Head + GRL
                                               |                    |
                                        disease logits         subject logits
                                               |                    |
                                           L_task             L_MI (adversarial)
                                                       +
                                                   L_IB (KL)

    Loss = L_task + beta * L_IB + lambda_adv * L_MI(via GRL)

Key improvements over MI finetuning:
  1. Token-level processing preserves spatial-temporal structure
  2. GRL explicitly minimizes I(Z; Subject) without needing expert features
  3. Per-token KL gives interpretable information retention maps
  4. No need for hand-crafted expert features (PSD, stats)
"""

import math
import os
import sys
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# Backbone factory (shared across all frameworks)
from backbone_factory import create_backbone


# ==============================================================================
# 1. Gradient Reversal Layer (GRL)
# ==============================================================================

class _GradientReversalFunction(Function):
    """Autograd function that reverses gradients during backward pass."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        return -lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer for adversarial training.

    During forward pass: identity function.
    During backward pass: reverses gradients scaled by lambda_.

    This allows joint training of the encoder and adversarial head without
    alternating min-max optimization. The encoder learns to REMOVE subject
    information because reversed gradients push it away from subject-predictive
    representations.
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.register_buffer('lambda_', torch.tensor(lambda_))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        self.lambda_.fill_(lambda_)


# ==============================================================================
# 3. Information Bottleneck Adapter (Token-Level)
# ==============================================================================

class CodeBrain_IB_Adapter(nn.Module):
    """
    Token-level Information Bottleneck adapter.

    Maps deterministic backbone tokens H (B, T, D) to stochastic latent
    variables Z (B, T, D') via the reparameterization trick:
        mu, log_var = f(H)
        Z = mu + sigma * eps,  eps ~ N(0, I)

    At test time, Z = mu (no sampling).

    Unlike the old VIBLayer which operated on a single flattened vector,
    this preserves the token structure, enabling:
      - Per-token information retention analysis
      - Spatial-temporal interpretability
      - More fine-grained compression control
    """

    def __init__(self, input_dim: int, latent_dim: int, dropout: float = 0.1):
        """
        Args:
            input_dim: Per-token input dimension (D from backbone, e.g. 200)
            latent_dim: Per-token latent dimension (D', compression target)
            dropout: Dropout probability
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Shared feature extractor across tokens
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Separate heads for mu and log_var
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_log_var = nn.Linear(input_dim, latent_dim)

    def forward(
        self, H: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            H: Backbone tokens (B, T, D)

        Returns:
            Z: Sampled latent (B, T, D')
            mu: Mean (B, T, D')
            log_var: Log variance (B, T, D'), clamped to [-10, 10]
        """
        feat = self.shared_encoder(H)  # (B, T, D)
        mu = self.fc_mu(feat)  # (B, T, D')
        log_var = torch.clamp(self.fc_log_var(feat), min=-10, max=10)  # (B, T, D')

        if self.training:
            std = torch.exp(0.5 * log_var)
            Z = mu + std * torch.randn_like(std)
        else:
            Z = mu

        return Z, mu, log_var


# ==============================================================================
# 4. Classification Heads
# ==============================================================================

class DiseaseClassifier(nn.Module):
    """Disease classification head operating on aggregated latent representation."""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.BatchNorm1d(input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, D') -> (B, num_classes)"""
        return self.head(z)


class SubjectClassifier(nn.Module):
    """
    Subject identification head for adversarial training.

    Paired with GRL: the encoder learns to REMOVE subject-specific information
    because GRL reverses gradients, making the encoder adversarial to this head.
    """

    def __init__(self, input_dim: int, num_subjects: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_subjects),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, D') -> (B, num_subjects)"""
        return self.head(z)


# ==============================================================================
# 5. MultiDisease CodeBrain Model
# ==============================================================================

class MultiDisease_CodeBrain_Model(nn.Module):
    """
    Frozen CodeBrain + IB Adapter + Dual Classification Heads.

    Architecture flow:
        1. Frozen backbone: (B, C, S, P) -> (B, C, S, 200)
        2. Reshape to tokens: (B, T, 200) where T = C * S
        3. IB Adapter: (B, T, 200) -> Z (B, T, D'), mu, log_var
        4. Aggregate tokens: Z (B, T, D') -> z_agg (B, D')
        5a. Disease head: z_agg -> disease logits
        5b. GRL + Subject head: z_agg -> subject logits (adversarial)

    Key design choices:
        - Token structure preserved for interpretability
        - Mean pooling for aggregation (simple, effective)
        - GRL for adversarial MI minimization (no alternating optimization)
    """

    def __init__(
        self,
        backbone: nn.Module,
        token_dim: int,
        num_classes: int,
        num_subjects: int,
        latent_dim: int = 128,
        lambda_adv: float = 1.0,
        dropout: float = 0.1,
    ):
        """
        Args:
            backbone: Pre-trained CodeBrain SSSM model (will be frozen)
            token_dim: Per-token dimension from backbone (200 for CodeBrain)
            num_classes: Number of disease classes
            num_subjects: Number of subjects for adversarial head
            latent_dim: IB bottleneck dimension per token
            lambda_adv: GRL scaling factor
            dropout: Dropout probability
        """
        super().__init__()

        self.backbone = backbone
        self.token_dim = token_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_subjects = num_subjects

        # Check if backbone has trainable adapter params (CBraMod with adapters)
        self.has_backbone_adapters = any(
            p.requires_grad for p in self.backbone.parameters()
        )

        # Token-level IB Adapter
        self.ib_adapter = CodeBrain_IB_Adapter(token_dim, latent_dim, dropout)

        # Disease classification head
        self.disease_head = DiseaseClassifier(latent_dim, num_classes, dropout)

        # Gradient Reversal Layer + Subject adversarial head
        self.grl = GradientReversalLayer(lambda_=lambda_adv)
        self.subject_head = SubjectClassifier(latent_dim, num_subjects, dropout)

        # Print parameter summary
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = frozen + trainable
        print(f"MultiDisease_CodeBrain_Model parameter summary:")
        print(f"  Frozen params:     {frozen:,}")
        print(f"  Trainable params:  {trainable:,}")
        print(f"  Trainable ratio:   {trainable / total * 100:.2f}%")
        if self.has_backbone_adapters:
            adapter_params = sum(
                p.numel() for name, p in self.backbone.named_parameters()
                if p.requires_grad
            )
            print(f"  Backbone adapters: {adapter_params:,}")
        print(f"  IB latent_dim:     {latent_dim}")
        print(f"  Disease classes:   {num_classes}")
        print(f"  Subject classes:   {num_subjects}")

    def forward(
        self, x: torch.Tensor, return_tokens: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: EEG input (B, C, S, P) for CodeBrain
            return_tokens: If True, also return per-token Z for analysis

        Returns:
            dict with keys:
                'disease_logits': (B, num_classes)
                'subject_logits': (B, num_subjects)
                'mu': (B, T, D')  - IB mean
                'log_var': (B, T, D')  - IB log variance
                'z_agg': (B, D')  - aggregated latent
                'z_tokens': (B, T, D') - per-token latent (only if return_tokens=True)
        """
        B, C, S, P = x.shape
        T = C * S  # number of tokens

        # 1. Backbone forward
        # When adapters have gradients, skip no_grad to allow backprop through them
        if self.has_backbone_adapters:
            backbone_out = self.backbone(x)  # (B, C, S, 200)
        else:
            with torch.no_grad():
                backbone_out = self.backbone(x)  # (B, C, S, 200)

        # 2. Reshape to token sequence: (B, C, S, 200) -> (B, T, D)
        H = backbone_out.reshape(B, T, self.token_dim)  # (B, T, 200)

        # 3. Token-level IB adapter
        Z, mu, log_var = self.ib_adapter(H)  # each (B, T, D')

        # 4. Aggregate across tokens (mean pooling)
        z_agg = Z.mean(dim=1)  # (B, D')

        # 5a. Disease classification
        disease_logits = self.disease_head(z_agg)  # (B, num_classes)

        # 5b. Subject adversarial classification (through GRL)
        z_reversed = self.grl(z_agg)  # identity forward, reversed gradient backward
        subject_logits = self.subject_head(z_reversed)  # (B, num_subjects)

        out = {
            'disease_logits': disease_logits,
            'subject_logits': subject_logits,
            'mu': mu,
            'log_var': log_var,
            'z_agg': z_agg,
        }
        if return_tokens:
            out['z_tokens'] = Z

        return out


# ==============================================================================
# 6. Loss Functions
# ==============================================================================

def compute_ib_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Token-level KL divergence: D_KL(q(z_t|h_t) || N(0,I)), averaged over
    batch and tokens.

    For token-level IB:
        mu, log_var: (B, T, D')
        KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        averaged over B*T

    This is the same formula as VIB KL but operates per-token, giving
    finer-grained compression control.
    """
    # (B, T, D') -> scalar
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    B, T = mu.shape[0], mu.shape[1]
    return kl / (B * T)


def compute_mi_loss(
    subject_logits: torch.Tensor,
    subject_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy loss for subject classification.

    This is the MI minimization term: the GRL ensures that gradients from
    this loss are REVERSED when flowing to the encoder, making the encoder
    learn to remove subject-specific information.

    The subject head itself is trained normally (to predict subjects),
    while the encoder is trained adversarially (to fool the subject head).

    Args:
        subject_logits: (B, num_subjects)
        subject_ids: (B,) integer subject labels

    Returns:
        L_MI: Cross-entropy loss for subject identification
    """
    return F.cross_entropy(subject_logits, subject_ids)


class InformationBottleneckLoss(nn.Module):
    """
    Complete loss function for IB + Disentanglement framework.

    L_total = L_task + beta * L_IB + lambda_adv * L_MI

    where:
        L_task: Disease classification (CE or BCE)
        L_IB:   Token-level KL divergence (information compression)
        L_MI:   Subject adversarial loss (through GRL, so gradients are reversed
                for the encoder, minimizing I(Z; Subject))

    Args:
        beta: Weight for IB (KL regularization). Higher = more compression.
              Typical range: 1e-4 to 1e-2
        lambda_adv: Weight for adversarial MI loss. Higher = stronger
                    subject disentanglement. Typical range: 0.1 to 1.0
        task_type: 'multiclass' or 'binary'
    """

    def __init__(
        self,
        beta: float = 1e-3,
        lambda_adv: float = 0.5,
        task_type: str = 'multiclass',
    ):
        super().__init__()
        self.beta = beta
        self.lambda_adv = lambda_adv
        self.task_type = task_type

    def forward(
        self,
        disease_logits: torch.Tensor,
        labels: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        subject_logits: Optional[torch.Tensor] = None,
        subject_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss.

        Args:
            disease_logits: (B, num_classes)
            labels: (B,) disease labels
            mu: (B, T, D') IB mean
            log_var: (B, T, D') IB log variance
            subject_logits: (B, num_subjects) or None if no subject info
            subject_ids: (B,) integer subject IDs or None

        Returns:
            total_loss: Scalar tensor for backward
            loss_dict: Dict of individual loss values for logging
        """
        # L_task: disease classification
        # Always use cross_entropy since disease_logits is (B, num_classes)
        # even for binary tasks where num_classes=2
        loss_task = F.cross_entropy(disease_logits, labels)

        # L_IB: token-level KL divergence
        loss_ib = compute_ib_loss(mu, log_var)

        # L_MI: adversarial subject classification (only if subject IDs available)
        if subject_logits is not None and subject_ids is not None:
            loss_mi = compute_mi_loss(subject_logits, subject_ids)
        else:
            loss_mi = torch.tensor(0.0, device=disease_logits.device)

        # Total loss
        # Note: GRL already handles gradient reversal for L_MI,
        # so we ADD lambda_adv * L_MI (the reversal happens in the gradient flow)
        total = loss_task + self.beta * loss_ib + self.lambda_adv * loss_mi

        loss_dict = {
            'total': total.item(),
            'task': loss_task.item(),
            'ib': loss_ib.item(),
            'mi': loss_mi.item(),
            'weighted_ib': (self.beta * loss_ib).item(),
            'weighted_mi': (self.lambda_adv * loss_mi).item(),
        }

        return total, loss_dict


# ==============================================================================
# 7. Interpretability: Token-wise Information Retention Heatmap
# ==============================================================================

def get_interpretability_heatmap(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    n_channels: int,
    seq_len: int,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-token information retention scores for clinical interpretability.

    For each token t, the information retained is measured by the KL divergence
    between the posterior q(z_t|h_t) = N(mu_t, sigma_t^2) and the prior N(0, I).
    Higher KL = more information retained from the backbone for that token.

    The token dimension T = n_channels * seq_len can be reshaped back to a
    (n_channels, seq_len) grid, giving a spatial-temporal heatmap of which
    EEG channels and time segments are most informative for the downstream task.

    Args:
        mu: (B, T, D') IB mean
        log_var: (B, T, D') IB log variance
        n_channels: Number of EEG channels
        seq_len: Number of temporal patches

    Returns:
        dict with:
            'kl_per_token': (B, T) per-token KL divergence
            'heatmap': (B, n_channels, seq_len) spatial-temporal heatmap
            'channel_importance': (B, n_channels) aggregated channel scores
            'temporal_importance': (B, seq_len) aggregated temporal scores
    """
    # Per-token KL: sum over latent dimensions, keep token dimension
    # (B, T, D') -> (B, T)
    kl_per_token = -0.5 * torch.sum(
        1 + log_var - mu.pow(2) - log_var.exp(), dim=-1
    )

    # Reshape to spatial-temporal grid
    B = mu.shape[0]
    heatmap = kl_per_token.reshape(B, n_channels, seq_len)  # (B, C, S)

    # Aggregate along dimensions
    channel_importance = heatmap.mean(dim=-1)  # (B, C) - average over time
    temporal_importance = heatmap.mean(dim=-2)  # (B, S) - average over channels

    return {
        'kl_per_token': kl_per_token,
        'heatmap': heatmap,
        'channel_importance': channel_importance,
        'temporal_importance': temporal_importance,
    }


# ==============================================================================
# 8. Training Step (demonstration)
# ==============================================================================

def train_step(
    model: MultiDisease_CodeBrain_Model,
    criterion: InformationBottleneckLoss,
    optimizer: torch.optim.Optimizer,
    eeg_data: torch.Tensor,
    labels: torch.Tensor,
    subject_ids: Optional[torch.Tensor] = None,
    clip_value: float = 5.0,
) -> Dict[str, float]:
    """
    Single training step demonstrating the full forward/backward with GRL.

    The key insight: we do NOT need alternating min-max optimization.
    The GRL automatically handles gradient reversal:
      - Subject head receives NORMAL gradients (learns to predict subjects)
      - Encoder receives REVERSED gradients from subject head
        (learns to remove subject information)
      - Disease head and IB adapter receive normal gradients

    Args:
        model: MultiDisease_CodeBrain_Model
        criterion: InformationBottleneckLoss
        optimizer: Optimizer for trainable parameters
        eeg_data: (B, C, S, P) raw EEG
        labels: (B,) disease labels
        subject_ids: (B,) subject IDs (optional)
        clip_value: Gradient clipping value

    Returns:
        loss_dict: Dictionary of loss values for logging
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(eeg_data)

    # Compute loss
    loss, loss_dict = criterion(
        disease_logits=outputs['disease_logits'],
        labels=labels,
        mu=outputs['mu'],
        log_var=outputs['log_var'],
        subject_logits=outputs['subject_logits'] if subject_ids is not None else None,
        subject_ids=subject_ids,
    )

    # Backward pass
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

    # Optimizer step
    optimizer.step()

    return loss_dict


# ==============================================================================
# 9. GRL Lambda Scheduler
# ==============================================================================

class GRLScheduler:
    """
    Schedule the GRL lambda using the standard domain adaptation ramp-up:
        lambda(p) = 2 / (1 + exp(-gamma * p)) - 1

    where p = epoch / total_epochs increases from 0 to 1.

    This starts with lambda~0 (no adversarial pressure, let encoder learn
    disease features first) and gradually increases to lambda~1 (full
    adversarial subject removal).
    """

    def __init__(self, model: MultiDisease_CodeBrain_Model, gamma: float = 10.0):
        self.model = model
        self.gamma = gamma

    def step(self, epoch: int, total_epochs: int):
        """Update GRL lambda based on training progress."""
        p = epoch / total_epochs
        lambda_ = 2.0 / (1.0 + math.exp(-self.gamma * p)) - 1.0
        self.model.grl.set_lambda(lambda_)
        return lambda_


# ==============================================================================
# 10. Optimizer Configuration
# ==============================================================================

def configure_optimizer(
    model: MultiDisease_CodeBrain_Model,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
) -> torch.optim.AdamW:
    """
    Configure AdamW optimizer for trainable parameters only.

    Uses different learning rates:
      - IB adapter: lr
      - Disease head: lr
      - Subject head: lr * 2 (adversarial head benefits from faster learning)
    """
    param_groups = [
        {
            'params': model.ib_adapter.parameters(),
            'lr': lr,
            'weight_decay': weight_decay,
            'name': 'ib_adapter',
        },
        {
            'params': model.disease_head.parameters(),
            'lr': lr,
            'weight_decay': weight_decay,
            'name': 'disease_head',
        },
        {
            'params': model.subject_head.parameters(),
            'lr': lr * 2,
            'weight_decay': weight_decay,
            'name': 'subject_head',
        },
    ]

    # Add backbone adapter params if present (CBraMod with inter-layer adapters)
    if model.has_backbone_adapters:
        adapter_params = [p for p in model.backbone.parameters() if p.requires_grad]
        if adapter_params:
            param_groups.append({
                'params': adapter_params,
                'lr': lr * 0.1,
                'weight_decay': weight_decay,
                'name': 'backbone_adapters',
            })

    return torch.optim.AdamW(param_groups)
