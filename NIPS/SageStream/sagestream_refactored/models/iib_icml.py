"""
ICML-style Information Invariant Bottleneck (IIB) Module.

This module implements the original IIB framework from ICML, which uses
dual prediction heads (invariant + environment-aware) and a Conditional
Independence (CI) loss to achieve subject-invariant representations.

Key differences from the NIPS (GRL-based) version:
- Dual prediction heads instead of GRL + Subject Discriminator
- CI loss: (L_inv - L_env)^2 enforces conditional independence
- Environment-aware head explicitly uses subject/domain label as input
- No gradient reversal layer needed

Loss function:
    L_total = L_inv + L_env + lambda * L_IB + beta * L_CI

where:
    L_inv: Classification loss from invariant head (Z only)
    L_env: Classification loss from environment-aware head (Z + domain)
    L_IB:  KL divergence (information bottleneck regularization)
    L_CI:  (L_inv - L_env)^2 (conditional independence constraint)

Reference:
    Li et al., "Invariant Information Bottleneck for Domain Generalization", AAAI 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class VariationalEncoder(nn.Module):
    """
    Variational Encoder for the ICML IIB framework.

    Maps input features to a latent Gaussian distribution (mu, logvar),
    then samples using the reparameterization trick.

    Compared to the NIPS version, this encoder is simpler (single linear
    projection for mu/logvar) since the backbone already provides
    high-level features.

    Args:
        input_dim: Dimension of input features from backbone
        latent_dim: Dimension of latent representation Z
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor, is_training: bool = True
    ) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        if is_training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu

    def forward(
        self, x: torch.Tensor, is_training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features (batch_size, input_dim)
            is_training: If True, sample from distribution; if False, use mean

        Returns:
            z: Sampled latent representation
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
        """
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, min=-10, max=10)
        z = self.reparameterize(mu, logvar, is_training)
        return z, mu, logvar


class InvariantHead(nn.Module):
    """
    Invariant prediction head.

    Predicts class labels using only the latent representation Z,
    without any domain/subject information.

    Args:
        latent_dim: Dimension of latent input
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes
    """

    def __init__(self, latent_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(z))
        return self.fc2(h)


class EnvironmentHead(nn.Module):
    """
    Environment-aware prediction head.

    Predicts class labels using both the latent representation Z
    and the domain/subject label d. If Z is truly subject-invariant,
    this head should perform similarly to the invariant head.

    Args:
        latent_dim: Dimension of latent input
        domain_dim: Dimension of domain label input (default 1)
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_classes: int,
        domain_dim: int = 1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + domain_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, z: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent features (batch_size, latent_dim)
            d: Domain/subject label (batch_size, domain_dim)
        """
        z_cat = torch.cat([z, d], dim=1)
        h = F.relu(self.fc1(z_cat))
        return self.fc2(h)


class IIB_ICML(nn.Module):
    """
    ICML-style Information Invariant Bottleneck (IIB) Module.

    This implements the original IIB framework with dual prediction heads
    and conditional independence (CI) loss. It is designed to be inserted
    between a backbone encoder and the downstream task.

    Architecture:
        Backbone features -> VariationalEncoder -> Z (latent)
                                                    |
                                    +---------------+---------------+
                                    |                               |
                            InvariantHead(Z)            EnvironmentHead(Z, d)
                                    |                               |
                                inv_logits                      env_logits
                                    |                               |
                                L_inv = CE(inv_logits, y)     L_env = CE(env_logits, y)
                                    |                               |
                                    +----------- L_CI = (L_inv - L_env)^2
                                    |
                            L_IB = KL(q(z|x) || p(z))
                                    |
                    L_total = L_inv + L_env + lambda * L_IB + beta * L_CI

    Args:
        input_dim: Dimension of input features from backbone
        latent_dim: Dimension of latent representation Z
        hidden_dim: Hidden dimension for prediction heads
        num_classes: Number of classification classes
        domain_dim: Dimension of domain label (default 1 for scalar subject ID)
        lambda_ib: Weight for IB (KL divergence) loss
        beta_ci: Weight for CI (conditional independence) loss
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 16,
        num_classes: int = 2,
        domain_dim: int = 1,
        lambda_ib: float = 0.1,
        beta_ci: float = 10.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.lambda_ib = lambda_ib
        self.beta_ci = beta_ci

        # Variational encoder
        self.encoder = VariationalEncoder(input_dim, latent_dim)

        # Dual prediction heads
        self.inv_head = InvariantHead(latent_dim, hidden_dim, num_classes)
        self.env_head = EnvironmentHead(latent_dim, hidden_dim, num_classes, domain_dim)

        # Output projection (back to original dimension for pipeline compatibility)
        self.output_projection = nn.Linear(latent_dim, input_dim)

        # Cache for loss computation
        self._cached_mu = None
        self._cached_logvar = None
        self._cached_z = None

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: encode features to subject-invariant Z and project back.

        This method is used in the main model pipeline. It replaces the input
        features with subject-invariant ones (same shape as input).

        Args:
            x: Input features (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            subject_ids: Optional subject IDs (not used in forward, only in compute_losses)

        Returns:
            output: Subject-invariant features of same shape as input
        """
        original_shape = x.shape
        is_3d = len(original_shape) == 3

        if is_3d:
            batch_size, seq_len, dim = original_shape
            x_flat = x.view(-1, dim)
        else:
            x_flat = x

        z, mu, logvar = self.encoder(x_flat, is_training=self.training)

        # Cache for loss computation
        self._cached_mu = mu
        self._cached_logvar = logvar
        self._cached_z = z

        # Project back to original dimension
        output_flat = self.output_projection(z)

        if is_3d:
            return output_flat.view(batch_size, seq_len, dim)
        return output_flat

    def compute_losses(
        self,
        labels: torch.Tensor,
        subject_ids: torch.Tensor,
        label_smoothing: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all IIB losses.

        Must be called after forward() which caches z, mu, logvar.

        Args:
            labels: Ground truth class labels (batch_size,)
            subject_ids: Subject/domain IDs (batch_size,)
            label_smoothing: Label smoothing for CE loss

        Returns:
            Dictionary with:
                - inv_loss: Invariant head classification loss
                - env_loss: Environment head classification loss
                - ib_loss: KL divergence (information bottleneck)
                - ci_loss: Conditional independence loss
                - total_iib_loss: Weighted sum of all losses
        """
        losses = {}

        if self._cached_z is None:
            device = next(self.parameters()).device
            zero = torch.tensor(0.0, device=device)
            return {
                'inv_loss': zero, 'env_loss': zero,
                'ib_loss': zero, 'ci_loss': zero, 'total_iib_loss': zero,
            }

        z = self._cached_z
        mu = self._cached_mu
        logvar = self._cached_logvar

        # Handle 3D case: if z was flattened from (B, S, D) -> (B*S, D),
        # we need to aggregate back to batch level for the classification heads.
        # For classification, we pool over the sequence dimension.
        if z.shape[0] != labels.shape[0]:
            # z is (B*S, latent_dim), reshape and mean-pool
            batch_size = labels.shape[0]
            seq_len = z.shape[0] // batch_size
            z_pooled = z.view(batch_size, seq_len, -1).mean(dim=1)
            mu_pooled = mu.view(batch_size, seq_len, -1).mean(dim=1)
            logvar_pooled = logvar.view(batch_size, seq_len, -1).mean(dim=1)
        else:
            z_pooled = z
            mu_pooled = mu
            logvar_pooled = logvar

        ce_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Invariant head loss
        inv_logits = self.inv_head(z_pooled)
        inv_loss = ce_fn(inv_logits, labels.long())
        losses['inv_loss'] = inv_loss

        # Environment head loss
        d = subject_ids.float().unsqueeze(1)  # (batch_size, 1)
        env_logits = self.env_head(z_pooled, d)
        env_loss = ce_fn(env_logits, labels.long())
        losses['env_loss'] = env_loss

        # IB loss (KL divergence)
        ib_loss = -0.5 * torch.mean(
            1 + logvar_pooled - mu_pooled.pow(2) - logvar_pooled.exp()
        )
        losses['ib_loss'] = ib_loss

        # CI loss (conditional independence)
        ci_loss = (inv_loss - env_loss) ** 2
        losses['ci_loss'] = ci_loss

        # Total IIB loss
        total = inv_loss + env_loss + self.lambda_ib * ib_loss + self.beta_ci * ci_loss
        losses['total_iib_loss'] = total

        return losses

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict using invariant head (for inference).

        Args:
            x: Input features (batch_size, input_dim) or (batch_size, seq_len, input_dim)

        Returns:
            logits: Classification logits (batch_size, num_classes)
        """
        original_shape = x.shape
        is_3d = len(original_shape) == 3

        if is_3d:
            batch_size, seq_len, dim = original_shape
            x_flat = x.view(-1, dim)
        else:
            x_flat = x

        z, _, _ = self.encoder(x_flat, is_training=False)

        if is_3d:
            z = z.view(batch_size, seq_len, -1).mean(dim=1)

        return self.inv_head(z)

    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent Z without projection."""
        is_3d = len(x.shape) == 3
        if is_3d:
            b, s, d = x.shape
            x_flat = x.view(-1, d)
        else:
            x_flat = x

        z, _, _ = self.encoder(x_flat, is_training=False)

        if is_3d:
            z = z.view(b, s, -1)
        return z

    def clear_cache(self):
        """Clear cached values."""
        self._cached_mu = None
        self._cached_logvar = None
        self._cached_z = None


class IIB_ICML_Config:
    """Configuration for ICML-style IIB module."""

    def __init__(
        self,
        input_dim: int = 512,
        latent_dim: int = 256,
        hidden_dim: int = 16,
        num_classes: int = 2,
        domain_dim: int = 1,
        lambda_ib: float = 0.1,
        beta_ci: float = 10.0,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.domain_dim = domain_dim
        self.lambda_ib = lambda_ib
        self.beta_ci = beta_ci

    def to_dict(self) -> Dict[str, Any]:
        return {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim,
            'num_classes': self.num_classes,
            'domain_dim': self.domain_dim,
            'lambda_ib': self.lambda_ib,
            'beta_ci': self.beta_ci,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "IIB_ICML_Config":
        return cls(**config_dict)
