"""
Information Invariant Bottleneck (IIB) Module.

This module implements a variational bottleneck with adversarial training
to achieve subject-invariant feature representation for improved cross-subject
generalization in medical time-series classification.

Components:
- GradientReversalLayer (GRL): Reverses gradients for adversarial training
- VariationalEncoder: Encodes features into a latent distribution (mu, sigma)
- SubjectDiscriminator: Predicts subject identity from latent features
- IIB: Main module combining all components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) Function.

    During forward pass, acts as identity.
    During backward pass, reverses the gradient and scales by alpha.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        output = grad_output.neg() * ctx.alpha
        return output, None


class GRL(nn.Module):
    """
    Gradient Reversal Layer Module.

    Wraps the GradientReversalFunction for use in nn.Sequential.
    """

    def __init__(self, alpha: float = 1.0):
        super(GRL, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)

    def set_alpha(self, alpha: float):
        """Update the gradient reversal strength."""
        self.alpha = alpha


class VariationalEncoder(nn.Module):
    """
    Variational Encoder that maps input features to a latent distribution.

    Outputs mean (mu) and log-variance (logvar) parameterizing a Gaussian
    distribution. Uses reparameterization trick for differentiable sampling.

    Args:
        input_dim: Dimension of input features
        hidden_dim: Dimension of hidden layer
        latent_dim: Dimension of latent representation Z
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float = 0.1
    ):
        super(VariationalEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Mean and log-variance heads
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in [self.fc_mu, self.fc_logvar]:
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon

        During training, samples from the distribution.
        During inference, returns the mean directly.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through variational encoder.

        Args:
            x: Input tensor of shape (batch_size, ..., input_dim)

        Returns:
            z: Sampled latent representation
            mu: Mean of the latent distribution
            logvar: Log-variance of the latent distribution
        """
        # Encode
        h = self.encoder(x)

        # Get distribution parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10, max=10)

        # Sample using reparameterization trick
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss: KL(q(z|x) || p(z))

        where p(z) = N(0, I) is the standard Gaussian prior.

        KL divergence for Gaussian:
        KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        Args:
            mu: Mean of the latent distribution
            logvar: Log-variance of the latent distribution

        Returns:
            kl_loss: Mean KL divergence across the batch
        """
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
            dim=-1
        ).mean()

        return kl_loss


class SubjectDiscriminator(nn.Module):
    """
    Subject Discriminator for adversarial training.

    Predicts subject identity from latent features.
    Used with Gradient Reversal Layer to make latent features
    subject-invariant.

    Args:
        latent_dim: Dimension of input latent features
        num_subjects: Number of unique subjects to classify
        hidden_dim: Dimension of hidden layer
        dropout: Dropout probability
    """

    def __init__(
        self,
        latent_dim: int,
        num_subjects: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.2
    ):
        super(SubjectDiscriminator, self).__init__()

        if hidden_dim is None:
            hidden_dim = latent_dim // 2

        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_subjects)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.discriminator.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through discriminator.

        Args:
            z: Latent features of shape (batch_size, latent_dim)

        Returns:
            logits: Subject classification logits of shape (batch_size, num_subjects)
        """
        return self.discriminator(z)


class IIB(nn.Module):
    """
    Information Invariant Bottleneck (IIB) Module.

    Combines variational encoding with adversarial subject discrimination
    to learn subject-invariant representations.

    Architecture:
        Input -> VariationalEncoder -> Z (latent)
                                        |
                                        +-> [To downstream (SA-MoE)]
                                        |
                                        +-> [GRL -> SubjectDiscriminator]

    Args:
        input_dim: Dimension of input features from MOMENT backbone
        hidden_dim: Hidden dimension for variational encoder
        latent_dim: Dimension of latent representation
        num_subjects: Number of unique subjects
        grl_alpha: Gradient reversal strength (default: 1.0)
        discriminator_hidden_dim: Hidden dim for discriminator (default: latent_dim // 2)
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_subjects: int,
        grl_alpha: float = 1.0,
        discriminator_hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super(IIB, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_subjects = num_subjects
        self.grl_alpha = grl_alpha

        # Variational encoder
        self.variational_encoder = VariationalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout
        )

        # Gradient reversal layer
        self.grl = GRL(alpha=grl_alpha)

        # Subject discriminator
        self.subject_discriminator = SubjectDiscriminator(
            latent_dim=latent_dim,
            num_subjects=num_subjects,
            hidden_dim=discriminator_hidden_dim,
            dropout=dropout
        )

        # Output projection (back to original dimension for compatibility)
        self.output_projection = nn.Linear(latent_dim, input_dim)

        # Cache for loss computation
        self._cached_mu = None
        self._cached_logvar = None
        self._cached_z = None

    def set_grl_alpha(self, alpha: float):
        """Update the gradient reversal strength."""
        self.grl_alpha = alpha
        self.grl.set_alpha(alpha)

    def forward(
        self,
        x: torch.Tensor,
        return_subject_prediction: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through IIB.

        Args:
            x: Input features of shape (batch_size, seq_len, input_dim)
               or (batch_size, input_dim)
            return_subject_prediction: Whether to return subject predictions

        Returns:
            output: Subject-invariant features of same shape as input
            kl_loss: KL divergence loss (if training)
            subject_logits: Subject prediction logits (if return_subject_prediction)
        """
        original_shape = x.shape
        is_3d_input = len(original_shape) == 3

        # Flatten if 3D input
        if is_3d_input:
            batch_size, seq_len, input_dim = original_shape
            x_flat = x.view(-1, input_dim)
        else:
            x_flat = x

        # Encode to latent distribution
        z, mu, logvar = self.variational_encoder(x_flat)

        # Cache for loss computation
        self._cached_mu = mu
        self._cached_logvar = logvar
        self._cached_z = z

        # Project back to original dimension
        output_flat = self.output_projection(z)

        # Reshape to original shape
        if is_3d_input:
            output = output_flat.view(batch_size, seq_len, input_dim)
        else:
            output = output_flat

        if return_subject_prediction:
            # Apply gradient reversal and predict subject
            z_reversed = self.grl(z)
            subject_logits = self.subject_discriminator(z_reversed)
            return output, subject_logits

        return output

    def compute_losses(
        self,
        subject_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute IIB losses.

        Args:
            subject_ids: Ground truth subject IDs for adversarial loss

        Returns:
            Dictionary containing:
                - kl_loss: KL divergence loss
                - adv_loss: Adversarial loss (cross-entropy for subject prediction)
        """
        losses = {}

        # KL divergence loss
        if self._cached_mu is not None and self._cached_logvar is not None:
            kl_loss = self.variational_encoder.compute_kl_loss(
                self._cached_mu, self._cached_logvar
            )
            losses['kl_loss'] = kl_loss
        else:
            losses['kl_loss'] = torch.tensor(0.0, device=next(self.parameters()).device)

        # Adversarial loss
        if subject_ids is not None and self._cached_z is not None:
            # Apply gradient reversal
            z_reversed = self.grl(self._cached_z)

            # Predict subject
            subject_logits = self.subject_discriminator(z_reversed)

            # Repeat subject_ids for each token if input was 3D
            if subject_ids.shape[0] != subject_logits.shape[0]:
                repeat_factor = subject_logits.shape[0] // subject_ids.shape[0]
                subject_ids_expanded = subject_ids.repeat_interleave(repeat_factor)
            else:
                subject_ids_expanded = subject_ids

            # Map subject_ids to valid range using modulo operation
            # This handles cases where subject_ids may be arbitrary hash values
            subject_ids_mapped = subject_ids_expanded % self.num_subjects

            # Cross-entropy loss
            adv_loss = F.cross_entropy(subject_logits, subject_ids_mapped)
            losses['adv_loss'] = adv_loss
        else:
            losses['adv_loss'] = torch.tensor(0.0, device=next(self.parameters()).device)

        return losses

    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation Z without projection back to input dim.

        Args:
            x: Input features

        Returns:
            z: Latent representation
        """
        original_shape = x.shape
        is_3d_input = len(original_shape) == 3

        if is_3d_input:
            batch_size, seq_len, input_dim = original_shape
            x_flat = x.view(-1, input_dim)
        else:
            x_flat = x

        z, _, _ = self.variational_encoder(x_flat)

        if is_3d_input:
            z = z.view(batch_size, seq_len, -1)

        return z

    def clear_cache(self):
        """Clear cached values."""
        self._cached_mu = None
        self._cached_logvar = None
        self._cached_z = None


class IIBConfig:
    """Configuration for IIB module."""

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        latent_dim: int = 256,
        grl_alpha: float = 1.0,
        discriminator_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        kl_loss_weight: float = 0.1,
        adv_loss_weight: float = 0.1
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.grl_alpha = grl_alpha
        self.discriminator_hidden_dim = discriminator_hidden_dim
        self.dropout = dropout
        self.kl_loss_weight = kl_loss_weight
        self.adv_loss_weight = adv_loss_weight

    def to_dict(self) -> Dict[str, Any]:
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim,
            'grl_alpha': self.grl_alpha,
            'discriminator_hidden_dim': self.discriminator_hidden_dim,
            'dropout': self.dropout,
            'kl_loss_weight': self.kl_loss_weight,
            'adv_loss_weight': self.adv_loss_weight
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "IIBConfig":
        return cls(**config_dict)
