"""
Variational Bottleneck for USBA.

Implements the core VIB mechanism:
  a = W_d(F)
  mu = W_mu(a), logvar = W_sigma(a)
  z = mu + sigma * eps   (training)
  z = mu                  (eval)

Also contains the residual write-back gate with configurable granularity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class VariationalBottleneck(nn.Module):
    """
    Token-level variational bottleneck.

    Input: (B, T, D)  → project down → mu/logvar → sample z → project up → delta
    Also computes per-layer KL divergence.

    The write-back gate g controls how much of the adapter correction is applied:
        H_adapt = H + g * W_up(z)

    Gate types:
      - 'layer_wise': single scalar g per adapter layer
      - 'token_wise': g of shape (B, T, 1) — one gate per token position,
            produced by a small projection from the fused input
      - 'channel_wise': g of shape (1, 1, D) — one gate per channel
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        gate_type: str = 'layer_wise',
        gate_init: float = 0.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.gate_type = gate_type

        # ── Encoder: project down ──────────────────────────────────────
        hidden = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

        # ── Decoder: project up to input dim ───────────────────────────
        self.decoder = nn.Linear(latent_dim, input_dim)
        # Bug-4 fix: use small random weights instead of zeros so that
        # task gradients can reach the encoder/branches through the decoder
        # from the very first step.
        nn.init.normal_(self.decoder.weight, std=0.02)
        nn.init.normal_(self.decoder.bias, std=0.02)

        # Learned scalar that multiplies delta.  Initialized to 0 so that
        # the adapter still starts as near-identity (scale * delta ≈ 0),
        # but gradients flow through the decoder immediately.
        self.residual_scale = nn.Parameter(torch.tensor(0.0))

        # ── Residual gate ──────────────────────────────────────────────
        # Learnable gate initialized near gate_init (passed through sigmoid)
        # So actual gate value starts at sigmoid(gate_init)
        if gate_type == 'layer_wise':
            self.gate_logit = nn.Parameter(torch.tensor(gate_init))
        elif gate_type == 'token_wise':
            # Bug-1 fix: a real per-token gate produced by projecting
            # the fused input to a scalar per token position.
            gate_hidden = max(input_dim // 4, 1)
            self.gate_proj = nn.Sequential(
                nn.Linear(input_dim, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, 1),
            )
            # Initialize the final projection bias so that the initial
            # output is close to gate_init (before sigmoid).
            nn.init.zeros_(self.gate_proj[-1].weight)
            nn.init.constant_(self.gate_proj[-1].bias, gate_init)
        elif gate_type == 'channel_wise':
            self.gate_logit = nn.Parameter(torch.full((input_dim,), gate_init))
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")

    def _get_gate(self, fused: torch.Tensor, B: int, T: int, D: int) -> torch.Tensor:
        """Compute gate value with appropriate shape for broadcasting.

        Args:
            fused: the fused input tensor (B, T, D), used only by token_wise.
        """
        if self.gate_type == 'layer_wise':
            return torch.sigmoid(self.gate_logit)  # scalar
        elif self.gate_type == 'token_wise':
            # Project fused input to per-token gate: (B, T, 1)
            return torch.sigmoid(self.gate_proj(fused))  # (B, T, 1)
        elif self.gate_type == 'channel_wise':
            return torch.sigmoid(self.gate_logit).unsqueeze(0).unsqueeze(0)  # (1, 1, D)

    def forward(
        self, fused: torch.Tensor, h_original: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            fused: output of fusion module (B, T, D)
            h_original: original hidden states before adapter (B, T, D)

        Returns:
            h_adapted: h_original + g * delta (B, T, D)
            aux: dict with 'mu', 'logvar', 'kl', 'gate_value', 'z'
                 kl is a scalar (mean over B, T)
                 gate_value is the effective gate for logging

        # ── SPCBA extension point ──────────────────────────────────────
        # Future: split z into z_d (domain) and z_s (shared), add nuisance
        # branch. Current single-z design does not block this: just replace
        # the encoder with a dual-head encoder.
        """
        B, T, D = fused.shape

        # Encode
        a = self.encoder(fused)                              # (B, T, D)
        mu = self.fc_mu(a)                                   # (B, T, latent)
        logvar = torch.clamp(self.fc_logvar(a), -10, 10)     # (B, T, latent)

        # Reparameterize
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu

        # Decode — residual_scale preserves near-identity init while
        # allowing gradients to flow through the decoder from step 1.
        delta = self.residual_scale * self.decoder(z)  # (B, T, D)

        # Gate
        g = self._get_gate(fused, B, T, D)

        # Write back to residual stream
        h_adapted = h_original + g * delta

        # KL divergence: averaged over B and T
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (B * T)

        # Gate statistics for logging
        with torch.no_grad():
            if self.gate_type == 'layer_wise':
                gate_val = torch.sigmoid(self.gate_logit).item()
            elif self.gate_type == 'token_wise':
                # g is (B, T, 1); report its mean for logging
                gate_val = g.mean().item()
            elif self.gate_type == 'channel_wise':
                gate_val = torch.sigmoid(self.gate_logit).mean().item()

        aux = {
            'mu': mu,
            'logvar': logvar,
            'kl': kl,
            'gate_value': gate_val,
            'z': z,
            # For BILO extension: could add rank allocation stats here
        }
        return h_adapted, aux
