"""
Baseline classifier head for DEB experiments.

Simple MLP: pool → Linear → BN → GELU → Dropout → Linear → logits
Supports linear probe, partial fine-tune, and full fine-tune.
"""

import torch
import torch.nn as nn


class BaselineHead(nn.Module):
    """
    Standard classification head that operates on backbone output.

    Input: (B, C, S, D) or (B, T, D)
    Output: (B, num_classes)
    """

    def __init__(self, token_dim: int, num_classes: int,
                 hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.token_dim = token_dim
        self.head = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, S, D) or (B, T, D) backbone output

        Returns:
            logits: (B, num_classes)
        """
        if features.dim() == 4:
            # (B, C, S, D) → mean pool to (B, D)
            z = features.mean(dim=(1, 2))
        elif features.dim() == 3:
            # (B, T, D) → mean pool to (B, D)
            z = features.mean(dim=1)
        else:
            z = features

        return self.head(z)
