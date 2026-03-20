"""
FlattenHead — CodeBrain-style classification head.

Replicates the original CodeBrain downstream head:
  flatten(B, C*S*D) → Linear → ELU → Dropout → Linear → ELU → Dropout → Linear → logits
"""

import torch
import torch.nn as nn


class FlattenHead(nn.Module):
    """
    CodeBrain-style flatten + 3-layer MLP head.

    Input: (B, C, S, D)
    Output: (B, num_classes)
    """

    def __init__(self, n_channels: int, seq_len: int, token_dim: int,
                 num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.token_dim = token_dim

        flat_dim = n_channels * seq_len * token_dim   # e.g. 16*10*200 = 32000
        mid_dim = seq_len * token_dim                  # e.g. 10*200 = 2000

        self.head = nn.Sequential(
            nn.Linear(flat_dim, mid_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, token_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, S, D)

        Returns:
            logits: (B, num_classes)
        """
        B = features.shape[0]
        z = features.contiguous().view(B, -1)
        return self.head(z)
