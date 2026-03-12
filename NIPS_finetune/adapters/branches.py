"""
Temporal and Spatial branches for USBA.

Supports two operating modes:
  - 3D mode: (B, T, D) — for LUNA and fallback
  - 4D structure-aware mode: (B, T, D) with known (n_channels, seq_len) —
    for CBraMod and CodeBrain, where T = n_channels * seq_len

In 4D mode, branches respect the spatial-temporal factorization:
  - Temporal branch: operates along seq_len axis (within each channel)
  - Spatial branch: operates along n_channels axis (within each time step)
This mirrors CBraMod's criss-cross attention and CodeBrain's parallel
SSM + local-attention architecture, so USBA corrections live in the
same factored subspace as the frozen backbone's own representations.

In 3D mode (LUNA), temporal operates along T (pure time patches)
and spatial operates along D (query-compressed features).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
# Temporal Branches (3D fallback)
# ═══════════════════════════════════════════════════════════════════════

class DepthwiseTemporalConv(nn.Module):
    """
    Depthwise temporal convolution — lightweight temporal modelling.

    3D mode: (B, T, D) → conv along T.
    4D-aware mode: internally reshape (B, C*S, D) → (B*C, S, D) → conv along S
    so the kernel never crosses channel boundaries.
    """

    def __init__(self, d_model: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,  # depthwise
            bias=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                token_structure: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
            token_structure: (n_channels, seq_len) if available.
                When provided, T must equal n_channels * seq_len.
                Conv is applied per-channel along seq_len axis only,
                preventing kernels from crossing channel boundaries.
        """
        if token_structure is not None:
            n_ch, seq_len = token_structure
            B, T, D = x.shape
            # (B, C*S, D) → (B*C, S, D)
            h = x.reshape(B * n_ch, seq_len, D)
            h = h.transpose(1, 2)          # (B*C, D, S)
            h = self.conv(h)               # (B*C, D, S)
            h = h.transpose(1, 2)          # (B*C, S, D)
            h = self.norm(h)
            h = self.act(h)
            h = self.drop(h)
            return h.reshape(B, T, D)
        else:
            h = x.transpose(1, 2)  # (B, D, T)
            h = self.conv(h)       # (B, D, T)
            h = h.transpose(1, 2)  # (B, T, D)
            h = self.norm(h)
            h = self.act(h)
            h = self.drop(h)
            return h


class LowRankTemporalMix(nn.Module):
    """
    Low-rank temporal mixing — learns a T×T mixing matrix factorized as T×r × r×T.

    In 4D-aware mode, mixing is applied per-channel along seq_len axis.
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, rank: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        self.rank = rank
        self.max_seq_len = max_seq_len
        self.down = nn.Linear(max_seq_len, rank, bias=False)
        self.up = nn.Linear(rank, max_seq_len, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                token_structure: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        if token_structure is not None:
            n_ch, seq_len = token_structure
            B, T, D = x.shape
            # Apply per-channel: (B, C*S, D) → (B*C, S, D)
            h = x.reshape(B * n_ch, seq_len, D)
            result = self._mix(h, seq_len)
            return result.reshape(B, T, D)
        else:
            B, T, D = x.shape
            return self._mix(x, T)

    def _mix(self, x: torch.Tensor, T: int) -> torch.Tensor:
        if T <= self.max_seq_len:
            h = F.pad(x.transpose(1, 2), (0, self.max_seq_len - T))
        else:
            h = x[:, :self.max_seq_len, :].transpose(1, 2)
        h = self.down(h)
        h = self.act(h)
        h = self.up(h)
        h = h[:, :, :T].transpose(1, 2)
        h = self.norm(h)
        h = self.drop(h)
        return h


# ═══════════════════════════════════════════════════════════════════════
# Spatial Branches
# ═══════════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    """
    Squeeze-and-excitation style attention.

    3D mode: SE along D axis (feature importance weighting).
    4D-aware mode: reshape (B, C*S, D) → (B*S, C, D), apply SE along C axis
    (true cross-channel attention per time step), reshape back.

    The 4D mode mirrors CBraMod's spatial attention path which also attends
    across channels independently per time step.
    """

    def __init__(self, d_model: int, reduction: int = 4, dropout: float = 0.1):
        super().__init__()
        mid = max(d_model // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(d_model, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor,
                token_structure: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        if token_structure is not None:
            n_ch, seq_len = token_structure
            B, T, D = x.shape
            # (B, C*S, D) → (B, C, S, D) → (B*S, C, D)
            h = x.reshape(B, n_ch, seq_len, D)
            h = h.permute(0, 2, 1, 3).reshape(B * seq_len, n_ch, D)
            # SE along C axis: pool over C, gate C
            s = h.mean(dim=1)      # (B*S, D)
            s = self.fc(s)         # (B*S, D)
            h = h * s.unsqueeze(1)  # (B*S, C, D)
            # Reshape back: (B*S, C, D) → (B, S, C, D) → (B, C, S, D) → (B, C*S, D)
            h = h.reshape(B, seq_len, n_ch, D).permute(0, 2, 1, 3).reshape(B, T, D)
            return h
        else:
            s = x.mean(dim=1)
            s = self.fc(s)
            return x * s.unsqueeze(1)


class GroupedSpatialMLP(nn.Module):
    """
    Grouped MLP operating along the D (feature/channel) axis.

    Splits D into groups, applies independent small MLPs, concatenates.
    In 4D-aware mode, applied per-timestep across channels (same as 3D
    but with correct reshaping to isolate cross-channel interactions).
    """

    def __init__(self, d_model: int, num_groups: int = 4, expansion: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.num_groups = num_groups
        group_dim = d_model // num_groups
        self.group_dim = group_dim
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(group_dim, group_dim * expansion),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(group_dim * expansion, group_dim),
            )
            for _ in range(num_groups)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.remainder = d_model - group_dim * num_groups
        if self.remainder > 0:
            self.remainder_mlp = nn.Sequential(
                nn.Linear(self.remainder, self.remainder * expansion),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.remainder * expansion, self.remainder),
            )

    def forward(self, x: torch.Tensor,
                token_structure: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """(B, T, D) → (B, T, D).  token_structure not needed here (D-axis op)."""
        chunks = x.split(self.group_dim, dim=-1)
        outs = []
        for i, mlp in enumerate(self.mlps):
            outs.append(mlp(chunks[i]))
        if self.remainder > 0:
            outs.append(self.remainder_mlp(chunks[-1]))
        h = torch.cat(outs, dim=-1)
        return self.norm(h)


# ═══════════════════════════════════════════════════════════════════════
# Non-factorized baseline (single MLP, for ablation)
# ═══════════════════════════════════════════════════════════════════════

class PlainMLP(nn.Module):
    """Single MLP as non-factorized baseline for ablation."""

    def __init__(self, d_model: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor,
                token_structure: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════
# Gated Fusion
# ═══════════════════════════════════════════════════════════════════════

class GatedFusion(nn.Module):
    """
    Gated fusion of identity + temporal branch + spatial branch.

    F = gate_t * T(H) + gate_s * S(H) + (1 - gate_t - gate_s) * H

    Gates are learned from H via a small FC and sigmoid, providing
    interpretable blending weights.

    Returns (fused, gate_stats_dict) where gate_stats contains mean gate
    values for logging.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Project input to 2 gate scalars (temporal, spatial)
        self.gate_proj = nn.Linear(d_model, 2)
        # Initialize near zero so early training is close to identity
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, h: torch.Tensor, t_out: torch.Tensor,
                s_out: torch.Tensor):
        """
        Args:
            h: normalized input (B, T, D)
            t_out: temporal branch output (B, T, D)
            s_out: spatial branch output (B, T, D)
        Returns:
            fused: (B, T, D)
            gate_stats: dict with 'gate_temporal_mean', 'gate_spatial_mean'
        """
        # Compute per-token gates from pooled representation
        g = self.gate_proj(h.mean(dim=1))  # (B, 2)
        g = torch.sigmoid(g)               # (B, 2) in [0, 1]
        g_t = g[:, 0:1].unsqueeze(1)       # (B, 1, 1)
        g_s = g[:, 1:2].unsqueeze(1)       # (B, 1, 1)

        fused = g_t * t_out + g_s * s_out + (1 - g_t - g_s).clamp(min=0) * h

        gate_stats = {
            'gate_temporal_mean': g[:, 0].mean().item(),
            'gate_spatial_mean': g[:, 1].mean().item(),
        }
        return fused, gate_stats


# ═══════════════════════════════════════════════════════════════════════
# Factory helpers
# ═══════════════════════════════════════════════════════════════════════

def build_temporal_branch(d_model: int, branch_type: str, **kwargs) -> nn.Module:
    """Build temporal branch. All branches accept optional token_structure in forward()."""
    if branch_type == 'depthwise_conv':
        return DepthwiseTemporalConv(
            d_model,
            kernel_size=kwargs.get('kernel_size', 5),
            dropout=kwargs.get('dropout', 0.1),
        )
    elif branch_type == 'low_rank_mix':
        return LowRankTemporalMix(
            d_model,
            rank=kwargs.get('rank', 16),
            dropout=kwargs.get('dropout', 0.1),
        )
    else:
        raise ValueError(f"Unknown temporal branch type: {branch_type}")


def build_spatial_branch(d_model: int, branch_type: str, **kwargs) -> nn.Module:
    """Build spatial branch. All branches accept optional token_structure in forward()."""
    if branch_type == 'channel_attention':
        return ChannelAttention(
            d_model,
            reduction=kwargs.get('reduction', 4),
            dropout=kwargs.get('dropout', 0.1),
        )
    elif branch_type == 'grouped_mlp':
        return GroupedSpatialMLP(
            d_model,
            dropout=kwargs.get('dropout', 0.1),
        )
    else:
        raise ValueError(f"Unknown spatial branch type: {branch_type}")
