"""
BrainPro: Brain State-Aware EEG Representation Learning

Reimplementation based on:
    "BrainPro: Towards Large-Scale Brain State-Aware EEG Representation Learning"
    Ding et al., arXiv:2509.22050v1, 2025

Architecture:
    Each encoder consists of 4 stages:
    1. Temporal Encoder: Channel-wise 1D CNN (GroupNorm + GELU)
    2. Retrieval-based Spatial Learner: Channel/region filter banks
    3. Patch Maker: Patchification + linear projection + positional embedding
    4. Transformer Encoder: Pre-norm MSA + MLP

    Pre-training: Shared encoder + K brain-state encoders + decoders
    Fine-tuning: Flexible encoder selection + MLP head
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Constants: Universal channel template and brain region definitions
# =============================================================================

# 60-channel universal template (SEED montage, excluding 2 reference channels)
UNIVERSAL_CHANNELS = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
    'O1', 'OZ', 'O2',
]
C_PRE = len(UNIVERSAL_CHANNELS)  # 60

# Channel name → index in universal template
CHANNEL_TO_IDX = {ch: i for i, ch in enumerate(UNIVERSAL_CHANNELS)}

# Common aliases (10-20 naming variants)
CHANNEL_ALIASES = {
    'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8',
    'TP9': 'TP7', 'TP10': 'TP8',
    'FP1': 'FP1', 'FP2': 'FP2',  # already canonical
}

# 24 brain regions (hemisphere-based, following LGGNet with BrainPro adjustments)
# Midline channels assigned to left hemisphere
BRAIN_REGIONS: Dict[int, Tuple[str, List[str]]] = {
    0:  ('Fp-L', ['FP1', 'FPZ']),
    1:  ('Fp-R', ['FP2']),
    2:  ('AF-L', ['AF3']),
    3:  ('AF-R', ['AF4']),
    4:  ('F-L',  ['F7', 'F5', 'F3', 'F1', 'FZ']),
    5:  ('F-R',  ['F2', 'F4', 'F6', 'F8']),
    6:  ('FC-L', ['FC5', 'FC3', 'FC1', 'FCZ']),
    7:  ('FC-R', ['FC2', 'FC4', 'FC6']),
    8:  ('FT-L', ['FT7']),
    9:  ('FT-R', ['FT8']),
    10: ('C-L',  ['C5', 'C3', 'C1', 'CZ']),
    11: ('C-R',  ['C2', 'C4', 'C6']),
    12: ('T-L',  ['T7']),
    13: ('T-R',  ['T8']),
    14: ('CP-L', ['CP5', 'CP3', 'CP1', 'CPZ']),
    15: ('CP-R', ['CP2', 'CP4', 'CP6']),
    16: ('TP-L', ['TP7']),
    17: ('TP-R', ['TP8']),
    18: ('P-L',  ['P7', 'P5', 'P3', 'P1', 'PZ']),
    19: ('P-R',  ['P2', 'P4', 'P6', 'P8']),
    20: ('PO-L', ['PO7', 'PO5', 'PO3', 'POZ']),
    21: ('PO-R', ['PO4', 'PO6', 'PO8']),
    22: ('O-L',  ['O1', 'OZ']),
    23: ('O-R',  ['O2']),
}
N_REGIONS = len(BRAIN_REGIONS)  # 24

# Build channel → region index mapping
CHANNEL_TO_REGION = {}
for region_idx, (region_name, channels) in BRAIN_REGIONS.items():
    for ch in channels:
        CHANNEL_TO_REGION[ch] = region_idx

# Brain-state-specific channel importance priors (for region-aware reconstruction)
BRAIN_STATE_PRIORS = {
    'affect': {  # Frontal/temporal emphasis
        0: 1.0, 1: 1.0, 2: 0.9, 3: 0.9,      # Fp, AF
        4: 0.8, 5: 0.8, 6: 0.4, 7: 0.4,       # F, FC
        8: 0.7, 9: 0.7, 10: 0.3, 11: 0.3,     # FT, C
        12: 0.7, 13: 0.7, 14: 0.2, 15: 0.2,   # T, CP
        16: 0.5, 17: 0.5, 18: 0.2, 19: 0.2,   # TP, P
        20: 0.2, 21: 0.2, 22: 0.2, 23: 0.2,   # PO, O
    },
    'motor': {  # Central/parietal emphasis
        0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2,
        4: 0.3, 5: 0.3, 6: 0.8, 7: 0.8,
        8: 0.5, 9: 0.5, 10: 1.0, 11: 1.0,
        12: 0.3, 13: 0.3, 14: 0.9, 15: 0.9,
        16: 0.4, 17: 0.4, 18: 0.7, 19: 0.7,
        20: 0.3, 21: 0.3, 22: 0.2, 23: 0.2,
    },
    'others': {i: 0.5 for i in range(N_REGIONS)},  # Uniform
}

# Default channel names for known datasets
DATASET_CHANNEL_NAMES = {
    'TUEV': ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4'],
    'TUAB': ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4'],
    'TUSZ': ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4',
             'P8', 'O1', 'OZ', 'O2', 'FT7', 'FT8'],
    'CHB-MIT': ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4'],
}


# =============================================================================
# 2. Utility functions
# =============================================================================

def resolve_channel_indices(channel_names: List[str]) -> List[int]:
    """Map dataset channel names to indices in the universal template.

    Returns list of indices (one per dataset channel). Channels not found
    in the template are assigned -1.
    """
    indices = []
    for ch in channel_names:
        ch_upper = ch.upper().strip()
        # Direct match
        if ch_upper in CHANNEL_TO_IDX:
            indices.append(CHANNEL_TO_IDX[ch_upper])
            continue
        # Alias match
        alias = CHANNEL_ALIASES.get(ch_upper)
        if alias and alias in CHANNEL_TO_IDX:
            indices.append(CHANNEL_TO_IDX[alias])
            continue
        indices.append(-1)  # Not found
    return indices


def get_region_mapping(channel_indices: List[int]) -> Dict[int, List[int]]:
    """Given universal template indices for present channels, compute region → local channel indices.

    Returns:
        region_to_local: dict mapping region_idx → list of local channel positions
    """
    region_to_local = {}
    for local_idx, template_idx in enumerate(channel_indices):
        if template_idx < 0:
            continue
        ch_name = UNIVERSAL_CHANNELS[template_idx]
        region_idx = CHANNEL_TO_REGION.get(ch_name)
        if region_idx is not None:
            region_to_local.setdefault(region_idx, []).append(local_idx)
    return region_to_local


def compute_importance_weights(
    brain_state: str,
    channel_indices: List[int],
    epoch: int = 0,
    total_epochs: int = 30,
) -> torch.Tensor:
    """Compute per-channel importance weights for region-aware reconstruction loss.

    Eq 17: weights(w) = 0.5 + sigmoid(T * (w - 0.5)), T = epoch
    """
    priors = BRAIN_STATE_PRIORS.get(brain_state, BRAIN_STATE_PRIORS['others'])
    weights = []
    for template_idx in channel_indices:
        if template_idx < 0:
            weights.append(0.5)
            continue
        ch_name = UNIVERSAL_CHANNELS[template_idx]
        region_idx = CHANNEL_TO_REGION.get(ch_name, -1)
        w = priors.get(region_idx, 0.5)
        weights.append(w)

    w_tensor = torch.tensor(weights, dtype=torch.float32)
    T = float(max(epoch, 1))
    return 0.5 + torch.sigmoid(T * (w_tensor - 0.5))


# =============================================================================
# 3. Temporal Encoder: Channel-wise multi-layer 1D CNN
# =============================================================================

class TemporalEncoder(nn.Module):
    """Multi-layer 1D CNN applied independently per EEG channel.

    Architecture (from Table 4):
        Conv1D(1→32, k=15, s=1, p=7) → GroupNorm → GELU
        Conv1D(32→32, k=3, s=1, p=1) → GroupNorm → GELU
        Conv1D(32→32, k=3, s=1, p=1) → GroupNorm → GELU

    Input: (B, C, T) — raw EEG, C channels, T time samples
    Output: (B, K_T, C, T) — K_T=32 temporal features per channel
    """

    def __init__(self, K_T: int = 32, n_groups: int = 8):
        super().__init__()
        self.K_T = K_T
        layers = []
        in_channels_list = [1, K_T, K_T]
        out_channels_list = [K_T, K_T, K_T]
        kernel_sizes = [15, 3, 3]
        paddings = [7, 1, 1]

        for in_ch, out_ch, k, p in zip(in_channels_list, out_channels_list, kernel_sizes, paddings):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=1, padding=p))
            layers.append(nn.GroupNorm(n_groups, out_ch))
            layers.append(nn.GELU())

        self.cnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, K_T, C, T)
        """
        B, C, T = x.shape
        # Process each channel independently: (B*C, 1, T)
        x = x.reshape(B * C, 1, T)
        h = self.cnn(x)  # (B*C, K_T, T)
        h = h.reshape(B, C, self.K_T, T)  # (B, C, K_T, T)
        h = h.permute(0, 2, 1, 3)  # (B, K_T, C, T)
        return h


# =============================================================================
# 4. Retrieval-based Spatial Learner
# =============================================================================

class SpatialLearner(nn.Module):
    """Retrieval-based spatial learning with channel and region filter banks.

    Maintains learnable filter banks W_C ∈ R^{C_pre×K_C} and W_R ∈ R^{N_region×K_R}
    aligned to the universal template. During forward pass, retrieves filters for
    channels present in the current dataset.

    For datasets without channel name mapping, falls back to a direct learnable
    projection (no retrieval).

    Input: (B, K_T, C, T) from TemporalEncoder
    Output: (B, K_T*K_total, T) where K_total = K_C + K_R
    """

    def __init__(
        self,
        K_T: int = 32,
        K_C: int = 32,
        K_R: int = 32,
        n_channels: int = 16,
        channel_indices: Optional[List[int]] = None,
        region_mapping: Optional[Dict[int, List[int]]] = None,
        random_retrieval: bool = False,
        n_groups: int = 8,
    ):
        super().__init__()
        self.K_T = K_T
        self.K_C = K_C
        self.K_R = K_R
        self.K_total = K_C + K_R
        self.n_channels = n_channels
        self.use_retrieval = channel_indices is not None and all(i >= 0 for i in channel_indices)

        if self.use_retrieval:
            # Full filter banks on universal template
            self.W_C = nn.Parameter(torch.randn(C_PRE, K_C) * 0.02)
            self.W_R = nn.Parameter(torch.randn(N_REGIONS, K_R) * 0.02)

            if random_retrieval:
                # Ablation: random fixed filters (not learned)
                self.W_C.requires_grad = False
                self.W_R.requires_grad = False

            # Store channel indices as buffer (not a parameter)
            self.register_buffer('ch_indices', torch.tensor(channel_indices, dtype=torch.long))

            # Compute region mapping
            if region_mapping is None:
                region_mapping = get_region_mapping(channel_indices)
            self.region_indices = sorted(region_mapping.keys())
            self.register_buffer('region_idx_tensor', torch.tensor(self.region_indices, dtype=torch.long))

            # Build local channel → region aggregation indices
            # For each region, store the local channel positions
            max_channels_per_region = max(len(v) for v in region_mapping.values()) if region_mapping else 1
            n_regions_present = len(self.region_indices)
            agg_indices = torch.zeros(n_regions_present, max_channels_per_region, dtype=torch.long)
            agg_mask = torch.zeros(n_regions_present, max_channels_per_region, dtype=torch.bool)
            for i, reg_idx in enumerate(self.region_indices):
                local_chs = region_mapping[reg_idx]
                for j, ch in enumerate(local_chs):
                    agg_indices[i, j] = ch
                    agg_mask[i, j] = True
            self.register_buffer('agg_indices', agg_indices)
            self.register_buffer('agg_mask', agg_mask)
        else:
            # Fallback: direct learnable spatial projection (no retrieval)
            self.W_C = nn.Parameter(torch.randn(n_channels, K_C) * 0.02)
            self.W_R = nn.Parameter(torch.randn(n_channels, K_R) * 0.02)

        # Normalization + activation for channel and region features
        self.ch_norm = nn.GroupNorm(min(n_groups, K_C), K_C)
        self.ch_act = nn.GELU()
        self.rg_norm = nn.GroupNorm(min(n_groups, K_R), K_R)
        self.rg_act = nn.GELU()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, K_T, C, T) from TemporalEncoder
        Returns:
            (B, K_T*K_total, T)
        """
        B, K_T, C, T = h.shape
        # Reshape: (B, K_T, C, T) → (B, C, K_T, T) → (B, C, K_T*T)
        h = h.permute(0, 2, 1, 3).reshape(B, C, K_T * T)

        if self.use_retrieval:
            # Retrieve channel filters for present channels
            W_ch = self.W_C[self.ch_indices]  # (C, K_C)
            # Channel features: H_C = σ(W_C[I_ch]^T @ H_reshp_temp)
            H_C = torch.einsum('ck,bcd->bkd', W_ch, h)  # (B, K_C, K_T*T)

            # Region aggregation: average channels within each region
            # agg_indices: (N_R_present, max_ch_per_region)
            # agg_mask: (N_R_present, max_ch_per_region)
            N_R = self.agg_indices.shape[0]
            # Gather channel features for each region
            expanded_indices = self.agg_indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, K_T * T)
            h_expanded = h.unsqueeze(1).expand(-1, N_R, -1, -1)
            gathered = torch.gather(h_expanded, 2, expanded_indices)  # (B, N_R, max_ch, K_T*T)
            mask_expanded = self.agg_mask.unsqueeze(0).unsqueeze(-1).float()  # (1, N_R, max_ch, 1)
            M_region = (gathered * mask_expanded).sum(dim=2) / mask_expanded.sum(dim=2).clamp(min=1)  # (B, N_R, K_T*T)

            # Retrieve region filters
            W_rg = self.W_R[self.region_idx_tensor]  # (N_R, K_R)
            H_R = torch.einsum('rk,brd->bkd', W_rg, M_region)  # (B, K_R, K_T*T)
        else:
            # Fallback: direct projection
            H_C = torch.einsum('ck,bcd->bkd', self.W_C, h)  # (B, K_C, K_T*T)
            H_R = torch.einsum('ck,bcd->bkd', self.W_R, h)  # (B, K_R, K_T*T)

        # Apply normalization + activation
        H_C = self.ch_act(self.ch_norm(H_C))
        H_R = self.rg_act(self.rg_norm(H_R))

        # Concatenate: (B, K_total, K_T*T)
        H_spatial = torch.cat([H_C, H_R], dim=1)

        # Reshape: (B, K_total, K_T*T) → (B, K_total, K_T, T) → (B, K_T*K_total, T)
        H_spatial = H_spatial.view(B, self.K_total, K_T, T)
        H_spatial = H_spatial.permute(0, 2, 1, 3).reshape(B, K_T * self.K_total, T)

        return H_spatial


# =============================================================================
# 5. Patch Maker: Patchification + projection + positional embedding
# =============================================================================

class PatchMaker(nn.Module):
    """Patchify spatial features and project to token embeddings.

    Splits H_spatial into temporal patches, flattens, and projects to d-dim tokens.
    Adds learnable positional embeddings.

    Input: (B, K_T*K_total, T)
    Output: (B, N_p, d) where N_p = (T - patch_len) // patch_stride + 1
    """

    def __init__(
        self,
        feat_dim: int,    # K_T * K_total
        d_model: int = 32,
        patch_len: int = 20,
        patch_stride: int = 20,
        max_patches: int = 200,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.patch_len = patch_len
        self.patch_stride = patch_stride

        # Linear projection: flatten(patch) → d_model
        self.proj = nn.Linear(feat_dim * patch_len, d_model)

        # Learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)

        # Learned [MASK] token for pre-training
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            h: (B, feat_dim, T)
            mask: optional (B, N_p) binary mask (1=masked, 0=visible)
        Returns:
            tokens: (B, N_p, d_model)
        """
        B, F, T = h.shape
        # Unfold into patches: (B, F, T) → (B, F, N_p, patch_len)
        patches = h.unfold(dimension=2, size=self.patch_len, step=self.patch_stride)
        N_p = patches.shape[2]
        # Reshape: (B, F, N_p, P) → (B, N_p, F*P)
        patches = patches.permute(0, 2, 1, 3).reshape(B, N_p, -1)

        # Project to d_model
        tokens = self.proj(patches)  # (B, N_p, d_model)

        # Apply mask (for pre-training)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # (B, N_p, 1)
            mask_tokens = self.mask_token.expand(B, N_p, -1)
            tokens = tokens * (1 - mask_expanded) + mask_tokens * mask_expanded

        # Add positional embedding
        tokens = tokens + self.pos_embedding[:, :N_p, :]
        tokens = self.dropout(tokens)

        return tokens

    def reset_pos_embedding(self):
        """Re-initialize positional embeddings (recommended for fine-tuning)."""
        nn.init.xavier_uniform_(self.pos_embedding)


# =============================================================================
# 6. Transformer Encoder Layer (Pre-norm)
# =============================================================================

class BrainProTransformerLayer(nn.Module):
    """Pre-norm Transformer block with MSA + MLP.

    Z' = Z + MSA(LN(Z))
    Z_out = Z' + MLP(LN(Z'))
    """

    def __init__(self, d_model: int = 32, nhead: int = 32, d_ff: int = 64, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MSA with pre-norm
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        # MLP with pre-norm
        x = x + self.mlp(self.norm2(x))
        return x


# =============================================================================
# 7. BrainPro Encoder (full pipeline)
# =============================================================================

class BrainProEncoder(nn.Module):
    """Complete BrainPro encoder: Temporal → Spatial → Patch → Transformer.

    Args:
        n_channels: Number of EEG channels in the dataset
        K_T: Temporal feature dimension (CNN output channels)
        K_C: Number of channel-wise spatial filters
        K_R: Number of region-wise spatial filters
        d_model: Transformer hidden dimension
        nhead: Number of attention heads
        d_ff: Transformer MLP dimension
        n_layers: Number of transformer layers
        patch_len: Temporal patch length
        patch_stride: Temporal patch stride
        max_patches: Maximum number of patches (for pos embedding)
        channel_indices: Universal template indices for present channels
        region_mapping: Region → local channel mapping
        random_retrieval: Use random (fixed) spatial filters (ablation)
        dropout: Dropout rate
    """

    def __init__(
        self,
        n_channels: int = 16,
        K_T: int = 32,
        K_C: int = 32,
        K_R: int = 32,
        d_model: int = 32,
        nhead: int = 32,
        d_ff: int = 64,
        n_layers: int = 4,
        patch_len: int = 20,
        patch_stride: int = 20,
        max_patches: int = 200,
        channel_indices: Optional[List[int]] = None,
        region_mapping: Optional[Dict[int, List[int]]] = None,
        random_retrieval: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        K_total = K_C + K_R
        feat_dim = K_T * K_total

        self.temporal_encoder = TemporalEncoder(K_T=K_T)
        self.spatial_learner = SpatialLearner(
            K_T=K_T, K_C=K_C, K_R=K_R,
            n_channels=n_channels,
            channel_indices=channel_indices,
            region_mapping=region_mapping,
            random_retrieval=random_retrieval,
        )
        self.patch_maker = PatchMaker(
            feat_dim=feat_dim, d_model=d_model,
            patch_len=patch_len, patch_stride=patch_stride,
            max_patches=max_patches, dropout=dropout,
        )
        self.transformer = nn.ModuleList([
            BrainProTransformerLayer(d_model=d_model, nhead=nhead, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) raw EEG
            mask: optional (B, N_p) binary mask for pre-training
        Returns:
            (B, N_p, d_model)
        """
        h = self.temporal_encoder(x)        # (B, K_T, C, T)
        h = self.spatial_learner(h)          # (B, K_T*K_total, T)
        tokens = self.patch_maker(h, mask)   # (B, N_p, d_model)
        for layer in self.transformer:
            tokens = layer(tokens)
        tokens = self.final_norm(tokens)
        return tokens

    def reset_pos_embedding(self):
        """Re-initialize temporal positional embeddings for fine-tuning."""
        self.patch_maker.reset_pos_embedding()


# =============================================================================
# 8. Reconstruction Decoder (for pre-training)
# =============================================================================

class ReconstructionDecoder(nn.Module):
    """Transformer-based decoder for masked reconstruction.

    Takes concatenated outputs from shared + state encoder, applies transformer
    layers, then projects each token back to channel-time space.

    Input: (B, 2*N_p, d_model) — concatenated encoder outputs
    Output: (B, C_pre, T) — reconstructed signal on universal template
    """

    def __init__(
        self,
        d_model: int = 32,
        nhead: int = 32,
        d_ff: int = 64,
        n_layers: int = 2,
        patch_len: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Project concatenated input (2*d → d)
        self.input_proj = nn.Linear(2 * d_model, d_model)

        self.transformer = nn.ModuleList([
            BrainProTransformerLayer(d_model=d_model, nhead=nhead, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Reconstruction head: project each token to C_pre * patch_len
        self.recon_head = nn.Linear(d_model, C_PRE * patch_len)
        self.patch_len = patch_len

    def forward(self, z_shared: torch.Tensor, z_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_shared: (B, N_p, d_model)
            z_state: (B, N_p, d_model)
        Returns:
            x_hat: (B, C_pre, T_reconstructed) where T_reconstructed = N_p * patch_len
        """
        B, N_p, d = z_shared.shape
        # Concatenate encoder outputs along feature dim → project back to d
        z_cat = torch.cat([z_shared, z_state], dim=-1)  # (B, N_p, 2*d)
        z = self.input_proj(z_cat)  # (B, N_p, d)

        for layer in self.transformer:
            z = layer(z)
        z = self.norm(z)

        # Reconstruct: (B, N_p, d) → (B, N_p, C_pre * P) → (B, C_pre, N_p * P)
        x_hat = self.recon_head(z)  # (B, N_p, C_pre * P)
        x_hat = x_hat.view(B, N_p, C_PRE, self.patch_len)
        x_hat = x_hat.permute(0, 2, 1, 3).reshape(B, C_PRE, N_p * self.patch_len)

        return x_hat


# =============================================================================
# 9. BrainPro Pre-training Model
# =============================================================================

class BrainProPretrainModel(nn.Module):
    """Full BrainPro model for pre-training.

    Consists of:
    - 1 shared encoder (E_S)
    - K brain-state-specific encoders (E_A, E_M, E_O for affect, motor, others)
    - K decoders (D_A, D_M, D_O)

    Pre-training objectives:
    - Region-aware masked reconstruction (Eq 18)
    - Brain-state decoupling loss (Eq 19)
    """

    BRAIN_STATES = ['affect', 'motor', 'others']

    def __init__(
        self,
        n_channels: int = 60,
        K_T: int = 32,
        K_C: int = 32,
        K_R: int = 32,
        d_model: int = 32,
        nhead: int = 32,
        d_ff: int = 64,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 2,
        patch_len: int = 20,
        patch_stride: int = 20,
        max_patches: int = 200,
        mask_ratio: float = 0.5,
        decoupling_margin: float = 0.1,
        channel_indices: Optional[List[int]] = None,
        region_mapping: Optional[Dict[int, List[int]]] = None,
        random_retrieval: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.decoupling_margin = decoupling_margin
        self.n_channels = n_channels
        self.patch_len = patch_len
        self.patch_stride = patch_stride

        encoder_kwargs = dict(
            n_channels=n_channels, K_T=K_T, K_C=K_C, K_R=K_R,
            d_model=d_model, nhead=nhead, d_ff=d_ff, n_layers=n_encoder_layers,
            patch_len=patch_len, patch_stride=patch_stride, max_patches=max_patches,
            channel_indices=channel_indices, region_mapping=region_mapping,
            random_retrieval=random_retrieval, dropout=dropout,
        )

        # Shared encoder
        self.encoder_shared = BrainProEncoder(**encoder_kwargs)
        # State-specific encoders
        self.encoders_state = nn.ModuleDict({
            state: BrainProEncoder(**encoder_kwargs) for state in self.BRAIN_STATES
        })
        # State-specific decoders
        self.decoders = nn.ModuleDict({
            state: ReconstructionDecoder(
                d_model=d_model, nhead=nhead, d_ff=d_ff,
                n_layers=n_decoder_layers, patch_len=patch_len, dropout=dropout,
            ) for state in self.BRAIN_STATES
        })

    def generate_mask(self, B: int, N_p: int, device: torch.device) -> torch.Tensor:
        """Generate random patch mask. Returns (B, N_p) with 1=masked, 0=visible."""
        n_masked = int(N_p * self.mask_ratio)
        mask = torch.zeros(B, N_p, device=device)
        for i in range(B):
            indices = torch.randperm(N_p, device=device)[:n_masked]
            mask[i, indices] = 1.0
        return mask

    def forward(
        self,
        x: torch.Tensor,
        brain_state: str,
        channel_indices: Optional[List[int]] = None,
        epoch: int = 0,
        total_epochs: int = 30,
        use_masking: bool = True,
        use_reconstruction: bool = True,
        use_decoupling: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, T) raw EEG
            brain_state: 'affect', 'motor', or 'others'
            channel_indices: template indices for current channels (for importance weights)
            epoch: current epoch (for importance weight schedule)
            total_epochs: total epochs
            use_masking: whether to apply masking (ablation toggle)
            use_reconstruction: whether to compute reconstruction loss
            use_decoupling: whether to compute decoupling loss
        Returns:
            dict with 'loss', 'loss_rec', 'loss_dec', 'z_shared', 'z_state'
        """
        B, C, T = x.shape
        N_p = (T - self.patch_len) // self.patch_stride + 1

        # Generate mask
        mask = self.generate_mask(B, N_p, x.device) if use_masking else None

        # Encode with shared encoder
        z_shared = self.encoder_shared(x, mask=mask)  # (B, N_p, d)

        # Encode with all state encoders
        z_states = {}
        for state in self.BRAIN_STATES:
            z = self.encoders_state[state](x, mask=mask)
            if state != brain_state:
                z = z.detach()  # Selective gradient: detach inactive encoders
            z_states[state] = z

        z_active = z_states[brain_state]

        losses = {}
        total_loss = torch.tensor(0.0, device=x.device)

        # 1. Region-aware masked reconstruction loss
        if use_reconstruction:
            x_hat = self.decoders[brain_state](z_shared, z_active)  # (B, C_pre, T_hat)
            T_hat = x_hat.shape[2]

            # Map dataset channels to universal template for loss
            if channel_indices is not None:
                valid_ch = [i for i in range(len(channel_indices)) if channel_indices[i] >= 0]
                template_ch = [channel_indices[i] for i in valid_ch]
                x_target = x[:, valid_ch, :T_hat]  # (B, C_valid, T_hat)
                x_pred = x_hat[:, template_ch, :T_hat]  # (B, C_valid, T_hat)

                # Importance weights
                w = compute_importance_weights(brain_state, channel_indices, epoch, total_epochs)
                w = w[valid_ch].to(x.device)  # (C_valid,)
            else:
                C_eff = min(C, C_PRE)
                x_target = x[:, :C_eff, :T_hat]
                x_pred = x_hat[:, :C_eff, :T_hat]
                w = torch.ones(C_eff, device=x.device) * 0.5

            # Weighted MSE on masked positions only
            if mask is not None:
                # Expand mask to cover patch positions in the time domain
                mask_time = mask.unsqueeze(1).repeat_interleave(self.patch_len, dim=2)  # (B, 1, T_hat)
                loss_rec = (w.view(1, -1, 1) * mask_time * (x_target - x_pred) ** 2).sum()
                loss_rec = loss_rec / (mask_time.sum() * len(w) + 1e-8)
            else:
                loss_rec = (w.view(1, -1, 1) * (x_target - x_pred) ** 2).mean()

            losses['loss_rec'] = loss_rec
            total_loss = total_loss + loss_rec

        # 2. Brain-state decoupling loss (Eq 19)
        if use_decoupling:
            # Pool to get segment-level representations
            z_shared_pool = z_shared.mean(dim=1)  # (B, d)
            z_active_pool = z_active.mean(dim=1)

            # Cosine similarity between shared and active state
            cos_shared_active = F.cosine_similarity(z_shared_pool, z_active_pool, dim=-1)
            dec_loss = F.relu(cos_shared_active - self.decoupling_margin).mean()

            # Cosine similarity between active and inactive states
            for state in self.BRAIN_STATES:
                if state != brain_state:
                    z_inactive_pool = z_states[state].mean(dim=1)
                    cos_active_inactive = F.cosine_similarity(z_active_pool, z_inactive_pool, dim=-1)
                    dec_loss = dec_loss + F.relu(cos_active_inactive - self.decoupling_margin).mean()

            losses['loss_dec'] = dec_loss
            total_loss = total_loss + dec_loss

        losses['loss'] = total_loss
        losses['z_shared'] = z_shared
        losses['z_state'] = z_active

        return losses


# =============================================================================
# 10. BrainPro Fine-tuning Model
# =============================================================================

class BrainProFinetuneModel(nn.Module):
    """BrainPro model for downstream fine-tuning.

    Supports flexible encoder selection and multiple token merge modes.

    Args:
        n_channels: Number of EEG channels
        num_classes: Number of output classes
        encoder_config: dict of encoder hyperparameters
        active_states: list of brain states to activate (e.g., ['affect'] or ['affect', 'motor'])
        token_merge: 'mean', 'aggr', or 'all'
        hidden_factor: MLP hidden dimension multiplier
        dropout: Dropout rate
        label_smoothing: Label smoothing for CE loss (0 = none)
        task_type: 'multiclass' or 'binary'
    """

    BRAIN_STATES = ['affect', 'motor', 'others']

    def __init__(
        self,
        n_channels: int = 16,
        num_classes: int = 6,
        K_T: int = 32,
        K_C: int = 32,
        K_R: int = 32,
        d_model: int = 32,
        nhead: int = 32,
        d_ff: int = 64,
        n_layers: int = 4,
        patch_len: int = 20,
        patch_stride: int = 20,
        max_patches: int = 200,
        channel_indices: Optional[List[int]] = None,
        region_mapping: Optional[Dict[int, List[int]]] = None,
        random_retrieval: bool = False,
        active_states: Optional[List[str]] = None,
        token_merge: str = 'mean',
        hidden_factor: int = 1,
        dropout: float = 0.1,
        label_smoothing: float = 0.0,
        task_type: str = 'multiclass',
    ):
        super().__init__()
        self.d_model = d_model
        self.token_merge = token_merge
        self.task_type = task_type
        self.label_smoothing = label_smoothing
        self.active_states = active_states or ['affect']  # Default: affect encoder
        self.n_active_encoders = 1 + len(self.active_states)  # shared + active states

        encoder_kwargs = dict(
            n_channels=n_channels, K_T=K_T, K_C=K_C, K_R=K_R,
            d_model=d_model, nhead=nhead, d_ff=d_ff, n_layers=n_layers,
            patch_len=patch_len, patch_stride=patch_stride, max_patches=max_patches,
            channel_indices=channel_indices, region_mapping=region_mapping,
            random_retrieval=random_retrieval, dropout=dropout,
        )

        # Shared encoder (always active)
        self.encoder_shared = BrainProEncoder(**encoder_kwargs)
        # State-specific encoders
        self.encoders_state = nn.ModuleDict({
            state: BrainProEncoder(**encoder_kwargs) for state in self.BRAIN_STATES
        })

        # MLP classification head
        # Input dim depends on token_merge mode
        # For 'mean' and 'aggr': d_model (pooled to single vector)
        # For 'all': will be computed dynamically in first forward pass
        self._head_initialized = False
        self._hidden_factor = hidden_factor
        self._num_classes = num_classes
        self._dropout = dropout

        if token_merge in ('mean', 'aggr'):
            head_input_dim = d_model
            self.classifier = self._build_mlp(head_input_dim, num_classes, d_model, hidden_factor, dropout)
            self._head_initialized = True
        else:
            # 'all' mode: lazy init since N_p depends on input length
            self.classifier = None

    def _build_mlp(self, input_dim, num_classes, d_model, hidden_factor, dropout):
        hidden_dim = d_model * hidden_factor
        output_dim = 1 if self.task_type == 'binary' else num_classes
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def _merge_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Merge token sequence into segment-level representation.

        Args:
            tokens: (B, N_tokens, d_model) — concatenated from all active encoders
        Returns:
            (B, feat_dim) — segment-level vector
        """
        if self.token_merge == 'mean':
            return tokens.mean(dim=1)
        elif self.token_merge == 'aggr':
            # Moving average: group every 5 tokens, then mean
            B, N, d = tokens.shape
            group_size = 5
            n_groups = N // group_size
            if n_groups > 0:
                tokens_trimmed = tokens[:, :n_groups * group_size, :]
                tokens_grouped = tokens_trimmed.view(B, n_groups, group_size, d).mean(dim=2)
            else:
                tokens_grouped = tokens
            return tokens_grouped.mean(dim=1)
        elif self.token_merge == 'all':
            # Flatten all tokens
            return tokens.reshape(tokens.shape[0], -1)
        else:
            raise ValueError(f"Unknown token_merge: {self.token_merge}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, S, P) or (B, C, T) — EEG input
        Returns:
            logits: (B, num_classes) or (B, 1) for binary
        """
        # Handle 4D input: (B, C, S, P) → (B, C, T)
        if x.dim() == 4:
            B, C, S, P = x.shape
            x = x.reshape(B, C, S * P)

        # Encode with shared encoder
        z_shared = self.encoder_shared(x)  # (B, N_p, d)

        # Encode with active state encoders
        z_list = [z_shared]
        for state in self.active_states:
            z_list.append(self.encoders_state[state](x))

        # Concatenate along token dimension
        z_cat = torch.cat(z_list, dim=1)  # (B, n_encoders * N_p, d)

        # Token merge
        features = self._merge_tokens(z_cat)  # (B, feat_dim)

        # Lazy init for 'all' mode
        if not self._head_initialized:
            feat_dim = features.shape[-1]
            self.classifier = self._build_mlp(
                feat_dim, self._num_classes, self.d_model,
                self._hidden_factor, self._dropout,
            ).to(x.device)
            self._head_initialized = True

        logits = self.classifier(features)
        return logits

    def load_pretrained(self, pretrained_path: str, reset_pos_emb: bool = True):
        """Load pre-trained weights and optionally reset positional embeddings.

        Args:
            pretrained_path: Path to pre-trained checkpoint
            reset_pos_emb: Whether to reset temporal positional embeddings (recommended)
        """
        print(f"Loading BrainPro pretrained weights from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location='cpu')

        # Handle checkpoint wrapping
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # Filter: only load encoder weights (not decoders)
        encoder_keys = {}
        for k, v in state_dict.items():
            if k.startswith('encoder_shared.') or k.startswith('encoders_state.'):
                encoder_keys[k] = v

        missing, unexpected = self.load_state_dict(encoder_keys, strict=False)
        if missing:
            # Filter out classifier keys (expected to be missing)
            real_missing = [k for k in missing if 'classifier' not in k]
            if real_missing:
                print(f"  Missing encoder keys: {len(real_missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")

        if reset_pos_emb:
            print("  Resetting temporal positional embeddings (Xavier uniform)")
            self.encoder_shared.reset_pos_embedding()
            for state in self.BRAIN_STATES:
                self.encoders_state[state].reset_pos_embedding()

        print("  Pre-trained weights loaded successfully")

    def get_param_groups(self, lr: float, weight_decay: float = 0.05):
        """Get parameter groups for optimizer.

        Returns separate groups for encoders and classifier with potentially
        different learning rates.
        """
        encoder_params = []
        classifier_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                encoder_params.append(param)

        return [
            {'params': encoder_params, 'lr': lr, 'weight_decay': weight_decay},
            {'params': classifier_params, 'lr': lr, 'weight_decay': weight_decay},
        ]


# =============================================================================
# 11. Helper: Create BrainPro model with sensible defaults
# =============================================================================

def create_brainpro_finetune(
    n_channels: int = 16,
    num_classes: int = 6,
    task_type: str = 'multiclass',
    channel_names: Optional[List[str]] = None,
    dataset_name: Optional[str] = None,
    active_states: Optional[List[str]] = None,
    token_merge: str = 'mean',
    hidden_factor: int = 1,
    random_retrieval: bool = False,
    # Architecture
    K_T: int = 32,
    K_C: int = 32,
    K_R: int = 32,
    d_model: int = 32,
    nhead: int = 32,
    d_ff: int = 64,
    n_layers: int = 4,
    patch_len: int = 20,
    patch_stride: int = 20,
    max_patches: int = 200,
    dropout: float = 0.1,
    label_smoothing: float = 0.0,
    # Pretrained weights
    pretrained_path: Optional[str] = None,
    reset_pos_emb: bool = True,
) -> BrainProFinetuneModel:
    """Create a BrainPro fine-tuning model with automatic channel mapping.

    Args:
        n_channels: Number of EEG channels
        num_classes: Number of output classes
        task_type: 'multiclass' or 'binary'
        channel_names: List of channel names (for spatial retrieval)
        dataset_name: Dataset name (to lookup default channel names)
        active_states: Brain states to activate (default: auto-select based on task)
        token_merge: Token merge mode ('mean', 'aggr', 'all')
        hidden_factor: MLP hidden dimension multiplier
        random_retrieval: Use random spatial filters (ablation)
        pretrained_path: Path to pretrained checkpoint (None = train from scratch)
        reset_pos_emb: Reset positional embeddings after loading
        ... (architecture params)
    """
    # Resolve channel names
    if channel_names is None and dataset_name is not None:
        channel_names = DATASET_CHANNEL_NAMES.get(dataset_name.upper())

    # Compute channel indices and region mapping
    channel_indices = None
    region_mapping = None
    if channel_names is not None:
        channel_indices = resolve_channel_indices(channel_names)
        if all(i >= 0 for i in channel_indices):
            region_mapping = get_region_mapping(channel_indices)
        else:
            # Some channels not found, fallback to no retrieval
            n_unmapped = sum(1 for i in channel_indices if i < 0)
            print(f"WARNING: {n_unmapped}/{len(channel_indices)} channels not in universal template, "
                  f"falling back to direct spatial projection")
            channel_indices = None

    # Default active states
    if active_states is None:
        active_states = ['affect']  # Default: shared + affect

    model = BrainProFinetuneModel(
        n_channels=n_channels,
        num_classes=num_classes,
        K_T=K_T, K_C=K_C, K_R=K_R,
        d_model=d_model, nhead=nhead, d_ff=d_ff, n_layers=n_layers,
        patch_len=patch_len, patch_stride=patch_stride, max_patches=max_patches,
        channel_indices=channel_indices,
        region_mapping=region_mapping,
        random_retrieval=random_retrieval,
        active_states=active_states,
        token_merge=token_merge,
        hidden_factor=hidden_factor,
        dropout=dropout,
        label_smoothing=label_smoothing,
        task_type=task_type,
    )

    # Load pretrained weights if available
    if pretrained_path is not None:
        model.load_pretrained(pretrained_path, reset_pos_emb=reset_pos_emb)

    # Print param summary
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nBrainPro Fine-tune Model:")
    print(f"  Active encoders: shared + {active_states}")
    print(f"  Token merge: {token_merge}, hidden_factor: {hidden_factor}")
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Spatial retrieval: {'yes' if channel_indices is not None else 'no (direct projection)'}")
    print()

    return model


def create_brainpro_pretrain(
    n_channels: int = 60,
    channel_names: Optional[List[str]] = None,
    random_retrieval: bool = False,
    # Architecture
    K_T: int = 32,
    K_C: int = 32,
    K_R: int = 32,
    d_model: int = 32,
    nhead: int = 32,
    d_ff: int = 64,
    n_encoder_layers: int = 4,
    n_decoder_layers: int = 2,
    patch_len: int = 20,
    patch_stride: int = 20,
    max_patches: int = 200,
    mask_ratio: float = 0.5,
    decoupling_margin: float = 0.1,
    dropout: float = 0.1,
) -> BrainProPretrainModel:
    """Create a BrainPro pre-training model."""
    channel_indices = None
    region_mapping = None
    if channel_names is not None:
        channel_indices = resolve_channel_indices(channel_names)
        if all(i >= 0 for i in channel_indices):
            region_mapping = get_region_mapping(channel_indices)
        else:
            channel_indices = None

    model = BrainProPretrainModel(
        n_channels=n_channels,
        K_T=K_T, K_C=K_C, K_R=K_R,
        d_model=d_model, nhead=nhead, d_ff=d_ff,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        patch_len=patch_len, patch_stride=patch_stride,
        max_patches=max_patches,
        mask_ratio=mask_ratio,
        decoupling_margin=decoupling_margin,
        channel_indices=channel_indices,
        region_mapping=region_mapping,
        random_retrieval=random_retrieval,
        dropout=dropout,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nBrainPro Pre-train Model:")
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")
    print()

    return model
