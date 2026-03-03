#!/usr/bin/env python3
"""
IMPROVED MSFT (Multi-Scale FineTuning) modules for CBraMod backbone.

Key improvements for CBraMod:
  1. Independent scale processing - preserves Criss-Cross Attention structure
  2. Scale-specific positional encoding refinement - adapts ACPE to different scales
  3. Criss-Cross-aware aggregator - respects spatial-temporal heterogeneity

Based on: "Multi-Scale Finetuning for Encoder-based Time Series Foundation Models"
Adapted for CBraMod's unique Criss-Cross Transformer architecture.
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Add CBraMod to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CBraMod'))
from models.cbramod import CBraMod


# ============================================================================
# Multi-Scale Generator (unchanged)
# ============================================================================
class MultiScaleGenerator(nn.Module):
    """Generate multi-scale EEG inputs via average-pooling downsampling.

    Downsamples along the seq_len (temporal patch) dimension by factors
    of 2^k for k = 0, 1, ..., num_scales-1.

    Input:  (B, n_channels, seq_len, patch_size)
    Output: list of num_scales tensors, each (B, n_channels, ceil(seq_len/2^k), patch_size)
    """

    def __init__(self, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales

    def forward(self, x):
        """
        Args:
            x: (B, n_ch, seq_len, patch_size)
        Returns:
            list of K tensors at different temporal scales
        """
        scales = [x]  # Scale 0 = original
        for k in range(1, self.num_scales):
            prev = scales[-1]
            B, C, S, P = prev.shape
            if S <= 1:
                scales.append(prev)
                continue
            # Pool along seq_len (dim=2)
            prev_flat = prev.permute(0, 1, 3, 2).reshape(B * C * P, 1, S)
            pooled = F.avg_pool1d(prev_flat, kernel_size=2, stride=2, ceil_mode=True)
            new_S = pooled.shape[-1]
            pooled = pooled.reshape(B, C, P, new_S).permute(0, 1, 3, 2)
            scales.append(pooled.contiguous())
        return scales


# ============================================================================
# Scale Adapter (unchanged)
# ============================================================================
class ScaleAdapter(nn.Module):
    """Scale-specific linear adapter applied after frozen PatchEmbedding.

    Projects patch embeddings through a learnable linear layer with residual
    connection to learn scale-specific representations.

    Input/Output: (B, n_ch, seq_len_k, d_model)
    """

    def __init__(self, d_model: int = 200):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

    def forward(self, x):
        return x + self.adapter(x)


# ============================================================================
# NEW: Scale-Specific Positional Encoding Refiner
# ============================================================================
class ScalePositionalRefiner(nn.Module):
    """Refine frozen ACPE for different scales.

    CBraMod's ACPE is learned during pretraining for a specific seq_len.
    Different scales have different seq_lens, so we add a lightweight
    refinement layer to adapt the positional encoding.

    Uses depthwise convolution (like ACPE) to maintain efficiency.

    Input/Output: (B, ch, seq_k, d_model)
    """

    def __init__(self, d_model: int = 200, kernel_size: int = 3):
        super().__init__()
        # Depthwise 2D conv on (ch, seq) dimensions
        # Similar to ACPE but simpler (smaller kernel)
        padding = (kernel_size // 2, kernel_size // 2)
        self.refiner = nn.Sequential(
            nn.Conv2d(
                d_model, d_model,
                kernel_size=(kernel_size, kernel_size),
                padding=padding,
                groups=d_model,  # depthwise
            ),
            nn.BatchNorm2d(d_model),
        )

    def forward(self, x):
        """
        Args:
            x: (B, ch, seq_k, d_model)
        Returns:
            x + refined_pos: (B, ch, seq_k, d_model)
        """
        # Permute to (B, d_model, ch, seq_k) for Conv2d
        x_permuted = x.permute(0, 3, 1, 2)  # (B, d, ch, seq)
        refined = self.refiner(x_permuted)
        refined = refined.permute(0, 2, 3, 1)  # back to (B, ch, seq, d)
        # Residual connection
        return x + refined


# ============================================================================
# IMPROVED: Criss-Cross-Aware Cross-Scale Aggregator
# ============================================================================
class CrissCrossAwareAggregator4D(nn.Module):
    """Bidirectional cross-scale aggregation that respects Criss-Cross structure.

    CBraMod models spatial and temporal dependencies separately.
    This aggregator performs:
      1. Temporal aggregation: C2F/F2C along seq dimension (different time resolutions)
      2. Spatial fusion: Channel-wise feature fusion across scales

    Key improvement: Recognizes that temporal and spatial dimensions are heterogeneous.
    """

    def __init__(self, d_model: int = 200, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales

        # === Temporal aggregation (along seq dimension) ===
        # C2F: project coarser scale i -> finer scale i-1
        self.temporal_c2f = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_scales - 1)
        ])
        # F2C: project finer scale i -> coarser scale i+1
        self.temporal_f2c = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_scales - 1)
        ])

        # === Spatial fusion (channel-wise) ===
        # Fuse features from all scales at each spatial location
        # This helps different scales learn complementary spatial patterns
        self.spatial_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
            ) for _ in range(num_scales)
        ])

    def _temporal_aggregation(self, scale_hiddens, scale_seq_lens):
        """Temporal C2F and F2C aggregation (original implementation)."""
        K = self.num_scales - 1
        B, ch = scale_hiddens[0].shape[:2]
        d = scale_hiddens[0].shape[3]

        # --- Coarse-to-Fine branch ---
        c2f_out = [h.clone() for h in scale_hiddens]
        for i in range(K, 0, -1):
            coarse = scale_hiddens[i]  # (B, ch, seq_i, d)
            projected = self.temporal_c2f[i - 1](coarse)
            # Upsample seq dimension
            seq_i = projected.shape[2]
            projected_3d = projected.permute(0, 1, 3, 2).reshape(B * ch, d, seq_i)
            upsampled_3d = F.interpolate(projected_3d, size=scale_seq_lens[i - 1], mode='nearest')
            upsampled = upsampled_3d.reshape(B, ch, d, scale_seq_lens[i - 1]).permute(0, 1, 3, 2)
            c2f_out[i - 1] = c2f_out[i - 1] + upsampled

        # --- Fine-to-Coarse branch ---
        f2c_out = [h.clone() for h in scale_hiddens]
        for i in range(0, K):
            fine = scale_hiddens[i]  # (B, ch, seq_i, d)
            projected = self.temporal_f2c[i](fine)
            seq_i = projected.shape[2]
            projected_3d = projected.permute(0, 1, 3, 2).reshape(B * ch, d, seq_i)
            downsampled_3d = F.adaptive_avg_pool1d(projected_3d, scale_seq_lens[i + 1])
            downsampled = downsampled_3d.reshape(B, ch, d, scale_seq_lens[i + 1]).permute(0, 1, 3, 2)
            f2c_out[i + 1] = f2c_out[i + 1] + downsampled

        # Average both branches
        merged = [(c + f) * 0.5 for c, f in zip(c2f_out, f2c_out)]
        return merged

    def _spatial_fusion(self, scale_hiddens):
        """Spatial feature fusion across scales.

        At each (channel, seq_position), we have features from different scales.
        We apply a scale-specific transformation and add it back.
        This allows spatial patterns to be shared across scales.
        """
        fused = []
        for k in range(self.num_scales):
            # Each scale: (B, ch, seq_k, d)
            h_k = scale_hiddens[k]
            # Apply spatial fusion (operates on d dimension, preserves spatial structure)
            h_k_fused = self.spatial_fusion[k](h_k)
            fused.append(h_k + h_k_fused)  # residual
        return fused

    def forward(self, scale_hiddens, scale_seq_lens):
        """
        Args:
            scale_hiddens: list of K tensors, each (B, ch, seq_k, d)
            scale_seq_lens: list of int [seq_0, seq_1, ..., seq_{K-1}]
        Returns:
            list of K tensors, aggregated
        """
        # 1. Temporal aggregation (C2F + F2C)
        temporal_agg = self._temporal_aggregation(scale_hiddens, scale_seq_lens)

        # 2. Spatial fusion
        spatial_agg = self._spatial_fusion(temporal_agg)

        return spatial_agg


# ============================================================================
# IMPROVED: MSFT Model for CBraMod
# ============================================================================
class ImprovedMSFTCBraModModel(nn.Module):
    """IMPROVED MSFT wrapper around a frozen CBraMod backbone.

    Key improvements:
      1. Independent scale processing - each scale goes through transformer separately
         This preserves CBraMod's Criss-Cross Attention structure

      2. Scale-specific positional encoding refinement
         Adapts frozen ACPE to different seq_lens at different scales

      3. Criss-Cross-aware aggregator
         Respects spatial-temporal heterogeneity in cross-scale fusion

    Architecture:
      Input -> MultiScaleGenerator -> [Scale 0, Scale 1, ..., Scale K]
        |
        v
      For each scale k:
        frozen PatchEmbedding -> trainable ScaleAdapter -> trainable PosRefiner
        |
        v
      Layer-by-layer processing:
        For each transformer layer:
          Each scale independently -> frozen CrissCrossTransformerLayer
          Cross-scale aggregation -> trainable CrissCrossAwareAggregator
        |
        v
      For each scale k:
        frozen proj_out -> trainable PerScaleClassifier
        |
        v
      Learnable softmax mixing -> Final logits

    Trainable components:
      - scale_adapters: per-scale linear adapters
      - scale_pos_refiners: per-scale positional encoding refiners (NEW)
      - cross_scale_aggs: per-layer criss-cross-aware aggregators (IMPROVED)
      - scale_mix_logits: learnable mixing weights
      - per_scale_classifiers: per-scale classification heads

    Frozen:
      - Entire CBraMod backbone
    """

    def __init__(
        self,
        backbone: CBraMod,
        num_scales: int = 3,
        num_classes: int = 6,
        task_type: str = 'multiclass',
        n_channels: int = 16,
        seq_len: int = 5,
        patch_size: int = 200,
        dropout: float = 0.1,
        n_layer: int = 12,
        use_pos_refiner: bool = True,
        use_criss_cross_agg: bool = True,
    ):
        super().__init__()

        self.backbone = backbone
        self.num_scales = num_scales
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.d_model = 200  # CBraMod d_model
        self.n_layer = n_layer
        self.task_type = task_type
        self.use_pos_refiner = use_pos_refiner
        self.use_criss_cross_agg = use_criss_cross_agg

        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Multi-scale input generator
        self.scale_generator = MultiScaleGenerator(num_scales)

        # Scale-specific adapters (after frozen PatchEmbedding)
        self.scale_adapters = nn.ModuleList([
            ScaleAdapter(d_model=self.d_model) for _ in range(num_scales)
        ])

        # NEW: Scale-specific positional encoding refiners
        if self.use_pos_refiner:
            self.scale_pos_refiners = nn.ModuleList([
                ScalePositionalRefiner(d_model=self.d_model, kernel_size=3)
                for _ in range(num_scales)
            ])

        # Cross-scale aggregators (one per transformer layer)
        num_encoder_layers = len(self.backbone.encoder.layers)
        if self.use_criss_cross_agg:
            self.cross_scale_aggs = nn.ModuleList([
                CrissCrossAwareAggregator4D(d_model=self.d_model, num_scales=num_scales)
                for _ in range(num_encoder_layers)
            ])
        else:
            # Fallback to simpler aggregator
            self.cross_scale_aggs = nn.ModuleList([
                SimpleCrossScaleAggregator4D(d_model=self.d_model, num_scales=num_scales)
                for _ in range(num_encoder_layers)
            ])

        # Learnable scale mixing logits -> softmax -> weights
        self.scale_mix_logits = nn.Parameter(torch.zeros(num_scales))

        # Per-scale classifiers
        output_dim = 1 if task_type == 'binary' else num_classes
        self.per_scale_classifiers = nn.ModuleList()
        for k in range(num_scales):
            seq_len_k = self._compute_seq_len(seq_len, k)
            input_dim_k = n_channels * seq_len_k * self.d_model
            hidden_dim = min(seq_len_k * self.d_model, 1000)
            self.per_scale_classifiers.append(nn.Sequential(
                nn.Linear(input_dim_k, hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.d_model),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(self.d_model, output_dim),
            ))

    @staticmethod
    def _compute_seq_len(seq_len, scale_idx):
        """Compute seq_len at scale k: ceil(seq_len / 2^k)."""
        s = seq_len
        for _ in range(scale_idx):
            s = math.ceil(s / 2)
        return s

    def forward(self, x):
        """
        IMPROVED FORWARD PASS:
        Key change: Each scale processes through transformer independently,
        preserving Criss-Cross Attention structure.

        Args:
            x: (B, n_channels, seq_len, patch_size=200)
        Returns:
            logits: (B, num_classes) or (B, 1) for binary
        """
        B = x.shape[0]

        # ---- Step 1: Multi-scale generation ----
        multi_scale_inputs = self.scale_generator(x)
        # multi_scale_inputs[k]: (B, n_ch, seq_len_k, 200)

        # ---- Step 2: Frozen PatchEmbedding + Trainable Adapters + Pos Refiners ----
        scale_embeddings = []
        scale_seq_lens = []
        for k, x_k in enumerate(multi_scale_inputs):
            # Frozen patch embedding (includes frozen ACPE)
            emb_k = self.backbone.patch_embedding(x_k)  # (B, ch, seq_k, 200)

            # Trainable scale adapter
            emb_k = self.scale_adapters[k](emb_k)

            # NEW: Trainable positional encoding refinement
            if self.use_pos_refiner:
                emb_k = self.scale_pos_refiners[k](emb_k)

            scale_seq_lens.append(emb_k.shape[2])
            scale_embeddings.append(emb_k)

        # ---- Step 3: IMPROVED Layer-by-layer processing ----
        # KEY CHANGE: Each scale goes through transformer independently
        scale_hiddens = scale_embeddings

        for layer_idx, layer in enumerate(self.backbone.encoder.layers):
            # Process each scale independently through the Criss-Cross Transformer layer
            new_scale_hiddens = []
            for h_k in scale_hiddens:
                # h_k: (B, ch, seq_k, d_model)
                # Frozen Criss-Cross Transformer layer
                # Criss-Cross Attention works correctly on fixed (ch, seq_k) structure
                h_k_out = layer(h_k)
                new_scale_hiddens.append(h_k_out)

            # Cross-scale aggregation between layers
            # This allows information flow between scales while preserving
            # the Criss-Cross structure within each scale
            scale_hiddens = self.cross_scale_aggs[layer_idx](
                new_scale_hiddens,
                scale_seq_lens
            )

        # ---- Step 4: Frozen proj_out for each scale ----
        scale_outputs = []
        for h_k in scale_hiddens:
            # h_k: (B, ch, seq_k, 200)
            out_k = self.backbone.proj_out(h_k)  # frozen
            scale_outputs.append(out_k)

        # ---- Step 5: Per-scale classifiers + mixing ----
        scale_logits = []
        for k, out_k in enumerate(scale_outputs):
            # Flatten: (B, ch, seq_k, 200) -> (B, ch * seq_k * 200)
            out_flat = out_k.contiguous().view(B, -1)
            logits_k = self.per_scale_classifiers[k](out_flat)
            scale_logits.append(logits_k)

        # Softmax-weighted mixing
        mix_weights = F.softmax(self.scale_mix_logits, dim=0)
        final_logits = sum(w * l for w, l in zip(mix_weights, scale_logits))

        return final_logits

    def get_scale_weights(self):
        """Return current scale mixing weights (for logging)."""
        return F.softmax(self.scale_mix_logits, dim=0).detach().cpu().tolist()


# ============================================================================
# Simplified aggregator (fallback)
# ============================================================================
class SimpleCrossScaleAggregator4D(nn.Module):
    """Simplified version without spatial fusion (for ablation studies)."""

    def __init__(self, d_model: int = 200, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        self.c2f_maps = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_scales - 1)
        ])
        self.f2c_maps = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_scales - 1)
        ])

    def forward(self, scale_hiddens, scale_seq_lens):
        K = self.num_scales - 1
        B, ch = scale_hiddens[0].shape[:2]
        d = scale_hiddens[0].shape[3]

        c2f_out = [h.clone() for h in scale_hiddens]
        for i in range(K, 0, -1):
            coarse = scale_hiddens[i]
            projected = self.c2f_maps[i - 1](coarse)
            seq_i = projected.shape[2]
            projected_3d = projected.permute(0, 1, 3, 2).reshape(B * ch, d, seq_i)
            upsampled_3d = F.interpolate(projected_3d, size=scale_seq_lens[i - 1], mode='nearest')
            upsampled = upsampled_3d.reshape(B, ch, d, scale_seq_lens[i - 1]).permute(0, 1, 3, 2)
            c2f_out[i - 1] = c2f_out[i - 1] + upsampled

        f2c_out = [h.clone() for h in scale_hiddens]
        for i in range(0, K):
            fine = scale_hiddens[i]
            projected = self.f2c_maps[i](fine)
            seq_i = projected.shape[2]
            projected_3d = projected.permute(0, 1, 3, 2).reshape(B * ch, d, seq_i)
            downsampled_3d = F.adaptive_avg_pool1d(projected_3d, scale_seq_lens[i + 1])
            downsampled = downsampled_3d.reshape(B, ch, d, scale_seq_lens[i + 1]).permute(0, 1, 3, 2)
            f2c_out[i + 1] = f2c_out[i + 1] + downsampled

        merged = [(c + f) * 0.5 for c, f in zip(c2f_out, f2c_out)]
        return merged


# ============================================================================
# Factory function
# ============================================================================
def create_improved_msft_cbramod_model(
    num_classes,
    task_type,
    n_channels,
    seq_len,
    patch_size=200,
    dropout=0.1,
    n_layer=12,
    num_scales=3,
    dim_feedforward=800,
    nhead=8,
    pretrained_weights_path=None,
    device='cuda:0',
    use_pos_refiner=True,
    use_criss_cross_agg=True,
):
    """Create an ImprovedMSFTCBraModModel with a frozen CBraMod backbone.

    Args:
        num_classes: number of output classes
        task_type: 'binary' or 'multiclass'
        n_channels: number of EEG channels
        seq_len: number of temporal patches
        patch_size: patch size (default 200)
        dropout: dropout rate
        n_layer: number of transformer layers in CBraMod
        num_scales: number of temporal scales (including original)
        dim_feedforward: feedforward dimension in transformer
        nhead: number of attention heads
        pretrained_weights_path: path to CBraMod pretrained weights
        device: target device string
        use_pos_refiner: whether to use positional encoding refinement (default True)
        use_criss_cross_agg: whether to use criss-cross-aware aggregator (default True)

    Returns:
        ImprovedMSFTCBraModModel instance
    """
    # Create CBraMod backbone
    backbone = CBraMod(
        in_dim=patch_size,
        out_dim=patch_size,
        d_model=200,
        dim_feedforward=dim_feedforward,
        seq_len=seq_len,
        n_layer=n_layer,
        nhead=nhead,
    )

    # Load pretrained weights
    if pretrained_weights_path and os.path.exists(pretrained_weights_path):
        print(f"Loading CBraMod pretrained weights from {pretrained_weights_path}")
        map_loc = torch.device(device)
        state_dict = torch.load(pretrained_weights_path, map_location=map_loc)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        missing, unexpected = backbone.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
    else:
        print("No pretrained weights loaded for CBraMod backbone")

    # Create IMPROVED MSFT wrapper
    model = ImprovedMSFTCBraModModel(
        backbone=backbone,
        num_scales=num_scales,
        num_classes=num_classes,
        task_type=task_type,
        n_channels=n_channels,
        seq_len=seq_len,
        patch_size=patch_size,
        dropout=dropout,
        n_layer=n_layer,
        use_pos_refiner=use_pos_refiner,
        use_criss_cross_agg=use_criss_cross_agg,
    )

    return model


# ============================================================================
# Utility function for ablation studies
# ============================================================================
def create_msft_cbramod_variants(
    num_classes,
    task_type,
    n_channels,
    seq_len,
    **kwargs
):
    """Create different variants for ablation studies.

    Returns:
        dict of models with keys:
            - 'baseline': No pos refiner, no criss-cross agg
            - 'pos_refiner': With pos refiner only
            - 'criss_cross_agg': With criss-cross agg only
            - 'full': All improvements
    """
    variants = {}

    # Baseline
    variants['baseline'] = create_improved_msft_cbramod_model(
        num_classes, task_type, n_channels, seq_len,
        use_pos_refiner=False,
        use_criss_cross_agg=False,
        **kwargs
    )

    # Pos refiner only
    variants['pos_refiner'] = create_improved_msft_cbramod_model(
        num_classes, task_type, n_channels, seq_len,
        use_pos_refiner=True,
        use_criss_cross_agg=False,
        **kwargs
    )

    # Criss-cross agg only
    variants['criss_cross_agg'] = create_improved_msft_cbramod_model(
        num_classes, task_type, n_channels, seq_len,
        use_pos_refiner=False,
        use_criss_cross_agg=True,
        **kwargs
    )

    # Full improvements
    variants['full'] = create_improved_msft_cbramod_model(
        num_classes, task_type, n_channels, seq_len,
        use_pos_refiner=True,
        use_criss_cross_agg=True,
        **kwargs
    )

    return variants
