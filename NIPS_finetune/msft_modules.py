#!/usr/bin/env python3
"""
MSFT (Multi-Scale FineTuning) modules for CodeBrain SSSM and CBraMod backbones.

Based on: "Multi-Scale Finetuning for Encoder-based Time Series Foundation Models"
(Qiao et al., 2025)

Adapted for EEG classification with:
  - CodeBrain's SSSM architecture
  - CBraMod's criss-cross transformer architecture

Freezes the pretrained backbone and adds trainable:
  - Scale-specific linear adapters after PatchEmbedding
  - Cross-scale aggregators (C2F + F2C) between layers
  - Per-scale classifiers with softmax-weighted mixing

Usage:
    # For CodeBrain SSSM
    from msft_modules import MSFTModel, create_msft_model
    model = create_msft_model(num_classes=6, ...)

    # For CBraMod
    from msft_modules import MSFTCBraModModel, create_msft_cbramod_model
    model = create_msft_cbramod_model(num_classes=6, ...)
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Add CodeBrain to path for SSSM import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CodeBrain'))
from Models.SSSM import SSSM

# Add CBraMod to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CBraMod'))
from models.cbramod import CBraMod


# ============================================================================
# Multi-Scale Generator
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
            list of K+1 tensors at different temporal scales
        """
        scales = [x]  # Scale 0 = original
        for k in range(1, self.num_scales):
            prev = scales[-1]
            B, C, S, P = prev.shape
            if S <= 1:
                # Cannot downsample further
                scales.append(prev)
                continue
            # Pool along seq_len (dim=2): reshape to use F.avg_pool1d
            # (B, C, S, P) -> (B*C*P, 1, S) for avg_pool1d on S
            prev_flat = prev.permute(0, 1, 3, 2).reshape(B * C * P, 1, S)
            pooled = F.avg_pool1d(prev_flat, kernel_size=2, stride=2, ceil_mode=True)
            new_S = pooled.shape[-1]
            pooled = pooled.reshape(B, C, P, new_S).permute(0, 1, 3, 2)
            scales.append(pooled.contiguous())
        return scales


# ============================================================================
# Scale Adapter
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
# Cross-Scale Aggregator
# ============================================================================
class CrossScaleAggregator(nn.Module):
    """Bidirectional cross-scale aggregation (Coarse-to-Fine + Fine-to-Coarse).

    For each pair of adjacent scales, applies a linear projection then
    upsamples (C2F via repeat/interpolate) or downsamples (F2C via avg_pool).
    Results from both branches are averaged.

    Operates on per-scale hidden representations in (B, d_model, L_k) format.
    """

    def __init__(self, d_model: int = 200, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        # C2F: project coarser scale i -> finer scale i-1
        self.c2f_maps = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_scales - 1)
        ])
        # F2C: project finer scale i -> coarser scale i+1
        self.f2c_maps = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_scales - 1)
        ])

    def forward(self, scale_hiddens, scale_lengths):
        """
        Args:
            scale_hiddens: list of K+1 tensors, each (B, d_model, L_k)
            scale_lengths: list of int [L_0, L_1, ..., L_K]
        Returns:
            list of K+1 tensors, aggregated
        """
        K = self.num_scales - 1

        # --- Coarse-to-Fine branch ---
        c2f_out = [h.clone() for h in scale_hiddens]
        for i in range(K, 0, -1):
            # Project coarse scale i tokens: (B, d, L_i) -> permute -> linear -> permute
            coarse = scale_hiddens[i]  # (B, d, L_i)
            projected = self.c2f_maps[i - 1](coarse.permute(0, 2, 1)).permute(0, 2, 1)
            # Upsample to match finer scale i-1
            upsampled = F.interpolate(projected, size=scale_lengths[i - 1], mode='nearest')
            c2f_out[i - 1] = c2f_out[i - 1] + upsampled

        # --- Fine-to-Coarse branch ---
        f2c_out = [h.clone() for h in scale_hiddens]
        for i in range(0, K):
            # Project fine scale i tokens
            fine = scale_hiddens[i]  # (B, d, L_i)
            projected = self.f2c_maps[i](fine.permute(0, 2, 1)).permute(0, 2, 1)
            # Downsample to match coarser scale i+1
            downsampled = F.adaptive_avg_pool1d(projected, scale_lengths[i + 1])
            f2c_out[i + 1] = f2c_out[i + 1] + downsampled

        # --- Average both branches ---
        merged = [(c + f) * 0.5 for c, f in zip(c2f_out, f2c_out)]
        return merged


# ============================================================================
# Cross-Scale Aggregator for 4D Tensors (CBraMod)
# ============================================================================
class CrossScaleAggregator4D(nn.Module):
    """Bidirectional cross-scale aggregation for 4D tensors (B, ch, seq, d).

    For CBraMod which keeps data in 4D format throughout.
    Aggregation happens along the seq (patch_num) dimension.
    """

    def __init__(self, d_model: int = 200, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        # C2F: project coarser scale i -> finer scale i-1
        self.c2f_maps = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_scales - 1)
        ])
        # F2C: project finer scale i -> coarser scale i+1
        self.f2c_maps = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_scales - 1)
        ])

    def forward(self, scale_hiddens, scale_seq_lens):
        """
        Args:
            scale_hiddens: list of K+1 tensors, each (B, ch, seq_k, d)
            scale_seq_lens: list of int [seq_0, seq_1, ..., seq_K]
        Returns:
            list of K+1 tensors, aggregated
        """
        K = self.num_scales - 1
        B, ch = scale_hiddens[0].shape[:2]
        d = scale_hiddens[0].shape[3]

        # --- Coarse-to-Fine branch ---
        c2f_out = [h.clone() for h in scale_hiddens]
        for i in range(K, 0, -1):
            # coarse: (B, ch, seq_i, d)
            coarse = scale_hiddens[i]
            # Project: apply linear on last dim
            projected = self.c2f_maps[i - 1](coarse)  # (B, ch, seq_i, d)
            # Upsample seq dimension: (B, ch, seq_i, d) -> (B, ch, seq_{i-1}, d)
            # Reshape to 3D for interpolate: (B*ch, d, seq_i)
            seq_i = projected.shape[2]
            projected_3d = projected.permute(0, 1, 3, 2).reshape(B * ch, d, seq_i)
            upsampled_3d = F.interpolate(projected_3d, size=scale_seq_lens[i - 1], mode='nearest')
            upsampled = upsampled_3d.reshape(B, ch, d, scale_seq_lens[i - 1]).permute(0, 1, 3, 2)
            c2f_out[i - 1] = c2f_out[i - 1] + upsampled

        # --- Fine-to-Coarse branch ---
        f2c_out = [h.clone() for h in scale_hiddens]
        for i in range(0, K):
            # fine: (B, ch, seq_i, d)
            fine = scale_hiddens[i]
            # Project
            projected = self.f2c_maps[i](fine)  # (B, ch, seq_i, d)
            # Downsample: use adaptive avg pool on seq dimension
            # Reshape to 3D: (B*ch, d, seq_i)
            seq_i = projected.shape[2]
            projected_3d = projected.permute(0, 1, 3, 2).reshape(B * ch, d, seq_i)
            downsampled_3d = F.adaptive_avg_pool1d(projected_3d, scale_seq_lens[i + 1])
            downsampled = downsampled_3d.reshape(B, ch, d, scale_seq_lens[i + 1]).permute(0, 1, 3, 2)
            f2c_out[i + 1] = f2c_out[i + 1] + downsampled

        # --- Average both branches ---
        merged = [(c + f) * 0.5 for c, f in zip(c2f_out, f2c_out)]
        return merged


# ============================================================================
# MSFT Model for SSSM
# ============================================================================
class MSFTModel(nn.Module):
    """MSFT wrapper around a frozen CodeBrain SSSM backbone.

    Trainable components:
      - scale_adapters: per-scale linear adapters after PatchEmbedding
      - cross_scale_aggs: per-layer cross-scale aggregators
      - scale_mix_logits: learnable weights for softmax mixing
      - per_scale_classifiers: per-scale classification heads

    Frozen:
      - Entire SSSM backbone (patch_embedding, init_conv, residual_layer, final_conv, norm)
    """

    def __init__(
        self,
        backbone: SSSM,
        num_scales: int = 3,
        num_classes: int = 6,
        task_type: str = 'multiclass',
        n_channels: int = 16,
        seq_len: int = 5,
        patch_size: int = 200,
        dropout: float = 0.1,
        n_layer: int = 8,
    ):
        super().__init__()

        self.backbone = backbone
        self.num_scales = num_scales
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.d_model = 200  # res_channels in SSSM
        self.n_layer = n_layer
        self.task_type = task_type

        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Multi-scale input generator
        self.scale_generator = MultiScaleGenerator(num_scales)

        # Scale-specific adapters (after frozen PatchEmbedding)
        self.scale_adapters = nn.ModuleList([
            ScaleAdapter(d_model=self.d_model) for _ in range(num_scales)
        ])

        # Cross-scale aggregators (one per residual block layer)
        num_res_layers = len(self.backbone.residual_layer.residual_blocks)
        self.cross_scale_aggs = nn.ModuleList([
            CrossScaleAggregator(d_model=self.d_model, num_scales=num_scales)
            for _ in range(num_res_layers)
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

    def _split_by_scale(self, h, scale_lengths):
        """Split concatenated (B, C, L_total) into per-scale list."""
        parts = []
        offset = 0
        for L_k in scale_lengths:
            parts.append(h[:, :, offset:offset + L_k])
            offset += L_k
        return parts

    def forward(self, x):
        """
        Args:
            x: (B, n_channels, seq_len, patch_size=200)
        Returns:
            logits: (B, num_classes) or (B, 1) for binary
        """
        B = x.shape[0]

        # ---- Step 1: Multi-scale generation ----
        multi_scale_inputs = self.scale_generator(x)
        # multi_scale_inputs[k]: (B, n_ch, seq_len_k, 200)

        # ---- Step 2: Frozen PatchEmbedding + Trainable ScaleAdapters ----
        scale_embeddings = []
        scale_seq_lens = []
        for k, x_k in enumerate(multi_scale_inputs):
            emb_k = self.backbone.patch_embedding(x_k)  # frozen (grad flows through)
            emb_k = self.scale_adapters[k](emb_k)       # trainable adapter
            scale_seq_lens.append(emb_k.shape[2])
            scale_embeddings.append(emb_k)

        # ---- Step 3: Rearrange to (B, 200, L_k) and concatenate ----
        scale_lengths = []
        scale_hiddens_list = []
        for emb_k in scale_embeddings:
            h_k = rearrange(emb_k, 'b c s p -> b p (c s)')  # (B, 200, L_k)
            scale_lengths.append(h_k.shape[2])
            scale_hiddens_list.append(h_k)

        h_concat = torch.cat(scale_hiddens_list, dim=2)  # (B, 200, L_total)

        # ---- Step 4: Frozen init_conv ----
        h_concat = self.backbone.init_conv(h_concat)  # (B, 200, L_total)

        # ---- Step 5: Layer-by-layer with cross-scale aggregation ----
        # Replicate Residual_group.forward() logic manually to inject aggregators
        noise = h_concat  # original input passed to every residual block
        h = h_concat
        skip_total = 0
        num_res_layers = len(self.backbone.residual_layer.residual_blocks)

        for n in range(num_res_layers):
            # Frozen residual block (no torch.no_grad — need grad flow for adapters)
            h_out, skip_n = self.backbone.residual_layer.residual_blocks[n]((h, noise))
            skip_total = skip_n + skip_total

            # Split h_out by scale
            h_scales = self._split_by_scale(h_out, scale_lengths)

            # Trainable cross-scale aggregation
            h_scales = self.cross_scale_aggs[n](h_scales, scale_lengths)

            # Re-concatenate for next layer
            h = torch.cat(h_scales, dim=2)

        # ---- Step 6: Frozen final_conv on accumulated skip ----
        skip_output = skip_total * math.sqrt(1.0 / num_res_layers)
        out = self.backbone.final_conv(skip_output)  # (B, 200, L_total)

        # ---- Step 7: Split by scale → per-scale classifiers → mixing ----
        out_scales = self._split_by_scale(out, scale_lengths)

        scale_logits = []
        for k, out_k in enumerate(out_scales):
            # Rearrange back: (B, 200, L_k) -> (B, n_ch, seq_len_k, 200)
            out_k = rearrange(
                out_k, 'b p (c s) -> b c s p',
                c=self.n_channels, s=scale_seq_lens[k],
            )
            # Frozen LayerNorm
            out_k = self.backbone.norm(out_k)
            # Flatten and classify
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
# Factory function
# ============================================================================
def create_msft_model(
    num_classes,
    task_type,
    n_channels,
    seq_len,
    patch_size=200,
    dropout=0.1,
    n_layer=8,
    num_scales=3,
    codebook_size_t=4096,
    codebook_size_f=4096,
    pretrained_weights_path=None,
    device='cuda:0',
):
    """Create an MSFTModel with a frozen SSSM backbone.

    Args:
        num_classes: number of output classes
        task_type: 'binary' or 'multiclass'
        n_channels: number of EEG channels
        seq_len: number of temporal patches
        patch_size: patch size (default 200)
        dropout: dropout rate
        n_layer: number of SSSM residual layers
        num_scales: number of temporal scales (including original)
        codebook_size_t: SSSM temporal codebook size
        codebook_size_f: SSSM frequency codebook size
        pretrained_weights_path: path to CodeBrain.pth
        device: target device string

    Returns:
        MSFTModel instance
    """
    # Compute s4_lmax for the original scale (must match pretrained weights)
    s4_lmax = n_channels * seq_len
    s4_lmax = ((s4_lmax + 18) // 19) * 19

    # Create SSSM backbone
    backbone = SSSM(
        in_channels=200,
        res_channels=200,
        skip_channels=200,
        out_channels=200,
        num_res_layers=n_layer,
        diffusion_step_embed_dim_in=200,
        diffusion_step_embed_dim_mid=200,
        diffusion_step_embed_dim_out=200,
        s4_lmax=s4_lmax,
        s4_d_state=64,
        s4_dropout=dropout,
        s4_bidirectional=True,
        s4_layernorm=True,
        codebook_size_t=codebook_size_t,
        codebook_size_f=codebook_size_f,
        if_codebook=False,
    )

    # Load pretrained weights
    if pretrained_weights_path and os.path.exists(pretrained_weights_path):
        print(f"Loading CodeBrain pretrained weights from {pretrained_weights_path}")
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
        print("No pretrained weights loaded for SSSM backbone")

    # Disable the codebook projection heads (not needed for classification)
    backbone.proj_out = nn.Sequential()

    # Create MSFT wrapper
    model = MSFTModel(
        backbone=backbone,
        num_scales=num_scales,
        num_classes=num_classes,
        task_type=task_type,
        n_channels=n_channels,
        seq_len=seq_len,
        patch_size=patch_size,
        dropout=dropout,
        n_layer=n_layer,
    )

    return model


# ============================================================================
# MSFT Model for CBraMod
# ============================================================================
class MSFTCBraModModel(nn.Module):
    """MSFT wrapper around a frozen CBraMod backbone.

    CBraMod architecture:
      - PatchEmbedding: Conv2d + spectral FFT + positional conv
      - TransformerEncoder: 12 layers of criss-cross attention
         - Each layer: spatial attention (across channels) + temporal attention (across patches)
      - proj_out: Linear(d_model, out_dim)

    Data stays in 4D format: (B, ch_num, patch_num, d_model) throughout.

    Trainable components:
      - scale_adapters: per-scale linear adapters after PatchEmbedding
      - cross_scale_aggs: per-layer cross-scale aggregators (4D version)
      - scale_mix_logits: learnable weights for softmax mixing
      - per_scale_classifiers: per-scale classification heads

    Frozen:
      - Entire CBraMod backbone (patch_embedding, encoder, proj_out)
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

        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Multi-scale input generator
        self.scale_generator = MultiScaleGenerator(num_scales)

        # Scale-specific adapters (after frozen PatchEmbedding)
        self.scale_adapters = nn.ModuleList([
            ScaleAdapter(d_model=self.d_model) for _ in range(num_scales)
        ])

        # Cross-scale aggregators (one per transformer layer)
        num_encoder_layers = len(self.backbone.encoder.layers)
        self.cross_scale_aggs = nn.ModuleList([
            CrossScaleAggregator4D(d_model=self.d_model, num_scales=num_scales)
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

    def _split_by_scale_4d(self, h, scale_seq_lens):
        """Split concatenated (B, ch, seq_total, d) into per-scale list."""
        parts = []
        offset = 0
        for L_k in scale_seq_lens:
            parts.append(h[:, :, offset:offset + L_k, :])
            offset += L_k
        return parts

    def forward(self, x):
        """
        Args:
            x: (B, n_channels, seq_len, patch_size=200)
        Returns:
            logits: (B, num_classes) or (B, 1) for binary
        """
        B = x.shape[0]

        # ---- Step 1: Multi-scale generation ----
        multi_scale_inputs = self.scale_generator(x)
        # multi_scale_inputs[k]: (B, n_ch, seq_len_k, 200)

        # ---- Step 2: Frozen PatchEmbedding + Trainable ScaleAdapters ----
        scale_embeddings = []
        scale_seq_lens = []
        for k, x_k in enumerate(multi_scale_inputs):
            emb_k = self.backbone.patch_embedding(x_k)  # frozen (B, ch, seq_k, 200)
            emb_k = self.scale_adapters[k](emb_k)       # trainable adapter
            scale_seq_lens.append(emb_k.shape[2])
            scale_embeddings.append(emb_k)

        # ---- Step 3: Concatenate along seq dimension ----
        h_concat = torch.cat(scale_embeddings, dim=2)  # (B, ch, seq_total, 200)

        # ---- Step 4: Layer-by-layer transformer with cross-scale aggregation ----
        # Replicate TransformerEncoder.forward() logic manually to inject aggregators
        h = h_concat
        for layer_idx, layer in enumerate(self.backbone.encoder.layers):
            # Frozen transformer layer
            h = layer(h)  # (B, ch, seq_total, 200)

            # Split by scale
            h_scales = self._split_by_scale_4d(h, scale_seq_lens)

            # Trainable cross-scale aggregation
            h_scales = self.cross_scale_aggs[layer_idx](h_scales, scale_seq_lens)

            # Re-concatenate for next layer
            h = torch.cat(h_scales, dim=2)

        # ---- Step 5: Frozen proj_out ----
        out = self.backbone.proj_out(h)  # (B, ch, seq_total, 200)

        # ---- Step 6: Split by scale → per-scale classifiers → mixing ----
        out_scales = self._split_by_scale_4d(out, scale_seq_lens)

        scale_logits = []
        for k, out_k in enumerate(out_scales):
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
# Factory function for CBraMod MSFT
# ============================================================================
def create_msft_cbramod_model(
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
):
    """Create an MSFTCBraModModel with a frozen CBraMod backbone.

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

    Returns:
        MSFTCBraModModel instance
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

    # Create MSFT wrapper
    model = MSFTCBraModModel(
        backbone=backbone,
        num_scales=num_scales,
        num_classes=num_classes,
        task_type=task_type,
        n_channels=n_channels,
        seq_len=seq_len,
        patch_size=patch_size,
        dropout=dropout,
        n_layer=n_layer,
    )

    return model
