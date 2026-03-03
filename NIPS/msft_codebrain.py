#!/usr/bin/env python3
"""
MSFT (Multi-Scale Fine-Tuning) for CodeBrain

CodeBrain uses SSSM backbone with 4D input: (Batch, Channels, Seq_Len, Patch_Size)
We adapt MSFT by downsampling along the sequence dimension.

Author: Claude
Date: 2026-02-26
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# Add CodeBrain path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CodeBrain'))
from Models.SSSM import SSSM


# ==============================================================================
# Multi-Scale Downsampling for CodeBrain (4D data)
# ==============================================================================

def downsample_4d_seq(x: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Downsample 4D EEG data along sequence dimension.

    Args:
        x: Input of shape (B, C, S, P) where S is sequence length
        factor: Downsampling factor (e.g., 2 means reduce S by half)

    Returns:
        Downsampled tensor of shape (B, C, S//factor, P)
    """
    if factor == 1:
        return x

    B, C, S, P = x.shape

    # Reshape for pooling: (B*C, S, P)
    x_reshaped = x.view(B * C, S, P)

    # Average pooling along sequence dimension
    # Input: (B*C, S, P) -> (B*C, 1, S, P)
    x_pooled = F.avg_pool2d(
        x_reshaped.unsqueeze(1),
        kernel_size=(factor, 1),
        stride=(factor, 1)
    ).squeeze(1)  # (B*C, S//factor, P)

    # Reshape back: (B, C, S//factor, P)
    new_S = x_pooled.shape[1]
    x_out = x_pooled.view(B, C, new_S, P)

    return x_out


# ==============================================================================
# Scale Adapter for CodeBrain
# ==============================================================================

class CodeBrainScaleAdapter(nn.Module):
    """
    Adapter for each scale in CodeBrain.

    Since CodeBrain backbone outputs (B, C, S, P), we add a lightweight
    adapter to process each scale independently.
    """

    def __init__(self, channels: int = 200, dropout: float = 0.1):
        super().__init__()
        self.channels = channels

        # Lightweight 1D conv adapter
        self.adapter = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, S, P)

        Returns:
            x_adapted: (B, C, S, P)
        """
        B, C, S, P = x.shape

        # Flatten S and P for conv: (B, C, S*P)
        x_flat = x.view(B, C, S * P)

        # Apply adapter
        x_adapted = self.adapter(x_flat)

        # Reshape back: (B, C, S, P)
        x_out = x_adapted.view(B, C, S, P)

        # Residual connection
        return x + x_out


# ==============================================================================
# Cross-Scale Aggregator for CodeBrain
# ==============================================================================

class CodeBrainCrossScaleAgg(nn.Module):
    """
    Aggregate features across different scales for CodeBrain.

    Since different scales have different sequence lengths, we use
    adaptive pooling to unify dimensions before aggregation.
    """

    def __init__(self, channels: int = 200, dropout: float = 0.1):
        super().__init__()
        self.channels = channels

        # Learnable aggregation weights
        self.aggregation = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        scale_features: List[torch.Tensor],
        target_seq_len: int
    ) -> List[torch.Tensor]:
        """
        Args:
            scale_features: List of (B, C, S_k, P) for each scale k
            target_seq_len: Target sequence length (use finest scale)

        Returns:
            aggregated_features: List of (B, C, S_k, P) with cross-scale info
        """
        num_scales = len(scale_features)
        B, C, _, P = scale_features[0].shape

        # Unify all scales to same sequence length for aggregation
        unified_features = []
        for k, feat_k in enumerate(scale_features):
            _, _, S_k, _ = feat_k.shape

            if S_k != target_seq_len:
                # Upsample to target length
                # (B, C, S_k, P) -> (B*C, S_k, P)
                feat_reshaped = feat_k.view(B * C, S_k, P)

                # Interpolate: (B*C, target_seq_len, P)
                feat_upsampled = F.interpolate(
                    feat_reshaped.unsqueeze(1),
                    size=(target_seq_len, P),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)

                # Reshape back: (B, C, target_seq_len, P)
                feat_unified = feat_upsampled.view(B, C, target_seq_len, P)
            else:
                feat_unified = feat_k

            unified_features.append(feat_unified)

        # Stack and aggregate: (num_scales, B, C, target_seq_len, P)
        stacked = torch.stack(unified_features, dim=0)

        # Mean aggregation across scales: (B, C, target_seq_len, P)
        aggregated = stacked.mean(dim=0)

        # Apply learnable aggregation
        # Flatten: (B, C, target_seq_len*P)
        agg_flat = aggregated.view(B, C, target_seq_len * P)
        agg_processed = self.aggregation(agg_flat)
        agg_reshaped = agg_processed.view(B, C, target_seq_len, P)

        # Broadcast back to each scale
        output_features = []
        for k, feat_k in enumerate(scale_features):
            _, _, S_k, _ = feat_k.shape

            if S_k != target_seq_len:
                # Downsample aggregated feature to scale k
                agg_flat = agg_reshaped.view(B * C, target_seq_len, P)
                agg_downsampled = F.interpolate(
                    agg_flat.unsqueeze(1),
                    size=(S_k, P),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
                agg_scale_k = agg_downsampled.view(B, C, S_k, P)
            else:
                agg_scale_k = agg_reshaped

            # Add aggregated info to original feature
            output_features.append(feat_k + agg_scale_k)

        return output_features


# ==============================================================================
# MSFT CodeBrain Model
# ==============================================================================

class MSFTCodeBrainModel(nn.Module):
    """
    MSFT wrapper for CodeBrain.

    Architecture:
        1. Multi-scale downsampling along sequence dimension
        2. Shared backbone (frozen) + scale-specific adapters
        3. Cross-scale aggregation
        4. Per-scale classifiers + mixing
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        num_scales: int = 3,
        channels: int = 200,
        dropout: float = 0.1,
        freeze_backbone: bool = True
    ):
        super().__init__()

        self.backbone = backbone
        self.num_scales = num_scales
        self.num_classes = num_classes
        self.channels = channels

        # Freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Remove original proj_out if exists
            if hasattr(self.backbone, 'proj_out'):
                self.backbone.proj_out = nn.Identity()

        # Scale-specific adapters
        self.scale_adapters = nn.ModuleList([
            CodeBrainScaleAdapter(channels, dropout)
            for _ in range(num_scales)
        ])

        # Cross-scale aggregators (one per layer, simplified to 1)
        self.cross_scale_agg = CodeBrainCrossScaleAgg(channels, dropout)

        # Per-scale classifiers
        self.scale_classifiers = nn.ModuleList()
        for k in range(num_scales):
            # Note: Output dimension depends on sequence length at each scale
            # We'll use adaptive pooling before classification
            classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # (B, C, S, P) -> (B, C, 1, 1)
                nn.Flatten(),  # (B, C)
                nn.Linear(channels, channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(channels, num_classes)
            )
            self.scale_classifiers.append(classifier)

        # Scale mixing weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input EEG of shape (B, C, S, P)

        Returns:
            logits: Classification logits of shape (B, num_classes)
        """
        B, C, S, P = x.shape

        # Step 1: Generate multi-scale inputs
        multi_scale_inputs = []
        scale_seq_lens = []

        for k in range(self.num_scales):
            factor = 2 ** k
            x_k = downsample_4d_seq(x, factor)
            multi_scale_inputs.append(x_k)
            scale_seq_lens.append(x_k.shape[2])

        # Step 2: Pass through backbone + adapters
        scale_features = []
        for k, x_k in enumerate(multi_scale_inputs):
            # Backbone expects (x, original) tuple, but we simplify
            # For frozen backbone, we just pass x_k
            with torch.no_grad():
                feat_k = self.backbone(x_k)  # (B, C, S_k, P)

            # Apply adapter (trainable)
            feat_k = self.scale_adapters[k](feat_k)
            scale_features.append(feat_k)

        # Step 3: Cross-scale aggregation
        scale_features = self.cross_scale_agg(scale_features, scale_seq_lens[0])

        # Step 4: Per-scale classification
        scale_logits = []
        for k, feat_k in enumerate(scale_features):
            logits_k = self.scale_classifiers[k](feat_k)  # (B, num_classes)
            scale_logits.append(logits_k)

        # Step 5: Mix predictions
        scale_logits_tensor = torch.stack(scale_logits, dim=0)  # (num_scales, B, num_classes)

        # Softmax weights
        weights = F.softmax(self.scale_weights, dim=0)

        # Weighted sum: (B, num_classes)
        mixed_logits = torch.einsum('k,kbc->bc', weights, scale_logits_tensor)

        return mixed_logits

    def get_scale_weights(self) -> List[float]:
        """Get current scale mixing weights."""
        weights = F.softmax(self.scale_weights, dim=0)
        return weights.detach().cpu().tolist()


# ==============================================================================
# Factory Function
# ==============================================================================

def create_msft_codebrain_model(
    num_classes: int,
    n_layer: int = 30,
    codebook_size_t: int = 512,
    codebook_size_f: int = 512,
    num_scales: int = 3,
    dropout: float = 0.1,
    pretrained_weights_path: str = None,
    device: str = 'cuda:0',
    freeze_backbone: bool = True
) -> MSFTCodeBrainModel:
    """
    Create MSFT CodeBrain model.

    Args:
        num_classes: Number of output classes
        n_layer: Number of SSSM layers
        codebook_size_t: Temporal codebook size
        codebook_size_f: Frequency codebook size
        num_scales: Number of scales for MSFT
        dropout: Dropout probability
        pretrained_weights_path: Path to pretrained weights
        device: Device to load model
        freeze_backbone: Whether to freeze backbone

    Returns:
        model: MSFTCodeBrainModel
    """
    # Create backbone
    backbone = SSSM(
        in_channels=200,
        res_channels=200,
        skip_channels=200,
        out_channels=200,
        num_res_layers=n_layer,
        diffusion_step_embed_dim_in=200,
        diffusion_step_embed_dim_mid=200,
        diffusion_step_embed_dim_out=200,
        s4_lmax=570,
        s4_d_state=64,
        s4_dropout=dropout,
        s4_bidirectional=True,
        s4_layernorm=True,
        codebook_size_t=codebook_size_t,
        codebook_size_f=codebook_size_f,
        if_codebook=False
    )

    # Load pretrained weights if provided
    if pretrained_weights_path and os.path.exists(pretrained_weights_path):
        print(f"Loading pretrained weights from {pretrained_weights_path}")
        map_location = torch.device(device)
        state_dict = torch.load(pretrained_weights_path, map_location=map_location)

        # Handle potential 'module.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        backbone.load_state_dict(new_state_dict, strict=False)
        print("✓ Pretrained weights loaded")

    # Create MSFT model
    model = MSFTCodeBrainModel(
        backbone=backbone,
        num_classes=num_classes,
        num_scales=num_scales,
        channels=200,
        dropout=dropout,
        freeze_backbone=freeze_backbone
    )

    return model


# ==============================================================================
# Test
# ==============================================================================

if __name__ == '__main__':
    print("Testing MSFT CodeBrain Model...")

    # Create model
    model = create_msft_codebrain_model(
        num_classes=2,
        n_layer=4,  # Small for testing
        num_scales=3,
        freeze_backbone=True,
        device='cpu'
    )

    # Dummy input: (B, C, S, P)
    x = torch.randn(2, 16, 5, 200)

    # Forward
    logits = model(x)

    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {logits.shape}")
    print(f"✓ Scale weights: {model.get_scale_weights()}")

    # Test backward
    loss = logits.sum()
    loss.backward()
    print("✓ Backward pass successful")

    print("\nMSFT CodeBrain model ready!")
