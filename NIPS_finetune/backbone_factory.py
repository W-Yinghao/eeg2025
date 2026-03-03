"""
Shared Backbone Factory for EEG Fine-Tuning Frameworks

Provides a unified interface to create frozen CodeBrain (SSSM), CBraMod, FEMBA, or LUNA
backbones with optional inter-layer bottleneck adapters for CBraMod.

Usage:
    from backbone_factory import create_backbone

    # CodeBrain (HEAD-ONLY)
    backbone, out_dim, token_dim = create_backbone('codebrain', ...)

    # CBraMod HEAD-ONLY
    backbone, out_dim, token_dim = create_backbone('cbramod', ...)

    # CBraMod with inter-layer adapters (INTER-LAYER)
    backbone, out_dim, token_dim = create_backbone('cbramod', use_layer_adapters=True, ...)

    # FEMBA (encoder-only, frozen)
    backbone, out_dim, token_dim = create_backbone('femba', ...)

    # LUNA (encoder-only, frozen)
    backbone, out_dim, token_dim = create_backbone('luna', ...)
"""

import os
import sys
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

# Add model repos to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CodeBrain'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CBraMod'))

from Models.SSSM import SSSM
from models.cbramod import CBraMod

# Add BioFoundation to path (for FEMBA and LUNA models)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'BioFoundation'))
try:
    from models.FEMBA import FEMBA
    FEMBA_AVAILABLE = True
except ImportError:
    FEMBA_AVAILABLE = False

try:
    from models.LUNA import LUNA
    LUNA_AVAILABLE = True
except ImportError:
    LUNA_AVAILABLE = False


# =============================================================================
# 1. Bottleneck Adapter (for CBraMod inter-layer injection)
# =============================================================================

class BottleneckAdapter(nn.Module):
    """
    Lightweight residual bottleneck adapter inserted after each frozen transformer layer.

    Architecture: Linear(d→d/r) → GELU → Linear(d/r→d) + residual
    Output layer is zero-initialized so the adapter acts as identity at init,
    preserving pretrained features before any training.

    Args:
        d_model: Input/output dimension (200 for CBraMod)
        reduction: Bottleneck reduction factor (default 4 → 200→50→200)
    """

    def __init__(self, d_model: int = 200, reduction: int = 4):
        super().__init__()
        bottleneck_dim = d_model // reduction
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, d_model)
        # Zero-init output so adapter = identity at init
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        # x: (B, C, S, D) — 4D tensor from CBraMod transformer layers
        return x + self.up(self.act(self.down(x)))


# =============================================================================
# 2. CBraMod Wrapper with Inter-Layer Adapters
# =============================================================================

class CBraModWithAdapters(nn.Module):
    """
    Wraps a frozen CBraMod backbone and inserts trainable BottleneckAdapters
    after each transformer encoder layer.

    Forward pass manually unrolls the transformer encoder loop:
        patch_embedding(x) → [frozen layer_i → trainable adapter_i] × N → output

    When use_adapters=False, behaves identically to standard frozen CBraMod
    (adapters are not created, layers run sequentially with no injection).

    Args:
        backbone: Frozen CBraMod model
        use_adapters: Whether to insert bottleneck adapters
        adapter_reduction: Reduction factor for bottleneck dimension
    """

    def __init__(self, backbone: CBraMod, use_adapters: bool = True,
                 adapter_reduction: int = 4):
        super().__init__()
        self.backbone = backbone
        self.use_adapters = use_adapters

        if use_adapters:
            d_model = backbone.patch_embedding.d_model  # 200
            n_layers = len(backbone.encoder.layers)
            self.adapters = nn.ModuleList([
                BottleneckAdapter(d_model=d_model, reduction=adapter_reduction)
                for _ in range(n_layers)
            ])

    def forward(self, x, mask=None):
        # Patch embedding (frozen)
        h = self.backbone.patch_embedding(x, mask)  # (B, C, S, D)

        if self.use_adapters:
            # Manually unroll encoder layers with adapter injection
            for layer, adapter in zip(self.backbone.encoder.layers, self.adapters):
                h = layer(h)       # Frozen transformer layer
                h = adapter(h)     # Trainable adapter
        else:
            # Standard frozen forward (no adapters)
            for layer in self.backbone.encoder.layers:
                h = layer(h)

        # proj_out is replaced with Identity — just pass through
        h = self.backbone.proj_out(h)
        return h


# =============================================================================
# 3. Factory Function
# =============================================================================

def create_backbone(
    model_type: str = 'codebrain',
    n_channels: int = 16,
    seq_len: int = 5,
    patch_size: int = 200,
    # CodeBrain-specific
    n_layer: int = 8,
    codebook_size_t: int = 4096,
    codebook_size_f: int = 4096,
    # CBraMod-specific
    n_layer_cbramod: int = 12,
    nhead: int = 8,
    dim_feedforward: int = 800,
    # Adapter options (CBraMod only)
    use_layer_adapters: bool = False,
    adapter_reduction: int = 4,
    # FEMBA-specific
    femba_embed_dim: int = 79,
    femba_num_blocks: int = 2,
    femba_exp: int = 4,
    femba_patch_size: Tuple[int, int] = (2, 16),
    femba_stride: Tuple[int, int] = (2, 16),
    # LUNA-specific
    luna_patch_size: int = 40,
    luna_embed_dim: int = 64,
    luna_depth: int = 8,
    luna_num_heads: int = 2,
    luna_num_queries: int = 4,
    luna_drop_path: float = 0.1,
    # Common
    dropout: float = 0.1,
    pretrained_weights_path: Optional[str] = None,
    device: str = 'cuda:0',
) -> Tuple[nn.Module, int, int]:
    """
    Create a frozen backbone for downstream fine-tuning.

    Args:
        model_type: 'codebrain', 'cbramod', 'femba', or 'luna'
        n_channels: Number of EEG channels
        seq_len: Number of temporal patches (segment_duration * sampling_rate / patch_size)
        patch_size: Samples per patch (always 200 for CodeBrain/CBraMod)
        n_layer: Number of SSSM residual layers (CodeBrain)
        codebook_size_t: Temporal codebook size (CodeBrain)
        codebook_size_f: Frequency codebook size (CodeBrain)
        n_layer_cbramod: Number of transformer layers (CBraMod, default 12)
        nhead: Number of attention heads (CBraMod)
        dim_feedforward: FFN dimension (CBraMod)
        use_layer_adapters: Insert trainable adapters after each CBraMod layer
        adapter_reduction: Bottleneck reduction factor for adapters
        femba_embed_dim: FEMBA embedding dimension
        femba_num_blocks: FEMBA number of BiMamba blocks
        femba_exp: FEMBA Mamba expansion factor
        femba_patch_size: FEMBA patch size tuple
        femba_stride: FEMBA stride tuple
        luna_patch_size: LUNA temporal patch size
        luna_embed_dim: LUNA embedding dimension
        luna_depth: LUNA Transformer depth
        luna_num_heads: LUNA attention heads
        luna_num_queries: LUNA learned queries
        luna_drop_path: LUNA drop path rate
        dropout: Dropout rate
        pretrained_weights_path: Path to pretrained weights (.pth or .safetensors)
        device: Target device string

    Returns:
        backbone: nn.Module (frozen)
        backbone_out_dim: Flattened output dimension
        token_dim: Per-token dimension
    """
    token_dim = patch_size  # 200 for CodeBrain/CBraMod

    if model_type == 'codebrain':
        backbone = _create_codebrain(
            n_channels=n_channels, seq_len=seq_len, patch_size=patch_size,
            n_layer=n_layer, codebook_size_t=codebook_size_t,
            codebook_size_f=codebook_size_f, dropout=dropout,
            pretrained_weights_path=pretrained_weights_path, device=device,
        )
        backbone_out_dim = n_channels * seq_len * token_dim
    elif model_type == 'cbramod':
        backbone = _create_cbramod(
            n_channels=n_channels, seq_len=seq_len, patch_size=patch_size,
            n_layer=n_layer_cbramod, nhead=nhead,
            dim_feedforward=dim_feedforward,
            use_layer_adapters=use_layer_adapters,
            adapter_reduction=adapter_reduction,
            pretrained_weights_path=pretrained_weights_path, device=device,
        )
        backbone_out_dim = n_channels * seq_len * token_dim
    elif model_type == 'femba':
        backbone, backbone_out_dim, token_dim = _create_femba(
            n_channels=n_channels, seq_len=seq_len, patch_size=patch_size,
            embed_dim=femba_embed_dim, num_blocks=femba_num_blocks, exp=femba_exp,
            femba_patch_size=femba_patch_size, femba_stride=femba_stride,
            pretrained_weights_path=pretrained_weights_path, device=device,
        )
    elif model_type == 'luna':
        backbone, backbone_out_dim, token_dim = _create_luna(
            n_channels=n_channels, seq_len=seq_len, patch_size=patch_size,
            luna_patch_size=luna_patch_size, embed_dim=luna_embed_dim,
            depth=luna_depth, num_heads=luna_num_heads,
            num_queries=luna_num_queries, drop_path=luna_drop_path,
            pretrained_weights_path=pretrained_weights_path, device=device,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. "
                         f"Choose 'codebrain', 'cbramod', 'femba', or 'luna'.")

    # Print parameter summary
    _print_param_summary(backbone, model_type, use_layer_adapters)

    return backbone, backbone_out_dim, token_dim


# =============================================================================
# Internal: CodeBrain creation
# =============================================================================

def _create_codebrain(
    n_channels, seq_len, patch_size, n_layer,
    codebook_size_t, codebook_size_f, dropout,
    pretrained_weights_path, device,
) -> SSSM:
    s4_lmax = n_channels * seq_len
    s4_lmax = ((s4_lmax + 18) // 19) * 19  # Round up to nearest multiple of 19

    backbone = SSSM(
        in_channels=200, res_channels=200,
        skip_channels=200, out_channels=200,
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

    if pretrained_weights_path and os.path.exists(pretrained_weights_path):
        print(f"Loading CodeBrain pretrained weights from {pretrained_weights_path}")
        state_dict = torch.load(pretrained_weights_path, map_location=torch.device(device))
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k[7:] if k.startswith('module.') else k
            new_state_dict[new_k] = v
        missing, unexpected = backbone.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
    else:
        print("WARNING: No pretrained weights loaded for CodeBrain backbone")

    # Freeze all backbone parameters (same as CBraMod branch)
    for param in backbone.parameters():
        param.requires_grad = False

    return backbone


# =============================================================================
# Internal: CBraMod creation
# =============================================================================

def _create_cbramod(
    n_channels, seq_len, patch_size, n_layer, nhead, dim_feedforward,
    use_layer_adapters, adapter_reduction,
    pretrained_weights_path, device,
) -> nn.Module:
    backbone = CBraMod(
        in_dim=patch_size,       # 200
        out_dim=patch_size,      # 200
        d_model=200,
        dim_feedforward=dim_feedforward,
        seq_len=30,              # Max sequence length (matches pretrained config)
        n_layer=n_layer,
        nhead=nhead,
    )

    if pretrained_weights_path and os.path.exists(pretrained_weights_path):
        print(f"Loading CBraMod pretrained weights from {pretrained_weights_path}")
        state_dict = torch.load(pretrained_weights_path, map_location=torch.device(device))
        # Handle 'module.' prefix from DataParallel training
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k[7:] if k.startswith('module.') else k
            new_state_dict[new_k] = v
        missing, unexpected = backbone.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
    else:
        print("WARNING: No pretrained weights loaded for CBraMod backbone")

    # Replace projection head with Identity (downstream tasks use their own heads)
    backbone.proj_out = nn.Identity()

    # Freeze all backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False

    if use_layer_adapters:
        # Wrap with trainable inter-layer adapters
        backbone = CBraModWithAdapters(
            backbone=backbone,
            use_adapters=True,
            adapter_reduction=adapter_reduction,
        )
    else:
        # Wrap without adapters (clean interface, still uses manual layer unroll)
        backbone = CBraModWithAdapters(
            backbone=backbone,
            use_adapters=False,
        )

    return backbone


# =============================================================================
# 4. FEMBA Encoder Wrapper (for framework integration)
# =============================================================================

class FEMBAEncoderWrapper(nn.Module):
    """
    Wraps a frozen FEMBA backbone (encoder-only mode, num_classes=0) to produce
    token-level features compatible with downstream frameworks.

    FEMBA operates on raw (B, C, T) input. In framework integration, our data
    arrives as (B, C, S, P) which we reshape to (B, C, S*P) = (B, C, T).

    The encoder output is (B, num_patches_w, grid_h * embed_dim) which we reshape
    to (B, C, S, token_dim) format to match CodeBrain/CBraMod output conventions.
    """

    def __init__(self, backbone: nn.Module, n_channels: int, seq_len: int, patch_size: int):
        super().__init__()
        self.backbone = backbone
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.temporal_length = seq_len * patch_size

        # Get encoder output shape
        grid_size = backbone.patch_embed.grid_size
        self.grid_h = grid_size[0]
        self.grid_w = grid_size[1]
        self.embed_dim = backbone.embed_dim
        # Encoder output: (B, grid_w, grid_h * embed_dim)
        self.encoder_out_dim = self.grid_h * self.embed_dim
        self.total_out_dim = self.grid_w * self.encoder_out_dim

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, C, S, P) patched input from data pipeline
        Returns:
            features: (B, C, S, token_dim) — reshaped to match expected format
                      where C=n_channels, S=seq_len, token_dim varies
        """
        B = x.shape[0]
        # Reshape to (B, C, T) for FEMBA
        if x.dim() == 4:
            x_raw = x.reshape(B, self.n_channels, self.temporal_length)
        else:
            x_raw = x

        # Generate fake mask
        fake_mask = torch.zeros(B, self.n_channels, self.temporal_length,
                                dtype=torch.bool, device=x.device)

        # FEMBA in decoder/encoder mode (num_classes=0): returns (reconstructed, original)
        encoded, _ = self.backbone(x_raw, fake_mask)
        # encoded shape: (B, grid_w, grid_h * embed_dim)

        return encoded


class LUNAEncoderWrapper(nn.Module):
    """
    Wraps a frozen LUNA backbone (encoder-only, but we use classifier mode and
    extract the latent representation before the classification head).

    LUNA operates on raw (B, C, T) + channel_locations (B, C, 3).
    We extract features from the encoder output before the classifier.
    """

    def __init__(self, backbone: nn.Module, n_channels: int, seq_len: int,
                 patch_size: int, channel_locations: np.ndarray):
        super().__init__()
        self.backbone = backbone
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.temporal_length = seq_len * patch_size
        self.luna_patch_size = backbone.patch_size
        self.num_patches = self.temporal_length // self.luna_patch_size

        # Register channel locations
        self.register_buffer('channel_locations',
                             torch.tensor(channel_locations, dtype=torch.float32))

        # Encoder output dim: num_patches * (num_queries * embed_dim)
        self.embed_dim = backbone.embed_dim
        self.num_queries = backbone.num_queries
        self.encoder_out_dim = self.num_patches * self.num_queries * self.embed_dim
        self.total_out_dim = self.encoder_out_dim

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, C, S, P) patched input from data pipeline
        Returns:
            features: (B, num_patches, Q*D) encoder output
        """
        B = x.shape[0]
        if x.dim() == 4:
            x_raw = x.reshape(B, self.n_channels, self.temporal_length)
        else:
            x_raw = x

        # Generate fake mask
        fake_mask = torch.zeros(B, self.n_channels, self.temporal_length,
                                dtype=torch.bool, device=x.device)

        # Expand channel locations
        ch_locs = self.channel_locations.unsqueeze(0).expand(B, -1, -1)

        # Run through LUNA encoder (without classifier head)
        # We need to extract the latent representation before classifier
        x_prepared, _ = self.backbone.prepare_tokens(x_raw, ch_locs, mask=fake_mask)
        x_cross, _ = self.backbone.cross_attn(x_prepared)
        x_cross = torch.einsum('btqd->bt(qd)', x_cross.reshape(B, self.num_patches,
                               self.num_queries, self.embed_dim))

        for blk in self.backbone.blocks:
            x_cross = blk(x_cross)
        x_latent = self.backbone.norm(x_cross)  # (B, num_patches, Q*D)

        return x_latent


def _compute_bipolar_channel_locations():
    """Compute 3D channel locations for 16-ch bipolar montage using MNE standard_1005."""
    TUEG_BIPOLAR_CHANNELS = [
        "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
        "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
        "T3-C3", "C3-CZ", "CZ-C4", "C4-T4",
        "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    ]
    try:
        import mne
        electrodes = list(set([part for ch in TUEG_BIPOLAR_CHANNELS for part in ch.split('-')]))
        ch_types = ['eeg'] * len(electrodes)
        info = mne.create_info(ch_names=electrodes, sfreq=200, ch_types=ch_types)
        info = info.set_montage(mne.channels.make_standard_montage("standard_1005"), match_case=False)
        positions = info.get_montage().get_positions()['ch_pos']
        locs = []
        for ch_name in TUEG_BIPOLAR_CHANNELS:
            e1, e2 = ch_name.split('-')
            locs.append((positions[e1] + positions[e2]) / 2.0)
        return np.array(locs, dtype=np.float32)
    except Exception:
        # Fallback: uniform sphere
        locs = []
        for i in range(16):
            theta = 2 * np.pi * i / 16
            locs.append([np.cos(theta) * 0.085, np.sin(theta) * 0.085, 0.0])
        return np.array(locs, dtype=np.float32)


# =============================================================================
# Internal: FEMBA creation
# =============================================================================

def _create_femba(
    n_channels, seq_len, patch_size, embed_dim, num_blocks, exp,
    femba_patch_size, femba_stride, pretrained_weights_path, device,
) -> Tuple[nn.Module, int, int]:
    if not FEMBA_AVAILABLE:
        raise ImportError("FEMBA requires mamba_ssm. Install with: pip install mamba-ssm")

    temporal_length = seq_len * patch_size

    # Create FEMBA in encoder-only mode (num_classes=0 -> decoder/reconstruction mode)
    backbone = FEMBA(
        seq_length=temporal_length,
        num_channels=n_channels,
        num_classes=0,  # Encoder-only (no classifier)
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        exp=exp,
        patch_size=femba_patch_size,
        stride=femba_stride,
    )

    # Load pretrained weights
    if pretrained_weights_path and os.path.exists(pretrained_weights_path):
        print(f"Loading FEMBA pretrained weights from {pretrained_weights_path}")
        if pretrained_weights_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(pretrained_weights_path)
        else:
            state_dict = torch.load(pretrained_weights_path, map_location=torch.device(device))
        new_sd = {}
        for k, v in state_dict.items():
            new_k = k[6:] if k.startswith('model.') else k
            new_sd[new_k] = v
        missing, unexpected = backbone.load_state_dict(new_sd, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    else:
        print("WARNING: No pretrained weights loaded for FEMBA backbone")

    # Freeze all backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False

    # Wrap in encoder wrapper
    wrapper = FEMBAEncoderWrapper(backbone, n_channels, seq_len, patch_size)
    backbone_out_dim = wrapper.total_out_dim
    token_dim = wrapper.encoder_out_dim

    return wrapper, backbone_out_dim, token_dim


# =============================================================================
# Internal: LUNA creation
# =============================================================================

def _create_luna(
    n_channels, seq_len, patch_size, luna_patch_size, embed_dim,
    depth, num_heads, num_queries, drop_path,
    pretrained_weights_path, device,
) -> Tuple[nn.Module, int, int]:
    if not LUNA_AVAILABLE:
        raise ImportError("LUNA requires timm and rotary-embedding-torch.")

    temporal_length = seq_len * patch_size

    # Create LUNA in pretrain mode (num_classes=0) so we can extract encoder features
    # Actually, we need classification mode to load pretrained weights properly
    # We'll create with num_classes=0 to get encoder-only mode
    backbone = LUNA(
        patch_size=luna_patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        num_queries=num_queries,
        drop_path=drop_path,
        num_classes=0,  # Encoder-only
    )

    # Load pretrained weights
    if pretrained_weights_path and os.path.exists(pretrained_weights_path):
        print(f"Loading LUNA pretrained weights from {pretrained_weights_path}")
        if pretrained_weights_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(pretrained_weights_path)
        else:
            state_dict = torch.load(pretrained_weights_path, map_location=torch.device(device))
        new_sd = {}
        for k, v in state_dict.items():
            new_k = k[6:] if k.startswith('model.') else k
            new_sd[new_k] = v
        missing, unexpected = backbone.load_state_dict(new_sd, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    else:
        print("WARNING: No pretrained weights loaded for LUNA backbone")

    # Freeze all backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False

    # Compute channel locations
    channel_locations = _compute_bipolar_channel_locations()

    # Wrap in encoder wrapper
    wrapper = LUNAEncoderWrapper(backbone, n_channels, seq_len, patch_size, channel_locations)
    backbone_out_dim = wrapper.total_out_dim
    token_dim = num_queries * embed_dim

    return wrapper, backbone_out_dim, token_dim


# =============================================================================
# Internal: Parameter summary
# =============================================================================

def _print_param_summary(backbone: nn.Module, model_type: str, use_adapters: bool):
    total = sum(p.numel() for p in backbone.parameters())
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"\nBackbone: {model_type}")
    print(f"  Total params:     {total:,}")
    print(f"  Frozen params:    {frozen:,}")
    print(f"  Trainable params: {trainable:,}")
    if use_adapters and model_type == 'cbramod':
        adapter_params = sum(
            p.numel() for name, p in backbone.named_parameters()
            if 'adapter' in name and p.requires_grad
        )
        print(f"  Adapter params:   {adapter_params:,}")
    print()
