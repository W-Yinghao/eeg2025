"""
True partial fine-tuning utilities.

Provides precise control over which backbone layers are trainable,
going beyond the existing binary frozen/partial/full regime.

Supported regimes:
    'frozen'    — head-only, backbone completely frozen
    'top1'      — last 1 block trainable
    'top2'      — last 2 blocks trainable
    'top4'      — last 4 blocks trainable
    'full'      — all backbone params trainable (for reference)

CodeBrain (SSSM): blocks = residual_layer.residual_blocks[0..n_layer-1]
CBraMod:          blocks = backbone.encoder.layers[0..n_layer-1]
                  (accessed through CBraModWithAdapters wrapper)
"""

import torch.nn as nn
from typing import Dict, List, Tuple


# ── Block enumeration per backbone ──────────────────────────────────────────

def _get_codebrain_blocks(backbone: nn.Module) -> List[nn.Module]:
    """Return list of residual blocks from CodeBrain SSSM."""
    if hasattr(backbone, 'residual_layer') and hasattr(backbone.residual_layer, 'residual_blocks'):
        return list(backbone.residual_layer.residual_blocks)
    raise ValueError("Cannot find residual_blocks in CodeBrain backbone")


def _get_cbramod_blocks(backbone: nn.Module) -> List[nn.Module]:
    """Return list of transformer layers from CBraMod (possibly wrapped)."""
    # CBraModWithAdapters wraps the actual backbone
    if hasattr(backbone, 'backbone') and hasattr(backbone.backbone, 'encoder'):
        return list(backbone.backbone.encoder.layers)
    if hasattr(backbone, 'encoder') and hasattr(backbone.encoder, 'layers'):
        return list(backbone.encoder.layers)
    raise ValueError("Cannot find encoder.layers in CBraMod backbone")


def _get_luna_blocks(backbone: nn.Module) -> List[nn.Module]:
    """Return transformer blocks from LUNA (via LUNAEncoderWrapper)."""
    if hasattr(backbone, 'backbone') and hasattr(backbone.backbone, 'blocks'):
        return list(backbone.backbone.blocks)
    raise ValueError("Cannot find blocks in LUNA backbone")


def get_backbone_blocks(backbone: nn.Module, model_type: str) -> List[nn.Module]:
    """Get ordered list of backbone blocks for the given model type."""
    if model_type == 'codebrain':
        return _get_codebrain_blocks(backbone)
    elif model_type == 'cbramod':
        return _get_cbramod_blocks(backbone)
    elif model_type == 'luna':
        return _get_luna_blocks(backbone)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ── Regime parsing ──────────────────────────────────────────────────────────

REGIME_MAP = {
    'frozen': 0,
    'top1': 1,
    'top2': 2,
    'top4': 4,
    'full': -1,  # sentinel: all blocks
}


def parse_regime(regime: str) -> int:
    """Parse regime string to number of blocks to unfreeze.

    Returns:
        Number of top blocks to unfreeze. 0 = frozen, -1 = all.
    """
    if regime in REGIME_MAP:
        return REGIME_MAP[regime]
    # Support 'topN' for arbitrary N
    if regime.startswith('top') and regime[3:].isdigit():
        return int(regime[3:])
    raise ValueError(
        f"Unknown regime '{regime}'. "
        f"Expected one of {list(REGIME_MAP.keys())} or 'topN'."
    )


# ── Apply regime ───────────────────────────────────────────────────────────

def apply_partial_ft_regime(
    backbone: nn.Module,
    model_type: str,
    regime: str,
    freeze_patch_embed: bool = True,
) -> Dict[str, object]:
    """
    Apply a true partial fine-tuning regime to the backbone.

    Steps:
        1. Freeze everything
        2. Unfreeze the last N blocks (based on regime)
        3. Optionally keep patch embedding frozen

    Args:
        backbone: The backbone module
        model_type: 'codebrain' | 'cbramod' | 'luna'
        regime: 'frozen' | 'top1' | 'top2' | 'top4' | 'full'
        freeze_patch_embed: If True, keep patch embedding frozen even in 'full'

    Returns:
        Dict with regime info:
            'regime': str
            'n_blocks_unfrozen': int
            'total_blocks': int
            'unfrozen_block_indices': list
            'unfrozen_layer_names': list
    """
    n_unfreeze = parse_regime(regime)

    # Step 1: Freeze everything
    for p in backbone.parameters():
        p.requires_grad = False

    if n_unfreeze == 0:
        # Fully frozen
        return {
            'regime': regime,
            'n_blocks_unfrozen': 0,
            'total_blocks': len(get_backbone_blocks(backbone, model_type)),
            'unfrozen_block_indices': [],
            'unfrozen_layer_names': [],
        }

    # Step 2: Get blocks and unfreeze top N
    blocks = get_backbone_blocks(backbone, model_type)
    total_blocks = len(blocks)

    if n_unfreeze == -1:
        n_unfreeze = total_blocks

    n_unfreeze = min(n_unfreeze, total_blocks)
    start_idx = total_blocks - n_unfreeze
    unfrozen_indices = list(range(start_idx, total_blocks))

    unfrozen_names = []
    for idx in unfrozen_indices:
        for name, param in blocks[idx].named_parameters():
            param.requires_grad = True
            unfrozen_names.append(f"block[{idx}].{name}")

    # Step 3: Handle patch embedding
    if freeze_patch_embed:
        _freeze_patch_embedding(backbone, model_type)

    return {
        'regime': regime,
        'n_blocks_unfrozen': n_unfreeze,
        'total_blocks': total_blocks,
        'unfrozen_block_indices': unfrozen_indices,
        'unfrozen_layer_names': unfrozen_names,
    }


def _freeze_patch_embedding(backbone: nn.Module, model_type: str):
    """Freeze patch embedding layers."""
    if model_type == 'codebrain':
        if hasattr(backbone, 'patch_embedding'):
            for p in backbone.patch_embedding.parameters():
                p.requires_grad = False
        # Also freeze init_conv, final_conv, lm_heads, norm
        for attr_name in ['init_conv', 'final_conv', 'lm_head_t', 'lm_head_f', 'norm',
                          'fc_t1', 'fc_t2', 'fc_t']:
            mod = getattr(backbone, attr_name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = False

    elif model_type == 'cbramod':
        # CBraModWithAdapters wraps backbone
        actual_bb = backbone.backbone if hasattr(backbone, 'backbone') else backbone
        if hasattr(actual_bb, 'patch_embedding'):
            for p in actual_bb.patch_embedding.parameters():
                p.requires_grad = False

    elif model_type == 'luna':
        actual_bb = backbone.backbone if hasattr(backbone, 'backbone') else backbone
        # LUNA patch embedding
        if hasattr(actual_bb, 'patch_embed'):
            for p in actual_bb.patch_embed.parameters():
                p.requires_grad = False
        if hasattr(actual_bb, 'pos_embed'):
            if actual_bb.pos_embed is not None and actual_bb.pos_embed.requires_grad:
                actual_bb.pos_embed.requires_grad = False


# ── Param summary ──────────────────────────────────────────────────────────

def compute_param_summary(model: nn.Module) -> Dict[str, object]:
    """Compute detailed parameter summary for a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    # Per-component breakdown
    breakdown = {}
    for name, module in model.named_children():
        mod_total = sum(p.numel() for p in module.parameters())
        mod_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
        breakdown[name] = {
            'total': mod_total,
            'trainable': mod_train,
            'frozen': mod_total - mod_train,
        }

    # Trainable layer names
    trainable_names = [
        name for name, p in model.named_parameters() if p.requires_grad
    ]

    return {
        'total_params': total,
        'trainable_params': trainable,
        'frozen_params': frozen,
        'trainable_ratio': trainable / max(total, 1),
        'breakdown': breakdown,
        'trainable_layer_names': trainable_names,
    }


def print_param_summary(summary: Dict, regime: str = ''):
    """Pretty-print parameter summary."""
    print(f"\n{'='*70}")
    print(f"  Parameter Summary {f'(regime={regime})' if regime else ''}")
    print(f"{'='*70}")
    print(f"  Total params:      {summary['total_params']:>12,}")
    print(f"  Trainable params:  {summary['trainable_params']:>12,}")
    print(f"  Frozen params:     {summary['frozen_params']:>12,}")
    print(f"  Trainable ratio:   {summary['trainable_ratio']:>12.6f} "
          f"({summary['trainable_ratio']*100:.4f}%)")

    print(f"\n  Component breakdown:")
    for comp, info in summary['breakdown'].items():
        print(f"    {comp:30s}: total={info['total']:>10,}  "
              f"trainable={info['trainable']:>10,}")

    # Show first/last few trainable layers
    names = summary['trainable_layer_names']
    if names:
        print(f"\n  Trainable layers ({len(names)} total):")
        for n in names[:5]:
            print(f"    + {n}")
        if len(names) > 10:
            print(f"    ... ({len(names) - 10} more) ...")
        for n in names[-5:]:
            if n not in names[:5]:
                print(f"    + {n}")
    print(f"{'='*70}\n")
