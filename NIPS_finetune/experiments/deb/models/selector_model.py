"""
SelectorModel — unified model for experiments 6/7/8.

Supports:
  - baseline mode: pool + MLP head (no gates)
  - selector mode: EnhancedSelectorHead (gates + fusion + classifier)

Fine-tuning regimes via true partial FT:
  - frozen: head-only
  - top1/top2/top4: last N backbone blocks trainable
  - full: all backbone trainable

Does NOT include VIB/DEB by default (extension point preserved).
"""

import torch
import torch.nn as nn
from typing import Dict, List

from .backbone_wrapper import BackboneWrapper
from .baseline_head import BaselineHead
from .flatten_head import FlattenHead
from .selector_enhanced import EnhancedSelectorHead
from ..training.partial_ft import (
    apply_partial_ft_regime,
    compute_param_summary,
    print_param_summary,
)


class SelectorModel(nn.Module):
    """
    Top-level model for selector experiments (Exp 6/7/8).

    mode='baseline': backbone -> mean-pool -> BaselineHead -> logits
    mode='selector': backbone -> EnhancedSelectorHead -> logits + gates
    """

    def __init__(
        self,
        backbone_wrapper: BackboneWrapper,
        mode: str,
        num_classes: int,
        cfg: dict,
    ):
        super().__init__()
        self.backbone_wrapper = backbone_wrapper
        self.mode = mode
        self.token_dim = backbone_wrapper.token_dim
        self.regime = cfg.get('regime', 'frozen')

        head_type = cfg.get('head_type', 'pool')

        if mode == 'baseline' and head_type == 'flatten':
            self.head = FlattenHead(
                n_channels=backbone_wrapper.n_channels,
                seq_len=backbone_wrapper.seq_len,
                token_dim=self.token_dim,
                num_classes=num_classes,
                dropout=cfg.get('head_dropout', 0.3),
            )
        elif mode == 'baseline':
            self.head = BaselineHead(
                token_dim=self.token_dim,
                num_classes=num_classes,
                hidden_dim=cfg.get('head_hidden', 512),
                dropout=cfg.get('head_dropout', 0.1),
            )
        elif mode == 'selector':
            self.head = EnhancedSelectorHead(
                token_dim=self.token_dim,
                num_classes=num_classes,
                gate_hidden=cfg.get('gate_hidden', 64),
                enable_temporal_gate=cfg.get('enable_temporal_gate', True),
                enable_frequency_gate=cfg.get('enable_frequency_gate', True),
                fusion=cfg.get('fusion', 'concat'),
                dropout=cfg.get('head_dropout', 0.1),
                enable_vib=cfg.get('enable_vib', False),
                vib_latent_dim=cfg.get('vib_latent_dim', 64),
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'baseline' or 'selector'.")

        # Apply true partial FT regime
        self.regime_info = apply_partial_ft_regime(
            backbone=self.backbone_wrapper.backbone,
            model_type=self.backbone_wrapper.model_type,
            regime=self.regime,
            freeze_patch_embed=cfg.get('freeze_patch_embed', True),
        )

        # Print summary
        self._summary = compute_param_summary(self)
        print_param_summary(self._summary, self.regime)

    def forward(self, x: torch.Tensor,
                return_gates: bool = False) -> Dict[str, torch.Tensor]:
        features = self.backbone_wrapper(x)

        if self.mode == 'baseline':
            logits = self.head(features)
            return {'logits': logits}
        else:
            return self.head(features, return_gates=return_gates)

    def get_param_groups(self, cfg: dict) -> List[Dict]:
        """
        Return optimizer parameter groups with differential LR.

        Head always gets lr_head.
        Unfrozen backbone layers get lr_backbone (= lr_head / lr_ratio).
        """
        head_params = [p for p in self.head.parameters() if p.requires_grad]
        bb_params = [p for p in self.backbone_wrapper.parameters() if p.requires_grad]

        groups = []
        if bb_params:
            lr_bb = cfg.get('lr_backbone')
            if lr_bb is None:
                lr_bb = cfg['lr_head'] / cfg.get('lr_ratio', 10.0)
            groups.append({'params': bb_params, 'lr': lr_bb})
        if head_params:
            groups.append({'params': head_params, 'lr': cfg['lr_head']})

        return groups

    def get_summary(self) -> Dict:
        """Return the parameter summary dict."""
        return self._summary

    def get_regime_info(self) -> Dict:
        """Return regime info dict."""
        return self.regime_info
