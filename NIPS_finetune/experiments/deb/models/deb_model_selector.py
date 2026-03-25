"""
DEBModel variant that adds 'selector' mode alongside baseline and deb.

Selector mode: gates + fusion + classifier, no VIB.
This is a standalone file — does NOT modify deb_model.py.
"""

import torch
import torch.nn as nn
from typing import Dict

from .backbone_wrapper import BackboneWrapper
from .baseline_head import BaselineHead
from .flatten_head import FlattenHead
from .evidence_bottleneck import EvidenceBottleneck
from .selector_head import SelectorHead


class DEBModelSelector(nn.Module):
    """
    Top-level model supporting baseline, deb, and selector modes.

    mode='baseline':  backbone → mean-pool → BaselineHead → logits
    mode='deb':       backbone → EvidenceBottleneck (gates + VIB) → logits + KL
    mode='selector':  backbone → SelectorHead (gates + fusion + classifier) → logits
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
        elif mode == 'deb':
            self.head = EvidenceBottleneck(
                token_dim=self.token_dim,
                num_classes=num_classes,
                latent_dim=cfg.get('deb_latent_dim', 64),
                gate_hidden=cfg.get('deb_gate_hidden', 64),
                enable_temporal_gate=cfg.get('deb_temporal_gate', True),
                enable_frequency_gate=cfg.get('deb_frequency_gate', True),
                fusion=cfg.get('deb_fusion', 'concat'),
                dropout=cfg.get('head_dropout', 0.1),
            )
        elif mode == 'selector':
            self.head = SelectorHead(
                token_dim=self.token_dim,
                num_classes=num_classes,
                gate_hidden=cfg.get('deb_gate_hidden', 64),
                enable_temporal_gate=cfg.get('deb_temporal_gate', True),
                enable_frequency_gate=cfg.get('deb_frequency_gate', True),
                fusion=cfg.get('deb_fusion', 'concat'),
                dropout=cfg.get('head_dropout', 0.1),
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self._print_summary()

    def forward(self, x: torch.Tensor,
                return_gates: bool = False) -> Dict[str, torch.Tensor]:
        features = self.backbone_wrapper(x)

        if self.mode == 'baseline':
            logits = self.head(features)
            return {'logits': logits}
        else:
            # Both 'deb' and 'selector' heads accept return_gates
            return self.head(features, return_gates=return_gates)

    def get_param_groups(self, cfg: dict):
        finetune = cfg.get('finetune', 'partial')
        head_params = list(self.head.parameters())

        if finetune == 'frozen':
            return [{'params': head_params, 'lr': cfg['lr_head']}]
        elif finetune == 'partial':
            bb_params = [p for p in self.backbone_wrapper.backbone.parameters()]
            for p in bb_params:
                p.requires_grad = True
            if cfg.get('freeze_patch_embed', True):
                self._freeze_patch_embed()
            return [
                {'params': bb_params, 'lr': cfg.get('lr_backbone', cfg['lr_head'] / cfg['lr_ratio'])},
                {'params': head_params, 'lr': cfg['lr_head']},
            ]
        else:  # full
            all_params = list(self.parameters())
            for p in all_params:
                p.requires_grad = True
            return [{'params': all_params, 'lr': cfg['lr_head']}]

    def _freeze_patch_embed(self):
        bb = self.backbone_wrapper.backbone
        if hasattr(bb, 'backbone') and hasattr(bb.backbone, 'patch_embedding'):
            for p in bb.backbone.patch_embedding.parameters():
                p.requires_grad = False
        elif hasattr(bb, 'patch_embedding'):
            for p in bb.patch_embedding.parameters():
                p.requires_grad = False
        if hasattr(bb, 'fc_t1'):
            for name in ['fc_t1', 'fc_t2', 'fc_t']:
                mod = getattr(bb, name, None)
                if mod is not None:
                    for p in mod.parameters():
                        p.requires_grad = False

    def _print_summary(self):
        total = sum(p.numel() for p in self.parameters())
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = total - frozen
        head_params = sum(p.numel() for p in self.head.parameters())

        print(f"\n{'='*60}")
        print(f"DEBModelSelector Summary ({self.mode})")
        print(f"{'='*60}")
        print(f"  Backbone:         {self.backbone_wrapper.model_type}")
        print(f"  Token dim:        {self.token_dim}")
        print(f"  Total params:     {total:,}")
        print(f"  Frozen params:    {frozen:,}")
        print(f"  Trainable params: {trainable:,}")
        print(f"  Head params:      {head_params:,}")
        if self.mode == 'selector':
            print(f"  Temporal gate:    {self.head.enable_temporal_gate}")
            print(f"  Frequency gate:   {self.head.enable_frequency_gate}")
            print(f"  Fusion:           {self.head.fusion}")
        elif self.mode == 'deb':
            print(f"  DEB latent dim:   {self.head.latent_dim}")
            print(f"  Temporal gate:    {self.head.enable_temporal_gate}")
            print(f"  Frequency gate:   {self.head.enable_frequency_gate}")
            print(f"  Fusion:           {self.head.fusion}")
        print(f"{'='*60}\n")
