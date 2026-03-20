"""
DEBModel — unified model class supporting baseline and DEB modes.

Wraps BackboneWrapper + head (BaselineHead or EvidenceBottleneck).
"""

import torch
import torch.nn as nn
from typing import Dict

from .backbone_wrapper import BackboneWrapper
from .baseline_head import BaselineHead
from .flatten_head import FlattenHead
from .evidence_bottleneck import EvidenceBottleneck


class DEBModel(nn.Module):
    """
    Top-level model for DEB experiments.

    mode='baseline':
        backbone → mean-pool → BaselineHead → logits

    mode='deb':
        backbone → EvidenceBottleneck (gates + VIB) → logits + KL + gates
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
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self._print_summary()

    def forward(self, x: torch.Tensor,
                return_gates: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, S, P) EEG patches
            return_gates: DEB mode only — include gate activations

        Returns:
            dict with at least 'logits' (B, num_classes).
            DEB mode also has 'mu', 'logvar', 'z_e', 'kl',
            and optionally 'temporal_gate', 'frequency_gate'.
        """
        features = self.backbone_wrapper(x)  # (B, C, S, D) or (B, T, D)

        if self.mode == 'baseline':
            logits = self.head(features)
            return {'logits': logits}
        else:
            return self.head(features, return_gates=return_gates)

    def get_param_groups(self, cfg: dict):
        """
        Return parameter groups for differential learning rates.

        backbone: lr_backbone (or 0 if frozen)
        head: lr_head
        """
        finetune = cfg.get('finetune', 'partial')

        head_params = list(self.head.parameters())

        if finetune == 'frozen':
            # Only train head
            return [{'params': head_params, 'lr': cfg['lr_head']}]
        elif finetune == 'partial':
            # Backbone at lower LR, head at higher LR
            bb_params = [p for p in self.backbone_wrapper.backbone.parameters()]
            # Unfreeze backbone
            for p in bb_params:
                p.requires_grad = True
            # Optionally re-freeze patch embedding
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
        """Freeze patch embedding / tokenizer layers if they exist."""
        bb = self.backbone_wrapper.backbone
        # CBraMod
        if hasattr(bb, 'backbone') and hasattr(bb.backbone, 'patch_embedding'):
            for p in bb.backbone.patch_embedding.parameters():
                p.requires_grad = False
        elif hasattr(bb, 'patch_embedding'):
            for p in bb.patch_embedding.parameters():
                p.requires_grad = False
        # CodeBrain — first conv layers
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
        print(f"DEBModel Summary ({self.mode})")
        print(f"{'='*60}")
        print(f"  Backbone:         {self.backbone_wrapper.model_type}")
        print(f"  Token dim:        {self.token_dim}")
        print(f"  Total params:     {total:,}")
        print(f"  Frozen params:    {frozen:,}")
        print(f"  Trainable params: {trainable:,}")
        print(f"  Head params:      {head_params:,}")
        if self.mode == 'deb':
            print(f"  DEB latent dim:   {self.head.latent_dim}")
            print(f"  Temporal gate:    {self.head.enable_temporal_gate}")
            print(f"  Frequency gate:   {self.head.enable_frequency_gate}")
            print(f"  Fusion:           {self.head.fusion}")
        print(f"{'='*60}\n")
