"""
USBAInjector — injects USBA adapters into frozen backbones.

Supports multiple injection strategies:
  1. 'output' mode: single USBA adapter after the backbone's final output
  2. 'all' / selected layers: USBA adapters inserted between backbone layers

Compatible with all backbone types in backbone_factory.py:
  - CodeBrain (SSSM): output-mode only (monolithic forward, layers not easily unrollable)
  - CBraMod (via CBraModWithAdapters): inter-layer or output mode
  - FEMBA: output-mode only
  - LUNA: output-mode only

Engineering adaptation:
  The backbone factory already handles freezing. USBAInjector wraps the
  backbone + adapter into a single module that:
    - Calls backbone.forward() (frozen)
    - Reshapes output to (B, T, D) tokens
    - Applies USBA adapter(s)
    - Returns adapted tokens + aux stats
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

from .usba_config import USBAConfig
from .usba import USBAAdapter


class USBAInjectedModel(nn.Module):
    """
    Backbone + USBA adapter + classification head.

    This is the main model class for USBA training. It:
      1. Runs frozen backbone to get token representations
      2. Reshapes to (B, T, D)
      3. Applies USBA adapter
      4. Mean-pools to get (B, D) aggregate
      5. Classifies with task head

    For inter-layer injection on CBraMod, it manually unrolls the
    transformer layers and inserts USBA after selected layers.
    """

    def __init__(
        self,
        backbone: nn.Module,
        config: USBAConfig,
        token_dim: int,
        num_classes: int,
        n_channels: int = 16,
        seq_len: int = 5,
    ):
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.token_dim = token_dim
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.num_classes = num_classes

        # Detect backbone type for inter-layer injection
        self._backbone_type = self._detect_backbone_type()
        self._has_trainable_backbone = any(
            p.requires_grad for p in backbone.parameters()
        )

        # Token structure for 4D-aware branches:
        # CBraMod / CodeBrain output (B, C, S, D) → flatten to (B, C*S, D)
        # Branches need (n_channels, seq_len) to unflatten internally.
        # LUNA outputs (B, S, Q*D) with no separate channel axis → None.
        if self._backbone_type in ('cbramod', 'codebrain'):
            self._token_structure = (n_channels, seq_len)
        else:
            # LUNA / FEMBA / unknown: pure temporal tokens, no channel structure
            self._token_structure = None

        # Freeze backbone if requested
        if config.freeze_backbone and not self._has_trainable_backbone:
            for p in backbone.parameters():
                p.requires_grad = False

        # ── Create USBA adapter(s) ─────────────────────────────────────
        if config.selected_layers == 'output' or self._backbone_type in ('codebrain', 'femba', 'luna'):
            # Single adapter after backbone output
            self.usba = USBAAdapter(token_dim, config, num_layers=1)
            self._injection_mode = 'output'
        elif config.selected_layers == 'all':
            # One adapter per backbone layer (CBraMod only for now)
            n_backbone_layers = self._count_backbone_layers()
            self.usba = USBAAdapter(token_dim, config, num_layers=n_backbone_layers)
            self._injection_mode = 'inter_layer'
        elif isinstance(config.selected_layers, list):
            self.usba = USBAAdapter(token_dim, config, num_layers=len(config.selected_layers))
            self._injection_mode = 'selected_layers'
            self._selected_indices = set(config.selected_layers)
        else:
            self.usba = USBAAdapter(token_dim, config, num_layers=1)
            self._injection_mode = 'output'

        # ── Classification head ────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.BatchNorm1d(token_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(token_dim * 2, num_classes),
        )

        self._print_summary()

    def _detect_backbone_type(self) -> str:
        """Auto-detect backbone type from class name."""
        name = type(self.backbone).__name__.lower()
        if 'cbramod' in name:
            return 'cbramod'
        elif 'sssm' in name or 'codebrain' in name:
            return 'codebrain'
        elif 'femba' in name:
            return 'femba'
        elif 'luna' in name:
            return 'luna'
        return 'unknown'

    def _count_backbone_layers(self) -> int:
        """Count backbone transformer layers for inter-layer injection."""
        if self._backbone_type == 'cbramod':
            if hasattr(self.backbone, 'backbone'):
                return len(self.backbone.backbone.encoder.layers)
            elif hasattr(self.backbone, 'encoder'):
                return len(self.backbone.encoder.layers)
        return 1  # fallback

    def _backbone_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone, handling gradients for frozen vs trainable.

        For CodeBrain (SSSM), the backbone ends with x.squeeze() which can
        collapse dimensions when any spatial dim is 1.  We recover the
        canonical (B, C, S, D) shape using the known n_channels / seq_len.
        """
        if self._has_trainable_backbone:
            out = self.backbone(x)
        else:
            with torch.no_grad():
                out = self.backbone(x)

        # CodeBrain squeeze() safety: always reshape back to 4D
        if self._backbone_type == 'codebrain' and out.dim() != 4:
            B = x.shape[0]
            out = out.reshape(B, self.n_channels, self.seq_len, self.token_dim)

        return out

    def _backbone_inter_layer_forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Manually unroll CBraMod layers with USBA injection.

        # Engineering adaptation for CBraMod:
        # CBraModWithAdapters has backbone.backbone.patch_embedding and
        # backbone.backbone.encoder.layers — we hook into those.
        """
        inner = self.backbone.backbone if hasattr(self.backbone, 'backbone') else self.backbone

        # Patch embedding (frozen)
        with torch.no_grad():
            h = inner.patch_embedding(x, None)  # (B, C, S, D)

        B = h.shape[0]
        # CBraMod layers work on 4D (B, C, S, D), but USBA works on 3D (B, T, D)
        # We reshape before/after each USBA layer
        all_aux = {}
        all_kls = []
        all_gate_vals = []

        layers = inner.encoder.layers
        usba_layer_idx = 0

        for i, layer in enumerate(layers):
            with torch.no_grad():
                h = layer(h)  # frozen transformer layer, (B, C, S, D)

            # Check if we should inject USBA here
            inject = False
            if self._injection_mode == 'inter_layer':
                inject = True
            elif self._injection_mode == 'selected_layers' and i in self._selected_indices:
                inject = True

            if inject and usba_layer_idx < len(self.usba.layers):
                # Reshape to (B, T, D) for USBA, pass structure for 4D-aware branches
                orig_shape = h.shape
                h_tokens = h.reshape(B, -1, self.token_dim)
                h_tokens, aux = self.usba.layers[usba_layer_idx](
                    h_tokens, token_structure=self._token_structure
                )
                h = h_tokens.reshape(orig_shape)

                all_kls.append(aux['_kl'])
                all_gate_vals.append(aux['_gate_value'])
                all_aux.update({k: v for k, v in aux.items() if not k.startswith('_')})
                usba_layer_idx += 1

        # proj_out (identity in our setup)
        with torch.no_grad():
            h = inner.proj_out(h)

        # Build merged aux
        all_aux['kl_total'] = sum(all_kls) if all_kls else torch.tensor(0.0, device=h.device)
        all_aux['kl_mean'] = sum(all_kls) / len(all_kls) if all_kls else torch.tensor(0.0, device=h.device)
        all_aux['gate_mean'] = sum(all_gate_vals) / len(all_gate_vals) if all_gate_vals else 0.0
        all_aux['_all_kls'] = all_kls
        all_aux['_all_gate_vals'] = all_gate_vals

        return h, all_aux

    def forward(
        self, x: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            x: (B, C, S, P) raw EEG input
            return_features: if True, also return per-token features

        Returns:
            dict with:
                'logits': (B, num_classes)
                'z_agg': (B, D) aggregated representation
                'adapter_aux': dict of adapter statistics
                Optionally: 'z_tokens' (B, T, D)
        """
        B = x.shape[0]

        if self._injection_mode in ('inter_layer', 'selected_layers') and self._backbone_type == 'cbramod':
            # Inter-layer injection path
            backbone_out, adapter_aux = self._backbone_inter_layer_forward(x)
            # Reshape to tokens
            H = backbone_out.reshape(B, -1, self.token_dim)  # (B, T, D)
            # z for downstream is the last layer's representation
            adapter_aux['_z_last'] = H[:, :, :self.config.latent_dim] if H.shape[-1] > self.config.latent_dim else H
            adapter_aux['_mu_last'] = adapter_aux.get('_z_last', H)
        else:
            # Output-mode injection
            backbone_out = self._backbone_forward(x)
            # Reshape to tokens: handle different backbone output shapes
            if backbone_out.dim() == 4:
                H = backbone_out.reshape(B, -1, self.token_dim)  # (B, T, D)
            elif backbone_out.dim() == 3:
                H = backbone_out  # already (B, T, D)
            else:
                H = backbone_out.reshape(B, -1, self.token_dim)

            # Apply USBA adapter (with structure info for 4D-aware branches)
            H, adapter_aux = self.usba(H, token_structure=self._token_structure)

        # Mean pool
        z_agg = H.mean(dim=1)  # (B, D)

        # Classification
        logits = self.head(z_agg)  # (B, num_classes)

        out = {
            'logits': logits,
            'z_agg': z_agg,
            'adapter_aux': adapter_aux,
        }
        if return_features:
            out['z_tokens'] = H

        return out

    def _print_summary(self):
        total = sum(p.numel() for p in self.parameters())
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = total - frozen
        adapter_params = self.usba.get_trainable_params()
        head_params = sum(p.numel() for p in self.head.parameters())

        print(f"\n{'='*60}")
        print(f"USBAInjectedModel Summary")
        print(f"{'='*60}")
        print(f"  Backbone type:     {self._backbone_type}")
        print(f"  Injection mode:    {self._injection_mode}")
        print(f"  Token structure:   {self._token_structure or '3D (flat temporal)'}")
        print(f"  Token dim:         {self.token_dim}")
        print(f"  USBA layers:       {len(self.usba.layers)}")
        print(f"  Latent dim:        {self.config.latent_dim}")
        print(f"  Gate type:         {self.config.gate_type}")
        print(f"  Factorized:        {self.config.factorized}")
        print(f"  Total params:      {total:,}")
        print(f"  Frozen params:     {frozen:,}")
        print(f"  Trainable params:  {trainable:,}")
        print(f"    Adapter params:  {adapter_params:,}")
        print(f"    Head params:     {head_params:,}")
        print(f"  Trainable ratio:   {trainable / max(total, 1) * 100:.2f}%")
        print(f"  CC-inv enabled:    {self.config.enable_cc_inv}")
        print(f"  Budget reg:        {self.config.enable_budget_reg}")
        print(f"{'='*60}\n")


class USBAInjector:
    """
    Static utility class for injecting USBA into backbones.

    Usage:
        from adapters.injection import USBAInjector
        model = USBAInjector.inject(backbone, config, token_dim, num_classes, ...)

    This is the main entry point for creating USBA-augmented models.
    Handles backbone detection, layer counting, and adapter creation.

    # ── Future BILO extension point ────────────────────────────────────
    # inject() could accept a bilo_config to create rank-gated LoRA layers
    # instead of USBA layers, reusing the same injection infrastructure.
    #
    # ── Future SPCBA extension point ───────────────────────────────────
    # inject() could accept spcba_config to create dual-branch adapters
    # with shared/private latent splits, using the same backbone wrapping.
    """

    @staticmethod
    def inject(
        backbone: nn.Module,
        config: USBAConfig,
        token_dim: int,
        num_classes: int,
        n_channels: int = 16,
        seq_len: int = 5,
    ) -> USBAInjectedModel:
        """
        Create a USBAInjectedModel wrapping the given backbone.

        Args:
            backbone: frozen backbone from backbone_factory.create_backbone()
            config: USBAConfig
            token_dim: per-token dimension (200 for CBraMod/CodeBrain)
            num_classes: number of task classes
            n_channels: EEG channels
            seq_len: temporal patches

        Returns:
            USBAInjectedModel ready for training
        """
        return USBAInjectedModel(
            backbone=backbone,
            config=config,
            token_dim=token_dim,
            num_classes=num_classes,
            n_channels=n_channels,
            seq_len=seq_len,
        )

    @staticmethod
    def get_trainable_params(model: USBAInjectedModel) -> Dict[str, int]:
        """Get parameter breakdown."""
        return {
            'total': sum(p.numel() for p in model.parameters()),
            'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'adapter': model.usba.get_trainable_params(),
            'head': sum(p.numel() for p in model.head.parameters()),
        }
