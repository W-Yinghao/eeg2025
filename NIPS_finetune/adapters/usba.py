"""
USBALayer and USBAAdapter — the core USBA adapter module.

USBALayer: single-layer adapter block
    LN → temporal branch + spatial branch → gated fusion → variational bottleneck → residual gate

USBAAdapter: multi-layer adapter stack wrapping a frozen backbone.
    Manages multiple USBALayers, collects KL/gate stats, and provides
    the adapted token representation for downstream heads.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

from .usba_config import USBAConfig
from .branches import (
    build_temporal_branch,
    build_spatial_branch,
    PlainMLP,
    GatedFusion,
)
from .bottleneck import VariationalBottleneck


class USBALayer(nn.Module):
    """
    Single USBA adapter block.

    Pipeline:
        H_tilde = LN(H_l)
        T_out   = TemporalBranch(H_tilde)
        S_out   = SpatialBranch(H_tilde)
        F_l     = GatedFusion(H_tilde, T_out, S_out)
        H_adapt, aux = VariationalBottleneck(F_l, H_l)

    Returns (H_adapt, aux_dict).
    """

    def __init__(self, d_model: int, config: USBAConfig, layer_idx: int = 0,
                 n_channels: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx

        # (1) LayerNorm
        self.ln = nn.LayerNorm(d_model)

        # (2) Branches
        if config.factorized and not config.non_factorized_baseline:
            self.temporal = build_temporal_branch(
                d_model,
                config.temporal_branch_type,
                kernel_size=config.temporal_kernel_size,
                rank=config.temporal_rank,
                dropout=config.dropout,
            )
            self.spatial = build_spatial_branch(
                d_model,
                config.spatial_branch_type,
                reduction=config.spatial_reduction,
                dropout=config.dropout,
                n_channels=n_channels,
            )
            # (3) Gated fusion
            self.fusion = GatedFusion(d_model)
            self._factorized = True
        else:
            # Non-factorized baseline: single MLP
            self.plain_mlp = PlainMLP(d_model, dropout=config.dropout)
            self._factorized = False

        # (4) Variational bottleneck + residual gate
        self.bottleneck = VariationalBottleneck(
            input_dim=d_model,
            latent_dim=config.latent_dim,
            gate_type=config.gate_type,
            gate_init=config.gate_init,
            dropout=config.dropout,
        )

    def forward(self, h: torch.Tensor,
                token_structure: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            h: (B, T, D) hidden states from backbone (or previous adapter layer)
            token_structure: optional (n_channels, seq_len) for 4D-aware mode.
                When provided, temporal branch operates per-channel along seq_len,
                spatial branch operates per-timestep along n_channels.
                This is critical for CBraMod/CodeBrain where T = C * S.
                For LUNA (pure temporal tokens), pass None.

        Returns:
            h_adapted: (B, T, D) adapted hidden states
            aux: dict with kl, gate stats, fusion gate stats, mu, logvar, z
        """
        # (1) Normalize
        h_tilde = self.ln(h)

        # (2-3) Branch + Fusion
        if self._factorized:
            t_out = self.temporal(h_tilde, token_structure=token_structure)
            s_out = self.spatial(h_tilde, token_structure=token_structure)
            fused, gate_stats = self.fusion(h_tilde, t_out, s_out)
        else:
            fused = self.plain_mlp(h_tilde, token_structure=token_structure)
            gate_stats = {}

        # (4) Variational bottleneck + residual write-back
        h_adapted, bn_aux = self.bottleneck(fused, h)

        # Merge aux info
        aux = {
            f'layer_{self.layer_idx}/kl': bn_aux['kl'],
            f'layer_{self.layer_idx}/gate_value': bn_aux['gate_value'],
        }
        aux.update({f'layer_{self.layer_idx}/{k}': v for k, v in gate_stats.items()})
        # Keep raw tensors for loss computation (not prefixed)
        aux['_mu'] = bn_aux['mu']
        aux['_logvar'] = bn_aux['logvar']
        aux['_kl'] = bn_aux['kl']
        aux['_z'] = bn_aux['z']
        aux['_gate_value'] = bn_aux['gate_value']

        return h_adapted, aux


class USBAAdapter(nn.Module):
    """
    Multi-layer USBA adapter that wraps a frozen backbone.

    Depending on config.selected_layers:
      - 'output': single USBALayer after backbone output (simplest, default)
      - 'all': one USBALayer per backbone transformer layer (inter-layer)
      - list of ints: USBALayer after specified backbone layers

    For 'output' mode, the backbone is called normally and USBA is applied
    to the final token representation. For inter-layer modes, requires
    the injector to unroll backbone layers.

    This class manages:
      - Collection of per-layer KL values
      - Aggregation of gate statistics
      - Mean pooling for downstream classification
    """

    def __init__(self, d_model: int, config: USBAConfig, num_layers: int = 1,
                 n_channels: Optional[int] = None):
        """
        Args:
            d_model: token dimension (e.g. 200 for CBraMod/CodeBrain)
            config: USBAConfig
            num_layers: number of USBA layers to create
            n_channels: EEG channel count for eager ChannelAttention init
        """
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        self.d_model = d_model

        self.layers = nn.ModuleList([
            USBALayer(d_model, config, layer_idx=i, n_channels=n_channels)
            for i in range(num_layers)
        ])

    def forward(
        self, h: torch.Tensor,
        token_structure: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Apply all USBA layers sequentially.

        Args:
            h: (B, T, D) token representation
            token_structure: optional (n_channels, seq_len) for 4D-aware branches.
                CBraMod/CodeBrain: pass (n_channels, seq_len) so temporal branch
                operates within-channel and spatial branch operates within-timestep.
                LUNA: pass None (tokens are already pure temporal).

        Returns:
            h_adapted: (B, T, D) after all adapter layers
            all_aux: merged dict of all layer aux info
        """
        all_aux = {}
        all_kls = []
        all_gate_vals = []

        for layer in self.layers:
            h, aux = layer(h, token_structure=token_structure)
            all_kls.append(aux['_kl'])
            all_gate_vals.append(aux['_gate_value'])
            # Store per-layer stats (prefixed keys)
            all_aux.update({k: v for k, v in aux.items() if not k.startswith('_')})

        # Aggregate stats
        all_aux['kl_total'] = sum(all_kls)  # tensor
        all_aux['kl_mean'] = sum(all_kls) / len(all_kls) if all_kls else torch.tensor(0.0)
        all_aux['gate_mean'] = sum(all_gate_vals) / len(all_gate_vals) if all_gate_vals else 0.0
        all_aux['_all_kls'] = all_kls
        all_aux['_all_gate_vals'] = all_gate_vals

        # Keep last layer's mu/logvar/z for loss (or concatenate if needed)
        # For output-mode (single layer) this is straightforward
        # For multi-layer, we sum KLs and use the last layer's z for cc-inv
        all_aux['_z_last'] = aux['_z']  # (B, T, latent)
        all_aux['_mu_last'] = aux['_mu']

        return h, all_aux

    def get_trainable_params(self) -> int:
        """Count trainable parameters in the adapter."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
