"""
Thin backbone wrapper that provides a unified forward_features() interface.

All backbones from backbone_factory.py output (B, C, S, D) or (B, T, D).
This wrapper standardizes the interface and attempts to expose temporal
and frequency sub-representations where possible.

CodeBrain (SSSM):
  SSSM interleaves S4 state-space layers (temporal) with spectral-gated
  convolutions (GConv, which uses FFT internally — frequency domain).
  These are fused within each Residual_block, so there are no cleanly
  separable H_t / H_f branches.  The wrapper returns:
    features: (B, C, S, D)  — full backbone output
    H_t: (B, S, D)          — mean-pooled over channels (temporal summary)
    H_f: (B, C, D)          — mean-pooled over time     (spatial/spectral proxy)
  This is a *degraded* factorization — see README.

CBraMod:
  Criss-cross transformer alternates temporal and spatial attention.
  Same factorization as CodeBrain (no explicit separate branches).

LUNA:
  Cross-attention with learned queries; outputs (B, T, Q*D).
  H_t = features, H_f = None (pure temporal tokens, no channel structure).
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class BackboneWrapper(nn.Module):
    """
    Wraps a frozen backbone to provide:
        forward_features(x, meta=None) -> dict

    The dict always contains 'features'.  'H_t' and 'H_f' are best-effort
    factorizations that may be None for backbones without dual branches.
    """

    def __init__(self, backbone: nn.Module, model_type: str,
                 n_channels: int, seq_len: int, token_dim: int):
        super().__init__()
        self.backbone = backbone
        self.model_type = model_type
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.token_dim = token_dim

        # Ensure backbone is frozen
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward_features(self, x: torch.Tensor,
                         meta: Optional[dict] = None) -> Dict[str, Optional[torch.Tensor]]:
        """
        Run backbone and return structured feature dict.

        Respects requires_grad on backbone parameters — when backbone is
        frozen (requires_grad=False), gradients are not needed; when unfrozen
        (partial/full fine-tune), gradients flow through normally.

        Args:
            x: (B, C, S, P) raw EEG patches
            meta: optional metadata dict (unused by current backbones)

        Returns:
            dict with keys:
              'features': (B, C, S, D) or (B, T, D) — full backbone output
              'hidden':   same as 'features' (no separate hidden for current backbones)
              'H_t':      (B, S, D) temporal summary (mean over channels) or None
              'H_f':      (B, C, D) spatial/spectral summary (mean over time) or None
              'token_structure': (n_channels, seq_len) or None
        """
        out = self.backbone(x)

        # CodeBrain squeeze() safety
        if self.model_type == 'codebrain' and out.dim() != 4:
            B = x.shape[0]
            out = out.reshape(B, self.n_channels, self.seq_len, self.token_dim)

        result = {
            'features': out,
            'hidden': out,
            'H_t': None,
            'H_f': None,
            'token_structure': None,
        }

        if out.dim() == 4:
            # (B, C, S, D) — CBraMod / CodeBrain
            result['H_t'] = out.mean(dim=1)  # (B, S, D) — temporal summary
            result['H_f'] = out.mean(dim=2)  # (B, C, D) — spatial/spectral proxy
            result['token_structure'] = (self.n_channels, self.seq_len)
        elif out.dim() == 3:
            # (B, T, D) — LUNA
            result['H_t'] = out  # pure temporal tokens
            # H_f remains None — no channel structure in LUNA output

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward (returns raw backbone output).

        Gradient flow is controlled by the requires_grad state on backbone
        parameters (set by DEBModel.get_param_groups based on finetune mode),
        NOT by a blanket no_grad context.
        """
        out = self.backbone(x)
        if self.model_type == 'codebrain' and out.dim() != 4:
            B = x.shape[0]
            out = out.reshape(B, self.n_channels, self.seq_len, self.token_dim)
        return out
