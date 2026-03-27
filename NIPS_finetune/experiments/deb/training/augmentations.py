"""
Light EEG augmentations for consistency regularization.

These augmentations are designed to be semantically-preserving:
they should not change the diagnostic content of the EEG signal.

Supported augmentations:
  - time_shift: shift temporal segments by a small offset
  - amplitude_jitter: add small Gaussian noise to amplitudes
  - time_mask: zero out a small contiguous time region

All operate on (B, C, S, P) patched EEG tensors.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class EEGAugmentor(nn.Module):
    """
    Applies a composition of light EEG augmentations.

    Each augmentation is applied with a given probability.
    The augmentations are designed to be light enough that
    diagnostic content is preserved.
    """

    def __init__(
        self,
        enable_time_shift: bool = True,
        enable_amplitude_jitter: bool = True,
        enable_time_mask: bool = True,
        time_shift_max: int = 1,
        jitter_std: float = 0.05,
        mask_ratio: float = 0.1,
        p_each: float = 0.5,
        # Per-augmentation probabilities (None → fall back to p_each)
        p_time_shift: float = None,
        p_jitter: float = None,
        p_mask: float = None,
    ):
        super().__init__()
        self.enable_time_shift = enable_time_shift
        self.enable_amplitude_jitter = enable_amplitude_jitter
        self.enable_time_mask = enable_time_mask
        self.time_shift_max = time_shift_max
        self.jitter_std = jitter_std
        self.mask_ratio = mask_ratio
        self.p_each = p_each
        self.p_time_shift = p_time_shift if p_time_shift is not None else p_each
        self.p_jitter = p_jitter if p_jitter is not None else p_each
        self.p_mask = p_mask if p_mask is not None else p_each

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to EEG tensor.

        Args:
            x: (B, C, S, P) patched EEG signal

        Returns:
            Augmented tensor of the same shape.
        """
        x_aug = x.clone()

        if self.enable_time_shift and torch.rand(1).item() < self.p_time_shift:
            x_aug = self._time_shift(x_aug)

        if self.enable_amplitude_jitter and torch.rand(1).item() < self.p_jitter:
            x_aug = self._amplitude_jitter(x_aug)

        if self.enable_time_mask and torch.rand(1).item() < self.p_mask:
            x_aug = self._time_mask(x_aug)

        return x_aug

    def _time_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Shift along the S (segment) dimension by a small amount."""
        B, C, S, P = x.shape
        if S <= 1:
            return x
        shift = torch.randint(-self.time_shift_max, self.time_shift_max + 1, (1,)).item()
        if shift == 0:
            return x
        return torch.roll(x, shifts=shift, dims=2)

    def _amplitude_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Add small Gaussian noise to amplitudes."""
        noise = torch.randn_like(x) * self.jitter_std
        return x + noise

    def _time_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Zero out a small contiguous region along the P (patch) dimension."""
        B, C, S, P = x.shape
        mask_len = max(1, int(P * self.mask_ratio))
        start = torch.randint(0, max(1, P - mask_len), (1,)).item()
        x[:, :, :, start:start + mask_len] = 0.0
        return x
