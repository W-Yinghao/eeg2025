"""
USBA Loss Functions.

L = L_task + beta * KL_total + lambda * L_cc_inv + eta * L_budget

Components:
  - L_task: class-weighted cross entropy (reuses project convention)
  - KL: per-layer KL aggregated from adapter aux dict
  - L_cc_inv: class-conditional HSIC penalty (lightweight)
  - L_budget: gate sparsity / activation budget regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


# ═══════════════════════════════════════════════════════════════════════
# Class-Conditional HSIC (lightweight approximation)
# ═══════════════════════════════════════════════════════════════════════

def _rbf_kernel(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """RBF kernel matrix. x: (N, D) → K: (N, N)"""
    xx = x @ x.t()
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    dists = rx.t() + rx - 2 * xx
    return torch.exp(-dists / (2 * sigma ** 2))


def _delta_kernel(ids: torch.Tensor) -> torch.Tensor:
    """
    Delta (categorical) kernel matrix.  ids: (N,) integer IDs.
    K(i, j) = 1 if ids[i] == ids[j] else 0.
    """
    return (ids.unsqueeze(0) == ids.unsqueeze(1)).float()


def _hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Biased HSIC estimator.  K, L: (N, N) kernel matrices.
    HSIC(X, Y) = trace(KHLH) / (N-1)^2
    where H = I - 1/N * 11^T (centering matrix).
    """
    N = K.shape[0]
    if N < 4:
        return torch.tensor(0.0, device=K.device)
    H = torch.eye(N, device=K.device) - 1.0 / N
    return (K @ H @ L @ H).trace() / ((N - 1) ** 2)


def class_conditional_hsic(
    z: torch.Tensor,
    labels: torch.Tensor,
    subject_ids: torch.Tensor,
    sigma_z: float = 1.0,
    sigma_s: float = 1.0,
) -> torch.Tensor:
    """
    Class-conditional HSIC: measures dependence between z and subject_id
    within each class, then averages.

    HSIC(Z, S | Y=y) for each class y, then mean.

    This penalizes subject-specific information that leaks into z
    within the same disease class, encouraging class-relevant-only features.

    Uses an RBF kernel on the z-side and a delta (categorical) kernel on the
    subject-ID side, since subject IDs are unordered categorical identifiers.

    Args:
        z: (B, D) aggregated latent representation
        labels: (B,) class labels
        subject_ids: (B,) subject IDs

    Returns:
        scalar HSIC penalty (lower = more invariant)
    """
    unique_labels = labels.unique()
    hsic_sum = torch.tensor(0.0, device=z.device)
    count = 0

    for y in unique_labels:
        mask = labels == y
        # Bug 10 fix: require >=4 samples AND >=2 unique subjects within each class
        if mask.sum() < 4:
            continue

        s_y = subject_ids[mask]
        if s_y.unique().numel() < 2:
            continue

        z_y = z[mask]

        K_z = _rbf_kernel(z_y, sigma=sigma_z)
        # Bug 5 fix: use delta/categorical kernel for subject IDs
        K_s = _delta_kernel(s_y)

        hsic_sum = hsic_sum + _hsic(K_z, K_s)
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=z.device)
    return hsic_sum / count


# ═══════════════════════════════════════════════════════════════════════
# Class-Conditional MMD (fallback)
# ═══════════════════════════════════════════════════════════════════════

def class_conditional_mmd(
    z: torch.Tensor,
    labels: torch.Tensor,
    subject_ids: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Class-conditional MMD: for each class, compute pairwise MMD between
    subject groups, encouraging features from different subjects (same class)
    to be similar.

    Lighter than HSIC when there are many subjects.
    """
    unique_labels = labels.unique()
    mmd_sum = torch.tensor(0.0, device=z.device)
    count = 0

    for y in unique_labels:
        mask_y = labels == y
        z_y = z[mask_y]
        s_y = subject_ids[mask_y]
        unique_subj = s_y.unique()

        if len(unique_subj) < 2:
            continue

        # Pairwise MMD between first two subject groups (for efficiency)
        s0, s1 = unique_subj[0], unique_subj[1]
        z0 = z_y[s_y == s0]
        z1 = z_y[s_y == s1]

        if len(z0) < 2 or len(z1) < 2:
            continue

        # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
        K00 = _rbf_kernel(z0, sigma).mean()
        K11 = _rbf_kernel(z1, sigma).mean()
        K01 = (_rbf_kernel(torch.cat([z0, z1]), sigma)[:len(z0), len(z0):]).mean()
        mmd_sq = K00 + K11 - 2 * K01
        mmd_sum = mmd_sum + mmd_sq.clamp(min=0)
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=z.device)
    return mmd_sum / count


# ═══════════════════════════════════════════════════════════════════════
# Budget Regularization
# ═══════════════════════════════════════════════════════════════════════

def compute_budget_loss(
    gate_values: List[float],
    adapter: nn.Module,
    mode: str = 'l1_gate',
) -> torch.Tensor:
    """
    Budget constraint to prevent adapter from becoming too expressive.

    Modes:
      - 'l1_gate': L1 penalty on gate activations (encourage sparsity)
      - 'activation_budget': penalize mean gate activation above a target

    Args:
        gate_values: list of gate activation values (floats) per layer
        adapter: the USBAAdapter module (for accessing gate parameters)

    Returns:
        scalar budget loss
    """
    device = next(adapter.parameters()).device

    if mode == 'l1_gate':
        # Collect all gate logits and penalize their sigmoid values
        total = torch.tensor(0.0, device=device)
        count = 0
        for layer in adapter.layers:
            g = torch.sigmoid(layer.bottleneck.gate_logit)
            total = total + g.mean()
            count += 1
        return total / max(count, 1)

    elif mode == 'activation_budget':
        # Penalize mean gate activation above target (0.3)
        target = 0.3
        mean_gate = sum(gate_values) / max(len(gate_values), 1)
        excess = max(mean_gate - target, 0.0)
        return torch.tensor(excess ** 2, device=device)

    else:
        return torch.tensor(0.0, device=device)


# ═══════════════════════════════════════════════════════════════════════
# Combined USBA Loss
# ═══════════════════════════════════════════════════════════════════════

class USBALoss(nn.Module):
    """
    Complete USBA loss:
        L = L_task + beta * KL + lambda_cc * L_cc_inv + eta * L_budget

    Handles:
      - Class-weighted cross entropy
      - Per-layer or global beta
      - Auto-disable cc-inv when no subject_id
      - Budget regularization
      - Detailed loss dict for logging
    """

    def __init__(
        self,
        beta: float = 1e-4,
        per_layer_beta: Optional[List[float]] = None,
        lambda_cc_inv: float = 0.01,
        eta_budget: float = 1e-3,
        enable_cc_inv: bool = True,
        enable_budget_reg: bool = True,
        task_type: str = 'multiclass',
        class_weights: Optional[torch.Tensor] = None,
        kl_reduction: str = 'mean',
        budget_warmup_epochs: int = 10,
    ):
        super().__init__()
        self.beta = beta
        self.per_layer_beta = per_layer_beta
        self.lambda_cc_inv = lambda_cc_inv
        self.eta_budget = eta_budget
        self.enable_cc_inv = enable_cc_inv
        self.enable_budget_reg = enable_budget_reg
        self.task_type = task_type
        self.kl_reduction = kl_reduction
        self.budget_warmup_epochs = budget_warmup_epochs

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

        self._cc_inv_active = False  # will be set at runtime

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        adapter_aux: Dict,
        adapter: nn.Module,
        subject_ids: Optional[torch.Tensor] = None,
        z_agg: Optional[torch.Tensor] = None,
        current_epoch: int = 0,
        beta_warmup_epochs: int = 5,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            logits: (B, num_classes) task logits
            labels: (B,) class labels
            adapter_aux: aux dict from USBAAdapter.forward()
            adapter: USBAAdapter module (for budget reg)
            subject_ids: (B,) optional subject IDs
            z_agg: (B, D) aggregated latent for cc-inv
            current_epoch: for beta warmup
            beta_warmup_epochs: warmup length

        Returns:
            total_loss: scalar
            loss_dict: detailed breakdown for logging
        """
        # ── L_task ─────────────────────────────────────────────────────
        loss_task = F.cross_entropy(logits, labels, weight=self.class_weights)

        # ── KL ─────────────────────────────────────────────────────────
        all_kls = adapter_aux.get('_all_kls', [])
        if self.per_layer_beta is not None and len(self.per_layer_beta) == len(all_kls):
            kl_weighted = sum(b * kl for b, kl in zip(self.per_layer_beta, all_kls))
        else:
            # Global beta with warmup
            warmup_factor = min(current_epoch / max(beta_warmup_epochs, 1), 1.0)
            effective_beta = self.beta * warmup_factor
            # Bug 8 fix: use kl_reduction to pick 'kl_mean' or 'kl_total'
            if self.kl_reduction == 'mean':
                kl_value = adapter_aux.get('kl_mean', adapter_aux.get('kl_total', torch.tensor(0.0, device=logits.device)))
            else:
                kl_value = adapter_aux.get('kl_total', torch.tensor(0.0, device=logits.device))
            kl_weighted = effective_beta * kl_value

        # ── L_cc_inv ───────────────────────────────────────────────────
        loss_cc = torch.tensor(0.0, device=logits.device)
        self._cc_inv_active = False
        if (self.enable_cc_inv and self.lambda_cc_inv > 0
                and subject_ids is not None and z_agg is not None):
            # Check that we have at least 2 unique subjects and 2 unique labels
            if subject_ids.unique().numel() >= 2 and labels.unique().numel() >= 2:
                loss_cc = class_conditional_hsic(z_agg, labels, subject_ids)
                self._cc_inv_active = True

        # ── L_budget ───────────────────────────────────────────────────
        loss_budget = torch.tensor(0.0, device=logits.device)
        if self.enable_budget_reg and self.eta_budget > 0:
            gate_vals = adapter_aux.get('_all_gate_vals', [])
            raw_budget = compute_budget_loss(gate_vals, adapter)
            # Bug 9 fix: budget warmup — zero before warmup, then linear ramp
            if current_epoch < self.budget_warmup_epochs:
                budget_factor = 0.0
            else:
                # Linear ramp over the warmup period after the start epoch
                ramp_progress = (current_epoch - self.budget_warmup_epochs) / max(self.budget_warmup_epochs, 1)
                budget_factor = min(ramp_progress, 1.0)
            loss_budget = raw_budget * budget_factor

        # ── Total ──────────────────────────────────────────────────────
        total = (
            loss_task
            + kl_weighted
            + self.lambda_cc_inv * loss_cc
            + self.eta_budget * loss_budget
        )

        loss_dict = {
            'total': total.item(),
            'task': loss_task.item(),
            'kl_total': adapter_aux.get('kl_total', torch.tensor(0.0)).item()
                if isinstance(adapter_aux.get('kl_total', 0), torch.Tensor)
                else adapter_aux.get('kl_total', 0.0),
            'kl_weighted': kl_weighted.item() if isinstance(kl_weighted, torch.Tensor) else kl_weighted,
            'cc_inv': loss_cc.item(),
            'cc_inv_active': self._cc_inv_active,
            'budget': loss_budget.item(),
            'gate_mean': adapter_aux.get('gate_mean', 0.0),
        }

        return total, loss_dict


# ═══════════════════════════════════════════════════════════════════════
# Metrics collection
# ═══════════════════════════════════════════════════════════════════════

def collect_usba_metrics(
    adapter_aux: Dict,
    adapter: nn.Module,
    loss_dict: Dict[str, float],
    total_params: int,
) -> Dict[str, float]:
    """
    Collect comprehensive USBA metrics for logging.

    Args:
        adapter_aux: aux dict from USBAAdapter.forward()
        adapter: USBAAdapter module
        loss_dict: loss breakdown dict
        total_params: total model parameters

    Returns:
        metrics dict for wandb/logging
    """
    trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)

    metrics = {
        'usba/trainable_params': trainable,
        'usba/trainable_ratio': trainable / max(total_params, 1) * 100,
        'usba/gate_mean': adapter_aux.get('gate_mean', 0.0),
        'usba/kl_total': loss_dict.get('kl_total', 0.0),
        'usba/cc_inv': loss_dict.get('cc_inv', 0.0),
        'usba/cc_inv_active': float(loss_dict.get('cc_inv_active', False)),
        'usba/budget': loss_dict.get('budget', 0.0),
    }

    # Per-layer stats
    all_kls = adapter_aux.get('_all_kls', [])
    all_gates = adapter_aux.get('_all_gate_vals', [])
    for i, kl in enumerate(all_kls):
        metrics[f'usba/layer_{i}/kl'] = kl.item() if isinstance(kl, torch.Tensor) else kl
    for i, g in enumerate(all_gates):
        metrics[f'usba/layer_{i}/gate'] = g

    return metrics
