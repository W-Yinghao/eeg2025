#!/usr/bin/env python3
"""
Hybrid Hyperbolic Fine-Tuning Framework for EEG Foundation Models

Architecture:
    Raw EEG (B, C, S, P=200)
        -> Frozen CodeBrain SSSM Backbone -> (B, C, S, 200)
        -> Flatten -> (B, C*S*200)
        -> EuclideanProjection (Linear) -> (B, hyp_dim)                    [Euclidean R^d]
        -> L2-Normalize + Pad time dim + projx -> (B, hyp_dim+1)           [Lorentz L^n_K]
        -> HyperbolicDSMDBN (domain-specific moment alignment) -> (B, hyp_dim+1)
        -> HMLR (Hyperbolic Multinomial Logistic Regression) -> (B, num_classes)

    Loss = L_CE + lambda * L_HHSW

Key References:
    [1] HEEGNet: Hyperbolic Embeddings for EEG (Li et al., 2026)
    [2] CodeBrain: EEG Foundation Model (ICLR 2026)

Mathematical Foundations:
    - Lorentz model: H^n_K = {x in R^{n+1} | <x,x>_L = 1/K, x_0 > 0}, K < 0
    - Minkowski inner product: <u,v>_L = -u_0*v_0 + sum_{i>0} u_i*v_i
    - Origin: L_0 = [1/sqrt(-K), 0, ..., 0]^T
    - Exponential map: exp_p(v) = cosh(alpha)*p + sinhdiv(alpha)*v, alpha = sqrt(-K)*||v||_L
    - HMLR: logit_c = sign(alpha_c) * beta_c * d_c, d_c = sqrt(K) * |arcsinh(sqrt(-K)*alpha_c/beta_c)|
    - HHSW: Busemann function projects Lorentz points to R, then 1D EMD
"""

import math
import os
import sys
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Backbone factory (shared across all frameworks)
from backbone_factory import create_backbone


# =============================================================================
# 0. Lorentz Manifold Operations (self-contained, no external geometry deps)
# =============================================================================

class LorentzManifold:
    """
    Lorentz (hyperboloid) model operations for K < 0.

    The manifold is defined as:
        L^n_K = {x in R^{n+1} | <x,x>_L = 1/K, x_0 > 0}

    where <u,v>_L = -u_0*v_0 + sum_{i>0} u_i*v_i (Minkowski inner product).

    All operations use K = -1 by default (unit hyperboloid).
    """

    def __init__(self, K: float = -1.0):
        self.K = K
        self.eps = 1e-8

    def ldot(self, u: torch.Tensor, v: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """Minkowski inner product: <u,v>_L = -u_0*v_0 + sum_{i>0} u_i*v_i"""
        m = u * v
        if keepdim:
            return torch.sum(m, dim=-1, keepdim=True) - 2.0 * m[..., 0:1]
        else:
            return torch.sum(m, dim=-1, keepdim=False) - 2.0 * m[..., 0]

    def calc_time(self, space: torch.Tensor) -> torch.Tensor:
        """Compute time component: x_0 = sqrt(||x_s||^2 - 1/K)"""
        return torch.sqrt(
            torch.sum(space * space, dim=-1, keepdim=True) - 1.0 / self.K
        ).clamp(min=self.eps)

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        """Project onto the hyperboloid by recomputing the time component."""
        space = x[..., 1:]
        t = self.calc_time(space)
        return torch.cat([t, space], dim=-1)

    def add_time(self, space: torch.Tensor) -> torch.Tensor:
        """Prepend computed time component to spatial coordinates."""
        t = self.calc_time(space)
        return torch.cat([t, space], dim=-1)

    def zero(self, dim: int, dtype=torch.float32, device='cpu') -> torch.Tensor:
        """Origin of Lorentz model: [1/sqrt(-K), 0, ..., 0]"""
        x = torch.zeros(dim, dtype=dtype, device=device)
        x[0] = 1.0 / (-self.K) ** 0.5
        return x

    def exp0(self, u: torch.Tensor) -> torch.Tensor:
        """
        Exponential map at origin: maps tangent vector u in T_{L_0} to the manifold.

        u has the form [0, u_1, ..., u_n] (time component is 0 at origin tangent space).
        Result: [cosh(theta)/sqrt(-K), sinhdiv(theta)*u_s]
        where theta = sqrt(-K) * ||u_s||
        """
        sqrtK = (-self.K) ** 0.5
        u_s = u[..., 1:]
        u_s_norm = torch.norm(u_s, p=2, dim=-1, keepdim=True).clamp(min=self.eps)
        theta = (u_s_norm * sqrtK).clamp(min=self.eps, max=20.0)

        res = torch.ones_like(u)
        res[..., :1] = torch.cosh(theta) / sqrtK
        # sinhdiv(theta) = sinh(theta) / theta
        res[..., 1:] = (torch.sinh(theta) / theta) * u_s
        return res

    def log0(self, x: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic map at origin: maps manifold point x to tangent vector at origin.

        Result: [0, arcosh(sqrt(-K)*x_0) / (||x_s||*sqrt(-K)) * x_s]
        """
        sqrtK = (-self.K) ** 0.5
        x_t = x[..., :1]
        x_s = x[..., 1:]
        dom = (torch.norm(x_s, p=2, dim=-1, keepdim=True) * sqrtK).clamp(min=self.eps)
        theta = (sqrtK * x_t).clamp_min(1.0 + self.eps)
        scale = torch.acosh(theta) / dom
        res = torch.zeros_like(x)
        res[..., 1:] = scale * x_s
        return res

    def dist(self, x: torch.Tensor, y: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """Geodesic distance: d(x,y) = acosh(K*<x,y>_L) / sqrt(-K)"""
        beta = (self.K * self.ldot(x, y, keepdim=keepdim)).clamp(min=1.0 + self.eps)
        return torch.acosh(beta) / (-self.K) ** 0.5

    def geodesic(self, x: torch.Tensor, y: torch.Tensor, t: float) -> torch.Tensor:
        """Geodesic interpolation: gamma(t) = exp_x(t * log_x(y))"""
        log_xy = self._log(x, y)
        return self._exp(x, t * log_xy)

    def _exp(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Exponential map at point x: exp_x(u) = cosh(alpha)*x + sinhdiv(alpha)*u"""
        un = self.ldot(u, u, keepdim=True).clamp(min=self.eps).sqrt()
        alpha = (un * (-self.K) ** 0.5).clamp(min=self.eps, max=20.0)
        sinhdiv_alpha = torch.sinh(alpha) / alpha
        return x * torch.cosh(alpha) + sinhdiv_alpha * u

    def _log(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map at point x: log_x(y)"""
        beta = (self.K * self.ldot(x, y, keepdim=True)).clamp(min=1.0 + self.eps)
        num = torch.acosh(beta)
        # divsinh(num) = num / sinh(num)
        divsinh_num = num / torch.sinh(num).clamp(min=self.eps)
        return divsinh_num * (y - beta * x)

    # ---- Gyrovector space operations ----

    def gyroinv(self, x: torch.Tensor) -> torch.Tensor:
        """Gyro-inverse: negate spatial components, keep time component.
        This is equivalent to: -1 ⊙_K^L x = [x_0, -x_1, ..., -x_n]"""
        xt = x[..., :1]
        xs = x[..., 1:]
        return torch.cat([xt, -xs], dim=-1)

    def gyroadd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Closed-form Lorentz gyroaddition: x ⊕_K^L y

        Based on Eq. 19 of HEEGNet paper, closed-form formula using:
        a = 1 + sqrt(|K|)*x_t, b = 1 + sqrt(|K|)*y_t
        """
        sqrt_absK = (-self.K) ** 0.5

        x_t = x[..., :1]
        y_t = y[..., :1]
        x_s = x[..., 1:]
        y_s = y[..., 1:]

        a = 1.0 + sqrt_absK * x_t
        b = 1.0 + sqrt_absK * y_t

        n_x = (x_s * x_s).sum(dim=-1, keepdim=True)
        n_y = (y_s * y_s).sum(dim=-1, keepdim=True)
        s_xy = (x_s * y_s).sum(dim=-1, keepdim=True)

        D = (a * a) * (b * b) - 2.0 * self.K * a * b * s_xy + (self.K ** 2) * n_x * n_y
        N = (a * a) * n_y + 2.0 * a * b * s_xy + (b * b) * n_x

        denom = D + self.K * N
        sign = torch.sign(denom)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        denom_safe = denom + sign * self.eps

        z_t = ((D - self.K * N) / denom_safe) / sqrt_absK

        A_x = a * (b * b) - 2.0 * self.K * b * s_xy - self.K * a * n_y
        A_y = b * (a * a + self.K * n_x)
        coef = 2.0 / denom_safe
        z_s = coef * (A_x * x_s + A_y * y_s)

        return torch.cat([z_t, z_s], dim=-1)

    def gyrotrans(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Left gyrotranslation: x ⊕ y"""
        return self.gyroadd(x, y)

    def gyroscalarprod(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Gyro scalar multiplication: r ⊙_K^L x
        = (1/sqrt(|K|)) * [cosh(r*theta), sinh(r*theta)/||x_s|| * x_s]
        where theta = acosh(sqrt(|K|) * x_t)
        """
        sqrtK = (-self.K) ** 0.5
        xt = x[..., 0:1]
        xs = x[..., 1:]
        xs_norm = torch.norm(xs, p=2, dim=-1, keepdim=True).clamp_min(self.eps)
        theta = torch.acosh((sqrtK * xt).clamp_min(1.0 + self.eps))

        if r.dim() == 0 or (r.dim() == 1 and r.shape[0] == 1):
            rt = r * theta
        else:
            rt = r.unsqueeze(-1) * theta if r.dim() < theta.dim() else r * theta

        # Clamp rt to prevent cosh/sinh overflow (cosh(20) ~ 2.4e8)
        rt = rt.clamp(-20.0, 20.0)

        out = torch.zeros_like(x)
        out[..., 0:1] = torch.cosh(rt)
        out[..., 1:] = (torch.sinh(rt) / xs_norm) * xs
        return out / sqrtK

    def frechet_mean(self, x: torch.Tensor, max_iter: int = 100) -> torch.Tensor:
        """
        Iterative Karcher mean (Frechet mean) on the hyperboloid.

        Args:
            x: Points on the manifold, shape (N, d+1)
            max_iter: Maximum number of iterations

        Returns:
            Mean point on the manifold, shape (d+1,)
        """
        w = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device) / x.shape[0]
        mu = x[0].clone()

        for _ in range(max_iter):
            tan_data = self._log(mu.unsqueeze(0).expand_as(x), x)  # (N, d+1)
            tan_mean = (tan_data * w).sum(dim=0)  # (d+1,)
            if tan_mean.norm() <= 1e-4:
                break
            mu = self._exp(mu, tan_mean)

        return mu

    def frechet_variance(self, x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Frechet variance: mean squared geodesic distance to the mean.

        Args:
            x: Points on the manifold, shape (N, d+1)
            mu: Mean point, shape (d+1,)

        Returns:
            Scalar variance
        """
        d = self.dist(x, mu.unsqueeze(0).expand_as(x))  # (N,)
        return (d ** 2).mean()


# =============================================================================
# 1. Hyperbolic Projection: R^d -> L^n_K (Euclidean to Lorentz manifold)
# =============================================================================

class HyperbolicProjection(nn.Module):
    """
    Projects Euclidean feature vectors onto the Lorentz manifold.

    Pipeline (following HEEGNet model.py lines 116-118):
        1. Linear projection: R^{backbone_out_dim} -> R^{hyp_dim}
        2. L2-normalize the spatial components
        3. Pad a leading zero for the time coordinate: R^{hyp_dim} -> R^{hyp_dim+1}
        4. projx: Recompute time component to place on L^n_K

    The output is a valid point on the Lorentz manifold L^{hyp_dim}_K with shape (B, hyp_dim+1).

    Args:
        in_dim: Input Euclidean dimension (backbone_out_dim)
        hyp_dim: Intrinsic hyperbolic dimension (spatial components)
        K: Negative curvature of the Lorentz manifold
        dropout: Dropout probability
    """

    def __init__(self, in_dim: int, hyp_dim: int = 128, K: float = -1.0, dropout: float = 0.1):
        super().__init__()
        self.hyp_dim = hyp_dim
        self.manifold = LorentzManifold(K=K)

        # Euclidean projection: reduce backbone_out_dim to hyp_dim
        self.projection = nn.Sequential(
            nn.Linear(in_dim, hyp_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hyp_dim * 2, hyp_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Euclidean features, shape (B, in_dim)

        Returns:
            Points on Lorentz manifold, shape (B, hyp_dim + 1)
        """
        # Clamp input to prevent extreme values from backbone during full fine-tuning
        x = x.clamp(-100.0, 100.0)

        # Step 1: Linear projection to hyp_dim spatial features
        x = self.projection(x)  # (B, hyp_dim)

        # Step 2: L2-normalize (following HEEGNet: x / (||x||_2 + eps))
        x = x / (x.norm(p=2, dim=-1, keepdim=True) + 1e-8)

        # Step 3: Pad a leading 0 for time coordinate -> (B, hyp_dim + 1)
        x = F.pad(x, pad=(1, 0))  # [0, x_1, ..., x_n]

        # Step 4: projx -> recompute time component to lie on L^n_K
        # x_0 = sqrt(||x_s||^2 - 1/K), placing point on the hyperboloid
        x = self.manifold.projx(x)

        return x


# =============================================================================
# 3. Hyperbolic Domain-Specific Momentum Batch Normalization (DSMDBN)
# =============================================================================

class HyperbolicBNSingle(nn.Module):
    """
    Single-domain hyperbolic batch normalization on the Lorentz manifold.

    Algorithm (from HEEGNet Algorithm 1):
        Training:
            1. Compute batch Frechet mean mu_k and variance nu^2_k
            2. Update running mean: rm = geodesic(rm, mu_k, eta_train)
            3. Update running variance: rv = (1 - eta_train)*rv + eta_train*nu^2_k
        Normalization:
            1. Center: X_n = (⊖ rm) ⊕ X      (gyrotranslation by inverse of mean)
            2. Scale:  X_n = (std / sqrt(rv + eps)) ⊙ X_n   (gyro scalar multiplication)

    Args:
        hyp_dim_plus1: Ambient dimension of the Lorentz manifold (hyp_dim + 1)
        K: Negative curvature
        eta_train: Training momentum (initial)
        eta_test: Test momentum (fixed)
        std: Shared learnable dispersion parameter (optional)
    """

    def __init__(self, hyp_dim_plus1: int, K: float = -1.0,
                 eta_train: float = 1.0, eta_test: float = 0.1,
                 std: Optional[nn.Parameter] = None):
        super().__init__()
        self.manifold = LorentzManifold(K=K)
        self.hyp_dim_plus1 = hyp_dim_plus1
        self.eta = eta_train
        self.eta_test = eta_test
        self.eps = 1e-5

        # Running statistics buffers (not learnable)
        origin = self.manifold.zero(hyp_dim_plus1)
        self.register_buffer('running_mean', origin.clone())
        self.register_buffer('running_var', torch.ones(1))
        self.register_buffer('running_mean_test', origin.clone())
        self.register_buffer('running_var_test', torch.ones(1))

        # Shared learnable dispersion parameter (passed from parent)
        if std is not None:
            self.std = std
        else:
            self.std = nn.Parameter(torch.ones(1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: Points on Lorentz manifold, shape (N, hyp_dim+1)

        Returns:
            Normalized points on Lorentz manifold, shape (N, hyp_dim+1)
        """
        if self.training:
            # Compute batch Frechet mean and variance (detached from running stats)
            batch_mean = self.manifold.frechet_mean(X.detach(), max_iter=100)
            batch_var = self.manifold.frechet_variance(X.detach(), batch_mean)

            # Compute normalization reference (detached running stats to avoid in-place issues)
            rm = self.manifold.geodesic(
                self.running_mean.detach(), batch_mean.detach(), self.eta
            )
            rv = (1.0 - self.eta) * self.running_var.detach() + self.eta * batch_var.detach()

            # Update all running stats buffers
            with torch.no_grad():
                self.running_mean.copy_(rm)
                self.running_var.copy_(rv)
                self.running_mean_test.copy_(
                    self.manifold.geodesic(self.running_mean_test, batch_mean, self.eta_test)
                )
                self.running_var_test.copy_(
                    (1.0 - self.eta_test) * self.running_var_test + self.eta_test * batch_var
                )
        else:
            # At test time, use buffered test running stats
            rm = self.running_mean_test
            rv = self.running_var_test

        # Normalize: center then scale
        # 1. Center: X_centered = (⊖ rm) ⊕ X
        inv_rm = self.manifold.gyroinv(rm.detach())
        X_centered = self.manifold.gyrotrans(inv_rm, X)

        # 2. Scale: X_scaled = (std / sqrt(rv + eps)) ⊙ X_centered
        # Gradients flow through self.std (learnable) and through X_centered -> X
        factor = self.std / (rv.detach() + self.eps).sqrt()
        X_normalized = self.manifold.gyroscalarprod(X_centered, factor)

        return X_normalized


class HyperbolicDSMDBN(nn.Module):
    """
    Domain-Specific Momentum-then-Distribution Batch Normalization (DSMDBN)
    on the Lorentz manifold.

    Stage 1 (DSMDBN-1): Moment alignment via domain-specific HBN
        - Each domain has its own running mean and variance
        - Learnable dispersion parameter std is SHARED across domains

    Stage 2 (DSMDBN-2): Distribution alignment via HHSW loss
        - Computed externally and added to the training loss

    At test time, only Stage 1 is applied.

    Args:
        hyp_dim_plus1: Ambient dimension of the Lorentz manifold (hyp_dim + 1)
        K: Negative curvature
        eta_train: Training momentum
        eta_test: Test momentum
    """

    def __init__(self, hyp_dim_plus1: int, K: float = -1.0,
                 eta_train: float = 1.0, eta_test: float = 0.1):
        super().__init__()
        self.hyp_dim_plus1 = hyp_dim_plus1
        self.K = K

        # Shared learnable dispersion parameter across all domains
        self.std = nn.Parameter(torch.ones(1))

        # Domain-specific BN layers, created on-the-fly
        self.domain_bns = nn.ModuleDict()
        self.eta_train = eta_train
        self.eta_test = eta_test

    def _get_or_create_bn(self, domain_key: str) -> HyperbolicBNSingle:
        """Get or lazily create a domain-specific BN layer."""
        if domain_key not in self.domain_bns:
            bn = HyperbolicBNSingle(
                self.hyp_dim_plus1, K=self.K,
                eta_train=self.eta_train, eta_test=self.eta_test,
                std=self.std,  # shared parameter
            )
            # Move to same device as std
            bn = bn.to(self.std.device)
            self.domain_bns[domain_key] = bn
        return self.domain_bns[domain_key]

    def forward(self, X: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: Points on Lorentz manifold, shape (B, hyp_dim+1)
            domains: Domain labels, shape (B,)

        Returns:
            Normalized points, shape (B, hyp_dim+1)
        """
        du = domains.unique()
        X_normalized = torch.empty_like(X)

        for domain in du:
            mask = (domains == domain)
            domain_key = str(domain.item())
            bn = self._get_or_create_bn(domain_key)

            X_domain = X[mask]
            X_out = bn(X_domain)

            indices = torch.nonzero(mask).flatten()
            X_normalized[indices] = X_out

        return X_normalized

    def set_eta(self, eta_train: Optional[float] = None, eta_test: Optional[float] = None):
        """Update momentum for all domain BNs (for scheduling)."""
        if eta_train is not None:
            self.eta_train = eta_train
        if eta_test is not None:
            self.eta_test = eta_test
        for bn in self.domain_bns.values():
            if eta_train is not None:
                bn.eta = eta_train
            if eta_test is not None:
                bn.eta_test = eta_test


# =============================================================================
# 4. HMLR: Hyperbolic Multinomial Logistic Regression
# =============================================================================

class HMLR(nn.Module):
    """
    Hyperbolic Multinomial Logistic Regression in the Lorentz model.

    Each class c has a decision hyperplane parameterized by:
        - a_c in R: offset along the hyperplane normal
        - z_c in R^{n-1}: spatial orientation of the hyperplane

    The logit for class c is:
        logit_c = sign(alpha_c) * beta_c * d_c

    where:
        alpha_c = cosh(sqrt(-K)*a_c) * <z_c, x_s> - sinh(sqrt(-K)*a_c)*||z_c||*x_t
        beta_c  = sqrt(||cosh(sqrt(-K)*a_c)*z_c||^2 - (sinh(sqrt(-K)*a_c)*||z_c||)^2)
        d_c     = sqrt(K) * |arcsinh(sqrt(-K)*alpha_c / beta_c)|

    Adapted from HEEGNet's LorentzMLR (lib/lorentz/layers/LMLR.py).

    Args:
        num_features: Ambient dimension of the Lorentz manifold (hyp_dim + 1)
        num_classes: Number of output classes
        K: Negative curvature (default -1)
    """

    def __init__(self, num_features: int, num_classes: int, K: float = -1.0):
        super().__init__()
        self.K = K
        self.k = abs(K)  # |K| = 1 for K = -1
        self.num_classes = num_classes
        self.num_features = num_features

        # Parameters: a_c (offset) and z_c (spatial hyperplane normal)
        # z has shape (num_classes, num_features - 2), padded with a leading 1
        self.a = nn.Parameter(torch.zeros(num_classes))
        self.z = nn.Parameter(
            F.pad(torch.zeros(num_classes, num_features - 2), pad=(1, 0), value=1.0)
        )
        self._init_weights()

    def _init_weights(self):
        stdv = 1.0 / math.sqrt(self.z.size(1))
        nn.init.uniform_(self.z, -stdv, stdv)
        nn.init.uniform_(self.a, -stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Points on Lorentz manifold, shape (B, num_features)

        Returns:
            logits: Shape (B, num_classes)
        """
        sqrt_mK = 1.0 / self.k ** 0.5  # sqrt(-K) = sqrt(|K|) = 1 for K=-1, so sqrt_mK = 1/sqrt(|K|)

        # Clamp parameter a to prevent sinh/cosh overflow
        a_clamped = self.a.clamp(-10.0, 10.0)

        # Hyperplane normal components
        norm_z = torch.norm(self.z, dim=-1)  # (num_classes,)
        w_t = torch.sinh(sqrt_mK * a_clamped) * norm_z  # (num_classes,)
        w_s = torch.cosh(sqrt_mK * a_clamped.unsqueeze(-1)) * self.z  # (num_classes, num_features-2)

        # beta = sqrt(-w_t^2 + ||w_s||^2) -- Lorentz norm of normal vector
        beta = torch.sqrt(
            (-w_t ** 2 + torch.norm(w_s, dim=-1) ** 2).clamp(min=1e-8)
        )  # (num_classes,)

        # alpha = -w_t * x_0 + cosh(sqrt(-K)*a) * <x_s, z>
        x_t = x.narrow(-1, 0, 1)  # (B, 1)
        x_s = x.narrow(-1, 1, x.shape[-1] - 1)  # (B, num_features-1)

        # Note: z has shape (num_classes, num_features-2), x_s has (B, num_features-1)
        # They must align: z was padded with leading 1, so z is (num_classes, num_features-1)
        alpha = (
            -w_t * x_t
            + torch.cosh(sqrt_mK * a_clamped) * torch.matmul(x_s, self.z.T)
        )  # (B, num_classes)

        # d = sqrt(K) * |arcsinh(sqrt(-K) * alpha / beta)|
        # Clamp the asinh argument to prevent extreme values
        asinh_arg = (sqrt_mK * alpha / beta.unsqueeze(0).clamp(min=1e-8)).clamp(-50.0, 50.0)
        d = self.k ** 0.5 * torch.abs(torch.asinh(asinh_arg))

        # logits = sign(alpha) * beta * d
        logits = torch.sign(alpha) * beta.unsqueeze(0) * d

        return logits


# =============================================================================
# 5. HHSW Loss: Hyperbolic Horospherical Sliced-Wasserstein
# =============================================================================

def _emd1d(u_values: torch.Tensor, v_values: torch.Tensor,
           u_weights=None, v_weights=None, p: int = 2) -> torch.Tensor:
    """
    1D Earth Mover's Distance (Wasserstein-p distance) via sorted CDFs.

    Args:
        u_values: Shape (num_projections, N)
        v_values: Shape (num_projections, M)
        p: Wasserstein exponent
    """
    n = u_values.shape[-1]
    m = v_values.shape[-1]
    device = u_values.device
    dtype = u_values.dtype

    if u_weights is None:
        u_weights = torch.full((n,), 1.0 / n, dtype=dtype, device=device)
    if v_weights is None:
        v_weights = torch.full((m,), 1.0 / m, dtype=dtype, device=device)

    u_values, u_sorter = torch.sort(u_values, -1)
    v_values, v_sorter = torch.sort(v_values, -1)

    u_weights = u_weights[..., u_sorter]
    v_weights = v_weights[..., v_sorter]

    u_cdf = torch.cumsum(u_weights, -1)
    v_cdf = torch.cumsum(v_weights, -1)

    cdf_axis, _ = torch.sort(torch.cat((u_cdf, v_cdf), -1), -1)

    u_index = torch.searchsorted(u_cdf, cdf_axis)
    v_index = torch.searchsorted(v_cdf, cdf_axis)

    u_icdf = torch.gather(u_values, -1, u_index.clip(0, n - 1))
    v_icdf = torch.gather(v_values, -1, v_index.clip(0, m - 1))

    cdf_axis = F.pad(cdf_axis, (1, 0))
    delta = cdf_axis[..., 1:] - cdf_axis[..., :-1]

    if p == 1:
        return torch.sum(delta * torch.abs(u_icdf - v_icdf), dim=-1)
    elif p == 2:
        return torch.sum(delta * (u_icdf - v_icdf) ** 2, dim=-1)
    else:
        return torch.sum(delta * torch.abs(u_icdf - v_icdf) ** p, dim=-1)


def _busemann_lorentz(v: torch.Tensor, z: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """
    Busemann function in the Lorentz model.

    B_v(z) = log(-<v + x0, z>_L)

    where x0 is the origin [1, 0, ..., 0] and v is a direction on the ideal boundary.

    Args:
        v: Projection directions, shape (num_projections, d)
        z: Points on manifold, shape (N, d)
        x0: Origin, shape (1, d)

    Returns:
        Shape (num_projections, N)
    """
    # <v+x0, z>_Minkowski = -(v+x0)_0 * z_0 + sum_{i>0} (v+x0)_i * z_i
    vx0 = v + x0  # (num_projections, d)
    ip = -vx0[:, 0:1] * z[:, 0:1].T + torch.matmul(vx0[:, 1:], z[:, 1:].T)
    # ip shape: (num_projections, N)
    # Clamp to prevent log(huge) which causes gradient explosion
    return torch.log((-ip).clamp(min=1e-8, max=1e6))


def compute_hhsw_loss(
    features: torch.Tensor,
    domains: torch.Tensor,
    manifold: LorentzManifold,
    num_projections: int = 1000,
    p: int = 2,
) -> torch.Tensor:
    """
    Hyperbolic Horospherical Sliced-Wasserstein (HHSW) loss.

    For each domain, compute HHSW distance between domain features and a
    standard hyperbolic Gaussian (Gaussian noise mapped to manifold via exp0).

    Algorithm (from HEEGNet Algorithm 2):
        For each domain d:
            1. Extract domain features P_d
            2. Sample Gaussian noise Z ~ N(0,I), normalize, map to L^n_K via exp0
            3. Compute HHSW(P_d, Q_d) via random Busemann projections

    Args:
        features: Points on Lorentz manifold, shape (B, hyp_dim+1)
        domains: Domain labels, shape (B,)
        manifold: LorentzManifold instance
        num_projections: Number of random slices (default 1000)
        p: Wasserstein exponent (default 2)

    Returns:
        Scalar HHSW loss
    """
    du = domains.unique()
    total_loss = torch.tensor(0.0, device=features.device, dtype=features.dtype)

    for domain in du:
        mask = (domains == domain)
        x = features[mask]  # (N_d, hyp_dim+1)

        if x.shape[0] < 2:
            continue

        n, d = x.shape

        # Sample target: Gaussian noise -> normalize -> exp0 (map to Lorentz)
        x_gaussian = torch.randn(n, d, dtype=x.dtype, device=x.device)
        x_gaussian = x_gaussian / (x_gaussian.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        # Create tangent vector at origin: [0, normalized_spatial...]
        x_gaussian_tangent = torch.zeros_like(x_gaussian)
        x_gaussian_tangent[..., 1:] = x_gaussian[..., 1:]
        x_gaussian = manifold.exp0(x_gaussian_tangent)

        # Random projection directions on the ideal boundary
        # Generate unit vectors in R^{d-1}, pad with 0 time component
        vs = np.random.normal(size=(num_projections, d - 1))
        vs = F.normalize(
            torch.from_numpy(vs).to(dtype=x.dtype, device=x.device),
            p=2, dim=-1
        )
        vs = F.pad(vs, (1, 0))  # (num_projections, d) with v_0 = 0

        # Origin point
        x0 = torch.zeros(1, d, dtype=x.dtype, device=x.device)
        x0[0, 0] = 1.0  # Lorentz origin for K=-1: [1, 0, ..., 0]

        # Busemann projections to R
        Xps = _busemann_lorentz(vs, x, x0)  # (num_projections, N_d)
        Xpt = _busemann_lorentz(vs, x_gaussian, x0)  # (num_projections, N_d)

        # 1D Wasserstein distance for each projection, then average
        domain_loss = torch.mean(_emd1d(Xps, Xpt, p=p))
        total_loss = total_loss + domain_loss

    return total_loss


# =============================================================================
# 6. Hyperbolic ELU Activation on the Lorentz Manifold
# =============================================================================

class LorentzELU(nn.Module):
    """
    ELU activation on the Lorentz manifold.

    Apply ELU to spatial components, then recompute time component
    to ensure the output remains on the manifold.

    From HEEGNet Eq. 21:
        f_ELU(x) = [sqrt(||ELU(x_s)||^2 - 1/K), ELU(x_s)]
    """

    def __init__(self, K: float = -1.0):
        super().__init__()
        self.manifold = LorentzManifold(K=K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ELU to spatial part, recompute time component."""
        x_s = F.elu(x[..., 1:])
        return self.manifold.add_time(x_s)


# =============================================================================
# 7. HybridCodeBrainFineTuner: Full Model Assembly
# =============================================================================

class HybridCodeBrainFineTuner(nn.Module):
    """
    Hybrid Euclidean-Hyperbolic Fine-Tuning Model for EEG Classification.

    Architecture:
        Frozen CodeBrain SSSM (Euclidean) -> HyperbolicProjection -> LorentzELU ->
        HyperbolicDSMDBN (moment alignment) -> HMLR (hyperbolic classifier)

    Training Loss:
        L_total = L_CE + lambda_hhsw * L_HHSW

    At test time:
        - Only moment alignment (DSMDBN Stage 1) is applied
        - No HHSW loss computation

    Args:
        backbone: Pre-trained CodeBrain SSSM model (will be frozen)
        backbone_out_dim: Flattened output dim of backbone (n_channels * seq_len * 200)
        num_classes: Number of classification targets
        hyp_dim: Intrinsic hyperbolic dimension (spatial components in Lorentz)
        K: Negative curvature (default -1.0)
        dropout: Dropout probability
        eta_train: DSMDBN training momentum
        eta_test: DSMDBN test momentum
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_out_dim: int,
        num_classes: int,
        hyp_dim: int = 128,
        K: float = -1.0,
        dropout: float = 0.1,
        eta_train: float = 1.0,
        eta_test: float = 0.1,
    ):
        super().__init__()

        self.backbone = backbone
        self.backbone_out_dim = backbone_out_dim
        self.hyp_dim = hyp_dim
        self.K = K
        self.manifold = LorentzManifold(K=K)

        # Check if backbone has trainable adapter params (CBraMod with adapters)
        self.has_backbone_adapters = any(
            p.requires_grad for p in self.backbone.parameters()
        )

        # Trainable: Euclidean -> Lorentz projection
        self.hyp_projection = HyperbolicProjection(
            in_dim=backbone_out_dim, hyp_dim=hyp_dim, K=K, dropout=dropout
        )

        # Trainable: Lorentz ELU activation
        self.lorentz_elu = LorentzELU(K=K)

        # Trainable: Domain-specific hyperbolic batch normalization
        self.dsmdbn = HyperbolicDSMDBN(
            hyp_dim_plus1=hyp_dim + 1, K=K,
            eta_train=eta_train, eta_test=eta_test
        )

        # Trainable: Hyperbolic MLR classifier
        self.hmlr = HMLR(
            num_features=hyp_dim + 1, num_classes=num_classes, K=K
        )

        # Print summary
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = frozen + trainable
        print(f"\nHybridCodeBrainFineTuner parameter summary:")
        print(f"  Frozen params:       {frozen:,}")
        print(f"  Trainable params:    {trainable:,}")
        print(f"  Trainable ratio:     {trainable / total * 100:.2f}%")
        if self.has_backbone_adapters:
            adapter_params = sum(
                p.numel() for name, p in self.backbone.named_parameters()
                if p.requires_grad
            )
            print(f"  Backbone adapters:   {adapter_params:,}")
        print(f"  Hyperbolic dim:      {hyp_dim} (ambient: {hyp_dim + 1})")
        print(f"  Curvature K:         {K}")
        print(f"  Output classes:      {num_classes}")

    def forward(
        self, x: torch.Tensor, domains: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Raw EEG input, shape (B, n_channels, seq_len, patch_size=200)
            domains: Subject/domain labels, shape (B,)

        Returns:
            logits: Classification logits, shape (B, num_classes)
            features: Hyperbolic features on Lorentz manifold, shape (B, hyp_dim+1)
        """
        B = x.shape[0]

        # 1. Backbone feature extraction
        # When adapters have gradients, skip no_grad to allow backprop through them
        if self.has_backbone_adapters:
            backbone_out = self.backbone(x)  # (B, C, S, 200)
        else:
            with torch.no_grad():
                backbone_out = self.backbone(x)  # (B, C, S, 200)

        # 2. Flatten backbone output
        backbone_flat = backbone_out.reshape(B, -1)  # (B, backbone_out_dim)

        # 3. Project to Lorentz manifold: R^d -> L^{hyp_dim}_K
        hyp_features = self.hyp_projection(backbone_flat)  # (B, hyp_dim + 1)

        # 4. Lorentz ELU activation (nonlinearity on manifold)
        hyp_features = self.lorentz_elu(hyp_features)  # (B, hyp_dim + 1)

        # 5. Domain-specific hyperbolic batch normalization
        hyp_features = self.dsmdbn(hyp_features, domains)  # (B, hyp_dim + 1)

        # 6. HMLR classifier
        logits = self.hmlr(hyp_features)  # (B, num_classes)

        return logits, hyp_features

    def domainadapt_finetune(self, x: torch.Tensor, domains: torch.Tensor):
        """
        Source-Free Unsupervised Domain Adaptation (SFUDA) at test time.
        Refit DSMDBN running statistics on target domain data.
        """
        self.eval()
        with torch.no_grad():
            for du in domains.unique():
                mask = (domains == du)
                self.forward(x[mask], domains[mask])


# =============================================================================
# 8. Loss Computation
# =============================================================================

def compute_hybrid_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    features: torch.Tensor,
    domains: torch.Tensor,
    manifold: LorentzManifold,
    lambda_hhsw: float = 0.5,
    num_projections: int = 1000,
    task_type: str = 'multiclass',
) -> Tuple[torch.Tensor, dict]:
    """
    Compute total hybrid loss: L_total = L_CE + lambda * L_HHSW

    Args:
        logits: Classification logits, shape (B, num_classes)
        labels: Ground truth labels, shape (B,)
        features: Hyperbolic features on Lorentz manifold, shape (B, hyp_dim+1)
        domains: Domain labels, shape (B,)
        manifold: LorentzManifold instance
        lambda_hhsw: Weight for HHSW loss (0.01 for emotion, 0.5 for others)
        num_projections: Number of HHSW slices
        task_type: 'multiclass' or 'binary'

    Returns:
        total_loss: Scalar loss
        loss_dict: Dictionary with individual loss components (for logging)
    """
    # Cross-entropy loss
    # Always use cross_entropy since HMLR outputs (B, num_classes) even for binary
    # (binary with num_classes=2 is just a special case of multiclass)
    loss_ce = F.cross_entropy(logits, labels)

    # HHSW distribution alignment loss
    loss_hhsw = compute_hhsw_loss(
        features, domains, manifold,
        num_projections=num_projections, p=2
    )

    total_loss = loss_ce + lambda_hhsw * loss_hhsw

    loss_dict = {
        'total': total_loss.item(),
        'ce': loss_ce.item(),
        'hhsw': loss_hhsw.item(),
        'weighted_hhsw': (lambda_hhsw * loss_hhsw).item(),
    }

    return total_loss, loss_dict


# =============================================================================
# 9. Optimizer Configuration
# =============================================================================

def configure_optimizer(
    model: HybridCodeBrainFineTuner,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    use_riemannian: bool = True,
    backbone_lr_ratio: float = 0.01,
) -> torch.optim.Optimizer:
    """
    Configure optimizer with separate parameter groups.

    Hyperbolic parameters (HMLR, DSMDBN) use zero weight decay
    following HEEGNet's convention. Euclidean parameters (projection)
    use standard weight decay.

    If use_riemannian=True and geoopt is available, uses RiemannianAdam.
    Otherwise falls back to standard Adam.

    Args:
        model: HybridCodeBrainFineTuner instance
        lr: Learning rate
        weight_decay: Weight decay for Euclidean parameters
        use_riemannian: Whether to use RiemannianAdam
        backbone_lr_ratio: Backbone lr = lr * ratio (for full fine-tuning)

    Returns:
        Configured optimizer
    """
    hyp_params = []     # Hyperbolic params -> zero weight decay
    eucl_params = []    # Euclidean params -> standard weight decay
    backbone_params = [] # Backbone params -> lower lr

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('backbone.'):
            backbone_params.append(param)
        elif name.startswith('hmlr') or name.startswith('dsmdbn') or name.startswith('lorentz_elu'):
            hyp_params.append(param)
        else:
            eucl_params.append(param)

    param_groups = [
        {'params': hyp_params, 'weight_decay': 0.0},
        {'params': eucl_params, 'weight_decay': weight_decay},
    ]
    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': lr * backbone_lr_ratio,
            'weight_decay': weight_decay,
        })

    if use_riemannian:
        try:
            from geoopt.optim import RiemannianAdam
            return RiemannianAdam(param_groups, lr=lr)
        except ImportError:
            print("geoopt not available, falling back to Adam")

    return torch.optim.Adam(param_groups, lr=lr)


# =============================================================================
# 10. Momentum Scheduler for DSMDBN
# =============================================================================

class DSMDBNMomentumScheduler:
    """
    Clamped exponential decay scheduler for DSMDBN training momentum.

    eta_train(k) = max(eta_min, eta_init * decay^k)

    Following HEEGNet's convention of adaptively decreasing momentum
    during training so that early batches have more influence initially,
    gradually transitioning to stable running statistics.
    """

    def __init__(self, model: HybridCodeBrainFineTuner,
                 eta_init: float = 1.0, eta_min: float = 0.1,
                 decay: float = 0.99):
        self.model = model
        self.eta_init = eta_init
        self.eta_min = eta_min
        self.decay = decay
        self.step_count = 0

    def step(self):
        """Update momentum after each training step."""
        self.step_count += 1
        eta = max(self.eta_min, self.eta_init * (self.decay ** self.step_count))
        self.model.dsmdbn.set_eta(eta_train=eta)

    def get_eta(self) -> float:
        """Get current momentum value."""
        return max(self.eta_min, self.eta_init * (self.decay ** self.step_count))
