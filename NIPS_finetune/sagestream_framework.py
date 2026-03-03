#!/usr/bin/env python3
"""
SageStream Fine-Tuning Framework for EEG Foundation Models

Combines Subject-Aware Mixture of Experts (SA-MoE) with Information Invariant
Bottleneck (IIB) for subject-invariant, adaptive fine-tuning of frozen
CodeBrain / CBraMod backbones.

Architecture:
    Backbone (frozen): (B, C, S, P) → (B, C, S, 200)
        ↓ reshape to tokens: (B, T, 200) where T = C*S
    SA-MoE layers: token-level expert routing + subject style alignment
        ↓ mean pool: (B, 200)
    IIB variational encoder: z ~ N(mu, σ²), KL loss
        ↓ GRL → subject discriminator (adversarial loss)
    Classification head: (B, num_classes)

References:
    - SageStream: SA-MoE + IIB (~/eeg2025/NIPS/SageStream)
    - backbone_factory.py: frozen backbone creation
    - ib_disentangle_framework.py: GRL, GRLScheduler
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ib_disentangle_framework import GradientReversalLayer, GRLScheduler


# =============================================================================
# SA-MoE Components
# =============================================================================

class GLUExpert(nn.Module):
    """Single FFN expert with GLU (Gated Linear Unit) gating."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.wi_0 = nn.Linear(d_model, d_ff, bias=False)
        self.wi_1 = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        # GLU: act(W0 x) * (W1 x)
        hidden = self.act(self.wi_0(x)) * self.wi_1(x)
        hidden = self.dropout(hidden)
        return self.wo(hidden)


class ExpertPool(nn.Module):
    """Shared pool of GLU experts, reused across SA-MoE layers."""

    def __init__(self, d_model, d_ff, num_experts=4, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            GLUExpert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])

    def forward(self, expert_id, x):
        return self.experts[expert_id](x)


class TopKRouter(nn.Module):
    """Token-to-expert router with top-k selection."""

    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_experts),
        )

    def forward(self, x):
        """
        Args:
            x: (*, d_model) token features
        Returns:
            top_k_gates: (*, top_k) normalized gate values
            top_k_indices: (*, top_k) expert indices
            router_probs: (*, num_experts) full softmax probabilities
        """
        logits = self.gate(x)
        router_probs = F.softmax(logits, dim=-1)
        top_k_gates, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_gates = top_k_gates / (top_k_gates.sum(dim=-1, keepdim=True) + 1e-8)
        return top_k_gates, top_k_indices, router_probs


class SubjectStyleAlignment(nn.Module):
    """Subject-specific style alignment via hypernetwork.

    Generates per-subject (gamma, beta) for instance normalization of token features.
    """

    def __init__(self, d_model, num_subjects, subject_embed_dim=64, hidden_dim=128):
        super().__init__()
        self.d_model = d_model
        self.subject_embedding = nn.Embedding(num_subjects, subject_embed_dim)
        nn.init.normal_(self.subject_embedding.weight, mean=0.0, std=0.02)

        self.hyper_net = nn.Sequential(
            nn.Linear(subject_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Output gamma and beta for d_model dimensions
        self.style_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, d_model * 2),
        )

    def forward(self, x, subject_ids):
        """
        Args:
            x: (B, T, D) token features
            subject_ids: (B,) integer subject IDs
        Returns:
            aligned: (B, T, D) style-aligned features
        """
        B, T, D = x.shape

        # Instance norm over the token dimension
        # (B, T, D) → (B, D, T) for instance_norm → (B, T, D)
        x_norm = F.instance_norm(x.permute(0, 2, 1), eps=1e-8).permute(0, 2, 1)

        # Generate subject-specific style params
        subj_embed = self.subject_embedding(subject_ids)  # (B, embed_dim)
        h = self.hyper_net(subj_embed)  # (B, hidden_dim)
        style_params = self.style_head(h)  # (B, D*2)
        gamma, beta = style_params.chunk(2, dim=-1)  # each (B, D)
        gamma = F.softplus(gamma) + 1e-8  # ensure positive scaling

        # Apply: gamma * norm(x) + beta
        aligned = x_norm * gamma.unsqueeze(1) + beta.unsqueeze(1)
        return aligned


class SAMoELayer(nn.Module):
    """Single SA-MoE layer: style alignment + expert routing + residual."""

    def __init__(self, d_model, expert_pool, router, num_subjects,
                 top_k=2, aux_loss_weight=0.01, use_style=True,
                 subject_embed_dim=64, style_hidden_dim=128):
        super().__init__()
        self.d_model = d_model
        self.expert_pool = expert_pool
        self.router = router
        self.num_experts = expert_pool.num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.use_style = use_style

        self.layer_norm = nn.LayerNorm(d_model)

        if use_style and num_subjects > 1:
            self.style_align = SubjectStyleAlignment(
                d_model, num_subjects, subject_embed_dim, style_hidden_dim
            )
        else:
            self.style_align = None

        self._last_router_probs = None
        self._last_indices = None

    def forward(self, x, subject_ids=None):
        """
        Args:
            x: (B, T, D) token features
            subject_ids: (B,) optional subject IDs
        Returns:
            output: (B, T, D) processed features with residual
        """
        residual = x
        h = self.layer_norm(x)

        # Subject style alignment
        if self.style_align is not None and subject_ids is not None:
            h = self.style_align(h, subject_ids)

        # Route tokens to experts
        B, T, D = h.shape
        h_flat = h.reshape(-1, D)  # (B*T, D)

        gates, indices, router_probs = self.router(h_flat)
        self._last_router_probs = router_probs
        self._last_indices = indices

        # Dispatch to experts and aggregate
        output_flat = torch.zeros_like(h_flat)
        gates_flat = gates  # (B*T, top_k)
        indices_flat = indices  # (B*T, top_k)

        for k in range(self.top_k):
            expert_indices = indices_flat[:, k]  # (B*T,)
            gate_values = gates_flat[:, k].unsqueeze(-1)  # (B*T, 1)

            for expert_idx in range(self.num_experts):
                mask = (expert_indices == expert_idx)
                if mask.any():
                    token_ids = torch.where(mask)[0]
                    expert_out = self.expert_pool(expert_idx, h_flat[token_ids])
                    output_flat[token_ids] += gate_values[token_ids] * expert_out

        output = output_flat.reshape(B, T, D)
        return residual + output

    def get_aux_loss(self):
        """Compute Switch Transformer auxiliary load-balancing loss."""
        if self._last_router_probs is None or self._last_indices is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        router_probs = self._last_router_probs  # (N, num_experts)
        indices = self._last_indices  # (N, top_k)
        num_experts = router_probs.shape[-1]

        # Average probability per expert
        P_i = router_probs.mean(dim=0)  # (num_experts,)

        # Fraction of tokens routed to each expert
        expert_mask = F.one_hot(indices, num_classes=num_experts).float()  # (N, top_k, E)
        f_i = expert_mask.mean(dim=0)  # (top_k, E)

        aux_loss = self.aux_loss_weight * num_experts * torch.sum(f_i * P_i.unsqueeze(0))
        return aux_loss

    def clear_aux_state(self):
        self._last_router_probs = None
        self._last_indices = None


# =============================================================================
# IIB Components
# =============================================================================

class VariationalEncoder(nn.Module):
    """Variational encoder for Information Invariant Bottleneck."""

    def __init__(self, input_dim, latent_dim, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        """
        Args:
            x: (B, D) aggregated features
        Returns:
            z: (B, latent_dim) sampled latent
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-10, 10)

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu
        return z, mu, logvar


class TaskClassifier(nn.Module):
    """Classification head for disease/event prediction."""

    def __init__(self, input_dim, num_classes, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.BatchNorm1d(input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SubjectClassifier(nn.Module):
    """Subject discriminator for adversarial training."""

    def __init__(self, input_dim, num_subjects, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_subjects),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Full SageStream Model
# =============================================================================

class SageStreamModel(nn.Module):
    """
    SageStream fine-tuning model: frozen backbone + SA-MoE + IIB.

    Combines Subject-Aware Mixture of Experts for adaptive token processing
    with Information Invariant Bottleneck for subject-invariant classification.
    """

    def __init__(
        self,
        backbone,
        token_dim,
        num_classes,
        num_subjects,
        latent_dim=128,
        n_moe_layers=2,
        num_experts=4,
        top_k=2,
        d_ff=None,
        aux_loss_weight=0.01,
        use_style=True,
        subject_embed_dim=64,
        style_hidden_dim=128,
        dropout=0.1,
        lambda_adv=1.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.token_dim = token_dim  # 200
        self.num_classes = num_classes
        self.num_subjects = num_subjects
        self.n_moe_layers = n_moe_layers

        if d_ff is None:
            d_ff = token_dim * 2  # 400

        # Check if backbone has trainable adapters
        self.has_backbone_adapters = hasattr(backbone, 'adapters') and backbone.adapters is not None

        # Shared expert pool and router across all MoE layers
        self.expert_pool = ExpertPool(token_dim, d_ff, num_experts, dropout)
        self.router = TopKRouter(token_dim, num_experts, top_k)

        # SA-MoE layers
        self.moe_layers = nn.ModuleList([
            SAMoELayer(
                d_model=token_dim,
                expert_pool=self.expert_pool,
                router=self.router,
                num_subjects=num_subjects,
                top_k=top_k,
                aux_loss_weight=aux_loss_weight,
                use_style=use_style,
                subject_embed_dim=subject_embed_dim,
                style_hidden_dim=style_hidden_dim,
            )
            for _ in range(n_moe_layers)
        ])

        # IIB variational encoder
        self.ib_encoder = VariationalEncoder(token_dim, latent_dim, dropout)

        # Classification head
        self.task_head = TaskClassifier(latent_dim, num_classes, dropout)

        # Adversarial subject head
        self.grl = GradientReversalLayer(lambda_=lambda_adv)
        self.subject_head = SubjectClassifier(latent_dim, num_subjects, dropout)

    def forward(self, x, subject_ids=None):
        """
        Args:
            x: (B, C, S, P) raw EEG input
            subject_ids: (B,) optional subject IDs
        Returns:
            dict with task_logits, subject_logits, mu, log_var, z_agg
        """
        # 1. Backbone forward
        if self.has_backbone_adapters:
            features = self.backbone(x)  # (B, C, S, 200)
        else:
            with torch.no_grad():
                features = self.backbone(x)  # (B, C, S, 200)

        # 2. Reshape to token sequence
        B, C, S, D = features.shape
        tokens = features.reshape(B, C * S, D)  # (B, T, 200)

        # 3. SA-MoE layers
        for moe_layer in self.moe_layers:
            tokens = moe_layer(tokens, subject_ids)

        # 4. Aggregate tokens via mean pooling
        pooled = tokens.mean(dim=1)  # (B, 200)

        # 5. IIB variational encoding
        z, mu, logvar = self.ib_encoder(pooled)  # (B, latent_dim)

        # 6. Classification
        task_logits = self.task_head(z)  # (B, num_classes)

        # 7. Adversarial subject prediction
        z_reversed = self.grl(z)
        subject_logits = self.subject_head(z_reversed)  # (B, num_subjects)

        return {
            'task_logits': task_logits,
            'subject_logits': subject_logits,
            'mu': mu,
            'log_var': logvar,
            'z_agg': z,
        }

    def get_total_aux_loss(self):
        """Sum auxiliary MoE load-balancing losses across all layers."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.moe_layers:
            total = total + layer.get_aux_loss()
        return total / max(len(self.moe_layers), 1)

    def clear_aux_state(self):
        for layer in self.moe_layers:
            layer.clear_aux_state()


# =============================================================================
# Loss Function
# =============================================================================

class SageStreamLoss(nn.Module):
    """Combined loss: L_task + alpha*L_KL + beta*L_adv + aux_weight*L_aux."""

    def __init__(self, alpha_kl=1e-3, beta_adv=0.5, aux_weight=0.01,
                 task_type='multiclass'):
        super().__init__()
        self.alpha_kl = alpha_kl
        self.beta_adv = beta_adv
        self.aux_weight = aux_weight
        self.task_type = task_type

    def forward(self, task_logits, labels, mu, log_var,
                subject_logits=None, subject_ids=None, aux_loss=None):
        """
        Returns:
            total_loss: scalar for backward
            loss_dict: dict with individual loss components
        """
        # Task loss
        if self.task_type == 'binary':
            task_loss = F.binary_cross_entropy_with_logits(
                task_logits.squeeze(), labels.float()
            )
        else:
            task_loss = F.cross_entropy(task_logits, labels)

        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        # Adversarial subject loss
        if subject_logits is not None and subject_ids is not None:
            adv_loss = F.cross_entropy(subject_logits, subject_ids)
        else:
            adv_loss = torch.tensor(0.0, device=task_logits.device)

        # MoE auxiliary loss
        if aux_loss is None:
            aux_loss = torch.tensor(0.0, device=task_logits.device)

        total = (task_loss
                 + self.alpha_kl * kl_loss
                 + self.beta_adv * adv_loss
                 + self.aux_weight * aux_loss)

        loss_dict = {
            'total': total.item(),
            'task': task_loss.item(),
            'kl': kl_loss.item(),
            'adv': adv_loss.item(),
            'aux': aux_loss.item(),
        }
        return total, loss_dict


# =============================================================================
# Optimizer Configuration
# =============================================================================

def configure_optimizer(model, lr=1e-3, weight_decay=1e-3):
    """Configure AdamW with grouped learning rates."""
    param_groups = []

    # Backbone adapters (if any) — lower LR
    if model.has_backbone_adapters:
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': lr * 0.1,
                'name': 'backbone_adapters',
            })

    # SA-MoE components (expert pool + router + style alignment)
    moe_params = []
    for name, p in model.expert_pool.named_parameters():
        if p.requires_grad:
            moe_params.append(p)
    for name, p in model.router.named_parameters():
        if p.requires_grad:
            moe_params.append(p)
    for layer in model.moe_layers:
        for name, p in layer.named_parameters():
            # Skip expert_pool and router params (already added above)
            if 'expert_pool' in name or 'router' in name:
                continue
            if p.requires_grad:
                moe_params.append(p)
    if moe_params:
        param_groups.append({
            'params': moe_params,
            'lr': lr,
            'name': 'sa_moe',
        })

    # IIB encoder
    ib_params = [p for p in model.ib_encoder.parameters() if p.requires_grad]
    if ib_params:
        param_groups.append({
            'params': ib_params,
            'lr': lr,
            'name': 'ib_encoder',
        })

    # Task head
    task_params = [p for p in model.task_head.parameters() if p.requires_grad]
    if task_params:
        param_groups.append({
            'params': task_params,
            'lr': lr,
            'name': 'task_head',
        })

    # Subject head — faster learning for adversarial stability
    subj_params = [p for p in model.subject_head.parameters() if p.requires_grad]
    if subj_params:
        param_groups.append({
            'params': subj_params,
            'lr': lr * 2,
            'name': 'subject_head',
        })

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
