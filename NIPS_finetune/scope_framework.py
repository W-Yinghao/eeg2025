"""
SCOPE: Structured COnfidence-aware Prototype-guided adaptation for EEG Foundation Models

Reimplementation based on:
    "Structured Prototype-Guided Adaptation for EEG Foundation Models"
    Ma, Wu et al., arXiv:2602.17251v1, 2026

Two-stage framework:
    Stage 1: External structured supervision construction
        - Task-Prior Network (TPN) with ETF regularization
        - Prototype clustering with Sinkhorn-Knopp balanced assignment
        - Confidence-aware fusion via Dempster-Shafer theory
    Stage 2: Prototype-conditioned adaptation
        - ProAdapter: Feature-wise modulation conditioned on prototypes
        - Warm-up strategy with confidence-weighted semi-supervised loss

Ablation studies (Table 2):
    Supervision construction: w/o ETF-guide, w/o Prototype Clustering, w/o Supervision construction
    ProAdapter design: w/o ProAdapter, w/o Confidence Weights, w/o Prototype Conditioning
    Training strategy: w/o Warm-up, sequential, Two-Stage
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sklearn.cluster import KMeans as _SKLearnKMeans
    SKLEARN_AVAILABLE = True
except (ImportError, AttributeError):
    SKLEARN_AVAILABLE = False


def _simple_kmeans(data: np.ndarray, n_clusters: int, n_init: int = 10,
                   max_iter: int = 100, seed: int = 42) -> np.ndarray:
    """Simple k-means fallback when sklearn is unavailable."""
    rng = np.random.RandomState(seed)
    best_centers = None
    best_inertia = float('inf')

    for _ in range(n_init):
        idx = rng.choice(len(data), min(n_clusters, len(data)), replace=False)
        centers = data[idx].copy()

        for _ in range(max_iter):
            dists = np.linalg.norm(data[:, None] - centers[None], axis=2)
            labels = dists.argmin(axis=1)
            new_centers = np.array([data[labels == k].mean(axis=0)
                                     if (labels == k).any() else centers[k]
                                     for k in range(n_clusters)])
            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers

        inertia = sum(np.linalg.norm(data[labels == k] - centers[k], axis=1).sum()
                      for k in range(n_clusters) if (labels == k).any())
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers

    return best_centers


# =============================================================================
# 1. Task-Prior Network (TPN)
# =============================================================================

class TaskPriorNetwork(nn.Module):
    """Lightweight EEGNet-style 2-block CNN for task-prior induction.

    Architecture (from Appendix J.1, Table 13):
        Block 1: Conv2D → BN → DepthwiseConv2D → BN → ELU → AvgPool → Dropout
        Block 2: SeparableConv2D → Conv2D(1x1) → BN → ELU → AvgPool → Dropout
        Classifier: Linear → K classes

    Also computes ETF regularization loss on classifier weights.
    """

    def __init__(
        self,
        n_channels: int = 16,
        chunk_size: int = 1000,  # T = sampling_rate * segment_duration
        num_classes: int = 5,
        # Architecture hyperparams (dataset-specific, Table 14)
        D: int = 2,
        F1: int = 16,
        F2: int = 32,
        kernel1: int = 192,
        kernel2: int = 48,
        pool1: int = 8,
        pool2: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Block 1: Temporal Conv → DepthwiseConv
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel1), padding=(0, kernel1 // 2), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise conv: each filter applied to each channel independently
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, pool1)),
            nn.Dropout(dropout),
        )

        # Block 2: Separable Conv → Pointwise Conv
        self.block2 = nn.Sequential(
            # Separable conv (depthwise + pointwise)
            nn.Conv2d(F1 * D, F1 * D, (1, kernel2), padding=(0, kernel2 // 2),
                      groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, pool2)),
            nn.Dropout(dropout),
        )

        # Compute output dimension
        T_out = chunk_size // (pool1 * pool2)
        self.feature_dim = F2 * T_out

        # Classifier
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings before classification.

        Args:
            x: (B, C, T) raw EEG or (B, 1, C, T)
        Returns:
            z: (B, feature_dim) embeddings
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)
        elif x.dim() == 4 and x.shape[1] != 1:
            # (B, C, S, P) format from LMDB → reshape to (B, 1, C, S*P)
            B, C, S, P = x.shape
            x = x.reshape(B, C, S * P).unsqueeze(1)

        h = self.block1(x)   # (B, F1*D, 1, T/pool1)
        h = self.block2(h)   # (B, F2, 1, T_out)
        z = h.reshape(h.size(0), -1)  # (B, feature_dim)
        return z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, K)
            embeddings: (B, feature_dim)
        """
        z = self.get_embeddings(x)
        logits = self.classifier(z)
        return logits, z

    def compute_etf_loss(self) -> torch.Tensor:
        """Compute ETF regularization loss on classifier weights (Eq 2).

        L_ETF = ||W̃ᵀW̃ - (K/(K-1)*I - 1/(K-1)*11ᵀ)||²_F
        """
        W = self.classifier.weight  # (K, d)
        W_norm = F.normalize(W, p=2, dim=1)  # (K, d)
        K = self.num_classes

        gram = W_norm @ W_norm.t()  # (K, K)
        target = (K / (K - 1)) * torch.eye(K, device=W.device) - \
                 (1 / (K - 1)) * torch.ones(K, K, device=W.device)

        return (gram - target).pow(2).sum()

    def compute_supervised_contrastive_loss(
        self, embeddings: torch.Tensor, labels: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """Supervised contrastive loss on TPN embeddings (replacement for ETF when K=2).

        Pulls same-class embeddings together and pushes different-class embeddings apart
        in the normalized embedding space. Works for any K including K=2.

        Args:
            embeddings: (B, d) TPN feature embeddings
            labels: (B,) class labels
            temperature: Temperature scaling (default 0.1)
        Returns:
            loss: scalar contrastive loss
        """
        z = F.normalize(embeddings, p=2, dim=1)  # (B, d)
        B = z.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=z.device)

        # Pairwise cosine similarity
        sim = z @ z.t() / temperature  # (B, B)

        # Mask: same class = positive, different class = negative
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        # Exclude self-pairs
        self_mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
        pos_mask = label_eq & self_mask
        neg_mask = ~label_eq & self_mask

        # Check we have both positives and negatives
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=z.device)

        # For numerical stability, subtract max
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # log-sum-exp over all non-self pairs (denominator)
        exp_sim = torch.exp(sim) * self_mask.float()
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)  # (B, 1)

        # Mean of log-prob over positive pairs
        log_prob = sim - log_denom  # (B, B)

        # Average over positive pairs for each anchor
        n_pos = pos_mask.float().sum(dim=1).clamp(min=1)  # (B,)
        loss = -(log_prob * pos_mask.float()).sum(dim=1) / n_pos  # (B,)

        return loss.mean()


# =============================================================================
# 2. Prototype Bank with Sinkhorn-Knopp Assignment
# =============================================================================

class PrototypeBank(nn.Module):
    """Learnable class-level prototypes with balanced assignment.

    Maintains M prototypes per class. Uses Sinkhorn-Knopp to enforce balanced
    sample-to-prototype assignment.

    Args:
        feature_dim: Embedding dimension from TPN
        num_classes: Number of classes
        num_prototypes_per_class: M prototypes per class (default 3)
        temperature: Temperature for similarity computation (default 10.0)
        sinkhorn_iters: Number of Sinkhorn iterations (default 3)
        sinkhorn_epsilon: Sinkhorn regularization (default 0.05)
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        num_prototypes_per_class: int = 3,
        temperature: float = 10.0,
        sinkhorn_iters: int = 3,
        sinkhorn_epsilon: float = 0.05,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.M = num_prototypes_per_class
        self.temperature = temperature
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_epsilon = sinkhorn_epsilon

        # Total prototypes: K * M
        total = num_classes * num_prototypes_per_class
        self.prototypes = nn.Parameter(torch.randn(total, feature_dim) * 0.01)

    def initialize_from_embeddings(self, embeddings: np.ndarray, labels: np.ndarray,
                                   max_per_class: int = 5000, n_init: int = 10):
        """Initialize prototypes via k-means on TPN embeddings grouped by predicted labels.

        Args:
            embeddings: (N, d) numpy array of TPN embeddings
            labels: (N,) numpy array of predicted prior labels
            max_per_class: Max samples per class for k-means
            n_init: Number of k-means initializations
        """
        with torch.no_grad():
            for k in range(self.num_classes):
                mask = labels == k
                class_embs = embeddings[mask]
                if len(class_embs) == 0:
                    continue

                # Subsample if too many
                if len(class_embs) > max_per_class:
                    idx = np.random.choice(len(class_embs), max_per_class, replace=False)
                    class_embs = class_embs[idx]

                n_clusters = min(self.M, len(class_embs))
                if SKLEARN_AVAILABLE:
                    kmeans = _SKLearnKMeans(n_clusters=n_clusters, n_init=n_init, random_state=42)
                    kmeans.fit(class_embs)
                    centers = kmeans.cluster_centers_
                else:
                    centers = _simple_kmeans(class_embs, n_clusters, n_init=n_init)

                centroids = torch.from_numpy(centers).float()
                start = k * self.M
                self.prototypes.data[start:start + n_clusters] = centroids

        print(f"  Prototypes initialized via k-means: {self.num_classes} classes x {self.M} prototypes")

    def compute_similarity(self, z: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between embeddings and all prototypes.

        Args:
            z: (B, d) embeddings
        Returns:
            sim: (B, K*M) similarity scores
        """
        z_norm = F.normalize(z, p=2, dim=1)
        p_norm = F.normalize(self.prototypes, p=2, dim=1)
        return z_norm @ p_norm.t()  # (B, K*M)

    def sinkhorn_assignment(self, sim_class: torch.Tensor) -> torch.Tensor:
        """Apply Sinkhorn-Knopp to get balanced assignment within a class.

        Args:
            sim_class: (B_k, M) similarity scores for samples of class k
        Returns:
            Q: (B_k, M) balanced soft assignment
        """
        B_k, M = sim_class.shape
        if B_k == 0 or M == 0:
            return sim_class

        Q = torch.exp(sim_class / self.sinkhorn_epsilon)

        # Sinkhorn iterations
        for _ in range(self.sinkhorn_iters):
            # Row normalization: each sample sums to 1
            Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-12)
            # Column normalization: each prototype gets B_k/M samples
            Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-12) * (B_k / M)

        # Final row normalization
        Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-12)
        return Q

    def compute_prototype_loss(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute prototype learning loss (Eq 7).

        Cross-entropy between Sinkhorn assignment and similarity-based softmax.

        Args:
            z: (B, d) embeddings
            labels: (B,) class labels
        Returns:
            loss: scalar
        """
        sim = self.compute_similarity(z)  # (B, K*M)
        loss = torch.tensor(0.0, device=z.device)
        n_classes_seen = 0

        for k in range(self.num_classes):
            mask = labels == k
            if mask.sum() == 0:
                continue

            # Get similarity to class k prototypes
            start, end = k * self.M, (k + 1) * self.M
            sim_k = sim[mask, start:end]  # (B_k, M)

            # Sinkhorn balanced assignment (target)
            with torch.no_grad():
                Q_k = self.sinkhorn_assignment(sim_k.detach())

            # Cross-entropy: CE(Q_k, softmax(sim_k / temperature))
            log_probs = F.log_softmax(sim_k / self.temperature, dim=1)
            loss_k = -(Q_k * log_probs).sum(dim=1).mean()
            loss = loss + loss_k
            n_classes_seen += 1

        return loss / max(n_classes_seen, 1)

    def predict(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get prototype-based class predictions.

        Args:
            z: (B, d) embeddings
        Returns:
            pred_labels: (B,) predicted class labels
            class_sim: (B, K) max similarity per class (used for fusion)
        """
        sim = self.compute_similarity(z)  # (B, K*M)
        # Reshape to (B, K, M) and take max across prototypes
        sim_reshaped = sim.view(-1, self.num_classes, self.M)
        class_sim, _ = sim_reshaped.max(dim=2)  # (B, K)
        pred_labels = class_sim.argmax(dim=1)    # (B,)
        return pred_labels, class_sim


# =============================================================================
# 3. Confidence-Aware Fusion (Dempster-Shafer)
# =============================================================================

class ConfidenceAwareFusion:
    """Fuse TPN and Prototype predictions via Dempster-Shafer theory.

    - Agreement check: pseudo-label assigned only when both sources agree
    - Dempster-Shafer combination of belief assignments
    - Entropy-based confidence score
    - Threshold filtering

    Args:
        num_classes: Number of classes
        confidence_threshold: Minimum confidence to accept pseudo-label (default 0.5)
        eps: Numerical stability epsilon
    """

    def __init__(self, num_classes: int, confidence_threshold: float = 0.5, eps: float = 1e-12):
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.eps = eps

    def fuse(
        self,
        prior_logits: torch.Tensor,
        prototype_sim: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fuse two prediction sources.

        Args:
            prior_logits: (N, K) TPN logits
            prototype_sim: (N, K) prototype similarity scores

        Returns:
            pseudo_labels: (N,) pseudo-labels (-1 for rejected samples)
            confidence: (N,) confidence scores
            agreement_mask: (N,) boolean mask (True = agreed)
        """
        # Convert to probability distributions (BBAs)
        m1 = F.softmax(prior_logits, dim=1)       # (N, K) TPN
        m2 = F.softmax(prototype_sim, dim=1)       # (N, K) Prototype

        # Check agreement
        prior_pred = prior_logits.argmax(dim=1)     # (N,)
        proto_pred = prototype_sim.argmax(dim=1)    # (N,)
        agreement_mask = (prior_pred == proto_pred)  # (N,)

        # Dempster-Shafer combination (singleton-only, Lemma 3.2)
        # m(ωc) = m1(ωc) * m2(ωc) / Σ_j m1(ωj) * m2(ωj)
        product = m1 * m2  # (N, K)
        normalizer = product.sum(dim=1, keepdim=True).clamp(min=self.eps)  # (N, 1)
        m_fused = product / normalizer  # (N, K)

        # Entropy-based confidence (Eq 9)
        # γ = 1 - H(m) / log(K)
        log_m = torch.log(m_fused.clamp(min=self.eps))
        entropy = -(m_fused * log_m).sum(dim=1)  # (N,)
        confidence = 1.0 - entropy / math.log(max(self.num_classes, 2))

        # Pseudo-labels
        pseudo_labels = m_fused.argmax(dim=1)  # (N,)

        # Apply agreement mask
        pseudo_labels[~agreement_mask] = -1

        # Apply confidence threshold
        low_conf = confidence < self.confidence_threshold
        pseudo_labels[low_conf] = -1

        return pseudo_labels, confidence, agreement_mask


# =============================================================================
# 4. ProAdapter: Prototype-Conditioned Feature-wise Modulation
# =============================================================================

class ProAdapter(nn.Module):
    """Prototype-conditioned feature-wise modulation adapter.

    Inserted into the last L layers of a frozen backbone. Performs:
        ProAdapter(h; ϖ) = α ⊙ h + β

    where α, β are generated from:
        1. Self-conditioning: c = [mean(h), std(h)] → Linear → [Δα⁰, β⁰]
        2. Prototype-conditioning: ϖ → gated modulation

    Args:
        feature_dim: Backbone feature dimension (e.g., 200 for CodeBrain/CBraMod)
        num_classes: Number of classes (for prototype similarity vector)
        lambda_proto: Scaling factor for prototype modulation (λ_ϖ)
        lambda_scale: Scaling factor for final alpha (λ)
    """

    def __init__(
        self,
        feature_dim: int = 200,
        num_classes: int = 5,
        lambda_proto: float = 0.1,
        lambda_scale: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.lambda_proto = lambda_proto
        self.lambda_scale = lambda_scale

        # Self-conditioning branch: [μ, σ] → [Δα⁰, β⁰]
        self.self_cond = nn.Linear(2 * feature_dim, 2 * feature_dim)

        # Prototype-conditioning branch: ϖ → modulation + gate
        self.proto_modulation = nn.Linear(num_classes, 2 * feature_dim)
        self.proto_gate = nn.Linear(num_classes, 2 * feature_dim)

        # Initialize to near-identity
        nn.init.zeros_(self.self_cond.weight)
        nn.init.zeros_(self.self_cond.bias)
        nn.init.zeros_(self.proto_modulation.weight)
        nn.init.zeros_(self.proto_modulation.bias)
        nn.init.zeros_(self.proto_gate.weight)
        nn.init.zeros_(self.proto_gate.bias)

    def forward(self, h: torch.Tensor, proto_sim: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            h: (B, ..., D) intermediate backbone representation
                For CodeBrain (SSSM): (B, C*S, D) after reshape, or (B, C, S, D)
                For CBraMod: (B, C, S, D)
            proto_sim: (B, K) prototype similarity vector (ϖ). None = no prototype conditioning.
        Returns:
            h_modulated: same shape as h
        """
        original_shape = h.shape
        D = h.shape[-1]

        # Compute temporal statistics: mean and std over all dims except batch and feature
        if h.dim() == 4:
            # (B, C, S, D) → compute stats over C and S
            h_flat = h.reshape(h.shape[0], -1, D)  # (B, C*S, D)
        elif h.dim() == 3:
            h_flat = h  # (B, T, D)
        else:
            h_flat = h.unsqueeze(1)  # (B, 1, D)

        mu = h_flat.mean(dim=1)     # (B, D)
        sigma = h_flat.std(dim=1)   # (B, D)

        # Self-conditioning
        c = torch.cat([mu, sigma], dim=1)  # (B, 2D)
        self_mod = self.self_cond(c)        # (B, 2D)
        delta_alpha_0, beta_0 = self_mod.chunk(2, dim=1)  # each (B, D)

        if proto_sim is not None:
            # Prototype-conditioning (Eq 12)
            proto_mod = self.lambda_proto * torch.tanh(self.proto_modulation(proto_sim))  # (B, 2D)
            gate = torch.sigmoid(self.proto_gate(proto_sim))  # (B, 2D)

            delta_alpha_proto, beta_proto = proto_mod.chunk(2, dim=1)  # each (B, D)
            gate_alpha, gate_beta = gate.chunk(2, dim=1)                # each (B, D)

            delta_alpha = delta_alpha_0 + gate_alpha * delta_alpha_proto
            beta = beta_0 + gate_beta * beta_proto
        else:
            delta_alpha = delta_alpha_0
            beta = beta_0

        # Final scale: α = 1 + λ * tanh(Δα)
        alpha = 1.0 + self.lambda_scale * torch.tanh(delta_alpha)  # (B, D)

        # Reshape for broadcasting
        if h.dim() == 4:
            alpha = alpha.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, D)
            beta = beta.unsqueeze(1).unsqueeze(1)
        elif h.dim() == 3:
            alpha = alpha.unsqueeze(1)  # (B, 1, D)
            beta = beta.unsqueeze(1)

        return alpha * h + beta


# =============================================================================
# 5. SCOPE Model (wrapping frozen backbone + ProAdapters)
# =============================================================================

class SCOPEModel(nn.Module):
    """SCOPE: Frozen EFM backbone with ProAdapters in last L layers + classifier.

    Architecture-agnostic: works with both Transformer and SSM backbones.
    ProAdapters are inserted after the backbone's layer outputs.

    For CodeBrain (SSSM): modulates residual layer outputs
    For CBraMod (Transformer): modulates transformer layer outputs

    Two classifier modes:
      - pooling (default): pool (B,C,S,D)→(B,D) then compact MLP (~3-10% params)
      - flatten: flatten (B,C,S,D)→(B,C*S*D) then 3-layer MLP (original, high param count)

    Args:
        backbone: Frozen EFM backbone
        num_classes: Number of classes
        backbone_out_dim: Flattened output dimension (used only by flatten classifier)
        token_dim: Per-token dimension (200)
        n_channels: Number of EEG channels
        seq_len: Number of temporal segments
        adapter_layers: Number of layers to insert adapters (L=3)
        classifier_hidden: Hidden dimension for classifier MLP
        dropout: Dropout rate
        use_prototype_conditioning: Whether to use prototype conditioning in ProAdapter
        pooling_classifier: If True (default), use pooling+compact MLP; if False,
            use the original flatten-based 3-layer MLP
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        backbone_out_dim: int,
        token_dim: int = 200,
        n_channels: int = 16,
        seq_len: int = 5,
        adapter_layers: int = 3,
        classifier_hidden: int = 200,
        dropout: float = 0.1,
        use_prototype_conditioning: bool = True,
        lambda_proto: float = 0.1,
        lambda_scale: float = 0.1,
        pooling_classifier: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.token_dim = token_dim
        self.adapter_layers = adapter_layers
        self.use_prototype_conditioning = use_prototype_conditioning
        self.pooling_classifier = pooling_classifier

        # Create ProAdapters for last L layers
        self.adapters = nn.ModuleList([
            ProAdapter(
                feature_dim=token_dim,
                num_classes=num_classes,
                lambda_proto=lambda_proto,
                lambda_scale=lambda_scale,
            )
            for _ in range(adapter_layers)
        ])

        out_dim = 1 if num_classes == 2 else num_classes

        if pooling_classifier:
            # Lightweight: pool (B,C,S,D)→(B,D), then compact 2-layer MLP.
            # Keeps trainable ratio at ~3-10% as recommended by the SCOPE paper.
            self.classifier = nn.Sequential(
                nn.Linear(token_dim, classifier_hidden),
                nn.ELU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden, out_dim),
            )
        else:
            # Original: flatten (B,C,S,D)→(B,C*S*D), 3-layer MLP.
            # Warning: high param count (50-93% trainable ratio).
            self.classifier = nn.Sequential(
                nn.Linear(backbone_out_dim, seq_len * token_dim),
                nn.ELU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(seq_len * token_dim, token_dim),
                nn.ELU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(token_dim, out_dim),
            )

    def _apply_adapters_codebrain(self, x: torch.Tensor,
                                   proto_sim: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward through CodeBrain (SSSM) with ProAdapter injection.

        CodeBrain's SSSM has residual blocks. We insert ProAdapters after the
        last L residual blocks.

        Args:
            x: (B, C, S, P) input EEG data
            proto_sim: (B, K) prototype similarity (or None)
        """
        B, C, S, P = x.shape
        backbone = self.backbone

        # Reshape for SSSM: (B, C, S, P) → (B*C*S, P) or appropriate format
        # CodeBrain's SSSM expects (B, D, L) where D=200, L=C*S
        x_reshaped = x.permute(0, 3, 1, 2).reshape(B, P, C * S)  # (B, 200, C*S)

        # Access residual layers
        if hasattr(backbone, 'residual_layers'):
            res_layers = backbone.residual_layers
        elif hasattr(backbone, 'res_layers'):
            res_layers = backbone.res_layers
        else:
            # Fallback: just run backbone normally and apply adapters to output
            with torch.no_grad():
                h = backbone(x)  # (B, C, S, D)
            # Restore batch dim if squeezed away (e.g. CodeBrain SSSM when B=1)
            if h.dim() == 3 and B == 1:
                h = h.unsqueeze(0)
            for adapter in self.adapters:
                h = adapter(h, proto_sim)
            return h

        n_layers = len(res_layers)
        adapter_start = max(0, n_layers - self.adapter_layers)

        # Manual forward through residual layers
        h = x_reshaped
        # Run through initial layers if backbone has them
        if hasattr(backbone, 'init_conv'):
            h = backbone.init_conv(h)

        for i, layer in enumerate(res_layers):
            h = layer(h)
            if i >= adapter_start:
                adapter_idx = i - adapter_start
                if adapter_idx < len(self.adapters):
                    # Reshape for adapter: (B, D, L) → (B, L, D) → adapter → (B, D, L)
                    h_t = h.permute(0, 2, 1)  # (B, L, D)
                    h_t = self.adapters[adapter_idx](h_t, proto_sim)
                    h = h_t.permute(0, 2, 1)  # (B, D, L)

        # Final processing
        if hasattr(backbone, 'final_conv'):
            h = backbone.final_conv(h)

        # Reshape back to (B, C, S, D)
        h = h.reshape(B, P, C, S).permute(0, 2, 3, 1)
        return h

    def _apply_adapters_simple(self, x: torch.Tensor,
                                proto_sim: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Simple adapter application: run backbone, then apply adapters to output.

        Used when we can't easily inject into backbone layers.
        """
        B = x.shape[0]
        with torch.no_grad():
            h = self.backbone(x)  # expected (B, C, S, D)

        # Some backbones (e.g. CodeBrain SSSM) call .squeeze() on output,
        # which removes the batch dim when B=1. Restore it to ensure 4D.
        if h.dim() == 3 and B == 1:
            h = h.unsqueeze(0)

        for adapter in self.adapters:
            h = adapter(h, proto_sim)

        return h

    def forward(self, x: torch.Tensor,
                proto_sim: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, S, P) EEG input
            proto_sim: (B, K) prototype similarity vector (from Stage 1)
        Returns:
            logits: (B, num_classes) or (B, 1) for binary
        """
        if not self.use_prototype_conditioning:
            proto_sim = None

        # Apply backbone with adapters → (B, C, S, D)
        h = self._apply_adapters_simple(x, proto_sim)

        if self.pooling_classifier:
            # Pool over channels and sequence → (B, D)
            if h.dim() == 4:
                h_input = h.mean(dim=[1, 2])         # (B, C, S, D) → (B, D)
            elif h.dim() == 3:
                h_input = h.mean(dim=1)               # (B, T, D) → (B, D)
            else:
                h_input = h                            # (B, D)
        else:
            # Flatten all spatial/temporal dims
            h_input = h.reshape(h.shape[0], -1)        # (B, C*S*D)
            expected_dim = self.classifier[0].in_features
            if h_input.shape[1] != expected_dim:
                if h_input.shape[1] > expected_dim:
                    h_input = h_input[:, :expected_dim]
                else:
                    padding = torch.zeros(h_input.shape[0],
                                          expected_dim - h_input.shape[1],
                                          device=h_input.device)
                    h_input = torch.cat([h_input, padding], dim=1)

        logits = self.classifier(h_input)
        return logits

    def get_trainable_params(self):
        """Get only trainable parameters (adapters + classifier)."""
        params = []
        for adapter in self.adapters:
            params.extend(adapter.parameters())
        params.extend(self.classifier.parameters())
        return params

    def get_param_groups(self, lr: float, weight_decay: float = 0.01):
        """Get parameter groups for optimizer."""
        adapter_params = []
        classifier_params = []
        for adapter in self.adapters:
            adapter_params.extend(adapter.parameters())
        classifier_params.extend(self.classifier.parameters())

        return [
            {'params': adapter_params, 'lr': lr, 'weight_decay': weight_decay},
            {'params': classifier_params, 'lr': lr, 'weight_decay': weight_decay},
        ]


# =============================================================================
# 6. TPN Configuration Presets (from Table 14)
# =============================================================================

# Dataset-specific TPN hyperparameters
TPN_CONFIGS = {
    # Sleep staging (ISRUC-like): 6ch, 30s@200Hz = 6000
    'ISRUC': {'D': 2, 'F1': 16, 'F2': 32, 'kernel1': 192, 'kernel2': 48,
              'pool1': 8, 'pool2': 8, 'dropout': 0.1},
    # Emotion (SEED-like): 62ch, 4s@200Hz = 800
    'SEED': {'D': 2, 'F1': 4, 'F2': 8, 'kernel1': 128, 'kernel2': 64,
             'pool1': 2, 'pool2': 4, 'dropout': 0.1},
    # Mental Arithmetic: 20ch, 5s@200Hz = 1000
    'MENTAL': {'D': 1, 'F1': 8, 'F2': 8, 'kernel1': 256, 'kernel2': 32,
               'pool1': 4, 'pool2': 8, 'dropout': 0.1},
    # Default for our datasets
    'DEFAULT': {'D': 2, 'F1': 8, 'F2': 16, 'kernel1': 64, 'kernel2': 16,
                'pool1': 4, 'pool2': 4, 'dropout': 0.1},
}


def get_tpn_config(dataset_name: str) -> dict:
    """Get TPN architecture config for a given dataset."""
    name_upper = dataset_name.upper()
    if name_upper in TPN_CONFIGS:
        return TPN_CONFIGS[name_upper]

    # Map our datasets to closest paper dataset
    if name_upper in ('TUEV',):
        return TPN_CONFIGS['DEFAULT']
    elif name_upper in ('TUAB', 'TUSZ', 'CHB-MIT'):
        return TPN_CONFIGS['DEFAULT']
    elif name_upper in ('DIAGNOSIS', 'AD_DIAGNOSIS', 'UNIFIED_DIAGNOSIS'):
        return TPN_CONFIGS['DEFAULT']
    elif name_upper in ('DEPRESSION', 'CVD_DEPRESSION_NORMAL'):
        return TPN_CONFIGS['DEFAULT']
    else:
        return TPN_CONFIGS['DEFAULT']


# =============================================================================
# 7. Factory: Create SCOPE components
# =============================================================================

def create_tpn(
    n_channels: int,
    chunk_size: int,
    num_classes: int,
    dataset_name: str = 'DEFAULT',
    dropout: float = 0.1,
) -> TaskPriorNetwork:
    """Create a Task-Prior Network with dataset-specific configuration."""
    config = get_tpn_config(dataset_name)
    config['dropout'] = dropout

    tpn = TaskPriorNetwork(
        n_channels=n_channels,
        chunk_size=chunk_size,
        num_classes=num_classes,
        **config,
    )

    total = sum(p.numel() for p in tpn.parameters())
    print(f"\nTask-Prior Network:")
    print(f"  n_channels={n_channels}, chunk_size={chunk_size}, num_classes={num_classes}")
    print(f"  Config: {config}")
    print(f"  Total params: {total:,}")
    print(f"  Feature dim: {tpn.feature_dim}")
    print()

    return tpn


def create_scope_model(
    backbone: nn.Module,
    num_classes: int,
    backbone_out_dim: int,
    token_dim: int = 200,
    n_channels: int = 16,
    seq_len: int = 5,
    adapter_layers: int = 3,
    dropout: float = 0.1,
    use_prototype_conditioning: bool = True,
    lambda_proto: float = 0.1,
    lambda_scale: float = 0.1,
    pooling_classifier: bool = True,
) -> SCOPEModel:
    """Create SCOPE model with frozen backbone and ProAdapters.

    Args:
        pooling_classifier: If True (default), pool before classifier for
            lightweight param count (~3-10%). If False, use original flatten-based
            3-layer MLP (high param count, 50-93%).
    """
    model = SCOPEModel(
        backbone=backbone,
        num_classes=num_classes,
        backbone_out_dim=backbone_out_dim,
        token_dim=token_dim,
        n_channels=n_channels,
        seq_len=seq_len,
        adapter_layers=adapter_layers,
        dropout=dropout,
        use_prototype_conditioning=use_prototype_conditioning,
        lambda_proto=lambda_proto,
        lambda_scale=lambda_scale,
        pooling_classifier=pooling_classifier,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    print(f"\nSCOPE Model:")
    print(f"  Adapter layers: {adapter_layers}")
    print(f"  Prototype conditioning: {use_prototype_conditioning}")
    print(f"  Total params:     {total:,}")
    print(f"  Frozen params:    {frozen:,}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Trainable ratio:  {100 * trainable / total:.2f}%")
    print()

    return model
