# USBA: Universal Sufficient Bottleneck Adapter

## 1. Overview

USBA is a parameter-efficient fine-tuning (PEFT) method for frozen pretrained EEG foundation models. It inserts a lightweight adapter between the frozen backbone output and the classification head, using a variational information bottleneck to compress representations into class-sufficient statistics while discarding subject-specific nuisance information.

**Supported backbones:** CBraMod (criss-cross transformer), CodeBrain/SSSM (state-space model), LUNA (cross-attention transformer)

**Key files:**
- `adapters/usba_config.py` — Configuration dataclass
- `adapters/branches.py` — Temporal and spatial branches + gated fusion
- `adapters/bottleneck.py` — Variational information bottleneck + residual gate
- `adapters/usba.py` — USBALayer and USBAAdapter (multi-layer stack)
- `adapters/losses.py` — Combined loss with KL, cc-HSIC, budget regularization
- `adapters/injection.py` — USBAInjector + USBAInjectedModel (backbone wrapping)
- `train_usba.py` — Training script

---

## 2. Architecture

### 2.1 Data Flow

```
Input EEG: (B, C, S, P)
  e.g. TUEV: (64, 16, 5, 200), DIAGNOSIS: (64, 58, 5, 200)

       ↓
┌──────────────────────────────┐
│  Frozen Backbone              │  All parameters frozen, no gradient
│  Output: (B, C, S, D)        │  D = token_dim = 200
└──────────────────────────────┘
       ↓ reshape to (B, T, D)     T = C × S
       ↓                          TUEV: T = 16×5 = 80
       ↓                          DIAGNOSIS: T = 58×5 = 290
┌──────────────────────────────┐
│  USBA Adapter                 │  Trainable
│  ┌────────────────────────┐  │
│  │ (1) LayerNorm(D)       │  │
│  │ (2) Temporal Branch    │──┐│
│  │ (3) Spatial Branch     │──┤│
│  │ (4) Gated Fusion       │←─┘│
│  │ (5) Variational BN     │  │
│  │ (6) Residual Gate      │  │
│  └────────────────────────┘  │
└──────────────────────────────┘
       ↓ (B, T, D)
       ↓ mean pool over T
       ↓ (B, D)
┌──────────────────────────────┐
│  Classification Head          │  Trainable
│  Linear(D, 2D) → BN → GELU  │
│  → Dropout → Linear(2D, K)   │
└──────────────────────────────┘
       ↓ (B, K)
```

### 2.2 USBALayer Detail

Given backbone hidden states $H_l \in \mathbb{R}^{B \times T \times D}$:

**Step 1: Normalize**
$$\tilde{H} = \text{LayerNorm}(H_l)$$

**Step 2: Factorized Branches**
$$T_{\text{out}} = f_{\text{temporal}}(\tilde{H}), \quad S_{\text{out}} = f_{\text{spatial}}(\tilde{H})$$

**Step 3: Gated Fusion**
$$(g_t, g_s) = \sigma\big(W_g \cdot \text{MeanPool}(\tilde{H})\big) \in [0,1]^2$$
$$F_l = g_t \cdot T_{\text{out}} + g_s \cdot S_{\text{out}} + \max(1 - g_t - g_s, 0) \cdot \tilde{H}$$

**Step 4: Variational Bottleneck**
$$a = \text{GELU}(W_{\text{enc}} \cdot F_l)$$
$$\mu = W_\mu \cdot a, \quad \log\sigma^2 = \text{clamp}(W_\sigma \cdot a, -10, 10)$$
$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) \quad \text{(training only; eval uses } z = \mu\text{)}$$
$$\delta = W_{\text{dec}}(z) \in \mathbb{R}^{B \times T \times D}$$

**Step 5: Residual Write-Back**
$$H_{\text{adapt}} = H_l + g \cdot \delta$$

where $g = \sigma(g_{\text{logit}})$ is a learnable gate (see Section 3.3).

---

## 3. Component Details

### 3.1 Temporal Branch

**Default: Depthwise Temporal Convolution** (`DepthwiseTemporalConv`)

Architecture:
```
Conv1d(D, D, kernel=5, padding=2, groups=D)  →  LayerNorm  →  GELU  →  Dropout
```

In **4D-aware mode** (CBraMod, CodeBrain):
- Input $(B, T, D)$ where $T = C \times S$
- Reshape to $(B \cdot C, S, D)$
- Apply conv along $S$ axis only (within each channel)
- Reshape back to $(B, T, D)$

This ensures the convolution kernel **never crosses channel boundaries** — it captures intra-channel temporal dynamics.

In **3D mode** (LUNA):
- Apply conv along $T$ directly

**Alternative: Low-Rank Temporal Mix** (`LowRankTemporalMix`)

Factorized $T \times T$ mixing matrix: $W_{\text{up}} \cdot W_{\text{down}}$ where $W_{\text{down}} \in \mathbb{R}^{T \times r}$, $W_{\text{up}} \in \mathbb{R}^{r \times T}$, $r = 16$.

### 3.2 Spatial Branch

**Default: Channel Attention** (`ChannelAttention`)

Squeeze-and-Excitation style:
```
MeanPool over axis  →  Linear(D, D/4)  →  GELU  →  Dropout  →  Linear(D/4, D)  →  Sigmoid  →  Scale
```

In **4D-aware mode** (CBraMod, CodeBrain):
- Reshape $(B, C \cdot S, D)$ → $(B \cdot S, C, D)$
- SE operates along $C$ axis: pool over channels, compute gate, apply per-channel scaling
- Each time step gets independent cross-channel attention
- Reshape back to $(B, T, D)$

This mirrors CBraMod's spatial attention path which attends across channels independently per time step.

In **3D mode** (LUNA):
- SE along $D$ axis (feature importance)

**Alternative: Grouped Spatial MLP** (`GroupedSpatialMLP`)

Splits $D$ into 4 groups, applies independent MLPs per group.

### 3.3 Gated Fusion

$$\mathbf{g} = \sigma\big(W_g \cdot \bar{H}\big) \in \mathbb{R}^{B \times 2}$$

where $\bar{H} = \frac{1}{T}\sum_t \tilde{H}_t$ (mean pool).

$$F = g_t \cdot T_{\text{out}} + g_s \cdot S_{\text{out}} + \text{clamp}(1 - g_t - g_s, \min=0) \cdot \tilde{H}$$

**Initialization:** $W_g = 0, b_g = 0$ → $\sigma(0) = 0.5$ → early training assigns roughly equal weight to all three terms.

**Interpretability:** Gate values are logged per batch, showing whether temporal or spatial adaptation dominates.

### 3.4 Variational Information Bottleneck

The bottleneck compresses fused features $F_l \in \mathbb{R}^D$ to a latent $z \in \mathbb{R}^{d'}$ where $d' \ll D$ (64 vs 200).

**Encoder:**
$$a = \text{GELU}(W_{\text{enc}} F_l + b_{\text{enc}})$$
$$\mu = W_\mu a + b_\mu \in \mathbb{R}^{d'}, \quad \log\sigma^2 = W_\sigma a + b_\sigma \in \mathbb{R}^{d'}$$

**Reparameterization:**
$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**Decoder:**
$$\delta = W_{\text{dec}} z + b_{\text{dec}} \in \mathbb{R}^D$$

**Key initialization:**
- $W_{\text{dec}}$ and $b_{\text{dec}}$ are **zero-initialized** → at training start, $\delta = 0$, so the adapter is an identity transformation
- This ensures stable early training

**KL divergence** (per-token, averaged over batch):
$$D_{\text{KL}}\big[q(z|x) \| p(z)\big] = -\frac{1}{2BT}\sum_{b,t}\sum_{j=1}^{d'}\Big(1 + \log\sigma^2_{b,t,j} - \mu^2_{b,t,j} - \sigma^2_{b,t,j}\Big)$$

where $p(z) = \mathcal{N}(0, I)$ is the standard Gaussian prior.

### 3.5 Residual Gate

Three granularity options:

| Gate Type | Parameter Shape | Behavior |
|-----------|----------------|----------|
| `layer_wise` (default) | scalar $g_{\text{logit}} \in \mathbb{R}$ | Single gate for entire layer |
| `token_wise` | $g_{\text{logit}} \in \mathbb{R}^1$ (broadcast to $T$) | Per-position gate |
| `channel_wise` | $g_{\text{logit}} \in \mathbb{R}^D$ | Per-feature gate |

$$g = \sigma(g_{\text{logit}})$$
$$H_{\text{adapt}} = H_{\text{original}} + g \cdot \delta$$

**Initialization:** $g_{\text{logit}} = 0$ → $g = 0.5$

---

## 4. Loss Function

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \beta \cdot \mathcal{L}_{\text{KL}} + \lambda \cdot \mathcal{L}_{\text{cc-inv}} + \eta \cdot \mathcal{L}_{\text{budget}}$$

### 4.1 Task Loss

$$\mathcal{L}_{\text{task}} = \text{CrossEntropy}(\hat{y}, y)$$

Used for both binary (TUAB, TUSZ) and multiclass (TUEV, DIAGNOSIS) tasks with 2 or K output logits respectively.

### 4.2 KL Divergence (Information Bottleneck)

$$\mathcal{L}_{\text{KL}} = \beta_{\text{eff}} \cdot \sum_l D_{\text{KL}}^{(l)}$$

with linear warmup:
$$\beta_{\text{eff}} = \beta \cdot \min\Big(\frac{\text{epoch}}{\text{warmup\_epochs}}, 1\Big)$$

**Theoretical motivation:** From the Information Bottleneck principle, we seek representations $Z$ that maximize $I(Z; Y)$ (task-relevant information) while minimizing $I(Z; X)$ (total information from input). The KL term upper-bounds $I(Z; X)$, forcing the bottleneck to discard nuisance information.

### 4.3 Class-Conditional HSIC Invariance

$$\mathcal{L}_{\text{cc-inv}} = \frac{1}{|\mathcal{Y}|}\sum_{y \in \mathcal{Y}} \text{HSIC}(Z_y, S_y)$$

where $Z_y, S_y$ are the latent representations and subject IDs for samples with class label $y$.

**HSIC (Hilbert-Schmidt Independence Criterion):**

For kernel matrices $K_Z, K_S \in \mathbb{R}^{N \times N}$:
$$\text{HSIC}(Z, S) = \frac{1}{(N-1)^2}\text{tr}(K_Z H K_S H)$$

where $H = I - \frac{1}{N}\mathbf{1}\mathbf{1}^\top$ is the centering matrix, and kernels are RBF:
$$K_Z(i,j) = \exp\Big(-\frac{\|z_i - z_j\|^2}{2\sigma_z^2}\Big), \quad K_S(i,j) = \exp\Big(-\frac{(s_i - s_j)^2}{2\sigma_s^2}\Big)$$

**Why class-conditional?** Global HSIC$(Z, S)$ would penalize any subject-dependent structure, including legitimate class-subject correlations (e.g., if certain subjects have specific diseases). By conditioning on class label, we only penalize subject information that is redundant given the disease label — the nuisance variation.

**Minimum sample guard:** Requires $\geq 4$ samples per class, $\geq 2$ unique subjects, and $\geq 2$ unique labels. Auto-disabled otherwise.

### 4.4 Budget Regularization

$$\mathcal{L}_{\text{budget}} = \frac{1}{L}\sum_{l=1}^{L} \sigma(g_{\text{logit}}^{(l)})$$

L1 penalty on gate activations, encouraging the adapter to stay close to identity where modification is unnecessary.

---

## 5. Hyperparameters

### 5.1 Current Default Settings

| Parameter | Value | Source |
|-----------|-------|--------|
| `latent_dim` | 64 | `usba_config.py` |
| `gate_type` | `layer_wise` | `usba_config.py` |
| `gate_init` | 0.0 (→ sigmoid = 0.5) | `usba_config.py` |
| `temporal_branch` | `depthwise_conv`, kernel=5 | `usba_config.py` |
| `spatial_branch` | `channel_attention`, reduction=4 | `usba_config.py` |
| `beta` (KL weight) | 1e-4 | `exp_record.txt` |
| `beta_warmup` | 5 epochs | `usba_config.py` |
| `lambda_cc_inv` | 0.01 | `exp_record.txt` |
| `eta_budget` | 1e-3 | `exp_record.txt` |
| `dropout` | 0.1 | `usba_config.py` |
| `factorized` | True | `usba_config.py` |

### 5.2 Training Settings (from `run_usba.sh`)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR | 1e-3 (all datasets) |
| Weight decay | 1e-3 |
| Epochs | 100 |
| Patience | 15 (early stopping on val balanced accuracy) |
| Gradient clipping | 5.0 |
| Scheduler | CosineAnnealingLR |
| Batch size | 64 |
| Seed | 3407 |

### 5.3 Per-Dataset Configuration

| Dataset | Classes | Channels | Seq Len | T = C×S | Batch |
|---------|---------|----------|---------|---------|-------|
| TUEV | 6 | 16 | 5 | 80 | 64 |
| TUAB | 2 | 16 | 10 | 160 | 64 |
| TUSZ | 2 | 22 | 5 | 110 | 64 |
| DIAGNOSIS | 4 | 58 | 5 | 290 | 64 |

---

## 6. Backbone-Specific Adaptations

### 6.1 4D-Aware Token Structure

CBraMod outputs $(B, C, S, D)$ where $C$ = EEG channels and $S$ = temporal patches. USBA detects this and passes `token_structure = (n_channels, seq_len)` to all branches:

```python
# injection.py line 73-74
if self._backbone_type in ('cbramod', 'codebrain'):
    self._token_structure = (n_channels, seq_len)
```

This enables structure-aware processing:
- **Temporal branch:** reshapes $(B, C \cdot S, D) \to (B \cdot C, S, D)$, conv along $S$ only
- **Spatial branch:** reshapes $(B, C \cdot S, D) \to (B \cdot S, C, D)$, SE along $C$ only

Without this, a flat $(B, T, D)$ convolution would mix signals from different channels, violating the spatial-temporal factorization that CBraMod's criss-cross attention was designed around.

### 6.2 Inter-Layer Injection (CBraMod Only)

When `selected_layers='all'` or a list of layer indices:

```python
# injection.py line 143-205
def _backbone_inter_layer_forward(self, x):
    # (1) Frozen patch embedding
    h = inner.patch_embedding(x, None)  # (B, C, S, D)

    # (2) For each transformer layer:
    for i, layer in enumerate(layers):
        h = layer(h)           # frozen transformer layer
        if should_inject(i):
            h_tokens = h.reshape(B, -1, D)
            h_tokens, aux = usba_layer(h_tokens, token_structure=(C, S))
            h = h_tokens.reshape(B, C, S, D)

    # (3) Frozen proj_out
    h = inner.proj_out(h)
```

This manually unrolls CBraMod's encoder layers and inserts USBA adapters between them, allowing progressive adaptation at multiple representation levels. Each USBA layer has its own KL, gates, and branch parameters.

**Not available for CodeBrain/LUNA** because their forward passes are monolithic and not easily unrollable.

### 6.3 Backbone Detection and Freezing

```python
# injection.py line 113-124
def _detect_backbone_type(self):
    name = type(self.backbone).__name__.lower()
    if 'cbramod' in name: return 'cbramod'
    elif 'sssm' in name or 'codebrain' in name: return 'codebrain'
    ...
```

For CBraMod specifically, the backbone wrapper from `backbone_factory.py` is `CBraModWrapper`, which is detected and handled with:
- 4D output reshape: `backbone_out.reshape(B, -1, self.token_dim)` from (B,C,S,D) to (B,C*S,D)
- Layer counting via `self.backbone.backbone.encoder.layers` for inter-layer injection
- Frozen `torch.no_grad()` context for backbone forward

### 6.4 Alignment with CBraMod's Criss-Cross Attention

CBraMod's encoder alternates between:
- **Temporal self-attention:** attends across $S$ positions within each channel
- **Spatial self-attention:** attends across $C$ channels within each time step

USBA's factorized branches mirror this:
- **Depthwise temporal conv:** operates within-channel along $S$
- **Channel attention (SE):** operates within-timestep along $C$

This alignment means USBA corrections live in the same factored subspace as CBraMod's own representations, rather than introducing cross-modal artifacts.

### 6.5 CodeBrain (SSSM) Adaptations

#### Architecture Differences

CodeBrain uses a **Structured State Space Model (S4)** backbone instead of transformers:
- Input $(B, C, S, P)$ → PatchEmbedding → rearrange to $(B, P, C \cdot S)$ → S4 residual layers → rearrange back to $(B, C, S, P)$
- S4 captures **long-range temporal dependencies** globally across the flattened $C \cdot S$ sequence
- No explicit spatial/temporal factorization in the backbone itself (unlike CBraMod's criss-cross attention)

#### squeeze() Safety Fix

`SSSM.forward()` ends with `return x.squeeze()`, which removes **any** size-1 dimension. This is a latent bug:
- `seq_len=1`: $(B, C, 1, D)$ → squeeze → $(B, C, D)$ — loses temporal axis
- `batch_size=1` (val/test tail): $(1, C, S, D)$ → squeeze → $(C, S, D)$ — loses batch axis

USBA's `_backbone_forward()` adds an explicit reshape guard:

```python
# injection.py
if self._backbone_type == 'codebrain' and out.dim() != 4:
    B = x.shape[0]
    out = out.reshape(B, self.n_channels, self.seq_len, self.token_dim)
```

This guarantees a consistent 4D output regardless of squeeze behavior.

#### 4D-Aware Branches

Like CBraMod, CodeBrain receives `token_structure = (n_channels, seq_len)`:
- **Temporal branch (depthwise conv):** complements S4's global temporal modeling with local intra-channel convolutions. S4 learns long-range dynamics; the conv captures local patterns within each channel.
- **Spatial branch (channel attention):** provides explicit cross-channel interaction per time step, which S4's flattened processing does not separate.

#### Output-Only Injection

CodeBrain is restricted to `injection_mode='output'` because SSSM's forward pass is monolithic — the S4 residual layers cannot be individually unrolled. A single USBA layer is applied after the backbone's final output.

#### CodeBrain-Specific Arguments

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--n_layer` | 8 | Number of S4 residual layers in SSSM |
| `--codebook_size_t` | 4096 | Temporal codebook (unused with `if_codebook=False`) |
| `--codebook_size_f` | 4096 | Frequency codebook (unused with `if_codebook=False`) |

---

## 7. Ablation Switches

| Flag | Effect | Command |
|------|--------|---------|
| `--usba_no_factorize` | Replace temporal+spatial branches with single MLP | Ablation: factorization contribution |
| `--usba_no_cc_inv` | Disable class-conditional HSIC | Ablation: subject invariance contribution |
| `--usba_no_budget` | Disable gate L1 penalty | Ablation: sparsity constraint contribution |
| `--no_subjects` | Skip subject ID loading | Run without subject information |
| `--usba_selected_layers all` | Inter-layer injection (CBraMod) | Compare output-only vs inter-layer |
| `--usba_gate_type token_wise` | Per-token gates instead of per-layer | Finer-grained gating |

---

## 8. Parameter Budget

For CBraMod + USBA output mode on TUEV (D=200, d'=64):

| Component | Parameters |
|-----------|-----------|
| LayerNorm(200) | 400 |
| DepthwiseTemporalConv(200, k=5) | 1,200 + 400 (LN) |
| ChannelAttention(200, r=4) | 50×2 + 50×200 + 200×50 = 20,200 |
| GatedFusion(200) | 200×2 + 2 = 402 |
| VIB encoder(200→200) | 200×200 + 200 = 40,200 |
| VIB μ/σ heads(200→64) | (200×64 + 64) × 2 = 25,728 |
| VIB decoder(64→200) | 64×200 + 200 = 13,000 |
| Gate logit | 1 |
| **USBA total** | **~101K** |
| Classification head | 200×400 + 400 + 400×K |
| **Backbone (frozen)** | **~5.5M** |

Trainable ratio: ~2% of total parameters.
