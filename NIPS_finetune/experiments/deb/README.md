# DEB — Disease Evidence Bottleneck + Selector Experiments

Experimental branch for EEG disease diagnosis with interpretable evidence selection, built on top of the existing fine-tuning framework.

## Research Direction Shift

Based on experimental results, the main research line has shifted from "complex DEB/VIB" to **"selector + interpretability enhancement"**:

1. **Finding**: Under frozen backbone, the Selector (gates + fusion, no VIB) consistently outperforms plain baseline heads. DEB/VIB did not stably surpass Selector.
2. **Finding**: The old "partial" fine-tuning unfreezes ~99.7% of parameters, drowning head differences in backbone updates.
3. **New priority**: Find a true partial FT regime where Selector advantages are preserved, then enhance Selector interpretability.

The DEB/VIB modules are **preserved as extension points** but are not the current main experiment line.

## Current Experiment Line

| Experiment | Goal | Status |
|------------|------|--------|
| **Exp 6**: True Partial FT Boundary Search | Find regime where Selector > Baseline is visible | Ready |
| **Exp 7**: Selector Interpretability Enhancement | Add sparse/consistency regularization | Ready |
| **Exp 8**: Formal Explainability Evaluation | Quantify interpretability (insertion/deletion, consistency, abnormal-focused) | Ready |

## Original Goal

Provide a clean baseline + minimal DEB head that:
1. Unifies the batch protocol with metadata placeholders
2. Uses subject-level splits to prevent data leakage
3. Monitors balanced accuracy / macro-F1 as primary metrics
4. Wraps existing backbones with a `forward_features()` interface
5. Adds temporal and frequency gates for interpretable evidence selection
6. Implements a variational bottleneck for compressed disease evidence

## What is implemented

| Component | Status | Details |
|-----------|--------|---------|
| Unified batch protocol | Done | `data/batch_protocol.py` — wraps existing dataset to output dict with `x`, `y`, `subject_id`, `channel_mask`, and placeholder metadata fields |
| Subject-level split | Done | `data/splits.py` — reuses `EEGLMDBDataset(cross_subject=True)` |
| Balanced accuracy / macro-F1 | Done | `evaluation/evaluator.py` — early stopping on balanced accuracy by default |
| Backbone wrapper | Done | `models/backbone_wrapper.py` — thin wrapper with `forward_features()` returning `features`, `H_t`, `H_f` |
| Baseline head | Done | `models/baseline_head.py` — pool → MLP classifier |
| DEB head | Done | `models/evidence_bottleneck.py` — temporal gate + frequency gate + VIB + classifier |
| Training loop | Done | `training/trainer.py` — supports frozen/partial/full fine-tune, KL warmup, differential LR |
| DEB loss | Done | `training/losses.py` — CE + beta·KL + sparse_lambda·L1_gate |
| Configs | Done | `configs/defaults.py` — `BASELINE_CONFIG` and `DEB_CONFIG` as Python dicts |
| Scripts | Done | `scripts/train.py`, `scripts/evaluate.py`, bash wrappers |

## What is NOT implemented

| Component | Notes |
|-----------|-------|
| Channel gate | Per-electrode attention conditioned on montage/coordinates — extension point marked in code |
| Shared-private disentanglement | z_shared / z_private split with adversarial loss — extension point in bottleneck |
| HSIC / MMD invariance | Subject-invariance regularization — see existing `adapters/losses.py` for reference |
| Montage consistency loss | Cross-montage agreement — interface reserved, default off |
| Multi-view parallel branches | Separate raw/spectral/spatial encoders — not applicable to current backbones |
| Complex query scorer | Attention-based evidence scoring — not implemented |
| Site-held-out split | LMDB metadata does not include site_id for most datasets — falls back to subject split with warning |

## Metadata field availability

| Field | Available | Source |
|-------|-----------|--------|
| `x` (EEG patches) | Yes | LMDB `signal` / `data` |
| `y` (class label) | Yes | LMDB `label` / `labels.disease` |
| `subject_id` | Yes | LMDB `subject_id` or parsed from `source_file` |
| `site_id` | No | Not in current LMDB metadata |
| `montage_id` | No | Not in current LMDB metadata |
| `reference_type` | No | Not in current LMDB metadata |
| `channel_coordinates` | No | Not in current LMDB metadata (LUNA has its own internal coords) |
| `channel_mask` | Yes | All-ones tensor `(B, C)` — all channels valid in current data |

## Backbone mapping

| Backbone | Type | H_t extraction | H_f extraction |
|----------|------|----------------|----------------|
| **CodeBrain (SSSM)** | State-space + spectral-gated conv | `features.mean(dim=1)` → (B, S, D) temporal summary | `features.mean(dim=2)` → (B, C, D) spatial/spectral proxy |
| **CBraMod** | Criss-cross transformer | Same factorization | Same factorization |
| **LUNA** | Cross-attention transformer | `features` (B, T, D) — pure temporal | None (no channel structure in output) |

CodeBrain does **not** have explicit separate temporal and frequency branches. Its Residual_blocks interleave S4 layers (temporal) with GConv (FFT-based spectral gating). The `H_t` / `H_f` returned by the wrapper are **degraded factorizations** obtained by mean-pooling over the channel and time axes respectively.

## Training

```bash
conda activate eeg2025
cd ~/eeg2025/NIPS_finetune

# Baseline: frozen CodeBrain + MLP head on TUEV
python experiments/deb/scripts/train.py \
    --mode baseline --dataset TUEV --model codebrain --cuda 0

# DEB: frozen CodeBrain + evidence bottleneck on TUEV
python experiments/deb/scripts/train.py \
    --mode deb --dataset TUEV --model codebrain --cuda 0

# DEB with partial fine-tune on DIAGNOSIS
python experiments/deb/scripts/train.py \
    --mode deb --dataset DIAGNOSIS --model codebrain \
    --finetune partial --lr_head 1e-3 --lr_ratio 10 --cuda 0

# Or use bash scripts:
./experiments/deb/scripts/run_baseline.sh codebrain TUEV 0
./experiments/deb/scripts/run_deb.sh codebrain TUEV 0
```

## Evaluation

```bash
python experiments/deb/scripts/evaluate.py \
    --checkpoint checkpoints_deb/best_TUEV_codebrain_deb_acc0.9500.pth \
    --dataset TUEV --model codebrain --mode deb
```

## Key config options

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | baseline | `baseline` or `deb` |
| `--finetune` | frozen | `frozen` / `partial` / `full` |
| `--lr_head` | 1e-3 | Head learning rate |
| `--lr_ratio` | 10.0 | Head LR / Backbone LR ratio (partial mode) |
| `--deb_latent_dim` | 64 | VIB latent dimension |
| `--deb_fusion` | concat | `concat` or `add` for temporal+frequency fusion |
| `--beta` | 1e-4 | KL divergence weight |
| `--beta_warmup_epochs` | 5 | Linear KL warmup |
| `--sparse_lambda` | 1e-3 | Gate sparsity L1 weight |
| `--split_strategy` | subject | `subject` / `random` / `site_held_out` |
| `--patience` | 15 | Early stopping patience (on balanced accuracy) |

## Directory structure

```
experiments/deb/
├── README.md
├── configs/
│   └── defaults.py              # BASELINE_CONFIG, DEB_CONFIG, make_config()
├── data/
│   ├── batch_protocol.py        # DEBDataset, load_deb_data()
│   └── splits.py                # SubjectSplitter
├── models/
│   ├── backbone_wrapper.py      # BackboneWrapper with forward_features()
│   ├── baseline_head.py         # BaselineHead (pool + MLP)
│   ├── evidence_bottleneck.py   # EvidenceBottleneck (gates + VIB) [DEB, preserved]
│   ├── deb_model.py             # DEBModel (original, preserved)
│   ├── deb_model_selector.py    # DEBModelSelector (original, preserved)
│   ├── selector_head.py         # Original SelectorHead (preserved)
│   ├── selector_enhanced.py     # [NEW] EnhancedSelectorHead (gate export + VIB extension)
│   └── selector_model.py        # [NEW] SelectorModel (true partial FT + baseline/selector)
├── training/
│   ├── losses.py                # DEBLoss (CE + KL + sparse) [preserved]
│   ├── trainer.py               # DEBTrainer [preserved]
│   ├── partial_ft.py            # [NEW] True partial FT regime utilities
│   ├── selector_loss.py         # [NEW] SelectorLoss (CE + sparse + consistency + VIB ext)
│   ├── selector_trainer.py      # [NEW] SelectorTrainer (Exp 6/7 training loop)
│   └── augmentations.py         # [NEW] Light EEG augmentations for consistency
├── evaluation/
│   ├── evaluator.py             # Evaluator (balanced acc, macro-F1, CM)
│   └── explainability.py        # [NEW] Insertion/Deletion, Consistency, Abnormal Analysis
└── scripts/
    ├── train.py                  # DEB training entry [preserved]
    ├── train_selector.py         # Original selector training [preserved]
    ├── train_partial_ft.py       # [NEW] Unified Exp 6/7 training entry
    ├── run_exp6_boundary_search.sh  # [NEW] Run all Exp 6 combinations locally
    ├── run_exp6_jeanzay.sh       # [NEW] Single Exp 6 job for SLURM
    ├── submit_exp6_seeds.sh      # [NEW] Submit all Exp 6 seeds to SLURM
    ├── run_exp7_interpretability.sh  # [NEW] Run all Exp 7 variants locally
    ├── run_exp7_jeanzay.sh       # [NEW] Single Exp 7 job for SLURM
    ├── submit_exp7_seeds.sh      # [NEW] Submit all Exp 7 seeds to SLURM
    ├── run_exp8_explainability.py  # [NEW] Exp 8 evaluation pipeline
    ├── aggregate_results.py      # [NEW] Results aggregation (markdown + CSV)
    └── evaluate.py               # DEB evaluation entry [preserved]
```

---

## Experiments 6/7/8: Selector + Interpretability

### True Partial Fine-Tuning (Exp 6)

The old `--finetune partial` unfreezes the **entire** backbone (minus patch embedding), resulting in ~99.7% trainable parameters. This drowns head differences.

The new `--regime` flag provides true partial FT:

| Regime | CodeBrain (8 blocks) | CBraMod (12 layers) | Trainable ratio |
|--------|---------------------|---------------------|-----------------|
| `frozen` | 0 blocks | 0 layers | ~0.7-1.2% (head only) |
| `top1` | block 7 | layer 11 | ~11-12% |
| `top2` | blocks 6-7 | layers 10-11 | ~22-23% |
| `top4` | blocks 4-7 | layers 8-11 | ~44% |
| `full` | all 8 blocks | all 12 layers | ~88% (patch embed still frozen) |

**Block mapping:**
- **CodeBrain**: `residual_layer.residual_blocks[i]` — each block contains S4 + GConv + MHA + skip/res connections
- **CBraMod**: `encoder.layers[i]` — each layer contains spatial MHA + temporal MHA + FFN

### Running Experiments

```bash
conda activate eeg2025
cd ~/eeg2025/NIPS_finetune

# ── Experiment 6: Boundary Search ──

# Single run: frozen selector on TUAB
python experiments/deb/scripts/train_partial_ft.py \
    --dataset TUAB --model codebrain --mode selector --regime frozen

# Single run: top2 baseline on TUAB
python experiments/deb/scripts/train_partial_ft.py \
    --dataset TUAB --model codebrain --mode baseline --regime top2

# All regimes x modes x seeds locally
bash experiments/deb/scripts/run_exp6_boundary_search.sh codebrain TUAB all 0

# Submit to Jean Zay
bash experiments/deb/scripts/submit_exp6_seeds.sh codebrain TUAB

# ── Experiment 7: Interpretability Enhancement ──

# Selector + sparse regularization
python experiments/deb/scripts/train_partial_ft.py \
    --dataset TUAB --model codebrain --mode selector --regime top2 \
    --enable_sparse --sparse_lambda 1e-3 --sparse_type l1

# Selector + consistency regularization
python experiments/deb/scripts/train_partial_ft.py \
    --dataset TUAB --model codebrain --mode selector --regime top2 \
    --enable_consistency --consistency_lambda 1e-2 --consistency_type l2

# Selector + both
python experiments/deb/scripts/train_partial_ft.py \
    --dataset TUAB --model codebrain --mode selector --regime top2 \
    --enable_sparse --sparse_lambda 1e-3 \
    --enable_consistency --consistency_lambda 1e-2

# All variants with best regime
bash experiments/deb/scripts/run_exp7_interpretability.sh codebrain TUAB top2 all 0

# ── Experiment 8: Explainability Evaluation ──

# Evaluate a trained checkpoint
python experiments/deb/scripts/run_exp8_explainability.py \
    --checkpoint checkpoints_selector/best_TUAB_codebrain_selector_top2_acc0.85_s3407.pth \
    --dataset TUAB --model codebrain \
    --n_insertion_steps 10 --n_augmentations 5 --abnormal_class 0

# Evaluate all checkpoints in a directory
python experiments/deb/scripts/run_exp8_explainability.py \
    --checkpoint_dir checkpoints_selector/ --dataset TUAB --model codebrain

# ── Result Aggregation ──

# Aggregate all
python experiments/deb/scripts/aggregate_results.py \
    --results_dir checkpoints_selector/ --experiment all

# Exp 6 only
python experiments/deb/scripts/aggregate_results.py \
    --results_dir checkpoints_selector/ --experiment exp6
```

### Sparse Regularization Design

Three sparse regularization types for selector gates:

| Type | Formula | Effect |
|------|---------|--------|
| `l1` | `mean(gate)` | Penalize average gate activation, encourage low values |
| `entropy` | `-mean(g*log(g) + (1-g)*log(1-g))` | Penalize uncertain gates, encourage binary 0/1 |
| `coverage` | `mean(gate > 0.5)` | Penalize fraction of active gates |

Default: `l1` with `sparse_lambda=1e-3` (conservative).

### Consistency Regularization Design

Applies light augmentations and constrains gate maps to be stable:

**Augmentations** (designed to preserve diagnostic content):
- Time shift: roll segments by +/-1
- Amplitude jitter: Gaussian noise (std=0.05)
- Time mask: zero 10% of patch dimension

**Consistency loss types:**
- `l2`: MSE between original and augmented gate maps
- `cosine`: 1 - cosine similarity
- `kl`: KL divergence between softmax-normalized gate maps

### Explainability Evaluation

**1. Insertion/Deletion:**
- Rank temporal segments by gate score
- Insertion: progressively reveal high-score segments, track target class probability
- Deletion: progressively mask high-score segments, track probability drop
- Report AUC of curves; per-class breakdown

**2. Augmentation Consistency:**
- Apply N augmentations per sample
- Compare gate maps via cosine similarity
- Report mean/std consistency, per-class breakdown

**3. Abnormal-Focused Analysis:**
- Abnormal recall, F1, precision
- Gate coverage on abnormal samples (fraction of active gates)
- Evidence length (number of high-gate segments)
- Comparison vs normal class

### Abnormal-Focused Analysis: Why It Matters

The project goal is disease diagnosis. Normal samples are easier to classify. What matters is:
1. Can the model detect abnormality? (recall)
2. Does the selector focus on diagnostically relevant regions? (coverage, evidence length)
3. Is the evidence consistent under perturbation? (consistency)

### VIB/DEB Extension Points

VIB is **not** enabled by default in the selector experiments. To enable:

```bash
python experiments/deb/scripts/train_partial_ft.py \
    --mode selector --regime top2 \
    --enable_vib --vib_latent_dim 64 --vib_beta 1e-4
```

The `EnhancedSelectorHead` has a `enable_vib` flag that adds mu/logvar/KL to the pipeline.
The `SelectorLoss` has `enable_vib` with warmup support.
The original DEB code (`evidence_bottleneck.py`, `deb_model.py`, `losses.py`, `trainer.py`) is fully preserved.

## Reused from main repo

| Module | Source | Usage |
|--------|--------|-------|
| `DATASET_CONFIGS` | `finetune_tuev_lmdb.py` | Dataset definitions (paths, channels, classes) |
| `EEGLMDBDataset` | `finetune_tuev_lmdb.py` | LMDB data loading, cross-subject split |
| `setup_seed()` | `finetune_tuev_lmdb.py` | Reproducibility |
| `EEGDatasetWithSubjects` | `train_ib_disentangle.py` | Subject ID extraction wrapper |
| `create_backbone()` | `backbone_factory.py` | Frozen backbone creation |
| `sklearn.metrics` | External | balanced_accuracy_score, f1_score |
