# DEB — Disease Evidence Bottleneck (Minimal)

Experimental branch implementing a minimal Disease Evidence Bottleneck for EEG disease diagnosis, built on top of the existing fine-tuning framework.

## Goal

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
│   └── defaults.py          # BASELINE_CONFIG, DEB_CONFIG, make_config()
├── data/
│   ├── batch_protocol.py    # DEBDataset, load_deb_data()
│   └── splits.py            # SubjectSplitter
├── models/
│   ├── backbone_wrapper.py  # BackboneWrapper with forward_features()
│   ├── baseline_head.py     # BaselineHead (pool + MLP)
│   ├── evidence_bottleneck.py  # EvidenceBottleneck (gates + VIB)
│   └── deb_model.py         # DEBModel (unified model class)
├── training/
│   ├── losses.py            # DEBLoss (CE + KL + sparse)
│   └── trainer.py           # DEBTrainer (training loop)
├── evaluation/
│   └── evaluator.py         # Evaluator (balanced acc, macro-F1, CM)
└── scripts/
    ├── train.py              # Main training entry
    ├── evaluate.py           # Evaluation entry
    ├── run_baseline.sh       # Bash wrapper for baseline
    └── run_deb.sh            # Bash wrapper for DEB
```

## Reused from main repo

| Module | Source | Usage |
|--------|--------|-------|
| `DATASET_CONFIGS` | `finetune_tuev_lmdb.py` | Dataset definitions (paths, channels, classes) |
| `EEGLMDBDataset` | `finetune_tuev_lmdb.py` | LMDB data loading, cross-subject split |
| `setup_seed()` | `finetune_tuev_lmdb.py` | Reproducibility |
| `EEGDatasetWithSubjects` | `train_ib_disentangle.py` | Subject ID extraction wrapper |
| `create_backbone()` | `backbone_factory.py` | Frozen backbone creation |
| `sklearn.metrics` | External | balanced_accuracy_score, f1_score |
