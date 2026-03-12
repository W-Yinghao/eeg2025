# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EEG foundation model fine-tuning research framework. Evaluates multiple fine-tuning strategies on frozen pretrained EEG backbones for downstream classification tasks (seizure detection, abnormality detection, disease diagnosis).

## Environment

```bash
conda activate eeg2025
```

## Common Commands

### Training

```bash
# Full fine-tune baseline (all backbones unfrozen)
scripts/run_full_finetune.sh [MODEL] [DATASET]  # MODEL: cbramod|codebrain|luna|all, DATASET: TUEV|TUAB|TUSZ|DIAGNOSIS|all

# MSFT fine-tuning
scripts/run_codebrain_msft.sh TUEV
python finetune_msft.py --model codebrain --dataset TUEV --cuda 0

# MI fine-tuning (VIB + InfoNCE)
python train_mi_finetuning.py --dataset TUEV --cuda 0 --alpha 1.0 --beta 1e-3

# SCOPE (semi-supervised with prototypes)
python train_scope.py --dataset TUEV --model codebrain --label_ratio 0.3

# IB disentanglement
python train_ib_disentangle.py --dataset TUEV --cuda 0

# SageStream (SA-MoE + IIB)
python train_sagestream.py --dataset TUEV --cuda 0

# Hyperbolic embeddings
python train_hyperbolic_finetuning.py --dataset TUEV --cuda 0

# CBraMod MSFT ablation
scripts/run_ablation_fixed_params.sh TUEV
```

### SLURM Submission

```bash
scripts/slurm_codebrain_msft.sh submit
scripts/slurm_ablation_fixed_params.sh submit
```

### Tests

```bash
python test_mi_framework.py
python test_ib_disentangle.py
```

## Architecture

### Backbone Factory (`backbone_factory.py`)

Central entry point for all frozen pretrained backbones. `create_backbone(model_type, ...)` returns `(backbone, out_dim, token_dim)`.

Supported backbones:
- **CodeBrain (SSSM)**: State-space model, weights at `CodeBrain/Checkpoints/CodeBrain.pth`
- **CBraMod**: Criss-cross transformer, weights at `Cbramod_pretrained_weights.pth`. Supports optional inter-layer `BottleneckAdapter` injection via `CBraModWithAdapters` wrapper
- **FEMBA**: BiMamba encoder (requires `mamba_ssm`), wrapped by `FEMBAEncoderWrapper`
- **LUNA**: Cross-attention transformer, wrapped by `LUNAEncoderWrapper`

All backbones are frozen. Their projection heads are replaced with `nn.Identity()`. Output shape convention: `(B, C, S, D)` where C=channels, S=temporal segments, D=token_dim (typically 200).

### Data Pipeline (`finetune_tuev_lmdb.py`)

Shared data loading from LMDB files. Provides `DATASET_CONFIGS`, `load_data()`, and `setup_seed()` used by all training scripts. EEG data is 16-channel bipolar montage at 200Hz, segmented into patches of 200 samples.

Datasets: TUEV (6-class), TUAB (binary), CHB-MIT (binary), TUSZ (binary), DIAGNOSIS (4-class), DEPRESSION (binary), CVD_DEPRESSION_NORMAL (3-class), UNIFIED_DIAGNOSIS (3-class), AD_DIAGNOSIS (4-class).

### Fine-Tuning Frameworks

Each framework defines its model architecture and loss function. Training scripts (`train_*.py`) handle the training loop, importing data infra from `finetune_tuev_lmdb.py`.

| Framework | Key File | Architecture | Loss |
|-----------|----------|-------------|------|
| MSFT | `msft_modules.py` | Multi-scale adapters + cross-scale aggregation | CE |
| MI | `mi_finetuning_framework.py` | VIB + InfoNCE contrastive alignment | CE + β·KL + α·InfoNCE |
| IB-Disentangle | `ib_disentangle_framework.py` | Token-level IB + GRL subject adversarial | CE + β·KL + λ·GRL |
| SageStream | `sagestream_framework.py` | SA-MoE (GLU experts) + IIB + GRL | CE + KL + adversarial |
| SCOPE | `scope_framework.py` | TPN + ETF + Sinkhorn prototypes + ProAdapter | CE + ETF + semi-supervised |
| Hyperbolic | `hyperbolic_finetuning.py` | Lorentz embeddings + HMLR + DSMBN | CE + λ·HHSW |
| BrainPro | `brainpro_model.py` | Multi-encoder (temporal CNN + spatial + transformer) | Pretrain + finetune |

### External Model Repos (symlinked)

- `CodeBrain/` — SSSM model (`Models.SSSM`), added to `sys.path` at runtime
- `CBraMod/` — Criss-cross transformer (`models.cbramod`), added to `sys.path` at runtime
- `BioFoundation/` — FEMBA and LUNA models, added to `sys.path` at runtime

### Logging

- WandB for experiment tracking (project names like `finetune_baseline`, `msft_codebrain`, etc.)
- Checkpoints saved to `checkpoints_*/` directories
- Logs saved to `logs_*/` directories
- SLURM logs in `slurm_log/`

## Key Conventions

- All frameworks share `backbone_factory.py` for backbone creation — never instantiate backbones directly
- All training scripts import data loading from `finetune_tuev_lmdb.py` — dataset configs and splits are centralized there
- Backbone parameters are always frozen; only adapter/head parameters are trainable (except full fine-tune baseline)
- Seed 3407 is the default across experiments
- Default patch_size=200 (matching 200Hz sampling rate × 1 second)
