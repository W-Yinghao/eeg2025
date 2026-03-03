# EEG Foundation Model Fine-Tuning

Fine-tuning framework for EEG foundation models (CodeBrain SSSM, CBraMod) on downstream tasks (TUEV, TUAB, etc.).

## Project Structure

```
NIPS_finetune/
├── CBraMod/                    -> symlink to model repo
├── CodeBrain/                  -> symlink to model repo
├── Cbramod_pretrained_weights.pth -> symlink to pretrained weights
│
├── finetune_tuev_lmdb.py       # Core data loading (LMDB datasets, DataLoader)
├── finetune_msft.py            # MSFT fine-tuning (CodeBrain + CBraMod)
├── finetune_msft_improved.py   # MSFT with ablation variants (pos_refiner, criss_cross_agg)
├── msft_modules.py             # MSFT model definitions
├── msft_modules_improved.py    # MSFT improved model definitions
├── mi_finetuning_framework.py  # Information-theoretic fine-tuning (VIB + InfoNCE)
├── train_mi_finetuning.py      # MI fine-tuning training script
├── test_mi_framework.py        # MI framework tests
│
├── scripts/                    # Shell scripts (local run + SLURM)
│   ├── run_codebrain_msft.sh
│   ├── slurm_codebrain_msft.sh
│   ├── run_ablation_fixed_params.sh
│   ├── run_ablation_study.sh
│   ├── slurm_ablation_fixed_params.sh
│   ├── slurm_single_experiment.sh
│   ├── slurm_submit_ablation.sh
│   ├── slurm_wandb_agent.sh
│   ├── run_cbramod_chu.sh
│   └── run_cbramod_tuab.sh
│
├── configs/                    # WandB sweep configs
│   ├── sweep_msft_cbramod.yaml
│   └── sweep_msft_cbramod_bayesian.yaml
│
├── preprocessing/              # Data preprocessing scripts
│   ├── segment_to_lmdb.py
│   ├── preprocess_all_eeg.py
│   └── ...
│
└── docs/                       # Documentation
    ├── MI_FINETUNING_README.md
    ├── CODEBRAIN_MSFT_GUIDE.md
    ├── ABLATION_STUDY_GUIDE.md
    ├── SLURM_USAGE.md
    └── ...
```

## Quick Start

### Environment

```bash
conda activate eeg2025
```

### MSFT Fine-Tuning (CodeBrain)

```bash
# Single scale
scripts/run_codebrain_msft.sh TUEV

# Or directly:
python finetune_msft.py --model codebrain --dataset TUEV --cuda 0 \
    --pretrained_weights CodeBrain/Checkpoints/CodeBrain.pth \
    --codebook_size_t 4096 --codebook_size_f 4096 --num_scales 3
```

### MI Fine-Tuning (VIB + InfoNCE)

```bash
# Full MI (VIB + InfoNCE)
python train_mi_finetuning.py --dataset TUEV --cuda 0 --alpha 1.0 --beta 1e-3

# Baseline (CE only)
python train_mi_finetuning.py --dataset TUEV --cuda 0 --alpha 0.0 --beta 0.0

# Test framework
python test_mi_framework.py
```

### MSFT Ablation Study (CBraMod)

```bash
scripts/run_ablation_fixed_params.sh TUEV
```

### SLURM Submission

```bash
scripts/slurm_codebrain_msft.sh submit
scripts/slurm_ablation_fixed_params.sh submit
```

## Backbones

| Backbone | Pretrained Weights | Key Params |
|----------|-------------------|------------|
| CodeBrain (SSSM) | `CodeBrain/Checkpoints/CodeBrain.pth` | `codebook_size_t=4096, codebook_size_f=4096` |
| CBraMod | `Cbramod_pretrained_weights.pth` | `n_layer=12, nhead=8` |

## Datasets

| Dataset | Task | Classes | Data Path |
|---------|------|---------|-----------|
| TUEV | Multiclass | 6 | `/projects/EEG-foundation-model/diagnosis_data/tuev_preprocessed` |
| TUAB | Binary | 2 | `/projects/EEG-foundation-model/diagnosis_data/tuab_preprocessed` |
