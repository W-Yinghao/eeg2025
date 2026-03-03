# Information-Theoretic Fine-Tuning Framework for EEG Foundation Models

Combines **Variational Information Bottleneck (VIB)** and **InfoNCE contrastive learning** on top of a frozen **CodeBrain (SSSM)** backbone.

## Core Idea

When fine-tuning pre-trained EEG foundation models, two challenges arise:

1. **Subject-specific noise**: Inter-subject variability (impedance, anatomy) hurts generalization
2. **Limited labels**: Small downstream datasets risk overfitting

This framework addresses both via information theory:
- **VIB** compresses representations, discarding subject-specific nuisance
- **InfoNCE** aligns learned representations with domain-expert features (PSD, stats)

## Architecture

```
EEG Input (B, C, S, 200)
       |
  Frozen CodeBrain (SSSM) Backbone
       |
  (B, C, S, 200) -> flatten -> (B, C*S*200)
       |
  RepProjection (trainable) -> Z_FM (B, hidden_dim)
       |                          |
   VIB Layer              ContrastHead (trainable)
       |                          |
  z (B, vib_dim)          z_proj (B, hidden_dim)
       |                          |
  Classifier              InfoNCE(z_proj, Z_expert)
       |
  logits -> CE Loss

Expert Features (PSD/stats from raw signal)
       |
  ExpertProjector (trainable) -> Z_expert (B, hidden_dim)

Total Loss = CE + beta * KL(VIB) + alpha * InfoNCE(z_proj, Z_expert)
```

Key design points:
- Backbone is **fully frozen** (no grad), saving GPU memory
- InfoNCE is applied on `Z_FM` (through a contrast head), so its gradients train `RepProjection` — the learned mapping from frozen backbone output to task-relevant representation
- VIB operates on `Z_FM`, compressing it before classification
- Expert features are computed from the **full time series** (4D data reshaped to 3D)

## Quick Start

```bash
# Test framework (CPU/GPU)
python test_mi_framework.py

# Train on TUEV with default settings
python train_mi_finetuning.py --dataset TUEV --cuda 0

# Train on TUAB with PSD expert features
python train_mi_finetuning.py --dataset TUAB --expert_feature psd --cuda 0

# Baseline (CE only, no MI)
python train_mi_finetuning.py --dataset TUEV --alpha 0.0 --beta 0.0 --cuda 0
```

## Ablation Study

```bash
# 1. Baseline (CE only)
python train_mi_finetuning.py --alpha 0.0 --beta 0.0

# 2. InfoNCE only
python train_mi_finetuning.py --alpha 1.0 --beta 0.0

# 3. VIB only
python train_mi_finetuning.py --alpha 0.0 --beta 1e-3

# 4. Full MI
python train_mi_finetuning.py --alpha 1.0 --beta 1e-3

# 5. Strong expert alignment
python train_mi_finetuning.py --alpha 2.0 --beta 1e-3

# 6. Strong compression
python train_mi_finetuning.py --alpha 1.0 --beta 1e-2
```

## Hyperparameters

| Param | Default | Range | Effect |
|-------|---------|-------|--------|
| `alpha` (InfoNCE) | 1.0 | [0, 5.0] | Expert alignment strength |
| `beta` (VIB) | 1e-3 | [0, 1e-2] | Information compression |
| `temperature` | 0.07 | [0.05, 0.1] | InfoNCE hardness |
| `vib_dim` | 128 | [32, 256] | Bottleneck dimension |
| `hidden_dim` | 256 | [128, 512] | Representation dimension |

## Expert Features

| Type | Dim (16ch) | Description |
|------|-----------|-------------|
| `psd` | 80 | Power in delta/theta/alpha/beta/gamma bands |
| `stats` | 64 | Mean/std/min/max per channel |
| `both` | 144 | Concatenation of PSD + stats |

Features are extracted from the full time series: 4D `(B, C, S, P)` is reshaped to `(B, C, S*P)` before FFT/stats.

## File Structure

| File | Description |
|------|-------------|
| `mi_finetuning_framework.py` | Core framework: backbone factory, MIFineTuner, VIB, losses |
| `train_mi_finetuning.py` | Training script with ExpertFeatureExtractor and MITrainer |
| `test_mi_framework.py` | Test script (construction + GPU forward/backward) |

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| VIB loss explodes | beta too high | Decrease beta (1e-3 -> 1e-4) |
| No InfoNCE benefit | Weak expert features | Try `both` features or decrease alpha |
| Training unstable | Temperature too low | Increase tau (0.07 -> 0.1) |
| Overfitting | Insufficient compression | Increase beta or decrease vib_dim |

## References

1. Alemi et al. (2017). "Deep Variational Information Bottleneck" - ICLR 2017
2. Oord et al. (2018). "Representation Learning with Contrastive Predictive Coding"
3. CodeBrain SSSM backbone
