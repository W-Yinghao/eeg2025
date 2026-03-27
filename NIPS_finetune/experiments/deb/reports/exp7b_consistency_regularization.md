# Exp 7B: Frozen Selector + Consistency Regularization

## Motivation

Selector 的 gate 作为病理证据选择器, 应当具有稳定性: 对同一样本的轻微扰动 (噪声、时移、遮盖) 不应显著改变 gate 的选择模式。

Consistency regularization 通过约束 gate(x) ≈ gate(aug(x)) 来增强这种稳定性。

## Design

在 frozen selector 基础上, loss 增加 consistency 项:

```
L = L_ce + lambda_cons * L_cons
L_cons = MSE(gate(x), gate(aug(x)))
```

其中 gate 默认指 temporal_gate (B, S, 1) 和 frequency_gate (B, C, 1)。

### Lambda Sweep

| Config | lambda_cons | Strength |
|--------|-----------|----------|
| l1e3 | 1e-3 | Light |
| l3e3 | 3e-3 | Medium |
| l1e2 | 1e-2 | Strong |

### Common Settings

| Parameter | Value |
|-----------|-------|
| Mode | selector |
| Regime | frozen |
| Dataset | TUAB |
| Model | CodeBrain |
| lr_head | 1e-3 |
| lr_backbone | 0.0 |
| epochs | 50 |
| patience | 12 |
| warmup_epochs | 3 |
| scheduler | cosine |
| clip_value | 1.0 |
| consistency_type | l2 |
| Seeds | 42, 2025, 3407 |

## Augmentation Strategy

All augmentations operate on (B, C, S, P) patched EEG tensors. They are designed to be semantically-preserving — light enough to not change diagnostic content.

| Augmentation | Probability | Parameter | Safety Rationale |
|-------------|------------|-----------|-----------------|
| time_shift | p=0.5 | max_shift=1 segment | 1s offset ≪ seizure duration; EEG diagnosis is position-invariant |
| amplitude_jitter | p=0.5 | sigma=0.03 | 3% std ≪ EEG SNR (~10-50μV signal); waveform morphology preserved |
| time_mask | p=0.3 | mask_ratio=0.08 | 16 samples = 80ms; too short to obscure spikes (≥70ms) or slow waves (>200ms) |

### Implementation

`training/augmentations.py` — `EEGAugmentor` class:
- Per-augmentation probabilities: `p_time_shift=0.5`, `p_jitter=0.5`, `p_mask=0.3`
- All augmentations are `@torch.no_grad()`, operate on input only, do not modify labels
- Backward-compatible: `p_time_shift=None` falls back to `p_each=0.5`

### Training Flow

```python
# In _train_epoch():
out = model(x, return_gates=True)           # original forward
x_aug = augmentor(x)                         # augment input
aug_out = model(x_aug, return_gates=True)    # augmented forward
loss = criterion(logits, y, model_out=out, aug_model_out=aug_out)
# CE loss + lambda * MSE(gate(x), gate(aug(x)))
```

Each batch requires **two forward passes** (original + augmented), ~1.5x compute vs non-consistency training.

## Logged Metrics

### Loss
- `loss_ce`, `loss_consistency`, `loss_total`

### Gate Statistics
- `gate_entropy`, `gate_coverage_0.5`
- `gate_top10_ratio`, `gate_top20_ratio`
- `abnormal_only_gate_coverage`

### Consistency-Specific (computed during evaluation)
- `gate_consistency_l2` — overall L2 distance between original and augmented gates
- `abnormal_only_consistency_l2` — consistency for abnormal class only
- `normal_only_consistency_l2` — consistency for normal class only

## Jean Zay Submission

```bash
# Submit all 9 jobs (3 lambdas × 3 seeds):
bash experiments/deb/scripts/submit_exp7b_cons_all_jeanzay.sh codebrain TUAB
```

### SLURM Resources

| Resource | Value |
|----------|-------|
| GPU | 1x V100-32GB |
| Account | ifd@v100 |
| Time limit | 20h (auto-requeue) |

## Scripts

| Script | Purpose |
|--------|---------|
| `run_exp7b_cons_l1e3_selector_jeanzay.sh` | lambda=1e-3 |
| `run_exp7b_cons_l3e3_selector_jeanzay.sh` | lambda=3e-3 |
| `run_exp7b_cons_l1e2_selector_jeanzay.sh` | lambda=1e-2 |
| `submit_exp7b_cons_all_jeanzay.sh` | Batch submit |
| `aggregate_exp7b.py` | Results aggregation |

## Output Paths

| Lambda | Log | Checkpoint |
|--------|-----|------------|
| 1e-3 | `deb_log/exp7b_cons_l1e3_selector/` | `checkpoints_selector/exp7b_cons_l1e3_selector/` |
| 3e-3 | `deb_log/exp7b_cons_l3e3_selector/` | `checkpoints_selector/exp7b_cons_l3e3_selector/` |
| 1e-2 | `deb_log/exp7b_cons_l1e2_selector/` | `checkpoints_selector/exp7b_cons_l1e2_selector/` |

## Aggregation

```bash
python experiments/deb/scripts/aggregate_exp7b.py
```

Outputs mean/std: BalAcc, Macro-F1, Abnormal F1, gate coverage, gate entropy, overall/abnormal/normal consistency.

## Total Jobs & Compute

| Item | Value |
|------|-------|
| Total jobs | 9 (3λ × 3 seeds) |
| Expected epochs/run | 25-35 |
| Time/epoch (V100) | ~1800s (2x forward pass) |
| Time/run | ~14-17h |
| Total | ~130 V100-hours |

## Code Changes (backward-compatible)

| File | Change |
|------|--------|
| `training/augmentations.py` | `EEGAugmentor`: added `p_time_shift`, `p_jitter`, `p_mask` (default=None → `p_each`) |
| `training/selector_trainer.py` | `_eval_consistency_per_class`: per-class consistency evaluation; augmentor init passes per-aug probs |
| `scripts/train_partial_ft.py` | Added `--aug_p_time_shift`, `--aug_p_jitter`, `--aug_p_mask` args |

All changes are backward-compatible — existing Exp6/7A runs are unaffected.

## Key Question

Does consistency regularization make gate selections more stable under perturbation, and does it improve or maintain classification accuracy?
