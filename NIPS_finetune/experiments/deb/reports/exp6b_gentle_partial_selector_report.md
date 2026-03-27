# Experiment 6B: Gentle Partial Fine-Tuning for Selector — Design Report

> **Date**: 2026-03-27
> **Platform**: Jean Zay (IDRIS), H100
> **Status**: Ready to submit

---

## 1. Motivation

Exp 6 boundary search revealed that partial FT (top1/top2/top4) with default LR settings causes **backbone updates too strong**, leading to gate collapse — gates saturate to near-all-open or near-all-closed across different seeds. This undermines the selector's interpretability.

Exp 6B tests whether **gentler partial FT** (lower backbone LR, fewer epochs, staged training) can preserve gate stability while still providing a small performance boost over frozen-only training.

**Key principle**: not chasing higher accuracy, but maintaining gate diversity and stability.

### 1.1 What This Experiment Does NOT Include

- No sparse regularization (L1/entropy/coverage)
- No consistency regularization
- No VIB bottleneck
- Pure CE loss only

---

## 2. Experimental Design

### 2.1 Fixed Regime

All configurations share:

| Parameter | Value |
|-----------|-------|
| `unfreeze_last_n_blocks` | 1 (regime=`top1`) |
| `freeze_patch_embed` | `true` |
| `mode` | `selector` |
| `backbone` | CodeBrain (SSSM) |
| `dataset` | TUAB |
| `batch_size` | 64 |
| `split_strategy` | subject |
| `AMP` | enabled |
| `return_gates` | `true` |

### 2.2 Seeds

3 seeds: **42, 2025, 3407**

### 2.3 Four Configurations

#### P1 — Gentle Partial (lr_backbone=1e-5)

| Parameter | Value |
|-----------|-------|
| `lr_head` | 1e-3 |
| `lr_backbone` | 1e-5 |
| `scheduler` | cosine |
| `warmup_epochs` | 3 |
| `max_epochs` | 12 |
| `early_stop_patience` | 4 |

#### P2 — Even Gentler Partial (lr_backbone=1e-6)

| Parameter | Value |
|-----------|-------|
| `lr_head` | 1e-3 |
| `lr_backbone` | 1e-6 |
| `scheduler` | cosine |
| `warmup_epochs` | 3 |
| `max_epochs` | 12 |
| `early_stop_patience` | 4 |

#### P3e1 — Staged Partial (stage1=1 epoch)

| Stage | Parameter | Value |
|-------|-----------|-------|
| Stage 1 | `unfreeze_last_n_blocks` | 1 |
| Stage 1 | `lr_head` | 1e-3 |
| Stage 1 | `lr_backbone` | 1e-5 |
| Stage 1 | `warmup_epochs` | 1 |
| Stage 1 | `epochs` | 1 |
| Stage 2 | backbone | frozen |
| Stage 2 | `lr_head` | 5e-4 |
| Stage 2 | `warmup_epochs` | 2 |
| Stage 2 | `epochs` | 20 |
| Stage 2 | `patience` | 6 |

#### P3e2 — Staged Partial (stage1=2 epochs)

Same as P3e1 except **stage1 epochs = 2**.

### 2.4 Total Jobs

4 configs x 3 seeds = **12 jobs** on H100, `--time=20:00:00` each.

---

## 3. Code Changes

### 3.1 Modified File

**`training/selector_trainer.py`** — `train_stage()` method enhanced:

- `_evaluate()` now called with `collect_gates=True` during stage1 (was `False`)
- Stage1 log line now prints `g_t=` and `g_f=` gate means
- WandB logging during stage1 now includes `train/temporal_gate_mean`, `train/frequency_gate_mean`, `train/temporal_active_ratio`, `train/frequency_active_ratio`, and all `val/gate_*` stats

This is backward-compatible — only adds information that was previously missing from stage1 output.

### 3.2 New Files

| File | Purpose |
|------|---------|
| `scripts/run_exp6b_gpartial_p1_selector_jeanzay.sh` | SLURM run script for P1 |
| `scripts/run_exp6b_gpartial_p2_selector_jeanzay.sh` | SLURM run script for P2 |
| `scripts/run_exp6b_gpartial_p3e1_selector_jeanzay.sh` | SLURM run script for P3e1 |
| `scripts/run_exp6b_gpartial_p3e2_selector_jeanzay.sh` | SLURM run script for P3e2 |
| `scripts/submit_exp6b_gpartial_selector_all_jeanzay.sh` | Submits all 12 jobs |
| `scripts/smoke_test_exp6b_gpartial_selector.sh` | Local sanity check script |

### 3.3 No Changes to Existing Exp6

All original Exp6/6A scripts, models, and training infrastructure are untouched.

---

## 4. Directory Structure

### 4.1 Logs

```
deb_log/
  exp6b_gpartial_p1_selector/     # P1 SLURM stdout/stderr
  exp6b_gpartial_p2_selector/     # P2
  exp6b_gpartial_p3e1_selector/   # P3e1
  exp6b_gpartial_p3e2_selector/   # P3e2
```

### 4.2 Checkpoints

```
checkpoints_selector/
  exp6b_gpartial_p1_selector/     # P1 best model, JSON, CSV, gate stats
  exp6b_gpartial_p2_selector/     # P2
  exp6b_gpartial_p3e1_selector/   # P3e1
  exp6b_gpartial_p3e2_selector/   # P3e2
```

Each checkpoint directory will contain per-seed:
- `best_TUAB_codebrain_selector_*.pth` — model checkpoint
- `*_summary.json` — structured results
- `*_gate_stats.json` — final test gate statistics
- `*_epoch_gate_stats.csv` — per-epoch gate tracking
- `*_classwise.json` — per-class F1/recall/precision
- `*_curve.csv` — training curve
- `*_summary.md` — human-readable summary

---

## 5. Tracked Metrics

### 5.1 Classification Metrics

- Balanced accuracy, macro F1, weighted F1
- Per-class F1, recall, precision
- Abnormal recall, abnormal F1, abnormal precision (binary tasks)
- CE loss

### 5.2 Gate Metrics (Selector-Specific)

| Metric | Description |
|--------|-------------|
| `g_t_mean` / `g_t_std` | Temporal gate activation mean / std |
| `g_f_mean` / `g_f_std` | Frequency gate activation mean / std |
| `gate_entropy` | Average binary entropy of gates (temporal + frequency) |
| `gate_coverage@0.5` | Fraction of gate values > 0.5 |
| `abnormal_only_gate_coverage` | Coverage computed on abnormal samples only |
| `abnormal_only_gate_entropy` | Entropy computed on abnormal samples only |
| `epoch_gate_stats.csv` | All above metrics tracked per epoch |

### 5.3 Infrastructure Metrics

- Trainable parameter ratio (logged at model creation and after stage switch)
- Per-component parameter breakdown (backbone vs head)
- Regime info (blocks unfrozen, unfrozen layer names)

---

## 6. Commands

### 6.1 Submit All (Jean Zay)

```bash
bash experiments/deb/scripts/submit_exp6b_gpartial_selector_all_jeanzay.sh codebrain TUAB
```

### 6.2 Individual Configs

```bash
# P1: lr_bb=1e-5
sbatch experiments/deb/scripts/run_exp6b_gpartial_p1_selector_jeanzay.sh codebrain TUAB <SEED>

# P2: lr_bb=1e-6
sbatch experiments/deb/scripts/run_exp6b_gpartial_p2_selector_jeanzay.sh codebrain TUAB <SEED>

# P3e1: staged, stage1=1ep
sbatch experiments/deb/scripts/run_exp6b_gpartial_p3e1_selector_jeanzay.sh codebrain TUAB <SEED>

# P3e2: staged, stage1=2ep
sbatch experiments/deb/scripts/run_exp6b_gpartial_p3e2_selector_jeanzay.sh codebrain TUAB <SEED>
```

### 6.3 Smoke Test (Local)

```bash
bash experiments/deb/scripts/smoke_test_exp6b_gpartial_selector.sh
```

---

## 7. Sanity Check Design

The smoke test (`smoke_test_exp6b_gpartial_selector.sh`) verifies:

1. **No NaN**: Checks all log output for `nan` strings
2. **No saturation**: Verifies gate values are not stuck at 0.000 or 1.000
3. **Gate stats in stage1**: Confirms `g_t=` appears in Stage1 log lines (new feature)
4. **Gate export after stage switch**: Confirms `*_gate_stats.json` and `*_epoch_gate_stats.csv` exist in P3e1 checkpoint dir
5. **Trainable ratio**: Extracts and prints trainable ratios for comparison

### 7.1 Expected Trainable Ratios

| Regime | Blocks Unfrozen | Approx. Trainable Ratio |
|--------|----------------|------------------------|
| `frozen` (Exp 6A) | 0 | ~0.2% (head only) |
| **`top1` (Exp 6B P1/P2)** | **1** | **~1.5%** |
| `top2` (old Exp 6) | 2 | ~3% |
| `top4` (old Exp 6) | 4 | ~6% |
| P3 stage1 | 1 | ~1.5% |
| P3 stage2 | 0 | ~0.2% |

Gentle partial (top1) has roughly **half** the trainable params of top2 and **quarter** of top4.

---

## 8. Expected Outcomes

| Config | Expected Behavior |
|--------|-------------------|
| P1 (lr_bb=1e-5) | Slight accuracy gain over frozen; moderate gate perturbation |
| P2 (lr_bb=1e-6) | Minimal accuracy change; near-frozen gate behavior |
| P3e1 (staged 1ep) | Brief backbone warm-start, then stable frozen training; gates should be stable |
| P3e2 (staged 2ep) | Slightly more backbone adaptation than P3e1; check if gates remain stable |

**Success criteria**: Gate entropy and coverage remain in a healthy range (not collapsed) across all 3 seeds for each config, while accuracy is at least comparable to frozen baseline.
