# Experiment 6B: Gentle Partial Fine-Tuning for Baseline

> **Date**: 2026-03-27
> **Status**: Implementation complete, ready to submit
> **Predecessor**: Exp 6 (block-level boundary search)

---

## 1. Motivation

Exp 6 showed that `top1` partial FT with `lr_backbone = 1e-4` (10x ratio) causes rapid overfitting in 1-3 epochs. The block count is not the issue — the backbone learning rate is too aggressive.

Exp 6B explores **gentler** partial FT: same block unfreezing (top1 only), but with 100-1000x lower backbone LR, warmup, and shorter training windows.

## 2. Experimental Design

### 2.1 Fixed Settings (all configs)

| Parameter | Value |
|-----------|-------|
| Backbone | CodeBrain (SSSM) |
| Dataset | TUAB (binary: normal vs abnormal) |
| Mode | baseline (pool + MLP head) |
| Unfrozen blocks | 1 (last block only) |
| Patch embedding | Frozen |
| Split | Subject-level |
| Batch size | 64 |
| Gradient clipping | 1.0 |
| Monitor metric | val_balanced_accuracy |
| Seeds | 42, 2025, 3407 |

### 2.2 Four Configurations

#### P1 — Gentle Continuous (100x ratio)

| Parameter | Value |
|-----------|-------|
| Regime | `top1_gentle` |
| lr_head | 1e-3 |
| lr_backbone | **1e-5** |
| Scheduler | cosine |
| Warmup | 3 epochs |
| Max epochs | 12 |
| Early stop patience | 4 |

#### P2 — Ultra-Gentle Continuous (1000x ratio)

| Parameter | Value |
|-----------|-------|
| Regime | `top1_gentle` |
| lr_head | 1e-3 |
| lr_backbone | **1e-6** |
| Scheduler | cosine |
| Warmup | 3 epochs |
| Max epochs | 12 |
| Early stop patience | 4 |

#### P3e1 — Staged Partial (1-epoch warm-start)

| Stage | Blocks | Epochs | lr_head | lr_backbone | Warmup | Patience |
|-------|--------|--------|---------|-------------|--------|----------|
| Stage 1 | top1 unfrozen | **1** | 1e-3 | 1e-5 | 1 | — |
| Stage 2 | all frozen | 20 | 5e-4 | 0 | 2 | 6 |

#### P3e2 — Staged Partial (2-epoch warm-start)

| Stage | Blocks | Epochs | lr_head | lr_backbone | Warmup | Patience |
|-------|--------|--------|---------|-------------|--------|----------|
| Stage 1 | top1 unfrozen | **2** | 1e-3 | 1e-5 | 1 | — |
| Stage 2 | all frozen | 20 | 5e-4 | 0 | 2 | 6 |

### 2.3 Comparison with Exp 6

| Aspect | Exp 6 (top1) | Exp 6B P1 | Exp 6B P2 | Exp 6B P3 |
|--------|-------------|-----------|-----------|-----------|
| Backbone LR | 1e-4 | 1e-5 | 1e-6 | 1e-5 (1-2ep only) |
| Head/BB ratio | 10x | 100x | 1000x | 100x → frozen |
| Warmup | 0 | 3 | 3 | 1 (stage1) + 2 (stage2) |
| Max epochs (BB active) | 100 | 12 | 12 | 1 or 2 |
| Total max epochs | 100 | 12 | 12 | 21 or 22 |
| Early stop | 15 | 4 | 4 | 6 (stage2 only) |

## 3. Implementation

### 3.1 Modified Files

| File | Change |
|------|--------|
| `training/partial_ft.py` | Added `top1_gentle: 1` alias to `REGIME_MAP` |
| `training/selector_trainer.py` | Refactored scheduler into `_build_scheduler()`; added `train_stage()` and `reset_for_stage()` for staged training; added preemption check in `train_stage()`; removed regime from resume checkpoint filename; added `lr_backbone` and `unfrozen_layer_names` to JSON summary |
| `scripts/train_partial_ft.py` | Added `top1_gentle` to regime choices; added `--staged_partial` + 8 stage args; added staged training flow with stage1 completion marker for safe resume |

### 3.2 New Files

| File | Purpose |
|------|---------|
| `scripts/run_exp6b_gpartial_p1_baseline_jeanzay.sh` | SLURM job: P1 config |
| `scripts/run_exp6b_gpartial_p2_baseline_jeanzay.sh` | SLURM job: P2 config |
| `scripts/run_exp6b_gpartial_p3e1_baseline_jeanzay.sh` | SLURM job: P3 stage1=1ep |
| `scripts/run_exp6b_gpartial_p3e2_baseline_jeanzay.sh` | SLURM job: P3 stage1=2ep |
| `scripts/submit_exp6b_gpartial_baseline_all_jeanzay.sh` | Submit all 12 jobs |

### 3.3 Staged Training Flow (`--staged_partial`)

```
main()
  │
  ├── staged=True, no .stage1_done marker
  │     │
  │     ├── Create model with regime=top1 (last block unfrozen)
  │     ├── Create trainer (stage1 LRs, no resume)
  │     ├── trainer.train_stage(N) → N epochs, no early stopping
  │     ├── Freeze all backbone params
  │     ├── Write .stage1_done marker
  │     ├── trainer.reset_for_stage(stage2_cfg)
  │     └── trainer.train() → stage2 with early stopping + final eval
  │
  └── staged=True, .stage1_done marker exists (resume)
        │
        ├── Create model with regime=frozen
        ├── Create trainer (stage2 LRs, --resume $SAVE_DIR)
        ├── Trainer auto-loads resume checkpoint
        └── trainer.train() → continues stage2
```

### 3.4 SLURM Preemption & Auto-Requeue

All scripts implement the full preemption pipeline:

```
#SBATCH --signal=B:USR1@120      ← SLURM sends signal 120s before timeout

bash trap forward_signal USR1     ← bash forwards SIGUSR1 to Python child
python ... &                      ← Python runs in background (required for trap)
TRAIN_PID=$!
wait loop                         ← bash waits, handles signal interruption

Python receives SIGUSR1:
  → _preempted = True
  → After current epoch: save resume checkpoint
  → sys.exit(124)

bash detects exit code 124:
  → sbatch "$0" "$MODEL" "$DATASET" "$SEED"  ← resubmit same script
```

Resume checkpoint contains: model state, optimizer state, scheduler state, best model, epoch, patience counter, training history.

### 3.5 Resume Path Design

Resume checkpoints use a **regime-independent** filename:

```
resume_{dataset}_{model}_{mode}_s{seed}.pth
```

This avoids path mismatch when staged training changes the regime from `top1` to `staged_partial` between stages.

## 4. Recorded Metrics

Each run produces a JSON summary (`_summary.json`) and checkpoint (`.pth`) containing:

- Total params / trainable params / trainable ratio
- Unfrozen layer names
- Best epoch
- Val/test balanced accuracy
- Macro F1
- Abnormal recall / abnormal F1

## 5. Directory Structure

```
checkpoints_selector/
├── exp6b_gpartial_p1_baseline/     ← P1 checkpoints + JSON summaries
├── exp6b_gpartial_p2_baseline/     ← P2
├── exp6b_gpartial_p3e1_baseline/   ← P3e1
└── exp6b_gpartial_p3e2_baseline/   ← P3e2

deb_log/
├── exp6b_gpartial_p1_baseline/     ← P1 SLURM logs (.out/.err)
├── exp6b_gpartial_p2_baseline/     ← P2
├── exp6b_gpartial_p3e1_baseline/   ← P3e1
└── exp6b_gpartial_p3e2_baseline/   ← P3e2
```

## 6. How to Run

```bash
# Submit all 12 jobs (4 configs x 3 seeds)
bash experiments/deb/scripts/submit_exp6b_gpartial_baseline_all_jeanzay.sh codebrain TUAB

# Or submit individual configs
sbatch experiments/deb/scripts/run_exp6b_gpartial_p1_baseline_jeanzay.sh codebrain TUAB 3407
sbatch experiments/deb/scripts/run_exp6b_gpartial_p2_baseline_jeanzay.sh codebrain TUAB 3407
sbatch experiments/deb/scripts/run_exp6b_gpartial_p3e1_baseline_jeanzay.sh codebrain TUAB 3407
sbatch experiments/deb/scripts/run_exp6b_gpartial_p3e2_baseline_jeanzay.sh codebrain TUAB 3407
```

## 7. Verification Checklist

- [x] `top1_gentle` maps to `n_unfreeze=1` (only last block)
- [x] `freeze_patch_embed=True` always (default)
- [x] P1/P2: backbone LR is 100x/1000x lower than head
- [x] P3: stage1 → stage2 transition freezes all backbone params
- [x] P3: `reset_for_stage()` rebuilds optimizer with head-only param group
- [x] P3: `.stage1_done` marker prevents re-running stage1 on resume
- [x] Signal forwarding: bash `trap` → `kill -USR1 $TRAIN_PID`
- [x] Auto-requeue: exit code 124 → `sbatch "$0"`
- [x] Resume: `--resume $SAVE_DIR` passed in all scripts
- [x] Directories: each config has isolated checkpoint + log dirs
- [x] Seeds: 42, 2025, 3407 (3 seeds only)
