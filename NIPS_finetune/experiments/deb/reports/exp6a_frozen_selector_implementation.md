# Exp 6A: Frozen Selector Supplementary Runs — Implementation Report

> **Date**: 2026-03-27
> **Platform**: Jean Zay (IDRIS), H100 (priority)
> **Seeds**: 42, 1234, 2025, 3407, 7777

---

## 1. Background & Motivation

Exp 6 (boundary search) ran a full factorial: 4 regimes x 2 modes x 5 seeds = 40 runs on V100.
Key findings:
- **Frozen selector**: gates stable (g_t ~ 0.10–0.27), good for interpretability
- **Partial FT (top1/top2/top4)**: gates destabilize (0.07–1.00 cross-seed variance)
- **Problem**: 7/10 frozen runs were **CANCELLED** at 20h V100 time limit (seeds 1234, 3407, 7777 for selector; others for baseline)

Exp 6A re-runs **frozen selector only** on H100 with:
- Enhanced gate metrics tracking (detailed per-epoch stats, abnormal-only breakdown)
- AMP (mixed precision) for faster training
- Auto-requeue mechanism to handle SLURM time limits gracefully

---

## 2. Changes Summary

### 2.1 New Files

| File | Description |
|------|-------------|
| `scripts/run_exp6a_frozen_selector_jeanzay.sh` | SLURM job script for H100, 20h time limit, auto-requeue on preemption |
| `scripts/submit_exp6a_frozen_selector_seeds_jeanzay.sh` | Batch submission for 5 seeds |

### 2.2 Modified Files

| File | Changes |
|------|---------|
| `training/selector_trainer.py` | +AMP, +gate collection in `_evaluate`, +`_compute_detailed_gate_stats`, +`_compute_topk_ratios`, +`_save_epoch_gate_csv`, +gate_stats in JSON/Markdown summary |
| `scripts/train_partial_ft.py` | +`--amp` flag for mixed precision |

### 2.3 Unchanged (Reused As-Is)

| File | Role |
|------|------|
| `models/selector_model.py` | SelectorModel with `apply_partial_ft_regime()` |
| `models/selector_enhanced.py` | EnhancedSelectorHead (TemporalGate + FrequencyGate + fusion) |
| `models/evidence_bottleneck.py` | TemporalGate / FrequencyGate implementations |
| `models/backbone_wrapper.py` | BackboneWrapper for CodeBrain |
| `data/batch_protocol.py` | `load_deb_data()` with subject-level splits |
| `training/partial_ft.py` | `apply_true_partial_ft()` regime logic |

---

## 3. Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | TUAB (binary: normal vs abnormal) |
| Model | CodeBrain (SSSM) |
| Mode | `selector` (temporal + frequency gating) |
| Regime | `frozen` (0 backbone blocks unfrozen) |
| `lr_head` | 1e-3 |
| `lr_backbone` | 0.0 (frozen) |
| Scheduler | Cosine with 3-epoch linear warmup |
| Max epochs | 50 |
| Patience | 12 (early stopping on val balanced accuracy) |
| Batch size | 64 |
| Grad clip | 1.0 |
| Split | Subject-level |
| AMP | Enabled (float16 forward, float32 backward) |
| Seeds | 42, 1234, 2025, 3407, 7777 |

---

## 4. Implementation Details

### 4.1 AMP (Automatic Mixed Precision)

Added to `SelectorTrainer.__init__`:
```python
self.use_amp = cfg.get('amp', False)
self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
```

Training loop (`_train_epoch`):
```python
with torch.amp.autocast('cuda', enabled=self.use_amp):
    out = self.model(x, return_gates=True)
    # ... loss computation ...

if self.scaler is not None:
    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
    self.scaler.step(self.optimizer)
    self.scaler.update()
```

Evaluation (`_evaluate`):
```python
with torch.amp.autocast('cuda', enabled=self.use_amp):
    out = self.model(x, return_gates=True)
```

Gate tensors are cast to `.float()` before stats computation to avoid float16 precision issues.

### 4.2 Gate Collection & Detailed Stats

`_evaluate()` now accepts `collect_gates: bool = False`. When True and mode is `selector`:
- Collects `temporal_gate` and `frequency_gate` tensors across all batches
- Calls `_compute_detailed_gate_stats()` to produce:

| Metric | Description |
|--------|-------------|
| `g_t_mean`, `g_t_std` | Temporal gate mean/std activation |
| `g_f_mean`, `g_f_std` | Frequency gate mean/std activation |
| `gate_entropy_temporal` | Binary entropy of temporal gate |
| `gate_entropy_frequency` | Binary entropy of frequency gate |
| `gate_entropy` | Average of temporal + frequency entropy |
| `gate_coverage_temporal_0.5` | Fraction of temporal gate > 0.5 |
| `gate_coverage_frequency_0.5` | Fraction of frequency gate > 0.5 |
| `gate_coverage_0.5` | Average of both coverages |
| `gate_top10_ratio_*` | Fraction of total gate mass in top 10% positions |
| `gate_top20_ratio_*` | Fraction of total gate mass in top 20% positions |
| `abnormal_only_*` | All above metrics filtered to class 0 (abnormal) only |

### 4.3 Per-Epoch Gate CSV

`_save_epoch_gate_csv()` saves a CSV with one row per epoch, columns include:
- `epoch`
- `train_g_t_mean`, `train_g_f_mean` (from batch-level averages during training)
- `val_g_t_mean`, `val_g_f_mean`, `val_gate_entropy`, `val_gate_coverage_0.5`, ... (from detailed collection)
- `test_*` variants (when `--eval_test_every_epoch` is set)

### 4.4 Auto-Requeue (SLURM Preemption Handling)

The full chain:

1. **SLURM**: `#SBATCH --signal=B:USR1@180` sends `SIGUSR1` 3 minutes before 20h timeout
2. **Trainer**: `_setup_signal_handler()` catches signal, sets `self._preempted = True`
3. **Trainer**: After current epoch finishes, calls `_save_resume_checkpoint()` with full state:
   - `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`
   - `best_val_metric`, `best_epoch`, `best_model_state` (deepcopy)
   - `patience_counter`, `train_history`, `epoch`
4. **Trainer**: `sys.exit(124)`
5. **Shell script**: Detects exit code 124, resubmits itself via `sbatch`
6. **Next job**: `--resume "$SAVE_DIR"` auto-detects `resume_TUAB_codebrain_selector_s{seed}.pth`, restores all state, continues from next epoch

Resume checkpoint is **deleted after loading** to prevent re-resume on completed runs.

---

## 5. Output Artifacts

Per seed, the following files are saved to `checkpoints_selector/exp6a_frozen_selector/`:

| File | Content |
|------|---------|
| `best_TUAB_codebrain_selector_frozen_acc{ACC}_s{SEED}.pth` | Best model checkpoint |
| `best_..._s{SEED}.json` | JSON summary (metrics + gate_stats + config + regime_info) |
| `best_..._s{SEED}_config.json` | Full config snapshot |
| `best_..._s{SEED}_classwise.json` | Per-class F1/recall/precision |
| `best_..._s{SEED}_gate_stats.json` | Detailed gate statistics (test set) |
| `best_..._s{SEED}_epoch_gate_stats.csv` | Per-epoch gate stats (train/val/test) |
| `best_..._s{SEED}_curve.csv` | Training curve (loss, accuracy per epoch) |
| `best_..._s{SEED}_summary.md` | Human-readable markdown summary |

---

## 6. Directory Layout

```
$WORK/yinghao/eeg2025/NIPS_finetune/
├── checkpoints_selector/
│   ├── exp6a_frozen_selector/     # <-- Exp 6A selector (NEW, independent)
│   ├── exp6a_frozen_baseline/     # <-- Exp 6A baseline (existing)
│   └── exp6/                      # <-- Original Exp 6 (untouched)
├── deb_log/
│   ├── exp6a_frozen_selector/     # <-- SLURM logs (NEW, independent)
│   ├── exp6a_frozen_baseline/     # <-- Existing
│   └── exp6/                      # <-- Original Exp 6 (untouched)
```

No files from previous experiments are overwritten.

---

## 7. Commands

### Submit all 5 seeds
```bash
bash experiments/deb/scripts/submit_exp6a_frozen_selector_seeds_jeanzay.sh codebrain TUAB
```

### Submit single seed
```bash
sbatch experiments/deb/scripts/run_exp6a_frozen_selector_jeanzay.sh codebrain TUAB 3407
```

### Local test (no SLURM)
```bash
python experiments/deb/scripts/train_partial_ft.py \
    --mode selector --dataset TUAB --model codebrain \
    --regime frozen --epochs 50 --batch_size 64 \
    --lr_head 1e-3 --lr_backbone 0.0 --patience 12 \
    --warmup_epochs 3 --clip_value 1.0 --scheduler cosine \
    --seed 3407 --cuda 0 --amp --eval_test_every_epoch \
    --save_dir checkpoints_selector/exp6a_frozen_selector
```

---

## 8. Verification Checklist

| Check | Status |
|-------|--------|
| Frozen backbone: all backbone params `requires_grad=False`, `lr_backbone=0.0` | PASS |
| SelectorModel returns gates when `return_gates=True` | PASS |
| Gate metrics collected in `_evaluate` with `collect_gates=True` | PASS |
| Detailed gate stats: g_t/g_f mean/std, entropy, coverage, top-K, abnormal-only | PASS |
| Gate CSV export via `_save_epoch_gate_csv` | PASS |
| `gate_stats` included in JSON summary (`_build_json_summary`) | PASS |
| AMP: `torch.amp.autocast` + `GradScaler`, backward-compatible (no-op when disabled) | PASS |
| Jean Zay scripts use independent `exp6a_frozen_selector/` directories | PASS |
| SLURM: `--time=20:00:00` + `--signal=B:USR1@180` + exit 124 requeue | PASS |
| No sparse/consistency/VIB regularization | PASS |
