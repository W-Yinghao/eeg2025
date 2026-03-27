# Exp 6A: Frozen Baseline Re-run

## Motivation

Exp6 的 frozen baseline 有 7/10 个 run 因 20h SLURM 时限被 CANCELLED (frozen 模式收敛慢, 需要 25-40+ epochs)。本实验补齐这些缺失结果, 形成可靠的 head-only baseline 对照。

## Design

| Item | Value |
|------|-------|
| Mode | baseline (pool + MLP head) |
| Regime | frozen (backbone 完全冻结) |
| Dataset | TUAB (binary: normal vs abnormal) |
| Model | CodeBrain (SSSM) |
| Seeds | 42, 1234, 2025, 3407, 7777 |

### Exp6A vs Exp6 配置差异

| Parameter | Exp6 (old) | Exp6A (new) | Rationale |
|-----------|-----------|-------------|-----------|
| SLURM time | 20h | 20h + **auto-requeue** | 超时自动续跑 |
| epochs | 100 | 50 | frozen baseline ~30 epochs 收敛 |
| patience | 15 | 12 | 更紧凑 |
| warmup_epochs | 0 | 3 | 稳定早期训练 |
| clip_value | 5.0 | 1.0 | 更保守的梯度裁剪 |
| lr_backbone | 1e-4 (via ratio) | 0.0 (explicit) | 明确 head-only |
| scheduler | cosine | cosine + linear warmup | 3 epoch warmup |

### Auto-Requeue 机制

```
SLURM --signal=B:USR1@180  →  Python 捕获信号  →  保存 resume checkpoint
→  sys.exit(124)  →  bash 检测 exit=124  →  sbatch 重新提交自身
→  新 job --resume 自动恢复  →  训练继续
```

## Training Configuration

```bash
python experiments/deb/scripts/train_partial_ft.py \
    --mode baseline --regime frozen \
    --dataset TUAB --model codebrain \
    --epochs 50 --batch_size 64 \
    --lr_head 1e-3 --lr_backbone 0.0 \
    --patience 12 --warmup_epochs 3 \
    --scheduler cosine --clip_value 1.0 \
    --seed {SEED} --cuda 0 \
    --save_dir checkpoints_selector/exp6a_frozen_baseline \
    --split_strategy subject --eval_test_every_epoch \
    --resume checkpoints_selector/exp6a_frozen_baseline
```

## Trainable Parameters

| Component | Total | Trainable |
|-----------|-------|-----------|
| Backbone (CodeBrain) | 15,065,200 | 0 (frozen) |
| Head (pool + MLP) | 104,962 | 104,962 |
| **Total** | **15,170,162** | **104,962 (0.69%)** |

## Jean Zay Submission

```bash
# Batch submit 5 seeds:
bash experiments/deb/scripts/submit_exp6a_frozen_baseline_seeds_jeanzay.sh codebrain TUAB
```

### SLURM Resources

| Resource | Value |
|----------|-------|
| GPU | 1x V100-32GB |
| Account | ifd@v100 |
| Constraint | v100-32g |
| Time limit | 20h (auto-requeue) |
| Signal | USR1@180s |

## Scripts

| Script | Purpose |
|--------|---------|
| `run_exp6a_frozen_baseline_jeanzay.sh` | Single seed SLURM job |
| `submit_exp6a_frozen_baseline_seeds_jeanzay.sh` | Batch submit 5 seeds |
| `run_exp6a_frozen_baseline_local.sh` | Local sequential runner |

## Output Paths

| Type | Path |
|------|------|
| SLURM logs | `$WORK/.../deb_log/exp6a_frozen_baseline/` |
| Checkpoints | `$WORK/.../checkpoints_selector/exp6a_frozen_baseline/` |

Per seed outputs:
- `best_TUAB_codebrain_baseline_frozen_acc{X}_s{SEED}.pth` — best model
- `...json` — structured summary
- `..._classwise.json` — per-class metrics
- `..._config.json` — config snapshot
- `..._curve.csv` — training curve
- `..._summary.md` — markdown summary

## Estimated Compute

| Item | Estimate |
|------|----------|
| Epochs per run | ~25-35 |
| Time per epoch (V100) | ~1200s |
| Time per run | ~10-12h |
| Total (5 seeds) | ~55 V100-hours |
