# Exp 6B: Gentle Partial Fine-Tuning

## Motivation

Exp6 发现 partial FT (top1/top2/top4) 严重过拟合: best epoch 都在 1-3, train-val gap 高达 15-19%。原因是 backbone LR=1e-4 太大, 导致预训练表征快速退化。

Exp6B 设计了 3 种 "gentle" 策略来解决这个问题, 在保留 partial FT 好处的同时控制过拟合。

## Design: 3 Protocols

### P1: Conservative LR (lr_bb=1e-5)

直接降低 backbone LR 100 倍。

| Parameter | Value |
|-----------|-------|
| Regime | top1 (last 1 block unfrozen) |
| lr_head | 1e-3 |
| lr_backbone | **1e-5** (ratio=100) |
| epochs | 12 |
| patience | 4 |
| warmup | 3 |

### P2: Ultra-Conservative LR (lr_bb=1e-6)

降低 backbone LR 1000 倍。

| Parameter | Value |
|-----------|-------|
| Regime | top1 |
| lr_head | 1e-3 |
| lr_backbone | **1e-6** (ratio=1000) |
| epochs | 12 |
| patience | 4 |
| warmup | 3 |

### P3: Staged Training

两阶段: 先短暂 partial FT warm-start, 再冻结 backbone 只训练 head。

| Stage | Regime | Epochs | lr_head | lr_backbone | Patience |
|-------|--------|--------|---------|-------------|----------|
| Stage 1 | top1 | 1 or 2 | 1e-3 | 1e-5 | N/A (no early stop) |
| Stage 2 | frozen | 20 | 5e-4 | 0.0 | 6 |

两个 variant:
- **P3e1**: stage1 = 1 epoch
- **P3e2**: stage1 = 2 epochs

## Modes

每个 protocol 都同时测试 **baseline** 和 **selector** head:
- baseline: 4 scripts × V100
- selector: 4 scripts × H100 (更快)

## Seeds

42, 2025, 3407 (3 seeds per config)

## Jean Zay Submission

```bash
# Baseline (V100):
bash experiments/deb/scripts/submit_exp6b_gpartial_baseline_all_jeanzay.sh codebrain TUAB

# Selector (H100):
bash experiments/deb/scripts/submit_exp6b_gpartial_selector_all_jeanzay.sh codebrain TUAB
```

### SLURM Resources

| Mode | GPU | Account | Partition | Constraint |
|------|-----|---------|-----------|------------|
| Baseline | V100-32GB | ifd@v100 | (default) | v100-32g |
| Selector | H100 | ifd@h100 | gpu_p6 | h100 |

## Scripts

### Baseline (V100)

| Script | Protocol |
|--------|----------|
| `run_exp6b_gpartial_p1_baseline_jeanzay.sh` | P1 (lr_bb=1e-5) |
| `run_exp6b_gpartial_p2_baseline_jeanzay.sh` | P2 (lr_bb=1e-6) |
| `run_exp6b_gpartial_p3e1_baseline_jeanzay.sh` | P3 staged (1ep warm) |
| `run_exp6b_gpartial_p3e2_baseline_jeanzay.sh` | P3 staged (2ep warm) |
| `submit_exp6b_gpartial_baseline_all_jeanzay.sh` | Batch submit all |

### Selector (H100)

| Script | Protocol |
|--------|----------|
| `run_exp6b_gpartial_p1_selector_jeanzay.sh` | P1 (lr_bb=1e-5) |
| `run_exp6b_gpartial_p2_selector_jeanzay.sh` | P2 (lr_bb=1e-6) |
| `run_exp6b_gpartial_p3e1_selector_jeanzay.sh` | P3 staged (1ep warm) |
| `run_exp6b_gpartial_p3e2_selector_jeanzay.sh` | P3 staged (2ep warm) |
| `submit_exp6b_gpartial_selector_all_jeanzay.sh` | Batch submit all |

## Output Paths

### Baseline

| Protocol | Log | Checkpoint |
|----------|-----|------------|
| P1 | `deb_log/exp6b_gpartial_p1_baseline/` | `checkpoints_selector/exp6b_gpartial_p1_baseline/` |
| P2 | `deb_log/exp6b_gpartial_p2_baseline/` | `checkpoints_selector/exp6b_gpartial_p2_baseline/` |
| P3e1 | `deb_log/exp6b_gpartial_p3e1_baseline/` | `checkpoints_selector/exp6b_gpartial_p3e1_baseline/` |
| P3e2 | `deb_log/exp6b_gpartial_p3e2_baseline/` | `checkpoints_selector/exp6b_gpartial_p3e2_baseline/` |

### Selector

| Protocol | Log | Checkpoint |
|----------|-----|------------|
| P1 | `deb_log/exp6b_gpartial_p1_selector/` | `checkpoints_selector/exp6b_gpartial_p1_selector/` |
| P2 | `deb_log/exp6b_gpartial_p2_selector/` | `checkpoints_selector/exp6b_gpartial_p2_selector/` |
| P3e1 | `deb_log/exp6b_gpartial_p3e1_selector/` | `checkpoints_selector/exp6b_gpartial_p3e1_selector/` |
| P3e2 | `deb_log/exp6b_gpartial_p3e2_selector/` | `checkpoints_selector/exp6b_gpartial_p3e2_selector/` |

## Total Jobs

- Baseline: 4 protocols × 3 seeds = **12 jobs** (V100)
- Selector: 4 protocols × 3 seeds = **12 jobs** (H100)
- **Total: 24 jobs**

## Estimated Compute

| Mode | Epochs/run | Time/epoch | Time/run | Total |
|------|-----------|-----------|---------|-------|
| Baseline P1/P2 (V100) | ~8 | ~1600s | ~3.6h | ~21 V100-h |
| Baseline P3 (V100) | ~15 | ~1400s | ~5.8h | ~35 V100-h |
| Selector P1/P2 (H100) | ~8 | ~530s | ~1.2h | ~7 H100-h |
| Selector P3 (H100) | ~15 | ~470s | ~2.0h | ~12 H100-h |
