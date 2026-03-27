# Exp 7A: Frozen Selector + Sparse Regularization

## Motivation

Exp6 显示 frozen selector 的 gate 行为稳定 (g_t ≈ 0.1-0.27), 说明 gate 在有选择地过滤信息。但 gate 值不够集中 — 很多位置有中等激活而非清晰的开/关。

Sparse regularization 通过 L1 惩罚鼓励 gate 值更低、更稀疏, 使证据选择更集中、可解释性更强。

## Design

在 frozen selector 基础上, loss 增加 L1 sparse 项:

```
L = L_ce + lambda_sparse * L_sparse
L_sparse = mean(temporal_gate.mean(), frequency_gate.mean())
```

### Lambda Sweep

| Config | lambda_sparse | Strength |
|--------|--------------|----------|
| l1e4 | 1e-4 | Light |
| l3e4 | 3e-4 | Medium |
| l1e3 | 1e-3 | Strong |

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
| Seeds | 42, 2025, 3407 |

## Sparse Loss Implementation

`selector_loss.py:141-143` (pre-existing, not modified):

```python
if self.sparse_type == 'l1':
    sparse_loss = sparse_loss + gate.mean()
```

Applied to both `temporal_gate` (B, S, 1) and `frequency_gate` (B, C, 1), averaged. Gate values are sigmoid outputs ∈ (0,1). L1 penalizes mean activation → encourages lower/sparser gate values.

## Logged Metrics

### Loss
- `loss_ce`, `loss_sparse`, `loss_total`

### Gate Statistics (per-epoch, train + val + test)
- `gate_entropy` — binary entropy of gate activations
- `gate_coverage_0.5` — fraction of gates > 0.5
- `gate_top10_ratio` — fraction of total gate mass in top 10% positions
- `gate_top20_ratio` — fraction of total gate mass in top 20% positions
- `abnormal_only_gate_coverage` — gate coverage for abnormal class only
- Per temporal/frequency breakdown

## Jean Zay Submission

```bash
# Submit all 9 jobs (3 lambdas × 3 seeds):
bash experiments/deb/scripts/submit_exp7a_sparse_all_jeanzay.sh codebrain TUAB
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
| `run_exp7a_sparse_l1e4_selector_jeanzay.sh` | lambda=1e-4 |
| `run_exp7a_sparse_l3e4_selector_jeanzay.sh` | lambda=3e-4 |
| `run_exp7a_sparse_l1e3_selector_jeanzay.sh` | lambda=1e-3 |
| `submit_exp7a_sparse_all_jeanzay.sh` | Batch submit |
| `aggregate_exp7a.py` | Results aggregation |

## Output Paths

| Lambda | Log | Checkpoint |
|--------|-----|------------|
| 1e-4 | `deb_log/exp7a_sparse_l1e4_selector/` | `checkpoints_selector/exp7a_sparse_l1e4_selector/` |
| 3e-4 | `deb_log/exp7a_sparse_l3e4_selector/` | `checkpoints_selector/exp7a_sparse_l3e4_selector/` |
| 1e-3 | `deb_log/exp7a_sparse_l1e3_selector/` | `checkpoints_selector/exp7a_sparse_l1e3_selector/` |

## Aggregation

```bash
python experiments/deb/scripts/aggregate_exp7a.py
```

Outputs mean/std across seeds for: BalAcc, Macro-F1, Abnormal F1, gate coverage, gate entropy, gate top10/top20 ratio.

## Total Jobs & Compute

| Item | Value |
|------|-------|
| Total jobs | 9 (3λ × 3 seeds) |
| Expected epochs/run | 25-35 |
| Time/epoch (V100) | ~1200s |
| Time/run | ~10-12h |
| Total | ~90 V100-hours |

## Key Question

Does L1 sparsity improve gate concentration (lower entropy, higher top-K ratio) without degrading classification accuracy?
