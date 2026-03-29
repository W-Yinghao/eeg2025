# Exp7C 实验结果总结

## 实验概述

Exp7C 共 3 个变体，均在 gpu-gw (本地集群) 上运行，每个变体 3 seeds (42, 2025, 3407)。

| 变体 | λ_sparse | λ_cons | Sparse 范围 | 目的 |
|------|----------|--------|------------|------|
| **7C-main** | 1e-3 | 3e-3 | both gates (uniform) | 标准联合正则主版 |
| **7C-control** | 3e-4 | 3e-3 | both gates (uniform) | 轻 sparse 对照 |
| **7C-asym** | λ_t=3e-4, λ_f=0 | 3e-3 | temporal only | branch-aware 消融 |

---

## 结果：全部 9 个 run 100% 失败 (NaN 崩溃)

### 7C-main (λ_sparse=1e-3, λ_cons=3e-3)

| Seed | Node | GPU | Ep 1 Train Acc | Early Stop | Test Bal Acc | Status |
|------|------|-----|----------------|-----------|-------------|--------|
| 42 | node10 | V100S-PCIE-32GB | 0.5636 | ep 13 | 0.5000 | **NaN from ep 1** |
| 2025 | node43 | V100S-PCIE-32GB | 0.5465 | ep 13 | 0.5000 | **NaN from ep 1** |
| 3407 | node43 | V100S-PCIE-32GB | 0.5612 | ep 13 | 0.5000 | **NaN from ep 1** |

Banner 确认配置正确：`Sparse: l1 lambda=0.001` + `Consistency: l2 lambda=0.003`

### 7C-control (λ_sparse=3e-4, λ_cons=3e-3)

| Seed | Node | GPU | Ep 1 Train Acc | Early Stop | Test Bal Acc | Status |
|------|------|-----|----------------|-----------|-------------|--------|
| 42 | node14 | V100-PCIE-16GB | 0.5636 | ep 13 | 0.5000 | **NaN from ep 1** |
| 2025 | node15 | V100S-PCIE-32GB | 0.5465 | ep 13 | 0.5000 | **NaN from ep 1** |
| 3407 | node09 | V100-PCIE-16GB | 0.5612 | ep 10 截断 | — | **NaN from ep 1, 被中断** |

Banner 确认：`Sparse: l1 lambda=0.0003` + `Consistency: l2 lambda=0.003`

### 7C-asym (λ_t=3e-4, λ_f=0, λ_cons=3e-3)

| Seed | Node | GPU | Ep 1 Train Acc | Early Stop | Test Bal Acc | Status |
|------|------|-----|----------------|-----------|-------------|--------|
| 42 | node43 | V100S-PCIE-32GB | 0.5624 | ep 13 | 0.5000 | **NaN from ep 1** |
| 2025 | node12 | V100S-PCIE-32GB | 0.5462 | ep 13 | 0.5000 | **NaN from ep 1** |
| 3407 | node13 | V100-PCIE-16GB | 0.5601 | ep 13 | 0.5000 | **NaN from ep 1** |

Banner 确认：`Sparse (branch-aware): l1 lambda_t=0.0003 lambda_f=0.0` + `Consistency: l2 lambda=0.003`

---

## 共性失败特征

所有 9 个 run 表现完全一致：

1. **Epoch 1**: train loss 立即变为 NaN, train acc 在 0.54-0.56 之间（首批之后崩溃）
2. **Epoch 2 起**: train acc 固定在 0.5000, 所有 loss/sparse/cons 指标全部 NaN
3. **Early stopping**: 在 epoch 13 触发 (patience=12, best=epoch 1, metric=0.5000)
4. **混淆矩阵**: `[[19907, 0], [17038, 0]]` — 全部预测为 class 0 (normal)
5. **Gate 统计**: 全部 NaN, coverage 全部 0.0000
6. **新增指标验证**: `delta_gf_abn_minus_norm`, `ratio_gf_abn_over_norm`, `normal_only_*` 字段全部存在但值为 NaN（代码改动正确工作，只是模型本身崩溃）

---

## 根因分析

### 这不是 7C 特有的 bug

**关键对比**：

| 实验 | 平台 | 环境 | Sparse | Consistency | 结果 |
|------|------|------|--------|-------------|------|
| Exp6A selector | gpu-gw | conda eeg2025 | 无 | 无 | **NaN 崩溃** |
| Exp7A sparse | **Jean Zay** | module pytorch-gpu/2.6.0 | 有 | 无 | **正常 (~0.80)** |
| Exp7B consistency | **Jean Zay** | module pytorch-gpu/2.6.0 | 无 | 有 | **正常 (~0.81)** |
| 7C-main | gpu-gw | conda eeg2025 | 有 | 有 | **NaN 崩溃** |
| 7C-control | gpu-gw | conda eeg2025 | 有(轻) | 有 | **NaN 崩溃** |
| 7C-asym | gpu-gw | conda eeg2025 | 有(仅temporal) | 有 | **NaN 崩溃** |

**结论**：Selector head 在 gpu-gw 本地环境下无法训练，无论正则化配置如何。在 Jean Zay 上则可以正常训练。

### 可能原因

1. **AMP (混合精度)**: gpu-gw 脚本使用了 `--amp`，Jean Zay 脚本没有。selector head 的 gate sigmoid 输出在 FP16 下可能产生数值不稳定
2. **PyTorch 版本差异**: gpu-gw 使用 conda env (`eeg2025`), Jean Zay 使用 `module load pytorch-gpu/py3/2.6.0`
3. **数据路径不同**: gpu-gw 用 `/projects/EEG-foundation-model/diagnosis_data/tuab_preprocessed/`, Jean Zay 用 `/lustre/.../TUAB/`，可能存在预处理差异

**最可能的原因是 AMP**。Exp6A frozen selector 在 gpu-gw 上也使用了 `--amp` 并崩溃，而 selector 的 gate sigmoid 在 FP16 下容易溢出。

---

## 下一步建议

1. **在 Jean Zay 上重跑所有 7C 变体**（不使用 AMP），预期应正常训练
2. **或在 gpu-gw 上去掉 `--amp` 重跑一次**以验证 AMP 是否是根因
3. 如果确认是 AMP 问题，后续 gpu-gw 脚本都应去掉 `--amp` 或对 gate 计算添加 `torch.float32` 强制转换

---

## 附：输出文件清单

所有 checkpoint/summary 已保存（虽然记录的是失败结果）：

**7C-main**: `checkpoints_selector/exp7c_main_selector/best_TUAB_codebrain_selector_frozen_acc0.5000_s{42,2025,3407}.*`
**7C-control**: `checkpoints_selector/exp7c_control_selector/best_TUAB_codebrain_selector_frozen_acc0.5000_s{42,2025}.*` (s3407 被中断)
**7C-asym**: `checkpoints_selector/exp7c_asym_sparse_cons_selector/best_TUAB_codebrain_selector_frozen_acc0.5000_s{42,2025,3407}.*`
