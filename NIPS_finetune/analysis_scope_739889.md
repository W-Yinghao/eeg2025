# SCOPE Fine-Tuning 实验分析报告 (slurm-739889)

> **日期**: 2026-03-03
> **脚本**: `scripts/run_scope_finetune.sh`
> **方法**: SCOPE (Structured COnfidence-aware Prototype-guided adaptation)
> **论文**: *Structured Prototype-Guided Adaptation for EEG Foundation Models* (arXiv:2602.17251)

---

## 一、方法简介

SCOPE 是一个两阶段的 EEG 基础模型微调框架，核心思想是在标签有限（30% labeled）的条件下，通过结构化原型引导来适配预训练模型：

- **Stage 1 — 外部监督构建**：
  1. **Task-Prior Network (TPN)**：轻量卷积网络 + ETF 正则化，在标注数据上训练，产生类间分离的嵌入
  2. **Prototype Learning**：k-means 初始化 + Sinkhorn 约束均衡分配，捕获类内变异
  3. **Confidence-Aware Fusion**：Dempster-Shafer 理论融合 TPN 和原型预测，仅保留双方一致且置信度 > ρ 的伪标签

- **Stage 2 — ProAdapter 训练**：
  - 冻结 backbone，插入轻量 ProAdapter 模块（FiLM 风格的特征调制）
  - 先 warmup（仅标注数据），再逐步引入伪标签
  - 论文推荐可训练参数比例：**2-5%**

---

## 二、实验配置

### 超参数

| 参数 | CodeBrain+TUEV | CodeBrain+TUAB | CBraMod+TUEV | CBraMod+TUAB |
|------|---------------|---------------|-------------|-------------|
| **Stage 2 epochs** | 60 | 30 | 60 | 50 |
| **batch_size** | 64 | 16 | 64 | 64 |
| **lr** | 5e-4 | 5e-5 | 1e-4 | 1e-4 |
| **weight_decay** | 0.01 | 0.001 | 0.01 | 0.001 |
| **dropout** | 0.1 | 0.1 | 0.1 | 0.1 |
| **adapter_L** | 3 | 3 | 3 | 3 |
| **warmup** | 10 | 10 | 10 | 10 |
| **pseudo_ratio** | 2.0 | 2.0 | 2.0 | 2.0 |

### Stage 1 通用配置

- TPN epochs: 50, TPN lr: 5e-4
- λ_ETF: 0.1, num_prototypes (M): 3, conf_threshold (ρ): 0.5

### 数据集

| 数据集 | 任务 | 类别数 | 通道 | 时长 | 训练/验证/测试 | 标注比例 |
|--------|------|--------|------|------|---------------|---------|
| **TUEV** | 事件分类 | 6 | 16 | 5s | 71343 / 12589 / 29421 | 30% (21402 labeled) |
| **TUAB** | 异常检测 | 2 | 16 | 10s | 297103 / 75407 / 36945 | 30% (89130 labeled) |

---

## 三、实验结果汇总

### 3.1 总体结果

| # | Backbone | 数据集 | 状态 | 测试指标 | Val 最佳 | Early Stop |
|---|----------|--------|------|----------|---------|------------|
| 1 | CodeBrain | TUEV (6类) | ✅ 完成 | **Kappa=0.5619**, BAcc=0.5572, F1=0.7739 | val_kappa=0.9626 (ep31) | ep46 |
| 2 | CodeBrain | TUAB (2类) | ❌ **崩溃** | N/A (test 评估报错) | val_auroc=0.8794 (ep1) | ep16 |
| 3 | CBraMod | TUEV (6类) | ✅ 完成 | **Kappa=0.4850**, BAcc=0.3900, F1=0.7162 | val_kappa=0.6192 (ep2) | ep17 |
| 4 | CBraMod | TUAB (2类) | ✅ 完成 | **AUROC=0.8181**, BAcc=0.7243, AUPRC=0.8146 | val_auroc=0.8338 (ep16) | ep31 |

### 3.2 Stage 1 伪标签生成质量

| # | Backbone | 数据集 | TPN 最终 loss | ETF loss | 原型 loss 变化 | 伪标签接受率 | 置信度 mean |
|---|----------|--------|--------------|----------|-------------|-------------|------------|
| 1 | CodeBrain | TUEV | 0.5001 | 0.0004 | 1.085→1.048 | **60.9%** ✅ | 0.814 |
| 2 | CodeBrain | TUAB | 0.4494 | **0.0000** ⚠️ | 1.095→1.093 | **0.4%** ❌ | 0.015 |
| 3 | CBraMod | TUEV | 0.5001 | 0.0004 | 1.085→1.048 | **60.9%** ✅ | 0.814 |
| 4 | CBraMod | TUAB | 0.4579 | **0.0000** ⚠️ | 1.057→1.016 | **5.1%** ❌ | 0.115 |

> **关键发现**：TUAB（二分类）上 ETF loss 全程为 0，伪标签接受率极低（<5%），Stage 1 几乎完全失效。

### 3.3 ProAdapter 参数量分析

| # | Backbone 参数 | ProAdapter 可训练参数 | 可训练比例 | 论文推荐 | 状态 |
|---|-------------|--------------------|-----------| --------| ----|
| 1 | 15.0M | 16.7M | **52.66%** | 2-5% | ⚠️ 超标 10x |
| 2 | 15.1M | 64.9M | **81.16%** | 2-5% | ⚠️ 超标 16x |
| 3 | 4.9M | 16.7M | **77.37%** | 2-5% | ⚠️ 超标 15x |
| 4 | 4.9M | 64.9M | **93.00%** | 2-5% | ⚠️ 超标 19x |

### 3.4 与其他方法对比

| 组合 | SCOPE | SageStream (moe-only) | SageStream (baseline-ce) | 论文 Full FT |
|------|-------|-----------------------|--------------------------|-------------|
| CodeBrain+TUEV (Kappa) | **0.562** | 0.525 | 0.480 | ~0.67 |
| CodeBrain+TUAB (AUROC) | 崩溃 | **0.814** | 0.79 | ~0.92 |
| CBraMod+TUEV (Kappa) | **0.485** | 0.465 | - | 0.677 |
| CBraMod+TUAB (AUROC) | **0.818** | 0.727 | 0.706 | 0.923 |

> SCOPE 在 TUEV 上略优于 SageStream，在 TUAB 上 CBraMod 的 SCOPE 表现最好，但所有方法都远低于论文 full finetune 的水平。

---

## 四、逐实验详细分析

### 实验1: CodeBrain + TUEV — 部分成功，严重过拟合

**表现**：
- Test Kappa=0.5619，在所有微调方法中属于中等偏上
- 但 val-test gap 巨大：val_kappa=0.9626 vs test_kappa=0.5619，差 **0.40**

**Stage 1**：正常。TPN loss 稳定下降，伪标签 60.9% 接受率，置信度 0.814。

**Stage 2 训练过程**：
```
Epoch  1/60 | loss=0.3443 | kappa=0.8960
Epoch 10/60 | loss=0.0480 | kappa=0.9511
Epoch 30/60 | loss=0.0253 | kappa=0.9570   ← 训练 loss 很低，val 很高
Epoch 45/60 | loss=0.0150 | kappa=0.9597
Early stopping at epoch 46 (best: 31)
Best val_kappa: 0.9626
Test kappa: 0.5619                          ← 巨大落差
```

**问题诊断**：
1. **验证集泄漏**：TUEV 的 val 和 train 来自同一文件 (`tuev_preprocessed/train`)，仅做 85/15 随机切分，导致 val 指标虚高
2. **ProAdapter 过大**：可训练参数 16.7M 接近 backbone 的 15M，相当于在做全微调而非轻量适配
3. 训练 loss 降到 0.015，但泛化不佳 → 典型过拟合

---

### 实验2: CodeBrain + TUAB — 失败（Bug + 伪标签崩溃）

**Bug — test 评估崩溃**：
```python
ValueError: Found input variables with inconsistent numbers of samples: [36945, 36960]
# 位置: train_scope.py:699, evaluate() 中的 roc_auc_score()
```

- 测试集 36945 样本，模型输出 36960 个预测
- **原因**：batch_size=16，36945 % 16 = 1，最后一个 batch 被填充到 16，多出 15 个预测
- test DataLoader 没有 `drop_last`，evaluate 函数未做截断处理

**伪标签崩溃（更严重）**：
```
Pseudo-labels: 813/207973 accepted (0.4%)    ← 几乎为零！
Confidence: mean=0.015, std=0.067            ← 极低
```

- TPN 的 ETF loss 在整个训练过程中为 **0.0000**（二分类时 ETF 被禁用）
- 原型学习 loss 几乎不下降（1.0951 → 1.0933）
- Stage 2 在 epoch 1 就达到 val 最优（auroc=0.8794），之后持续退化
- 说明 ProAdapter 仅依赖标注数据，没有伪标签辅助

**训练过程**：
```
Epoch  1/30 | loss=0.4373 | auroc=0.8794   ← 最佳在第1个 epoch
Epoch  5/30 | loss=0.1530 | auroc=0.8623   ← 开始退化
Epoch 10/30 | loss=0.0539 | auroc=0.8599
Epoch 15/30 | loss=0.0284 | auroc=0.8624
Early stopping at epoch 16 (best: 1)        ← best 在 epoch 1
```

---

### 实验3: CBraMod + TUEV — 完成但表现差

**表现**：Kappa=0.4850，远低于 CBraMod 论文报告的 0.677（full finetune）。

**训练极不稳定**：
```
Epoch  1/60 | loss=0.8610 | kappa=0.4974
Epoch  5/60 | loss=0.5411 | kappa=0.5371
Epoch 10/60 | loss=0.4689 | kappa=0.4218   ← 反而下降
Epoch 15/60 | loss=0.4090 | kappa=0.5354
Early stopping at epoch 17 (best: 2)        ← best 在 epoch 2
```

**问题**：
- Val kappa 大幅波动（0.4 ↔ 0.6），训练不收敛
- Loss 下降缓慢（epoch 17 时仍有 0.41），对比 CodeBrain+TUEV 的 epoch 1 仅 0.34
- 可训练比例 77.37% → ProAdapter 严重过大
- LR=1e-4 对 CBraMod (4.9M backbone) 可能过高，导致 adapter 参数剧烈震荡

---

### 实验4: CBraMod + TUAB — 相对最好

**表现**：AUROC=0.8181, BAcc=0.7243 — 四组实验中最稳定。

**训练过程**：
```
Epoch  1/50 | loss=0.6002 | auroc=0.7995
Epoch 10/50 | loss=0.5226 | auroc=0.8238
Epoch 16/50 | ...         | auroc=0.8338   ← best
Epoch 30/50 | loss=0.4750 | auroc=0.8142   ← 开始退化
Early stopping at epoch 31 (best: 16)
```

**正面**：
- 训练前期稳步上升，相对最稳定
- 原型接受率 5.1%（虽低但比 CodeBrain TUAB 的 0.4% 好）

**不足**：
- 仍远低于论文 full finetune 的 AUROC=0.923（差 0.105）
- 可训练比例 93%，本质上已接近全微调但效果更差
- 伪标签利用率依然很低（5.1%）

---

## 五、核心问题诊断

### 问题1: 二分类任务 Stage 1 完全失效 [严重]

SCOPE 的核心创新之一是 ETF (Equiangular Tight Frame) 正则化，确保 TPN 嵌入空间中类间最大分离。**但 ETF 仅对 K≥3 类有效**（数学上，二分类的 simplex ETF 退化为对径向量，无法提供有效约束）。

**影响链**：
```
ETF 不工作 → TPN 嵌入类间分离不足 → 原型学习失败（loss 不下降）
→ 置信度极低 → 伪标签接受率 < 5% → Stage 2 缺少伪标签 → 退化为普通 frozen backbone + adapter
```

这是 TUAB 上所有实验表现差的根本原因。

### 问题2: ProAdapter 参数量超标 10-20 倍 [严重]

论文设计 ProAdapter 为轻量模块（2-5% 可训练参数），但当前实现中：
- ProAdapter 内部维度过大（可能直接使用了 backbone 的完整特征维度而非降维后的表示）
- TUAB (seq_len=10) 的特征维度比 TUEV (seq_len=5) 更大，导致 TUAB 上 ProAdapter 更加膨胀（64.9M vs 16.7M）

**后果**：参数过多 → 过拟合 → 训练不稳定 → 泛化差

### 问题3: 验证集划分不当 [中等]

TUEV 的 val 与 train 来自同一源文件（`tuev_preprocessed/train`），仅做随机 split。这导致：
- Val 指标虚高（val_kappa=0.96 vs test_kappa=0.56）
- Early stopping 选择的模型不一定是泛化最好的
- 无法信赖 val 指标进行超参数调优

### 问题4: Test DataLoader batch 对齐 Bug [低但影响结果]

`evaluate()` 函数中，test DataLoader 没有 `drop_last=True`，最后一个不完整 batch 的预测被填充，导致 predictions 数量 > labels 数量，`roc_auc_score` 报错。

---

## 六、改进方案

### P0 — 紧急修复

#### 6.1 修复 evaluate batch padding bug

在 `train_scope.py` 的 `evaluate()` 函数中，截断到实际样本数：

```python
n_samples = len(dataloader.dataset)
all_preds = all_preds[:n_samples]
all_labels = all_labels[:n_samples]
all_probs = all_probs[:n_samples]
```

或在创建 test DataLoader 时避免 padding。

#### 6.2 修复二分类伪标签生成

ETF 在 K=2 时失效，需要替代方案：

**方案 A — 对比学习替代 ETF**：
```python
# 用 Supervised Contrastive Loss 替代 ETF
L_TPN = L_CE + λ_contrastive * SupConLoss(embeddings, labels)
```
SupConLoss 对任意类别数（包括 K=2）都有效，且同样促进类间分离。

**方案 B — Angular Margin (ArcFace)**：
```python
# 用 ArcFace/CosFace loss 替代 CE + ETF
L_TPN = ArcFaceLoss(embeddings, labels, margin=0.5, scale=30)
```
在嵌入空间的角度上施加类间间距约束，天然适配二分类。

**方案 C — 降低置信度阈值**：
```python
# 二分类时降低 ρ（从 0.5 → 0.3 或 0.2）
conf_threshold = 0.3 if num_classes == 2 else 0.5
```
作为临时缓解措施，放宽接受条件以获得更多伪标签。

### P1 — 关键改进

#### 6.3 缩小 ProAdapter 参数量

**目标**：可训练比例从当前 50-93% 降至 5-10%

**具体措施**：
1. 在 ProAdapter 前加 **特征池化/降维**（如 AdaptiveAvgPool 或线性投影），将 (C × seq_len × 200) 压缩为低维表示
2. ProAdapter 内部隐藏维度设为 backbone 维度的 **1/8 ~ 1/4**（如 200 → 32 或 50）
3. 减少 adapter_L：CodeBrain (8层) 用 **L=1-2**，CBraMod (12层) 用 **L=2-3**

**预期效果**：大幅减少过拟合，缩小 val-test gap。

#### 6.4 修复验证集划分

**TUEV**：使用 `tuev_preprocessed/eval` 的一部分作为验证集，或基于 subject-level 切分 train，确保 val 与 train 无重叠。

**TUAB**：当前已使用独立的 `tuab_preprocessed/val`，划分合理。

### P2 — 进一步优化

#### 6.5 超参数调整

| 参数 | 当前值 | 建议值 | 原因 |
|------|-------|-------|------|
| CBraMod LR | 1e-4 | **5e-5** | 当前训练不稳定，降低 LR 提高稳定性 |
| TUAB warmup | 10 | **15-20** | 伪标签质量低时，延长纯监督阶段 |
| TUAB num_prototypes (M) | 3 | **1-2** | 二分类不需要过多原型 |
| TUAB conf_threshold (ρ) | 0.5 | **0.3** | 放宽接受条件 |

#### 6.6 混合方法探索

鉴于 SageStream 的 moe-only 在 CodeBrain+TUAB 上达到 0.814（优于 SCOPE），可考虑：
- 将 SCOPE 的原型条件化引入 MoE 的 expert routing
- 或用 MoE 的 gating 机制替代 ProAdapter 的 prototype conditioning

#### 6.7 更多 backbone 适配

SCOPE 论文在 5 个 backbone 上验证（LaBraM, CBraMod, CSBrain, EEGMamba, CodeBrain），当前仅测试了 2 个。如果资源允许，可以测试 LaBraM（Transformer 架构，可能与 ProAdapter 更兼容）。

---

## 七、总结与优先级

| 优先级 | 改进项 | 预期收益 | 工作量 |
|--------|--------|---------|-------|
| **P0** | 修复 evaluate batch padding bug | 解锁 CodeBrain+TUAB 结果 | 5 min |
| **P0** | 修复二分类伪标签（替代 ETF） | 大幅提升 TUAB 性能 | 2-4 h |
| **P1** | 缩小 ProAdapter 至 5-10% 可训练参数 | 减少过拟合，提升泛化 | 2-3 h |
| **P1** | 修复 TUEV 验证集划分 | 可靠的模型选择 | 30 min |
| **P2** | 调整二分类超参数 (LR, M, ρ) | 进一步提升 | 1-2 h (实验) |
| **P2** | 混合 SCOPE + MoE | 探索性提升 | 4-8 h |

**核心瓶颈**：
1. **二分类 Stage 1 失效** — ETF 不工作 → 伪标签接受率 < 5% → Stage 2 退化
2. **ProAdapter 过大** — 可训练参数 50-93%（论文推荐 2-5%）→ 严重过拟合

修复这两个核心问题后，SCOPE 在 TUAB 上的 AUROC 预期可从 0.82 提升至 0.87+，在 TUEV 上的 Kappa 预期可从 0.48-0.56 提升至 0.60+。
