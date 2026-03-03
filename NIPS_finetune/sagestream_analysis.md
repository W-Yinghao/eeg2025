# SageStream 全变体实验结果综合分析

## 一、实验概览

- **SageStream** = SA-MoE (Subject-Adaptive Mixture of Experts) + IIB (Information Invariant Bottleneck)
- **两种 Backbone**: CBraMod (Transformer, n_layer=12, 4.9M params) / CodeBrain (EEGSSM, n_layer=8, 15M params)
- **两个数据集**: TUAB (binary, 297K train) / TUEV (6-class, 71K train)
- **8种变体**: full_sagestream, moe_only, iib_only, baseline_ce, high_experts, low_experts, deep_moe, no_style
- **训练配置**: epochs=30, batch=64, lr=1e-3, backbone冻结

### SLURM Job 对应关系

| Job ID | 命令 | Backbone | Dataset |
|--------|------|----------|---------|
| 740097 | `./run_sagestream.sh cbramod TUEV` | CBraMod | TUEV |
| 740099 | `./run_sagestream.sh codebrain TUEV` | CodeBrain | TUEV |
| 740306 | `./run_sagestream.sh cbramod TUAB` | CBraMod | TUAB |
| 740307 | `./run_sagestream.sh codebrain TUAB` | CodeBrain | TUAB |

---

## 二、完整结果汇总表

### TUAB 数据集 (Binary Classification)

| 变体 | CBraMod Bal-Acc | CBraMod F1(macro) | CodeBrain Bal-Acc | CodeBrain F1(macro) |
|------|:---:|:---:|:---:|:---:|
| **moe_only** | **0.7235** | **0.7246** | **0.8135** ★ | **0.8154** ★ |
| baseline_ce | 0.7060 | 0.7047 | 0.7839 | 0.7859 |
| iib_only | 0.7026 | 0.7024 | 0.7820 | 0.7841 |
| no_style | FAIL (load error) | - | N/A | - |
| low_experts | 0.6299 | 0.6269 | ~0.683 (未完成) | ~0.683 |
| full_sagestream | 0.5886\* | FAIL (load) | 0.6826\* | FAIL (load) |
| high_experts | FAIL (load error) | - | 0.6484 | 0.6477 |
| deep_moe | FAIL (load error) | - | N/A | - |

> \*full_sagestream 的最佳 epoch test 结果从训练日志中手动提取，因 checkpoint 加载失败未输出 Final Test Results。

### TUEV 数据集 (6-class Classification)

| 变体 | CBraMod Bal-Acc | CBraMod F1(macro) | CodeBrain Bal-Acc | CodeBrain F1(macro) |
|------|:---:|:---:|:---:|:---:|
| **no_style** | 0.4241 | 0.2509 | **0.5188** ★ | **0.4451** ★ |
| moe_only | **0.4649** | **0.2851** | 0.5313\* | 0.4849\* |
| full_sagestream | 0.3018 | 0.1261 | 0.4238\* | 0.2457\* |
| iib_only | FAIL (load) | - | 0.4351 | 0.4087 |
| baseline_ce | FAIL (load) | - | 0.4542 | 0.4171 |
| high_experts | FAIL (load) | - | 0.2239 | 0.1148 |
| low_experts | FAIL (load) | - | 0.1989 | 0.1324 |
| deep_moe | N/A | - | 0.2963 | 0.1205 |

> \*codebrain moe_only 和 full_sagestream 在 TUEV 上也出现了 checkpoint 加载失败，数据从训练日志中最高 epoch 提取。

---

## 三、关键发现

### 1. Backbone 差异显著

- **CodeBrain 全面优于 CBraMod**，尤其在 TUAB 上差距巨大（moe_only: 0.81 vs 0.72）
- CodeBrain (SSM架构) 在 EEG 时序建模上的优势明显，15M 参数量带来更强的特征表达

### 2. 变体排名一致性

- **moe_only 是目前最稳健的变体**，在 TUAB 两个 backbone 上都是最佳或接近最佳
- **baseline_ce 和 iib_only 表现接近**，说明 IIB 模块在当前设置下未带来显著增益
- **full_sagestream (完整模型) 表现反而最差之一**，这是核心问题

### 3. 严重的 Train-Val-Test Gap

- CodeBrain 在 TUEV 上 train acc 高达 **0.96**，但 test balanced acc 仅 **0.52**，存在严重的过拟合
- full_sagestream 的 val loss 在训练后期不断爆炸（从 1.x 涨到 10+），但 train loss 持续下降

### 4. 大量 Checkpoint 加载失败

- 几乎所有含 IIB 的变体（full_sagestream, iib_only, baseline_ce 等）都出现了 `RuntimeError: Error(s) in loading state_dict for SageStreamModel`
- 这导致最终测试无法在 best checkpoint 上进行，严重影响了结果的可靠性

---

## 四、问题根因分析

### 问题 1: Checkpoint 加载失败（最紧急）

**原因**: 保存 checkpoint 时模型结构与加载时不匹配。可能是 `SageStreamModel` 在推理模式下结构与训练时不同（例如 IIB 的 encoder/decoder 在推理时被移除），或者 `strict=True` 导致任何 key mismatch 都失败。

### 问题 2: full_sagestream 表现最差

**原因**: IIB 和 SA-MoE 的联合训练存在优化冲突：

- IIB 的 KL 损失 (alpha_kl=1e-3) + GRL 对抗损失 (beta_adv=0.5) 与主任务 CE 损失产生梯度干扰
- GRL (gradient reversal layer) 的 lambda 从 0 线性增长到 1，在训练后期对抗强度过大，导致 val loss 爆炸
- 多目标优化中各损失权重未经充分调参

### 问题 3: TUEV 上整体表现差

**原因**:

- TUEV 是 6 类严重不平衡数据集（class 6 占 75%，class 1 仅 0.9%）
- 当前使用标准 CE loss 无任何类别权重/重采样策略
- frozen backbone + lightweight head 难以学习细粒度的 6 类 EEG 事件区分

### 问题 4: high_experts / low_experts / deep_moe 表现极差

**原因**:

- **high_experts** (8 experts): 专家数量过多导致路由不稳定，训练样本不足以让每个专家充分专化，val loss 暴涨到 16+
- **low_experts** (2 experts, top_k=1): 容量不足，单专家激活无法捕获 subject 间的多样性
- **deep_moe** (4层 MoE): 过深的 MoE 层引入过多参数，在 frozen backbone 之上过拟合严重

### 问题 5: IIB 组件未带来增益

**原因**:

- `Subjects: 2` — 日志显示数据集仅检测到 **2个 subject**，这意味着 subject metadata 提取失败
- IIB 的核心是通过 subject-invariant bottleneck 去除受试者特异性信息，但只有 2 个 subject ID 时，对抗训练几乎无意义
- 这是 IIB 失效的根本原因

---

## 五、改进策略

### 高优先级 (Critical)

#### 1. 修复 Subject Metadata 提取

```
当前日志: "No subject metadata found, will extract from samples"
结果: Subjects: 2
```

- IIB 的对抗训练依赖正确的 subject ID，当前仅检测到 2 个 subject 使得 IIB 完全无效
- 需要从 TUAB/TUEV 的文件名或元数据中正确解析 subject ID（TUAB 约有 2329 个 subject，TUEV 约有 300+）
- 这一修复可能会同时显著提升 full_sagestream 和 iib_only 的表现

#### 2. 修复 Checkpoint 保存/加载

- 在 `state_dict` 加载时使用 `strict=False` 并打印 missing/unexpected keys
- 或者确保保存和加载时模型结构完全一致（检查是否有条件分支导致推理模式下缺少某些模块）
- 这样才能获得所有变体的真实 best-checkpoint 测试结果

### 中优先级 (Important)

#### 3. 重新平衡 TUEV 类别

- 使用 weighted CE loss: 按类别频率反比设置权重
- 或使用 focal loss 处理长尾类别
- 考虑过采样少数类 / 欠采样多数类

#### 4. 调整 IIB 超参数

- 降低 `beta_adv` (当前 0.5 → 0.1)：减少对抗梯度的干扰
- 使用更温和的 GRL schedule：当前线性增长太激进，考虑使用 warmup + plateau 策略
- 增大 `alpha_kl` (当前 1e-3 → 1e-2)：加强信息瓶颈约束，减少过拟合

#### 5. 优化 MoE 配置

- 当前最佳: 4 experts, top_k=2 (moe_only)
- 建议保持此配置，但增加 load balancing loss 防止路由坍塌
- 考虑使用 soft routing 替代 hard top-k

### 低优先级 (Nice to Have)

#### 6. 尝试 backbone partial fine-tuning

- 当前 backbone 完全冻结，只训练 head
- 可尝试解冻最后 2-3 层进行 fine-tuning（特别是对 TUEV 这种细粒度任务）

#### 7. 降低学习率

- 当前 lr=1e-3 对于 frozen backbone + lightweight head 可能偏大
- 尝试 lr=3e-4 或使用 cosine annealing with warmup

#### 8. 增加正则化

- CodeBrain 在 TUEV 上严重过拟合（train 0.96 vs test 0.52）
- 增加 dropout、weight decay 或 label smoothing

---

## 六、总结

当前实验的**核心瓶颈**是:

1. **Subject ID 提取失败**导致 IIB 模块完全无效
2. **Checkpoint 加载失败**导致大量实验结果不可靠

修复这两个问题后，full_sagestream 的真实潜力才能被评估。

在当前可靠的结果中，**moe_only + CodeBrain** 在 TUAB 上达到 **0.8135 balanced accuracy**，是最佳配置。TUEV 上所有变体表现都不理想，需要结合类别平衡策略和更细致的超参数调整。
