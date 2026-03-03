# IB + Disentanglement 全变体实验结果综合分析

## 一、实验概览

- **方法**: IB (Information Bottleneck) + GRL (Gradient Reversal Layer) Adversarial Subject Removal
- **Backbone**: CodeBrain (EEGSSM, n_layer=8, 15M params, frozen)
- **两个数据集**: TUAB (binary, 297K train) / TUEV (6-class, 71K train)
- **8种变体**: full_ib_adv, ib_only, adv_only, baseline_ce, high_beta, low_beta, high_lambda, low_lambda
- **训练配置**: epochs=30, batch=64, lr=1e-3, latent_dim=128, patience=15

### SLURM Job 对应关系

| Job ID | 命令 | Backbone | Dataset | 状态 |
|--------|------|----------|---------|------|
| 739416 | `run_ib.sh TUAB` | CodeBrain | TUAB | 7/8 完成, low_lambda 被 SLURM 时限取消 |
| 739420 | `run_ib.sh TUEV` | CodeBrain | TUEV | 8/8 全部完成 |

### 变体配置说明

| 变体 | beta (IB) | lambda_adv | 说明 |
|------|:---------:|:----------:|------|
| full_ib_adv | 1e-3 | 0.5 | 完整模型: IB + adversarial |
| ib_only | 1e-3 | 0.0 | 仅 IB, 无对抗训练 |
| adv_only | 0.0 | 0.5 | 仅对抗训练, 无 IB |
| baseline_ce | 0.0 | 0.0 | 仅 CE loss (基线) |
| high_beta | 1e-2 | 0.5 | 高 IB 权重 (10x) |
| low_beta | 1e-4 | 0.5 | 低 IB 权重 (0.1x) |
| high_lambda | 1e-3 | 1.0 | 高对抗权重 (2x) |
| low_lambda | 1e-3 | 0.1 | 低对抗权重 (0.2x) |

---

## 二、完整结果汇总表

### TUAB 数据集 (Binary Classification) — Job 739416

| 变体 | Bal-Acc | F1(macro) | F1(weighted) | F1(normal) | F1(abnormal) | Best Epoch | Best Val Acc |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **high_lambda** | **0.8124** | **0.8143** | **0.8161** | 0.8374 | **0.7912** | 26 | 0.8145 |
| ib_only | 0.8116 | 0.8135 | 0.8153 | 0.8367 | 0.7903 | 26 | 0.8146 |
| low_beta | 0.8110 | 0.8128 | 0.8145 | 0.8352 | 0.7903 | 11 | 0.8124 |
| full_ib_adv | 0.8101 | 0.8124 | 0.8145 | **0.8387** | 0.7862 | 11 | 0.8130 |
| high_beta | 0.8098 | 0.8118 | 0.8136 | 0.8356 | 0.7879 | 11 | 0.8125 |
| adv_only | 0.8096 | 0.8118 | 0.8138 | 0.8374 | 0.7862 | 14 | 0.8145 |
| baseline_ce | 0.8087 | 0.8107 | 0.8126 | 0.8354 | 0.7860 | 11 | 0.8120 |
| low_lambda | — | — | — | — | — | (epoch 4 取消) | — |

> low_lambda 在 epoch 4/30 时被 SLURM 时限取消，未完成训练。

### TUEV 数据集 (6-class Classification) — Job 739420

| 变体 | Bal-Acc | F1(macro) | F1(weighted) | F1(spsw) | F1(gped) | F1(pled) | F1(eyem) | F1(artf) | F1(bckg) | Best Epoch | Best Val Acc |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **high_lambda** | **0.5150** | **0.4648** | 0.7117 | **0.0578** | 0.4281 | 0.4325 | 0.5094 | 0.5070 | 0.8542 | 29 | 0.9396 |
| high_beta | 0.5122 | 0.4549 | 0.7052 | 0.0000 | 0.4557 | 0.4455 | 0.4580 | 0.5107 | 0.8593 | 29 | 0.9149 |
| low_lambda | 0.4994 | 0.4643 | 0.7125 | 0.0000 | 0.4638 | 0.4482 | 0.5094 | 0.5011 | 0.8632 | 29 | 0.9383 |
| full_ib_adv | 0.4965 | 0.4580 | 0.7114 | 0.0000 | 0.4268 | 0.4656 | 0.5086 | 0.4947 | 0.8523 | 29 | 0.9386 |
| ib_only | 0.4965 | 0.4643 | 0.7130 | 0.0000 | 0.4774 | 0.4515 | 0.5049 | 0.4866 | 0.8657 | 29 | 0.9408 |
| baseline_ce | 0.4934 | 0.4642 | 0.7135 | 0.0000 | 0.3515 | **0.5256** | **0.5588** | 0.4920 | 0.8575 | 28 | 0.9460 |
| adv_only | 0.4835 | 0.4546 | 0.7131 | 0.0000 | 0.4651 | 0.3620 | 0.5489 | 0.4682 | **0.8836** | 25 | 0.9416 |
| low_beta | 0.4815 | 0.4546 | 0.7086 | 0.0000 | 0.4137 | 0.4411 | 0.5664 | 0.4495 | 0.8572 | 29 | 0.9400 |

---

## 三、关键发现

### 1. Subject ID 提取失败 — 两个数据集均确认

所有实验均显示 `Subjects: 2`，而实际 TUAB 有 ~2329 subjects，TUEV 有 ~243 subjects。

```
Wrapping datasets with subject ID support:
  No subject metadata found, will extract from samples
Subjects: 2
```

**影响**: adversarial subject removal (GRL) 和 IIB 的核心机制依赖正确的 subject ID。仅 2 个 subject 使得对抗训练几乎无意义，所有 IB/adversarial 相关变体的效果被严重削弱。

### 2. TUAB: 所有变体表现极度相似

| 指标 | 最佳 (high_lambda) | 最差 (baseline_ce) | 差距 |
|------|:---:|:---:|:---:|
| Bal-Acc | 0.8124 | 0.8087 | **0.37%** |
| F1(macro) | 0.8143 | 0.8107 | **0.36%** |

- 8 个变体的 Bal-Acc 差距仅 0.37%，远在统计误差范围内
- **IB 和 adversarial 组件未带来任何有意义的增益**
- baseline_ce (纯 CE loss) 排名最末，但与最佳仅差 0.37%
- 这进一步证实：在 `Subjects: 2` 的情况下，所有 subject-aware 模块失效

### 3. TUEV: 整体表现差，存在严重过拟合

| 阶段 | Acc 范围 |
|------|---------|
| Train acc (epoch 30) | 0.83 - 0.91 |
| Val acc (best) | 0.91 - 0.95 |
| **Test balanced acc** | **0.48 - 0.52** |

- **Train-Val-Test Gap 巨大**: train 0.88 → val 0.94 → test 0.50
- Test loss 在训练过程中从 ~1.1 持续上涨到 **2.0-2.5**，而 val loss 下降到 0.06
- 说明 val 和 test 分布存在严重偏移，或 val split 存在数据泄漏问题

### 4. TUEV 类别严重不平衡

| 类别 | Train 样本数 | 占比 | Test F1 (best) |
|------|:-----------:|:----:|:--------------:|
| bckg (6) | 53,726 | 75.3% | 0.88 |
| eyem (2) | 11,254 | 15.8% | 0.57 |
| artf (5) | 11,053 | 15.5% | 0.51 |
| gped (3) | 6,184 | 8.7% | 0.48 |
| pled (4) | 1,070 | 1.5% | 0.53 |
| **spsw (1)** | **645** | **0.9%** | **0.00 (7/8 变体)** |

- **spsw (class 1) F1 = 0.0000**: 7/8 变体完全无法识别此类
- 唯一非零的是 high_lambda (F1=0.0578)，但也极低
- bckg 占 75%，模型偏向预测多数类
- 当前使用标准 CE loss 无类别权重/重采样策略

### 5. TUAB vs TUEV 难度差异

| 数据集 | 最佳 Bal-Acc | Train-Test Gap | 类别 | 核心挑战 |
|--------|:-----------:|:--------------:|:----:|---------|
| TUAB | 0.8124 | ~2% (小) | 2 (balanced) | 相对简单，baseline 已接近 0.81 |
| TUEV | 0.5150 | ~40% (巨大) | 6 (极不平衡) | 类别不平衡 + 严重过拟合 + 分布偏移 |

### 6. TUAB Checkpoint 可能存在覆盖问题

- full_ib_adv 和 ib_only 的 Test Bal-Acc 完全相同 (0.4965 on TUEV)
- 多个变体的 Best Epoch 相同 (TUAB 上 5 个变体的 best epoch 都是 11)
- 需要检查 `train_ib_disentangle.py` 的 checkpoint 保存逻辑是否存在与 SageStream 相同的文件名覆盖问题

---

## 四、与 SageStream 实验结果对比

### TUAB 最佳结果对比 (CodeBrain backbone)

| 方法 | 最佳变体 | Bal-Acc | F1(macro) |
|------|---------|:-------:|:---------:|
| **SageStream** (SA-MoE) | moe_only | **0.8135** | **0.8154** |
| **IB + Disentangle** | high_lambda | 0.8124 | 0.8143 |
| SageStream baseline_ce | — | 0.7839 | 0.7859 |
| IB baseline_ce | — | 0.8087 | 0.8107 |

- SageStream moe_only 略优于 IB high_lambda，但差距极小 (0.11%)
- **IB 的 baseline_ce (0.8087) 显著高于 SageStream 的 baseline_ce (0.7839)**，说明 IB 框架的分类头设计更优

### TUEV 最佳结果对比 (CodeBrain backbone)

| 方法 | 最佳变体 | Bal-Acc | F1(macro) |
|------|---------|:-------:|:---------:|
| **SageStream** | moe_only | 0.5313* | 0.4849* |
| **IB + Disentangle** | high_lambda | 0.5150 | 0.4648 |

> *SageStream moe_only 结果从训练日志提取（checkpoint 加载失败）

- 两个框架在 TUEV 上均表现不佳（Bal-Acc ~0.50-0.53）
- 共同瓶颈：TUEV 类别不平衡 + frozen backbone 特征不够细粒度

---

## 五、问题根因分析

### 问题 1: Subject ID 提取失败（根本问题）

**原因**: `EEGDatasetWithSubjects._build_subject_map()` 检查 LMDB metadata 的 `subjects` key（不存在），然后 lazy fallback 机制有 bug，最终只检测到 2 个 subject。

**状态**: ✅ **已修复** — `train_ib_disentangle.py` 中的 `_build_subject_map()` 已重写，从 LMDB 样本的 `source_file` 字段提取 subject ID（如 `aaaaamde_s001_t001.edf` → subject `aaaaamde`）。

**预期影响**: 修复后 TUAB 应检测到 ~2329 subjects，TUEV ~243 subjects。IB adversarial training 和 full_ib_adv 变体应该表现出与 baseline_ce 的显著差异。

### 问题 2: TUEV 类别不平衡

**原因**: bckg 占 75%，spsw 仅占 0.9%。标准 CE loss 使模型偏向多数类。

**改进**:
- 使用 weighted CE loss（按类别频率反比加权）
- 或使用 focal loss 处理长尾类别
- 考虑过采样少数类

### 问题 3: TUEV 上 Val-Test 分布偏移

**现象**: Val acc 0.94 vs Test bal-acc 0.50，差距巨大。

**可能原因**:
- Val split 来自 train LMDB（`/tuev_preprocessed/train`），test 来自 eval LMDB（`/tuev_preprocessed/eval`）
- Val acc 使用标准 accuracy（受类别不平衡影响），Test 使用 balanced accuracy
- Train/Val 的 subject 可能有重叠（同一 subject 的不同 segment 分到 train 和 val），而 test 是完全不同的 subject
- 需要检查 Val split 的创建逻辑是否存在数据泄漏

### 问题 4: SLURM 时间限制

**现象**: Job 739416 (TUAB) 的 low_lambda 在 epoch 4/30 被取消。每个实验约 3.6 小时 (30 epochs × ~433s/epoch)，8 个实验总计 ~29 小时。

**改进**: ✅ **已修复** — `run_ib.sh` 已添加 `batch1`/`batch2` 参数支持，可将 8 个实验拆成两批提交。

### 问题 5: Checkpoint 可能存在覆盖问题

**证据**: TUAB 上 5 个变体的 best epoch 都是 11，TUEV 上 full_ib_adv 和 ib_only 的 Bal-Acc 完全相同。

**需要检查**: `train_ib_disentangle.py` 中 checkpoint 的保存文件名是否像 SageStream 一样使用固定名称（如 `best_TUAB_ib.pth`），导致后续变体覆盖前序变体的 checkpoint。

---

## 六、改进策略（按优先级）

### 高优先级 (Critical)

1. **✅ 修复 Subject ID 提取** — 已完成
2. **✅ 添加 batch 拆分支持** — 已完成
3. **检查并修复 IB checkpoint 保存逻辑** — 确保每个变体使用唯一文件名

### 中优先级 (Important)

4. **TUEV 类别权重/Focal Loss** — 解决 spsw F1=0 的问题
5. **检查 Val split 逻辑** — 排查 TUEV 上 val-test 巨大偏移的原因
6. **添加 CBraMod backbone 实验** — 当前 IB 只跑了 CodeBrain，缺少 CBraMod 对比

### 低优先级 (Nice to Have)

7. **降低学习率** — lr=1e-3 可能偏大，尝试 3e-4 + cosine annealing
8. **增加正则化** — TUEV 严重过拟合，增加 dropout/label smoothing
9. **尝试 partial fine-tuning** — 解冻 backbone 最后 2-3 层

---

## 七、总结

### 核心瓶颈

1. **Subject ID 提取失败** → IB/adversarial 模块完全无效 → 所有变体表现趋同
2. **TUEV 类别严重不平衡** → spsw F1=0, 整体 bal-acc 仅 ~0.50
3. **TUEV val-test 分布偏移** → train-test gap 高达 40%

### 当前可靠结论

- **TUAB**: IB 框架 + CodeBrain 在 binary classification 上 bal-acc ~0.81，与 SageStream moe_only 持平
- **TUEV**: 所有方法均表现不佳 (~0.50 bal-acc)，需要类别平衡 + subject 修复后重新评估
- **IB baseline_ce (0.8087) > SageStream baseline_ce (0.7839)**，说明 IB 的 latent bottleneck 架构本身对分类有益

### 下一步实验计划

1. 使用修复后的 subject ID + batch 拆分，重跑 CodeBrain TUAB/TUEV
2. 新增 CBraMod backbone 实验 (`run_ib.sh cbramod TUAB/TUEV`)
3. 在 TUEV 上加入 weighted CE loss
4. 检查并修复 IB checkpoint 覆盖问题
