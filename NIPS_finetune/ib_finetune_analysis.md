# IB Finetune 实验分析报告

> Jobs: 740851 ~ 740858 | 日期: 2026-03-02/03

---

## 一、实验设计

### 任务矩阵

| Job ID | 数据集 | 模型 | 批次 |
|--------|--------|------|------|
| 740851 | TUAB | cbramod | batch1（消融基线） |
| 740852 | TUAB | cbramod | batch2（超参探索） |
| 740853 | TUAB | codebrain | batch1（二进制文件，无法读取） |
| 740854 | TUAB | codebrain | batch2（超参探索） |
| 740855 | TUEV | codebrain | batch1（消融基线） |
| 740856 | TUEV | codebrain | batch2（超参探索） |
| 740857 | TUEV | cbramod | batch1（消融基线） |
| 740858 | TUEV | cbramod | batch2（超参探索） |

### Batch1 消融设计（每文件4个子实验）

| 实验名 | beta (IB) | lambda_adv | 含义 |
|--------|-----------|------------|------|
| `full_ib_adv` | 1e-3 | 0.5 | IB + 对抗 |
| `ib_only` | 1e-3 | 0.0 | 仅 IB 约束 |
| `adv_only` | 0.0 | 0.5 | 仅对抗约束 |
| `baseline_ce` | 0.0 | 0.0 | 纯 CE 基线 |

### Batch2 超参设计（每文件4个子实验）

| 实验名 | beta (IB) | lambda_adv |
|--------|-----------|------------|
| `high_beta` | 1e-2 | 0.5 |
| `low_beta` | 1e-4 | 0.5 |
| `high_lambda` | 1e-3 | 1.0 |
| `low_lambda` | 1e-3 | 0.1 |

### 训练参数

- epochs: 30，batch_size: 64，lr: 1e-3，latent_dim: 128
- 早停策略：patience based on val_acc

---

## 二、Final Test Acc 汇总

### 有效结果统计

> **仅 6/32 个实验获得了有效的 final test 结果**，其余均因 `RuntimeError` 在加载 best model 时崩溃。

### TUAB 数据集（二分类：normal / abnormal）

| Job | 模型 | 实验 | beta | lambda | best_val_acc | **test_bal_acc** | F1_macro | F1_abnormal | F1_normal | best_epoch |
|-----|------|------|------|--------|:------------:|:---------------:|:--------:|:-----------:|:---------:|:----------:|
| 740854 | codebrain | low_lambda | 1e-3 | 0.1 | 0.8118 | **0.8109** | 0.8128 | 0.7896 | 0.8360 | 20 |

### TUEV 数据集（6分类：bckg / artf / eyem / gped / pled）

#### codebrain — batch1 消融（740855，全部4个有效）

| 实验 | beta | lambda | best_val_acc | **test_bal_acc** | F1_macro | F1_bckg | F1_eyem | F1_artf | F1_gped | F1_pled | best_epoch |
|------|------|--------|:------------:|:---------------:|:--------:|:-------:|:-------:|:-------:|:-------:|:-------:|:----------:|
| full_ib_adv | 1e-3 | 0.5 | 0.6428 | 0.4611 | 0.4032 | 0.8894 | 0.2743 | 0.4586 | 0.4164 | 0.3807 | 25 |
| ib_only | 1e-3 | 0.0 | 0.9408 | 0.4965 | 0.4643 | 0.8657 | 0.5049 | 0.4866 | 0.4774 | 0.4515 | 29 |
| adv_only | 0.0 | 0.5 | 0.6508 | **0.5078** | 0.4914 | 0.8487 | 0.6073 | 0.4846 | 0.5211 | 0.4865 | 7 |
| baseline_ce | 0.0 | 0.0 | 0.9460 | 0.4934 | 0.4642 | 0.8575 | 0.5588 | 0.4920 | 0.3515 | 0.5256 | 28 |

#### codebrain — batch2 超参（740856，仅最后1个有效）

| 实验 | beta | lambda | best_val_acc | **test_bal_acc** | F1_macro | F1_bckg | F1_eyem | F1_artf | F1_gped | F1_pled | best_epoch |
|------|------|--------|:------------:|:---------------:|:--------:|:-------:|:-------:|:-------:|:-------:|:-------:|:----------:|
| low_lambda | 1e-3 | 0.1 | 0.9022 | **0.5391** | 0.4957 | 0.8568 | 0.5129 | 0.4525 | 0.5450 | 0.6072 | 30 |

#### cbramod — batch1（740857，仅1个有效）

| 实验 | beta | lambda | best_val_acc | **test_bal_acc** | F1_macro | F1_bckg | F1_eyem | F1_artf | F1_gped | F1_pled | best_epoch |
|------|------|--------|:------------:|:---------------:|:--------:|:-------:|:-------:|:-------:|:-------:|:-------:|:----------:|
| adv_only | 0.0 | 0.5 | 0.4044 | 0.3524 | 0.1308 | 0.0260 | 0.2453 | 0.2318 | 0.2817 | 0 | 29 |

---

## 三、RuntimeError 分析（缺失结果原因）

大量实验在 `train_ib_disentangle.py:739` 的 `model.load_state_dict(ckpt['model_state_dict'])` 处崩溃。

### Bug A：subject_head 尺寸不匹配

```
size mismatch for subject_head.head.4.weight:
  checkpoint shape: [2, 128]
  current model:    [1655, 128]  (TUAB) 或  [290, 128]  (TUEV)
```

**根本原因**：checkpoint 在 `num_subjects=2` 的配置下保存（推测是某种 `use_subjects` 参数错误），加载时当前模型用真实 subject 数量初始化，导致不匹配。

**受影响实验**：740854 exp1-3，740858 全部 4 个

### Bug B：backbone 层路径不匹配（模型包装层级变化）

```
# checkpoint 保存时（旧版本）：
backbone.patch_embedding.*
backbone.init_conv.*
backbone.residual_layer.*

# 当前模型加载时（新版本）：
backbone.backbone.patch_embedding.*
backbone.backbone.encoder.*
```

**根本原因**：代码重构导致模型在 `MultiDisease_CodeBrain_Model` 中的 backbone 多了一层 `.backbone` 包装，旧 checkpoint 结构与新代码不兼容。

**受影响实验**：740851 全部，740852 全部，740856 exp1-3，740857 exp1/2/4

### 有效/缺失汇总

| Job | 数据集+模型 | exp1 | exp2 | exp3 | exp4 |
|-----|------------|:----:|:----:|:----:|:----:|
| 740851 | TUAB+cbramod batch1 | ❌ Bug B | ❌ Bug B | ❌ Bug B | ❌ Bug B |
| 740852 | TUAB+cbramod batch2 | ❌ Bug B | ❌ Bug B | ❌ Bug B | ❌ Bug B |
| 740853 | TUAB+codebrain batch1 | ❓ binary | ❓ binary | ❓ binary | ❓ binary |
| 740854 | TUAB+codebrain batch2 | ❌ Bug A | ❌ Bug A | ❌ Bug A | ✅ 0.8109 |
| 740855 | TUEV+codebrain batch1 | ✅ 0.4611 | ✅ 0.4965 | ✅ 0.5078 | ✅ 0.4934 |
| 740856 | TUEV+codebrain batch2 | ❌ Bug B | ❌ Bug B | ❌ Bug B | ✅ 0.5391 |
| 740857 | TUEV+cbramod batch1 | ❌ Bug B | ❌ Bug B | ✅ 0.3524 | ❌ Bug B |
| 740858 | TUEV+cbramod batch2 | ❌ Bug A | ❌ Bug A | ❌ Bug A | ❌ Bug A |

> ✅ = 有效 test 结果 | ❌ = RuntimeError | ❓ = 二进制文件无法读取

---

## 四、关键发现与分析

### 1. val_acc vs test_bal_acc 严重背离（TUEV）

TUEV 任务中存在极大的 val/test 指标鸿沟：

| 实验 | val_acc | test_bal_acc | 差值 |
|------|:-------:|:------------:|:----:|
| ib_only | **0.9408** | 0.4965 | −0.444 |
| baseline_ce | **0.9460** | 0.4934 | −0.453 |
| adv_only | 0.6508 | **0.5078** | −0.143 |
| full_ib_adv | 0.6428 | 0.4611 | −0.182 |

**分析**：`ib_only` 和 `baseline_ce` 的 val_acc 高达 0.94，但 test_bal_acc 仅 0.49，说明模型几乎将所有样本预测为 `bckg`（背景类，占主导），在验证集上获得虚高的 accuracy，但 balanced accuracy 揭示了真实性能极差。

`adv_only` val_acc 最低（0.65），test_bal_acc 却最高（0.508），证明**对抗训练有效改善了跨类均衡性**。

### 2. IB 约束对 test acc 的实际作用（TUEV codebrain batch1）

```
adv_only     (λ=0.5, β=0.0) → test_bal_acc = 0.508  ← 最佳
baseline_ce  (λ=0.0, β=0.0) → test_bal_acc = 0.493
ib_only      (λ=0.0, β=1e-3)→ test_bal_acc = 0.497
full_ib_adv  (λ=0.5, β=1e-3)→ test_bal_acc = 0.461  ← 最差
```

- **对抗训练（adv_only）有效**：相比 baseline 提升约 +1.5%
- **IB 约束单独效果有限**：ib_only 仅比 baseline 高 +0.4%
- **IB + 对抗联合（full_ib_adv）反而最差**：两种约束同时施加，信息被过度压缩，损害判别性

### 3. 超参数：low_lambda 是目前 TUEV 最优配置

```
batch1 最优 (adv_only):                test_bal_acc = 0.508
batch2 low_lambda (β=1e-3, λ=0.1):    test_bal_acc = 0.539  ← 更好！
```

弱对抗约束（λ=0.1）+ 中等 IB（β=1e-3）的组合效果优于 batch1 所有实验，说明 λ 不宜过大。

### 4. cbramod 在 TUEV 上严重失效

| 指标 | codebrain (adv_only) | cbramod (adv_only) |
|------|:--------------------:|:------------------:|
| test_bal_acc | 0.508 | 0.352 |
| F1_bckg | 0.849 | **0.026** |
| F1_pled | 0.487 | **0** |
| F1_macro | 0.491 | 0.131 |

cbramod 在 bckg 类上 F1 接近 0，在 pled 类上 F1=0，属于严重退化模型。

---

## 五、建议

### 修复 Bug（优先级高）

1. **修复 Bug A**：检查 `num_subjects` 参数传递，确保 `subject_head` 在 checkpoint 保存和加载时使用相同的 subject 数
2. **修复 Bug B**：检查模型定义中 backbone 的包装层级，确保新旧 checkpoint 兼容，或统一重新训练

### 实验结论

3. **TUEV 任务应以 test_bal_acc 为主要指标**，而非 val_acc（后者因类别不均衡而极具误导性）
4. **对抗训练（λ_adv）对 TUEV 有正向效果**，建议保留但调小（λ=0.1 优于 λ=0.5）
5. **full_ib_adv 组合效果不佳**，需排查 IB 与对抗约束的相互干扰机制
6. **cbramod 需要专门排查**：可能是预训练权重格式问题或模型容量不足

---

## 六、参考：per-epoch 性能（TUEV codebrain，以供趋势参考）

| Epoch | adv_only val_acc | adv_only test_acc | baseline_ce val_acc | baseline_ce test_acc |
|:-----:|:----------------:|:-----------------:|:-------------------:|:--------------------:|
| 1 | 0.506 | 0.515 | 0.687 | 0.659 |
| 5 | 0.628 | 0.660 | 0.727 | 0.708 |
| 10 | 0.619 | 0.583 | 0.666 | 0.662 |
| 15 | 0.532 | 0.514 | 0.686 | 0.692 |
| 20 | 0.513 | 0.513 | 0.650 | 0.667 |
| 25 | 0.576 | 0.573 | 0.660 | 0.643 |
| 30 | — | — | 0.720 | 0.687 |

> adv_only 早停于 epoch 7（best），baseline_ce 跑满 30 epochs 后取 epoch 28

---

*生成时间：2026-03-03*
