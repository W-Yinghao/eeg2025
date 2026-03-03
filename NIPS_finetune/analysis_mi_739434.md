# MI Fine-Tuning 实验分析报告 (slurm-739434)

> **日期**: 2026-03-03
> **脚本**: `scripts/run_mi_finetune.sh`
> **方法**: MI Fine-Tuning (VIB + InfoNCE)
> **论文基础**: CodeBrain (arXiv:2506.09110v2)

---

## 一、方法简介

MI Fine-Tuning 是一种基于**信息论**的 EEG 基础模型微调框架，冻结预训练 backbone，仅训练轻量 MI head。核心思想：

- **InfoNCE (MI Maximization)**：通过对比学习将基础模型表征与 PSD 专家特征对齐，鼓励模型捕获频谱相关的临床特征
- **VIB (Variational Information Bottleneck)**：通过 KL 散度正则化压缩表征中的受试者特异性噪声，提升跨受试者泛化能力

**架构**：
```
Raw EEG (B,C,S,P) → Frozen Backbone → RepProjection → Z_FM
                                                         │
                                               ┌─────────┴──────────┐
                                               │                    │
                                          VIB Layer           ContrastHead
                                               │                    │
                                          z (B, V)          z_proj (B, H)
                                               │                    │
                                          Classifier          InfoNCE loss
                                               │            with Z_expert
                                          CE loss

Expert Features (PSD) → ExpertProjector → Z_expert (B, H)
```

**Total Loss** = CE + β × VIB + α × InfoNCE

**PSD 专家特征**：对原始 EEG 做 FFT，计算 5 个标准频段（Delta/Theta/Alpha/Beta/Gamma）的 log band power，维度 = C × 5（如 16 通道 = 80 维）。

---

## 二、实验配置

### 消融设计（4 种损失组合）

| 配置 | alpha (NCE) | beta (VIB) | 含义 |
|------|-------------|------------|------|
| `full_vib_nce` | 1.0 | 1e-3 | 完整模型（VIB + InfoNCE）|
| `nce_only` | 1.0 | 0.0 | 仅 InfoNCE，无 VIB |
| `vib_only` | 0.0 | 1e-3 | 仅 VIB，无 InfoNCE |
| `baseline_ce` | 0.0 | 0.0 | 纯 CE baseline |

### 超参数

| 参数 | CodeBrain+TUEV | CodeBrain+TUAB |
|------|---------------|---------------|
| **epochs** | 50 | 50 |
| **batch_size** | 64 | 64 |
| **lr** | 2e-5 | 1e-5 |
| **weight_decay** | 5e-4 | 5e-5 |
| **vib_dim** | 128 | 128 |
| **hidden_dim** | 256 | 256 |
| **temperature** | 0.07 | 0.07 |
| **expert** | psd | psd |

### 数据集

| 数据集 | 任务 | 类别数 | 通道 | 时长 | 样本量 |
|--------|------|--------|------|------|--------|
| **TUEV** | 事件分类 | 6 | 16 | 5s | ~112K |
| **TUAB** | 异常检测 | 2 | 16 | 10s | ~409K |

### 计划实验

共计划 16 个实验：4 配置 × 2 数据集 × 2 模型（CodeBrain + CBraMod），脚本按顺序执行：先 CodeBrain 全部 8 个，再 CBraMod 全部 8 个。

---

## 三、实验结果汇总

### 3.1 总体结果

#### CodeBrain + TUEV（6 分类）

| # | 配置 | 状态 | Test BalAcc | Test Kappa | Test F1 | 备注 |
|---|------|------|-----------|------------|---------|------|
| 1 | **vib_only** | ✅ 完成 | **0.6005** | **0.5441** | **0.7612** | **最佳** |
| 2 | full_vib_nce | ✅ 完成 | 0.5960 | 0.4882 | 0.7273 | |
| 3 | nce_only | ✅ 完成 | 0.5794 | 0.5197 | 0.7509 | |
| 4 | baseline_ce | ✅ 完成 | 0.5308 | 0.4672 | 0.7237 | 最差 |

#### CodeBrain + TUAB（2 分类）

| # | 配置 | 状态 | Test BalAcc | Test Kappa | Test F1 | 备注 |
|---|------|------|-----------|------------|---------|------|
| 5 | **full_vib_nce** | ✅ 完成 | **0.8022** | **0.6095** | **0.7785** | **最佳** |
| 6 | nce_only | ✅ 完成 | 0.7917 | 0.5868 | 0.7697 | |
| 7 | vib_only | ✅ 完成 | 0.7879 | 0.5785 | 0.7661 | 提前收敛(ep17) |
| 8 | baseline_ce | ❌ **未完成** | ~0.786* | ~0.574* | ~0.766* | epoch 13/50 时超时 |

> \* baseline_ce on TUAB 在 epoch 12 被 SLURM 终止，数值为截止时刻的 test 指标，非最终结果。

#### CBraMod（全部 8 个实验）

| # | 状态 | 原因 |
|---|------|------|
| 9-16 | ❌ **全部未运行** | SLURM 24h 时限耗尽，脚本按顺序执行，CodeBrain 8 个实验已耗完时间 |

**终止日志**：
```
slurmstepd: error: *** JOB 739434 ON node03 CANCELLED AT 2026-02-28T23:37:28 DUE TO TIME LIMIT ***
```

### 3.2 消融对比分析

#### TUEV 上的消融结论

```
排序（Test BalAcc）: vib_only (0.600) > full_vib_nce (0.596) > nce_only (0.579) > baseline_ce (0.531)
排序（Test Kappa）:  vib_only (0.544) > nce_only (0.520) > full_vib_nce (0.488) > baseline_ce (0.467)
```

- **VIB 贡献最大**：vib_only 相对 baseline_ce 提升 **+7.0% BalAcc, +7.7% Kappa**
- **InfoNCE 有帮助但不如 VIB**：nce_only 相对 baseline 提升 +4.9% BalAcc, +5.3% Kappa
- **full model 并非最优**：full_vib_nce 的 Kappa (0.488) 反而低于 vib_only (0.544) 和 nce_only (0.520)，说明两个损失组合时存在冲突

#### TUAB 上的消融结论

```
排序（Test BalAcc）: full_vib_nce (0.802) > nce_only (0.792) > vib_only (0.788) > baseline_ce (~0.786)
排序（Test Kappa）:  full_vib_nce (0.610) > nce_only (0.587) > vib_only (0.579) > baseline_ce (~0.574)
```

- **InfoNCE 贡献最大**：nce_only 相对 baseline 提升 **+0.6% BalAcc, +1.3% Kappa**
- **VIB + NCE 组合有效**：full model 是 TUAB 上的最优，两者互补
- **整体差距较小**：四种配置在 TUAB 上差距仅 ~1.6% BalAcc，不如 TUEV 显著

### 3.3 与其他方法对比

| 组合 | MI (best) | SCOPE | SageStream (moe-only) | 论文 Full FT |
|------|-----------|-------|-----------------------|-------------|
| CodeBrain+TUEV (Kappa) | **0.544** | 0.562 | 0.525 | ~0.560 |
| CodeBrain+TUAB (BalAcc) | **0.802** | 崩溃 | 0.814 | 0.829 |

> MI 方法在 TUEV 上 Kappa 接近论文 full finetune 水平，在 TUAB 上 BalAcc 略低于论文 2.7%。

### 3.4 训练曲线特征

#### 典型过拟合：Val-Test 巨大差距

**CodeBrain + TUEV**（所有配置均有此问题）：
```
full_vib_nce: Val BalAcc ~0.98 vs Test BalAcc 0.596  → 差距 38%
vib_only:     Val BalAcc ~0.98 vs Test BalAcc 0.601  → 差距 38%
nce_only:     Val BalAcc ~0.97 vs Test BalAcc 0.579  → 差距 39%
baseline_ce:  Val BalAcc ~0.97 vs Test BalAcc 0.531  → 差距 44%
```

**CodeBrain + TUAB**（差距较小）：
```
full_vib_nce: Val BalAcc ~0.81 vs Test BalAcc 0.802  → 差距 ~1%
nce_only:     Val BalAcc ~0.81 vs Test BalAcc 0.792  → 差距 ~2%
```

#### 训练后期收敛停滞

- TUEV：约 epoch 10 后 Test BalAcc 进入平台（~0.59-0.61），此后 30+ epoch 无明显提升
- TUAB：前 3 epoch 快速提升，之后持续缓慢退化

---

## 四、逐实验详细分析

### 实验1: full_vib_nce + CodeBrain + TUEV — 完成，中等偏上

**训练关键节点**：
```
Epoch  1/50 | Val BalAcc=0.620  | Test BalAcc=0.434
Epoch  5/50 | Val BalAcc=0.807  | Test BalAcc=0.569  ← 快速提升
Epoch 10/50 | Val BalAcc=0.971  | Test BalAcc=0.557  ← Val 虚高，Test 停滞
Epoch 25/50 | Val BalAcc=0.982  | Test BalAcc=0.607  ← 最佳 Test 附近
Epoch 50/50 | Val BalAcc=0.978  | Test BalAcc=0.599  ← 略有退化
```

**最终指标**（wandb final_test）：BalAcc=0.596, Kappa=0.488, F1=0.727

**问题**：full model 的 Kappa (0.488) 不如 vib_only (0.544)，说明 InfoNCE 在 TUEV 上可能引入了与 VIB 矛盾的梯度方向。

---

### 实验2: nce_only + CodeBrain + TUEV — 完成，Kappa 中等

**最终指标**：BalAcc=0.579, Kappa=0.520, F1=0.751

**特点**：
- Kappa 排第二（0.520），优于 full model (0.488)
- 但 BalAcc 排第三（0.579），说明 InfoNCE 对部分类别的提升不均匀
- 训练后期 Test Kappa 震荡剧烈（0.42-0.54），不如 VIB 稳定

---

### 实验3: vib_only + CodeBrain + TUEV — 完成，TUEV 上最佳

**最终指标**：BalAcc=0.601, Kappa=0.544, F1=0.761

**亮点**：
- 三项指标全面最优（TUEV）
- epoch 16 出现峰值 Test BalAcc=0.635, Kappa=0.568，但未被保存（基于 val 选 best）
- VIB 有效压缩了受试者噪声，在多分类场景中收益最大

**值得注意**：
- 在 epoch 9 达到 Test Kappa=0.586 的阶段高点，之后虽有波动但整体稳定
- 说明 VIB 的正则化效果使训练相对稳定

---

### 实验4: baseline_ce + CodeBrain + TUEV — 完成，最差

**最终指标**：BalAcc=0.531, Kappa=0.467, F1=0.724

**问题**：
- 全场最差，验证了 MI 组件（VIB/InfoNCE）确实有用
- Test BalAcc 始终在 0.49-0.55 之间低位震荡
- 纯 CE + frozen backbone 的能力上限有限

---

### 实验5: full_vib_nce + CodeBrain + TUAB — 完成，TUAB 上最佳

**最终指标**：BalAcc=0.802, Kappa=0.610, F1=0.778

**训练关键节点**：
```
Epoch  1/50 | Val BalAcc=0.793  | Test BalAcc=0.796
Epoch  3/50 | Val BalAcc=0.815  | Test BalAcc=0.802  ← Val 峰值附近
Epoch 10/50 | Val BalAcc=0.801  | Test BalAcc=0.785  ← 开始退化
Epoch 50/50 | Val BalAcc=0.795  | Test BalAcc=0.772  ← 持续退化
```

**亮点**：
- 四种配置中最佳，VIB + InfoNCE 在二分类上互补有效
- Test 峰值在 epoch 3（0.802），与 val 峰值基本一致
- PSD 频谱特征对正常/异常 EEG 区分有较强指导意义

**问题**：
- 训练后期持续退化，epoch 50 比 epoch 3 下降约 3%
- 说明 50 epoch 过多，10-15 epoch 足够

---

### 实验6: nce_only + CodeBrain + TUAB — 完成

**最终指标**：BalAcc=0.792, Kappa=0.587, F1=0.770

**特点**：与 full model 差距较小（BalAcc -1.0%），说明 InfoNCE 是 TUAB 上的主要贡献者。

---

### 实验7: vib_only + CodeBrain + TUAB — 完成，提前收敛

**最终指标**：BalAcc=0.788, Kappa=0.579, F1=0.766

**特点**：
- 仅训练到 epoch 17 即出现 wandb 最终结果，可能触发了某种停止条件
- VIB 在二分类上的收益不如多分类，因为受试者噪声对二分类影响相对较小

---

### 实验8: baseline_ce + CodeBrain + TUAB — 未完成（超时）

**截止状态**：epoch 13/50，最后完整 epoch 12 的 Test BalAcc ~0.786

**推测最终结果**：
- 基于 epoch 12 的趋势，最终 BalAcc 预计 ~0.786-0.790
- 与其他三种配置差距在 1-2% 内
- TUAB 上四种配置差距较小（~1.6%），说明 frozen backbone 本身已提取了大部分有用信息

---

## 五、核心问题诊断

### 问题1: TUEV 上的严重过拟合 [严重]

Val BalAcc 达到 ~0.98 但 Test 仅 ~0.60，差距高达 38 个百分点。

**原因链**：
```
TUEV train/val 来自同源 → val 指标虚高（~0.98）
    ↓
Early stopping 基于虚高 val → 选择的模型不一定泛化最好
    ↓
50 epoch 对 frozen backbone + linear head 过多 → 后 40 epoch 徒增过拟合
    ↓
RepProjection 将 (C×S×200) flatten 后维度很大 → head 参数量不小
```

### 问题2: TUAB 训练后期退化 [中等]

所有 TUAB 实验都在前 3 epoch 达到 Test 峰值，之后持续下降。

**原因**：
- TUAB 数据量大（~297K train, 4642 batches/epoch），每 epoch 遍历量大
- lr=1e-5 + 50 epoch → 过度训练
- frozen backbone 提取的特征已足够区分正常/异常，head 进一步训练反而过拟合

### 问题3: InfoNCE 与 VIB 在 TUEV 上的冲突 [中等]

full_vib_nce 的 Kappa (0.488) < vib_only (0.544) 和 nce_only (0.520)。

**原因推测**：
- VIB 鼓励压缩表征（最小化信息），而 InfoNCE 鼓励保留与 PSD 专家对齐的信息（最大化互信息），两者可能产生矛盾梯度
- PSD 对 TUEV 中某些类别（artf 伪迹、eyem 眼动）区分度不高，InfoNCE 可能引入了误导信号
- beta=1e-3 和 alpha=1.0 的权重比例可能不当

### 问题4: CBraMod 实验全部缺失 [严重]

SLURM 24h 时限不够完成 8 个 CodeBrain 实验，CBraMod 完全未执行。

**原因**：
- TUAB 数据量大（4642 batches × 50 epochs × 4 configs），单个 TUAB 实验约 3h
- 脚本使用 `set -e` 顺序执行，无并行

### 问题5: PSD 专家特征的局限性 [低]

PSD 仅包含 5 个标准频段的 band power，对于 TUEV 中的瞬态事件（spike, sharp wave）可能不够。

---

## 六、改进方案

### P0 — 紧急修复

#### 6.1 增加 SLURM 时限或拆分作业

**当前问题**：24h 不够跑完 8 个 CodeBrain 实验。

**方案**：
```bash
# 方案 A: 拆分为独立 job（推荐）
sbatch --time=12:00:00 run_mi_finetune.sh codebrain TUEV
sbatch --time=12:00:00 run_mi_finetune.sh codebrain TUAB
sbatch --time=12:00:00 run_mi_finetune.sh cbramod TUEV
sbatch --time=12:00:00 run_mi_finetune.sh cbramod TUAB

# 方案 B: 增加时限到 48h
#SBATCH --time=48:00:00
```

#### 6.2 减少 epoch 数 + Early Stopping

| 数据集 | 当前 epochs | 建议 epochs | patience | 原因 |
|--------|------------|------------|---------|------|
| TUEV | 50 | **20** | 5 | epoch 10 后 Test 已收敛 |
| TUAB | 50 | **10-15** | 3 | 前 3 epoch 即达峰值，CodeBrain 论文推荐 TUAB 用 early stop at epoch 10 |

预计可将每个实验时间缩短 60%+，4 个 TUAB 实验从 ~12h 降至 ~4h。

### P1 — 关键改进

#### 6.3 增强正则化（针对 TUEV 过拟合）

| 措施 | 当前值 | 建议值 | 预期效果 |
|------|-------|-------|---------|
| VIB beta | 1e-3 | **1e-2, 1e-1** | 增强信息压缩，减少过拟合 |
| Dropout | 未设置 | **0.3-0.5** | 增加 head 的正则化 |
| hidden_dim | 256 | **128 或 64** | 缩小 head 容量 |
| Weight decay | 5e-4 | **1e-3** | 增强参数正则化 |

#### 6.4 调整 VIB + InfoNCE 权重比

当前 full_vib_nce 在 TUEV 上表现不如单独 VIB，说明 alpha=1.0 与 beta=1e-3 的比例失衡。

**建议消融**：
```
alpha=0.1, beta=1e-3  （降低 InfoNCE 权重）
alpha=0.1, beta=1e-2  （同时调整两者）
alpha=1.0, beta=1e-2  （增强 VIB）
```

#### 6.5 改进 PSD 专家特征

| 改进方向 | 具体措施 | 预期收益 |
|---------|---------|---------|
| 增加时域特征 | `--expert_type both`（统计矩 + PSD）| 提升对瞬态事件的区分 |
| Connectivity 特征 | 增加 PLV、coherence 等通道间特征 | 捕获脑区协同信息 |
| Wavelet 特征 | 用小波变换替代 FFT，保留时频局部信息 | 更好的时频分辨率 |

#### 6.6 调整 InfoNCE 温度

| 当前值 | 建议值 | 原因 |
|-------|-------|------|
| 0.07 | **0.1, 0.2, 0.5** | 当前过低，对比学习过于"尖锐"，可能导致训练不稳定 |

### P2 — 进一步优化

#### 6.7 为 CBraMod 启用 Layer Adapters

脚本已支持 `--use_layer_adapters` 参数，可在 CBraMod Transformer 各层之间插入可训练的 bottleneck adapter，比纯 frozen backbone 更灵活：

```bash
python train_mi_finetuning.py --model cbramod --use_layer_adapters ...
```

#### 6.8 混合微调策略

- **部分解冻 backbone**：冻结前 N 层、解冻后 2 层
- **LoRA**：参数高效微调，仅在 attention 层插入低秩矩阵
- **渐进式解冻**：前 10 epoch 全冻结，后续逐层解冻

#### 6.9 Cross-subject 数据增强

针对 TUEV 过拟合问题：
- 时间扰动（jittering, time warping）
- 通道 dropout（随机掩码部分通道）
- 频谱混合（Mixup on frequency domain）

#### 6.10 多专家集成

不仅用 PSD，同时引入 wavelet、CSP 等多种专家特征，各自一个对比分支：

```
Z_FM ──→ ContrastHead_1 ──→ InfoNCE(Z_FM, Z_psd)
     ├──→ ContrastHead_2 ──→ InfoNCE(Z_FM, Z_wavelet)
     └──→ ContrastHead_3 ──→ InfoNCE(Z_FM, Z_csp)

Total InfoNCE = mean(InfoNCE_1, InfoNCE_2, InfoNCE_3)
```

---

## 七、与 CodeBrain 论文参考指标对比

| 数据集 | 论文 Full FT | MI best (frozen) | 差距 | 说明 |
|--------|-------------|-----------------|------|------|
| TUEV (Kappa) | ~0.560 | **0.544** (vib_only) | **-1.6%** | 接近论文水平 |
| TUEV (BalAcc) | ~0.60 | **0.601** (vib_only) | **+0.1%** | 持平 |
| TUAB (BalAcc) | 0.829 | **0.802** (full_vib_nce) | **-2.7%** | 有差距，backbone 解冻可能弥补 |
| TUAB (Kappa) | ~0.66 | **0.610** (full_vib_nce) | **-5.0%** | 差距较大 |

> MI frozen-backbone 方法在 TUEV 上已接近 full finetune 水平，但在 TUAB 上仍有 2.7-5.0% 差距，说明 TUAB 可能需要部分 backbone 解冻或更强的适配模块。

---

## 八、总结与优先级

| 优先级 | 改进项 | 预期收益 | 工作量 |
|--------|--------|---------|-------|
| **P0** | 拆分/增加 SLURM 时限，补跑 CBraMod 实验 | 补全消融数据 | 10 min 改脚本 |
| **P0** | 减少 epoch 数 + Early Stopping | 节省 60%+ 时间，防止退化 | 10 min |
| **P1** | 增强正则化（增大 beta, 加 dropout, 缩小 hidden_dim）| 缓解 TUEV 过拟合 | 1-2 h |
| **P1** | 调整 VIB+InfoNCE 权重比 | 消除 full model 冲突 | 2-4 h (消融) |
| **P1** | 丰富专家特征（时域统计 + wavelet）| 提升 InfoNCE 锚点质量 | 2-3 h |
| **P2** | 调整 InfoNCE 温度参数 | 稳定对比学习 | 1-2 h (消融) |
| **P2** | CBraMod 启用 layer adapters | 增强 CBraMod 适配能力 | 30 min |
| **P2** | 混合微调 / LoRA / 渐进解冻 | 缩小与 full FT 差距 | 4-8 h |

### 核心结论

1. **VIB 是 TUEV 上的核心组件**：单独 VIB 即为最佳配置，有效压缩受试者噪声
2. **InfoNCE 在 TUAB 上更有价值**：PSD 频谱特征对正常/异常二分类有强指导
3. **VIB + InfoNCE 组合需要更精细的权重调节**：当前 alpha/beta 比例在 TUEV 上导致梯度冲突
4. **训练时间过长是主要浪费**：50 epoch 中仅前 10-15 有效，之后过拟合/退化
5. **CBraMod 数据缺失**：需紧急补跑，以完成完整消融分析
