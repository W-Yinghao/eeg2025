# EEG Foundation Model Fine-Tuning: 方法总览

> 本文档整理了 `NIPS_finetune/` 中所有 fine-tuning 方法的设计、架构、损失函数和实验状态。

---

## 目录

1. [Base Models (Backbone)](#1-base-models)
2. [Datasets](#2-datasets)
3. [Method 1: MSFT (Multi-Scale Fine-Tuning)](#3-method-1-msft)
4. [Method 2: Improved MSFT (CBraMod-Specific)](#4-method-2-improved-msft)
5. [Method 3: MI Fine-Tuning (VIB + InfoNCE)](#5-method-3-mi-fine-tuning)
6. [Method 4: Hybrid Hyperbolic Fine-Tuning](#6-method-4-hybrid-hyperbolic)
7. [Method 5: IB + Disentanglement (Token-Level IB + GRL)](#7-method-5-ib--disentanglement)
8. [方法对比表](#8-方法对比表)
9. [实验状态与结果](#9-实验状态与结果)

---

## 1. Base Models

| Backbone | 架构 | 层数 | Feature Dim | 预训练权重 | 关键参数 |
|----------|------|------|-------------|-----------|---------|
| **CodeBrain (SSSM)** | Structured State Space Model | 8 | 200 | `CodeBrain/Checkpoints/CodeBrain.pth` | `codebook_size_t=4096, codebook_size_f=4096` |
| **CBraMod** | Criss-Cross Transformer (时间+空间分离注意力) | 12 | 200 | `Cbramod_pretrained_weights.pth` | `nhead=8, dim_feedforward=800` |

两个 backbone 在所有方法中均为 **frozen**（冻结全部参数），只训练下游 adapter / head。

---

## 2. Datasets

| Dataset | 任务类型 | 类别数 | 通道数 | 时长 | 采样率 | 数据路径 |
|---------|---------|-------|-------|------|-------|---------|
| **TUEV** | 癫痫事件分类 (Multiclass) | 6 | 16 | 5s | 200Hz | `/projects/.../tuev_preprocessed` |
| **TUAB** | 脑电异常检测 (Binary) | 2 | 16 | 10s | 200Hz | `/projects/.../tuab_preprocessed` |
| **CHB-MIT** | 癫痫发作检测 (Binary) | 2 | 16 | 10s | 200Hz | LMDB格式 |
| **TUSZ** | 癫痫发作检测 (Binary) | 2 | 16 | 10s | 200Hz | LMDB格式 |
| **AD_DIAGNOSIS** | 阿尔茨海默诊断 | 4 | 58 | 1s | 200Hz | LMDB格式 |
| **DIAGNOSIS** | 通用诊断 | 4 | 16 | 10s | 200Hz | LMDB格式 |

所有数据集通过 `finetune_tuev_lmdb.py` 中的 `DATASET_CONFIGS` 统一配置，使用 LMDB 格式存储，支持跨被试划分。

---

## 3. Method 1: MSFT (Multi-Scale Fine-Tuning)

**文件:** `finetune_msft.py`, `msft_modules.py`
**参考论文:** Qiao et al., "Multi-Scale Finetuning for Encoder-based Time Series Foundation Models" (2025)

### 核心思想

在不同时间尺度（1x, 2x, 4x, 8x）下处理 EEG，通过 frozen backbone 提取多尺度特征，再用可训练的跨尺度聚合器融合信息。

### 架构

```
Input EEG (B, C, S, P)
    │
    ├── Scale 1 (1x): 原始分辨率
    ├── Scale 2 (2x): avg_pool 2倍下采样
    ├── Scale 3 (4x): avg_pool 4倍下采样
    └── Scale 4 (8x): avg_pool 8倍下采样
    │
    ▼ (每个scale)
Frozen PatchEmbedding + Trainable ScaleAdapter (Linear+GELU residual)
    │
    ▼ 拼接所有scale沿seq维度
Frozen Backbone Layers (逐层)
    │
    ├── Frozen Layer_i
    └── Trainable CrossScaleAggregator_i (C2F↑ + F2C↓ 双向)
    │
    ▼ 按scale拆分
Per-Scale Classifiers (3层MLP) → Softmax加权混合 (learnable scale_mix_logits)
    │
    ▼
Final Prediction
```

### 可训练模块

| 模块 | 描述 | 参数量级 |
|------|------|---------|
| `MultiScaleGenerator` | avg_pool1d 生成多尺度输入 | 0 (无参数) |
| `ScaleAdapter` (per scale) | Linear(200,200)+GELU residual | ~40K/scale |
| `CrossScaleAggregator` (per layer) | C2F+F2C 双向融合, Linear(200,200) | ~80K/layer |
| `per_scale_classifiers` | 3层MLP per scale | ~50K/scale |
| `scale_mix_logits` | Learnable (num_scales,) | num_scales |

### 损失函数

```
L = CrossEntropy(label_smoothing=0.1)  # multiclass
L = BCEWithLogits                       # binary
```

### 关键超参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `num_scales` | 1/2/3/4 (ablation) | 时间尺度数量 |
| `lr` | 1e-3 | |
| `weight_decay` | 5e-2 | |
| `dropout` | 0.1 | |
| `epochs` | 50 | |
| `batch_size` | 64 | |
| `label_smoothing` | 0.1 | |
| Scheduler | CosineAnnealingLR (eta_min=1e-6) | |

### 支持的 Backbone

- **CodeBrain (SSSM)**: 8层, CrossScaleAggregator 3D
- **CBraMod**: 12层, CrossScaleAggregator4D

### 独特功能

- Scale mixing weights 可视化（解释哪个时间尺度最重要）
- t-SNE 特征可视化
- WandB 实验追踪

---

## 4. Method 2: Improved MSFT (CBraMod-Specific)

**文件:** `finetune_msft_improved.py`, `msft_modules_improved.py`

### 核心思想

针对 CBraMod 的 Criss-Cross Attention 结构进行架构感知改进。原始 MSFT 将不同 scale 拼接后统一过 Transformer，破坏了 CBraMod 的时间-空间分离注意力模式。改进版让每个 scale 独立通过每层 Transformer。

### 三项关键改进

| 改进 | 模块 | 描述 |
|------|------|------|
| **独立Scale处理** | 核心架构 | 每个scale独立通过frozen Transformer层，保持正确的 (channels × seq_k) 结构 |
| **Scale位置编码精化** | `ScalePositionalRefiner` | Depthwise Conv2d(200,200,k=3) + BN，适应不同scale的seq_len |
| **Criss-Cross感知聚合** | `CrissCrossAwareAggregator4D` | 在temporal C2F/F2C之外额外添加spatial fusion (per-scale Linear+GELU) |

### 4种Ablation变体

| 变体 | `use_pos_refiner` | `use_criss_cross_agg` | 描述 |
|------|-------------------|----------------------|------|
| `baseline` | ✗ | ✗ | 与vanilla MSFT相同 |
| `pos_refiner` | ✓ | ✗ | 仅添加位置精化 |
| `criss_cross_agg` | ✗ | ✓ | 仅添加空间融合 |
| `full` | ✓ | ✓ | 完整改进模型 |

### 支持的 Backbone

**仅 CBraMod** — 专为 Criss-Cross Attention 设计

---

## 5. Method 3: MI Fine-Tuning (VIB + InfoNCE)

**文件:** `mi_finetuning_framework.py`, `train_mi_finetuning.py`

### 核心思想

结合 Variational Information Bottleneck (VIB) 进行噪声抑制和 InfoNCE 对比学习对齐专家特征。VIB 压缩表示去除冗余，InfoNCE 确保学到的表示与领域先验知识（PSD、统计特征）对齐。

### 架构

```
Frozen CodeBrain Backbone
    │
    ▼ Flatten: (B, C, S, 200) → (B, C*S*200)
RepProjection: Linear → GELU → Dropout → Linear  →  h (B, 256)
    │
    ├──→ VIBLayer: fc_mu, fc_log_var → Z ~ N(mu, sigma²)  →  ClassifierHead → disease logits
    │
    └──→ ContrastHead: Linear → GELU → Linear → L2norm  →  Z_contrast
                                                              │
                                                              ▼ InfoNCE
                                     ExpertProjector(expert_features) → L2norm
```

### 损失函数

```
L_total = L_CE + beta × L_VIB + alpha × L_InfoNCE
```

| 损失项 | 公式 | 作用 |
|--------|------|------|
| `L_CE` | CrossEntropy / BCE | 分类监督 |
| `L_VIB` | KL(q(z\|x) ‖ N(0,I)) | 信息瓶颈压缩 |
| `L_InfoNCE` | Symmetric InfoNCE (cosine/temp) | 与专家特征对齐 |

### 专家特征 (Expert Features)

| 类型 | 维度 (16ch) | 描述 |
|------|------------|------|
| `psd` | 80 | FFT-based PSD, 5个EEG频段 (delta/theta/alpha/beta/gamma), log1p |
| `stats` | 64 | Mean/std/min/max per channel |
| `both` | 144 | PSD + Stats 拼接 |

> **注意:** 专家特征仅在训练时需要（用于InfoNCE），推理时只需VIB路径。

### 关键超参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `alpha` (InfoNCE weight) | 1.0 | 0=禁用InfoNCE |
| `beta` (VIB weight) | 1e-3 | 0=禁用VIB |
| `temperature` | 0.07 | InfoNCE温度 |
| `vib_dim` | 128 | 瓶颈维度 |
| `hidden_dim` | 256 | 隐藏层维度 |
| `expert_feature` | 'psd' | psd/stats/both |

### Ablation 配置

| 配置 | alpha | beta | 描述 |
|------|-------|------|------|
| `baseline_ce` | 0.0 | 0.0 | 纯CE监督 |
| `infonce_only` | 1.0 | 0.0 | 仅对比学习 |
| `vib_only` | 0.0 | 1e-3 | 仅信息瓶颈 |
| `full_mi` | 1.0 | 1e-3 | 完整模型 |

### 支持的 Backbone

**仅 CodeBrain (SSSM)**

---

## 6. Method 4: Hybrid Hyperbolic Fine-Tuning

**文件:** `hyperbolic_finetuning.py`, `train_hyperbolic_finetuning.py`
**参考论文:** HEEGNet (Li et al., 2026), 灵感来自 Lorentz 流形几何

### 核心思想

将欧氏空间的 backbone 特征投影到 Lorentz（双曲面）流形上，利用双曲空间天然的层次结构进行分类。DSMDBN (Domain-Specific Momentum-Distribution Batch Normalization) 在流形上实现域自适应。

### 架构

```
Frozen CodeBrain Backbone
    │
    ▼ Flatten: (B, C, S, 200) → (B, D_flat)
HyperbolicProjection: Linear → ELU → Dropout → Linear → L2-norm → Lorentz projx
    │
    ▼ 在 Lorentz 流形 L^d_K 上
LorentzELU: 对空间分量做ELU，重新计算时间分量保持在流形上
    │
    ▼
HyperbolicDSMDBN: 域特定BN (每个域/被试有独立running Fréchet mean和variance)
    │
    ▼
HMLR (Hyperbolic MLR): 双曲空间决策超平面，通过到类超平面的带符号双曲距离计算logits
    │
    ▼
Disease Logits (B, num_classes)
```

### 自包含 Lorentz 流形实现

无需外部几何库（geoopt 仅用于 RiemannianAdam 优化器，可选）。实现包括：
- Minkowski 内积、流形投影 (projx)
- 指数/对数映射 (exp0, log0)
- 测地线距离
- Gyrovector 运算 (gyroinv, gyroadd, gyrotrans, gyroscalarprod)
- 迭代 Karcher Fréchet mean/variance

### 损失函数

```
L_total = L_CE + lambda_hhsw × L_HHSW
```

| 损失项 | 公式 | 作用 |
|--------|------|------|
| `L_CE` | F.cross_entropy(logits, labels) | 分类监督（统一多类/二分类） |
| `L_HHSW` | Hyperbolic Horospherical Sliced-Wasserstein | 域对齐：每个域特征与标准双曲高斯分布之间的Wasserstein距离 |

HHSW 通过 Busemann 函数将双曲点投影到 R，然后计算1D Earth Mover's Distance。

### DSMDBN 详解

```
训练时:
  1. 对每个域计算batch Fréchet mean (迭代Karcher均值, max 100步)
  2. 通过测地线插值更新running mean: gamma(t) on geodesic
  3. 归一化: gyro-inverse中心化 + gyro-scalar缩放

推理时:
  使用running statistics, 不需要batch统计
```

### 关键超参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `hyp_dim` | 128 | 双曲空间维度 |
| `curvature` | -1.0 | 曲率 K |
| `lambda_hhsw` | auto (大部分0.5, 情感数据集0.01) | HHSW权重 |
| `num_projections` | 1000 | HHSW随机投影数 |
| `eta_train_init` | 1.0 | DSMDBN初始动量 |
| `eta_decay` | 0.99 | 动量衰减率 |
| `epochs` | 100 | |
| `patience` | 15 | Early stopping |
| Optimizer | RiemannianAdam (geoopt) / Adam fallback | |

### 优化器配置

| 参数组 | Weight Decay | 说明 |
|--------|-------------|------|
| 双曲参数 (HMLR, DSMDBN, LorentzELU) | 0 | 流形参数不施加weight decay |
| 欧氏参数 (projection) | weight_decay | 标准正则化 |

### 支持的 Backbone

**仅 CodeBrain (SSSM)**

### 独特功能

- Source-Free UDA: `domainadapt_finetune()` 用于测试时域自适应
- Per-module gradient norm 监控（调试双曲训练稳定性）
- WandB 记录: confusion matrix, per-class F1, gradient norms

---

## 7. Method 5: IB + Disentanglement (Token-Level IB + GRL)

**文件:** `ib_disentangle_framework.py`, `train_ib_disentangle.py`

### 核心思想

保留 backbone 输出的 token 结构 (B, T, D)，在 token 级别做信息瓶颈压缩；同时使用 Gradient Reversal Layer (GRL) 进行对抗性被试解耦。与 Method 3 (MI) 的区别在于：

| 区别 | MI Fine-Tuning (Method 3) | IB + Disentangle (Method 5) |
|------|--------------------------|----------------------------|
| 粒度 | Flatten到单一向量 | **Token-level** (B,T,D) |
| 去噪策略 | VIB (全局) + InfoNCE (专家特征) | Token IB + GRL (对抗被试) |
| 需要专家特征 | 是 (PSD/Stats) | **否** |
| 需要被试ID | 否 | **是** |
| 可解释性 | 无 | **Per-token KL heatmap** |

### 架构

```
Frozen CodeBrain Backbone: (B, C, S, P) → (B, C, S, 200)
    │
    ▼ Reshape to tokens: (B, T, 200)  where T = C × S
    │
CodeBrain_IB_Adapter (Token-level):
    Shared Encoder: Linear → LayerNorm → GELU → Dropout
    ├── fc_mu → mu (B, T, D')
    └── fc_log_var → log_var (B, T, D')
    Z = mu + sigma × eps  (reparameterization, per-token)
    │
    ▼ Mean Pooling: (B, T, D') → z_agg (B, D')
    │
    ├──→ DiseaseClassifier: Linear→BN→GELU→Dropout→Linear → disease logits
    │
    └──→ GRL (gradient reversal: forward=identity, backward=-lambda×grad)
         │
         └──→ SubjectClassifier: Linear→BN→GELU→Dropout→Linear → subject logits
```

### GRL (Gradient Reversal Layer)

```python
Forward:  y = x                    # 恒等映射
Backward: dx = -lambda × dy       # 梯度反转
```

- Encoder 从 subject head 收到反转梯度 → 学习去除被试信息
- Subject head 正常训练 → 学习预测被试
- 单次 forward-backward 即可，无需交替 min-max 优化

**Lambda Schedule:** `lambda(p) = 2 / (1 + exp(-gamma × p)) - 1`, p = epoch/total_epochs

### 损失函数

```
L_total = L_task + beta × L_IB + lambda_adv × L_MI
```

| 损失项 | 公式 | 作用 |
|--------|------|------|
| `L_task` | CrossEntropy(disease_logits, labels) | 疾病分类监督 |
| `L_IB` | Token-level KL: -0.5×sum(1+log_var-mu²-exp(log_var))/(B×T) | 信息压缩 |
| `L_MI` | CrossEntropy(subject_logits, subject_ids) | 被试分类 (GRL反转梯度) |

### Per-Token 临床可解释性

```python
get_interpretability_heatmap(mu, log_var, n_channels, seq_len):
    kl_per_token = 0.5 * (mu² + exp(log_var) - 1 - log_var).sum(dim=-1)  # (B, T)
    heatmap = kl_per_token.reshape(B, n_channels, seq_len)  # 空间-时间热力图
    channel_importance = heatmap.mean(dim=2)   # 哪些EEG通道保留最多信息
    temporal_importance = heatmap.mean(dim=1)   # 哪些时间段保留最多信息
```

### 关键超参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `latent_dim` | 128 | Token-level 瓶颈维度 |
| `beta` (IB weight) | 1e-3 | 0=禁用IB |
| `lambda_adv` | 0.5 | 0=禁用GRL |
| `grl_gamma` | 10.0 | Lambda schedule 陡峭度 |
| `lr` | 1e-3 | |
| `patience` | 15 | Early stopping |
| `heatmap_interval` | 10 | 每N个epoch计算heatmap |

### 8种 Ablation 配置

| 配置 | beta | lambda_adv | 描述 |
|------|------|-----------|------|
| `full_ib_adv` | 1e-3 | 0.5 | 完整模型 |
| `ib_only` | 1e-3 | 0.0 | 仅IB压缩 |
| `adv_only` | 0.0 | 0.5 | 仅对抗去被试 |
| `baseline_ce` | 0.0 | 0.0 | 纯CE基线 |
| `high_beta` | 1e-2 | 0.5 | 强压缩 |
| `low_beta` | 1e-4 | 0.5 | 弱压缩 |
| `high_lambda` | 1e-3 | 1.0 | 强去被试 |
| `low_lambda` | 1e-3 | 0.1 | 弱去被试 |

### 支持的 Backbone

**仅 CodeBrain (SSSM)**

---

## 8. 方法对比表

### 8.1 总体对比

| 特性 | MSFT | Improved MSFT | MI Fine-Tuning | Hyperbolic | IB+Disentangle |
|------|------|---------------|----------------|------------|----------------|
| **支持的Backbone** | SSSM + CBraMod | CBraMod only | SSSM only | SSSM only | SSSM only |
| **核心技术** | 多时间尺度 | 多尺度+架构感知 | VIB + InfoNCE | Lorentz 流形 | Token IB + GRL |
| **损失函数** | CE | CE | CE+VIB+InfoNCE | CE+HHSW | CE+IB+MI(GRL) |
| **需要专家特征** | 否 | 否 | 是 (PSD/stats) | 否 | 否 |
| **需要被试ID** | 否 | 否 | 否 | 是 (域标签) | 是 |
| **域自适应** | 否 | 否 | 否 | 是 (DSMDBN+HHSW) | 是 (GRL) |
| **可解释性** | Scale权重 | Scale权重 | 无 | 无 | **Per-token heatmap** |
| **几何空间** | 欧氏 | 欧氏 | 欧氏 | **双曲 (Lorentz)** | 欧氏 |
| **特征处理** | 多尺度 | 多尺度 | Flatten→单向量 | Flatten→流形投影 | **Token-level保持** |
| **默认epochs** | 50 | 50 | 50 | 100 | 100 |
| **Scheduler** | CosineAnnealing | CosineAnnealing | CosineAnnealing | Early Stop | Early Stop |

### 8.2 各方法在不同 Backbone 上的适用性

| Method | CodeBrain (SSSM) | CBraMod |
|--------|------------------|---------|
| MSFT | ✓ (CrossScaleAggregator 3D) | ✓ (CrossScaleAggregator4D) |
| Improved MSFT | ✗ | ✓ (专为Criss-Cross设计) |
| MI Fine-Tuning | ✓ | ✗ |
| Hyperbolic | ✓ | ✗ |
| IB + Disentangle | ✓ | ✗ |

### 8.3 各方法在不同 Dataset 上的适用性

所有方法理论上支持 `DATASET_CONFIGS` 中定义的全部 10 个数据集。主要实验在 TUEV 和 TUAB 上进行：

| Method | TUEV (6-class) | TUAB (Binary) | 其他 |
|--------|---------------|---------------|------|
| MSFT | ✓ | ✓ | 全支持 |
| Improved MSFT | ✓ | ✓ | 全支持 |
| MI Fine-Tuning | ✓ | ✓ | 全支持 |
| Hyperbolic | ✓ (已有checkpoint) | ✓ (已修复二分类bug) | 全支持 |
| IB + Disentangle | ✓ | ✓ | 需LMDB中有subject_id字段 |

---

## 9. 实验状态与结果

### 9.1 已完成的实验

| 实验 | Backbone | Dataset | 状态 | Checkpoint | 结果 |
|------|----------|---------|------|-----------|------|
| Hyperbolic Fine-Tuning | CodeBrain | TUEV | ✅ 完成 | `checkpoints_hyperbolic/best_TUEV_hyp.pth` (109MB) | 见WandB |
| CBraMod Full Finetune | CBraMod | AD_DIAGNOSIS | ✅ 完成 | 外部 (bal_acc=0.744, kappa=0.769, F1=0.865 @ epoch 6) | 作为transfer learning上游 |

### 9.2 已提交但失败的实验

| 实验 | 原因 | SLURM Job |
|------|------|-----------|
| IB + Disentanglement | mkdir权限错误 (SLURM路径问题) | 738679 |

### 9.3 脚本已就绪、待运行

| 实验 | 脚本 | 实验数量 | 描述 |
|------|------|---------|------|
| MSFT CodeBrain Scale Ablation | `run_codebrain_msft.sh` | 2×4=8 | 1/2/3/4 scales × TUEV/TUAB |
| MSFT CBraMod Variant Ablation | `run_ablation_fixed_params.sh` | 2×4=8 | 4 variants × TUEV/TUAB |
| MSFT CBraMod Grid Sweep | `configs/sweep_msft_cbramod.yaml` | 576 | WandB Sweep |
| MSFT CBraMod Bayesian Sweep | `configs/sweep_msft_cbramod_bayesian.yaml` | Bayesian | WandB Sweep |
| MI Fine-Tuning | `train_mi_finetuning.py` | ~8 | 4 ablation configs × 2 datasets |
| IB + Disentangle Ablation | `run_ib.sh` | 2×8=16 | 8 configs × TUEV/TUAB |
| CBraMod → CHB-MIT Transfer | `run_cbramod_chu.sh` | 3 | Linear probe / Full FT / Pretrained probe |
| CBraMod TUAB | `run_cbramod_tuab.sh` | 1 | 基线微调 |

### 9.4 WandB 项目

| WandB Project | 对应方法 | 状态 |
|---------------|---------|------|
| `codebrain-msft-ablation` | MSFT CodeBrain | 待运行 |
| `eeg-msft-improved-ablation` | Improved MSFT | 待运行 |
| `eeg-msft-improved-bayes` | MSFT Bayesian Sweep | 待运行 |
| `codebrain-ib-disentangle` | IB + Disentangle | 待运行 |
| `diagnosis_foundation_model` | CBraMod Transfer | 待运行 |

---

## 附录: 文件索引

### 框架文件

| 文件 | 方法 | 类型 |
|------|------|------|
| `finetune_tuev_lmdb.py` | 所有 | 数据加载基础设施 |
| `finetune_msft.py` | MSFT | 训练脚本 (CodeBrain+CBraMod) |
| `msft_modules.py` | MSFT | 模型定义 |
| `finetune_msft_improved.py` | Improved MSFT | 训练脚本 (CBraMod only) |
| `msft_modules_improved.py` | Improved MSFT | 模型定义 |
| `mi_finetuning_framework.py` | MI Fine-Tuning | 框架 (VIB+InfoNCE) |
| `train_mi_finetuning.py` | MI Fine-Tuning | 训练脚本 |
| `test_mi_framework.py` | MI Fine-Tuning | 测试脚本 |
| `hyperbolic_finetuning.py` | Hyperbolic | 框架 (Lorentz+DSMDBN+HMLR) |
| `train_hyperbolic_finetuning.py` | Hyperbolic | 训练脚本 |
| `ib_disentangle_framework.py` | IB+Disentangle | 框架 (IB+GRL) |
| `train_ib_disentangle.py` | IB+Disentangle | 训练脚本 |
| `test_ib_disentangle.py` | IB+Disentangle | 测试脚本 |

### 运行脚本

| 脚本 | 方法 | 类型 |
|------|------|------|
| `scripts/run_codebrain_msft.sh` | MSFT CodeBrain | 本地运行 |
| `scripts/slurm_codebrain_msft.sh` | MSFT CodeBrain | SLURM提交 |
| `scripts/run_ablation_fixed_params.sh` | Improved MSFT | 本地运行 |
| `scripts/slurm_ablation_fixed_params.sh` | Improved MSFT | SLURM提交 |
| `scripts/run_ablation_study.sh` | MSFT全量搜索 | 本地运行 |
| `scripts/slurm_submit_ablation.sh` | MSFT全量搜索 | SLURM提交 |
| `scripts/run_ib.sh` | IB+Disentangle | 本地运行/SLURM |
| `scripts/run_cbramod_chu.sh` | Transfer Learning | SLURM |
| `scripts/run_cbramod_tuab.sh` | CBraMod基线 | SLURM |
| `scripts/slurm_wandb_agent.sh` | WandB Sweep | SLURM |
