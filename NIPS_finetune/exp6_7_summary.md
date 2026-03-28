# 实验总结：Exp6A / Exp6B / Exp7A / Exp7B

## 通用设置

- **Backbone**: CodeBrain (SSSM), 15,065,200 params, 预训练权重冻结
- **数据集**: TUAB (Temple University Abnormal EEG), 二分类 (normal vs abnormal)
  - Train: 297,103 samples (1,655 subjects), 标签分布: {normal: 147,894, abnormal: 149,209}
  - Val: 75,407 samples (424 subjects), 标签分布: {normal: 35,815, abnormal: 39,592}
  - Test: 36,945 samples (253 subjects), 标签分布: {normal: 19,907, abnormal: 17,038}
- **GPU**: Tesla V100-SXM2-32GB (Jean Zay) / Tesla V100-PCIE-16GB (local)

---

## 1. Exp6A: Frozen Baseline (仅训练 head, 无 selector gate)

**配置**: 50 epochs, patience=12, 104,962 trainable params (0.69%), head = simple classifier

### 逐 Epoch (展示 seed=42 代表性曲线, 其余种子趋势一致)

所有5个种子的最终测试结果如下:

| Seed | Best Epoch | Test Bal Acc | Macro F1 | Weighted F1 | Abnormal Recall | Abnormal F1 |
|------|-----------|-------------|----------|-------------|-----------------|-------------|
| 42   | 34        | 0.7888      | 0.7890   | 0.7920      | 0.8534          | 0.8165      |
| 1234 | 30        | 0.7908      | 0.7928   | 0.7949      | 0.8648          | 0.8208      |
| 2025 | 14        | 0.7915      | 0.7936   | 0.7960      | 0.8766          | 0.8239      |
| 3407 | 29        | 0.7889      | 0.7927   | 0.7955      | 0.8629          | 0.8214      |
| 7777 | 26        | 0.7932      | 0.7948   | 0.7967      | 0.8515          | 0.8194      |
| **Mean±Std** | | **0.7906±0.0018** | **0.7926±0.0021** | **0.7950±0.0018** | **0.8618±0.0094** | **0.8204±0.0027** |

**关键发现**: Baseline 在冻结 backbone 下稳定达到 ~79% balanced accuracy, 方差极小 (<0.2%)

---

## 2. Exp6A: Frozen Selector (selector head, 无正则化)

**配置**: 50 epochs, patience=12, 187,860 trainable params (1.23%)
- Head 结构: temporal_gate + frequency_gate + classifier (14 trainable layers)
- 无额外正则化 (无 sparse, 无 consistency)

### 所有种子全部失败

| Seed | Best Epoch | Test Bal Acc | Macro F1 | Status |
|------|-----------|-------------|----------|--------|
| 42   | 1         | 0.5000      | 0.3502   | **NaN loss from epoch 1** |
| 1234 | 1         | 0.5000      | 0.3502   | **NaN loss from epoch 1** |
| 2025 | 1         | 0.5000      | 0.3502   | **NaN loss from epoch 1** |
| 3407 | 1         | 0.5000      | 0.3502   | **NaN loss from epoch 1** |
| 7777 | 1         | 0.5000      | 0.3502   | **NaN loss from epoch 1** |

**逐 Epoch 示例 (seed=42)**:
```
Ep  1/50 | Train: loss=nan acc=0.5678 | Val: acc=0.5000 f1=0.3220 | Test: acc=0.5000
Ep  2/50 | Train: loss=nan acc=0.5000 | Val: acc=0.5000 f1=0.3220 | Test: acc=0.5000
...
Ep 13/50 | Early stopping (best: epoch 1, metric=0.5000)
```

- 混淆矩阵: 全部预测为 normal, abnormal recall=0.0000
- Gate 统计: 全部 NaN, gate_coverage=0.0000
- **结论**: Selector head 在无正则化时训练完全崩溃, loss 立即变为 NaN

---

## 3. Exp6B Baseline: Gentle Partial Fine-tuning (baseline head + 解冻 top-1 层)

**通用配置**: CodeBrain backbone 解冻 top-1 层, baseline head, 1,756,162~1,844,460 trainable params (~11.6%)

### 3.1 Exp6B P1 Baseline (lr_backbone=1e-5, lr_head=1e-3, 12 epochs, patience=4, warmup=3)

| Seed | Best Epoch | Test Bal Acc | Macro F1 | Weighted F1 | Abnormal Recall | Abnormal F1 |
|------|-----------|-------------|----------|-------------|-----------------|-------------|
| 42   | 11        | 0.8165      | 0.8190   | 0.8210      | 0.8962          | 0.8449      |
| 2025 | 11        | 0.8204      | 0.8228   | 0.8246      | 0.8938          | 0.8471      |
| 3407 | 10        | 0.8197      | 0.8216   | 0.8233      | 0.8795          | 0.8436      |
| **Mean±Std** | | **0.8189±0.0021** | **0.8211±0.0019** | **0.8230±0.0018** | **0.8898±0.0090** | **0.8452±0.0018** |

### 3.2 Exp6B P2 Baseline (lr_backbone=1e-6, lr_head=1e-3, 12 epochs, patience=4)

| Seed | Best Epoch | Test Bal Acc | Macro F1 | Weighted F1 | Abnormal Recall | Abnormal F1 |
|------|-----------|-------------|----------|-------------|-----------------|-------------|
| 42   | 12        | 0.8084      | 0.8106   | 0.8126      | 0.8825          | 0.8365      |
| 2025 | 11        | 0.8078      | 0.8094   | 0.8112      | 0.8644          | 0.8322      |
| 3407 | 12        | 0.8096      | 0.8116   | 0.8134      | 0.8743          | 0.8356      |
| **Mean±Std** | | **0.8086±0.0009** | **0.8105±0.0011** | **0.8124±0.0011** | **0.8737±0.0091** | **0.8348±0.0022** |

### 3.3 Exp6B P3e1 Baseline (Stage1: 1ep partial top-1 → Stage2: 20ep frozen, lr_head=5e-4)

| Seed | Stage1 Best | Stage2 Best Ep | Test Bal Acc | Macro F1 | Weighted F1 | Abnormal Recall | Abnormal F1 |
|------|------------|---------------|-------------|----------|-------------|-----------------|-------------|
| 42   | Ep 1       | 15            | 0.7944      | 0.7965   | 0.7987      | 0.8739          | 0.8252      |
| 2025 | Ep 1       | 7             | 0.7859      | 0.7872   | 0.7890      | 0.8364          | 0.8109      |
| 3407 | Ep 1       | 10            | 0.7883      | 0.7894   | 0.7912      | 0.8349          | 0.8121      |
| **Mean±Std** | | | **0.7895±0.0044** | **0.7910±0.0049** | **0.7930±0.0051** | **0.8484±0.0221** | **0.8161±0.0079** |

### 3.4 Exp6B P3e2 Baseline (Stage1: 2ep partial top-1 → Stage2: 20ep frozen, lr_head=5e-4)

| Seed | Stage1 Best | Stage2 Best Ep | Test Bal Acc | Macro F1 | Weighted F1 | Abnormal Recall | Abnormal F1 |
|------|------------|---------------|-------------|----------|-------------|-----------------|-------------|
| 42   | Ep 2       | 13            | 0.8171      | 0.8192   | 0.8211      | 0.8853          | 0.8430      |
| 2025 | Ep 2       | 18            | 0.8133      | 0.8153   | 0.8171      | 0.8782          | 0.8389      |
| 3407 | Ep 2       | 13            | 0.8112      | 0.8135   | 0.8156      | 0.8879          | 0.8396      |
| **Mean±Std** | | | **0.8139±0.0030** | **0.8160±0.0029** | **0.8179±0.0028** | **0.8838±0.0051** | **0.8405±0.0022** |

---

## 4. Exp6B Selector: Gentle Partial Fine-tuning (selector head + 解冻 top-1 层)

**通用配置**: 1,844,460 trainable params (12.09%), selector head with temporal/frequency gates

### 全部失败 — 与 Exp6A Selector 相同的 NaN 问题

| 实验 | Seeds | Test Bal Acc | Macro F1 | Status |
|------|-------|-------------|----------|--------|
| P1 (lr_bb=1e-5) | 42, 2025, 3407 | 0.5000 | 0.3502 | NaN loss, early stop ep 5 |
| P2 (lr_bb=1e-6) | 42, 2025, 3407 | 0.5000 | 0.3502 | NaN loss, early stop ep 5 |
| P3e1 (1ep+20ep) | 42, 2025, 3407 | 0.5000 | 0.3502 | Stage1 train acc~0.54-0.57, Stage2 NaN, early stop ep 7 |
| P3e2 (2ep+20ep) | 42, 2025, 3407 | 0.5000 | 0.3502 | Stage1 train acc~0.54-0.57, Stage2 NaN, early stop ep 7 |

**共 12 个 run 全部失败**, 模型坍塌到全部预测 normal, abnormal recall=0.0000

**结论**: Selector head 在无正则化时无论是 frozen 还是 partial fine-tuning 都无法训练

---

## 5. Exp7A: Frozen Selector + L1 Sparsity 正则化

**配置**: 50 epochs, patience=12, 187,860 trainable params (1.23%), frozen backbone
- 正则化: L1 sparse loss on gate activations
- 测试了三个 lambda: 1e-3, 1e-4, 3e-4

### 5.1 Exp7A λ_sparse=1e-3 (3 seeds)

**逐 Epoch (seed=42)**:
```
Ep  1/50 | Train: loss=0.5072 acc=0.7521 | Val: acc=0.7664 f1=0.7648 | sp=0.5884 | g_t=0.558 g_f=0.619 | Test: acc=0.7769
Ep  2/50 | Train: loss=0.4684 acc=0.7759 | Val: acc=0.8004 f1=0.7999 | sp=0.3959 | g_t=0.402 g_f=0.390 | Test: acc=0.7916
Ep  3/50 | Train: loss=0.4518 acc=0.7857 | Val: acc=0.7866 f1=0.7863 | sp=0.2706 | g_t=0.279 g_f=0.262 | Test: acc=0.7986
Ep  5/50 | Train: loss=0.4326 acc=0.7975 | Val: acc=0.7938 f1=0.7936 | sp=0.2051 | g_t=0.206 g_f=0.204 | Test: acc=0.8003
Ep 10/50 | Train: loss=0.4082 acc=0.8112 | Val: acc=0.8024 f1=0.7999 | sp=0.1767 | g_t=0.179 g_f=0.174 | Test: acc=0.7960
Ep 15/50 | Train: loss=0.3940 acc=0.8193 | Val: acc=0.8070 f1=0.8068 | sp=0.1756 | g_t=0.180 g_f=0.172 | Test: acc=0.7998
Ep 20/50 | Train: loss=0.3834 acc=0.8257 | Val: acc=0.8104 f1=0.8093 | sp=0.1800 | g_t=0.186 g_f=0.173 | Test: acc=0.8015  ← BEST
Ep 25/50 | Train: loss=0.3730 acc=0.8308 | Val: acc=0.8043 f1=0.8029 | sp=0.1790 | g_t=0.187 g_f=0.171 | Test: acc=0.7951
Ep 30/50 | Train: loss=0.3645 acc=0.8355 | Val: acc=0.8044 f1=0.8028 | sp=0.1750 | g_t=0.184 g_f=0.166 | Test: acc=0.8035
Ep 32/50 | Early stopping (best: epoch 20, metric=0.8104)
```

**逐 Epoch (seed=2025)**:
```
Ep  1/50 | Train: loss=0.5083 acc=0.7494 | Val: acc=0.7601 f1=0.7577 | sp=0.5100 | g_t=0.546 g_f=0.474 | Test: acc=0.7734
Ep  5/50 | Train: loss=0.4280 acc=0.8005 | Val: acc=0.7950 f1=0.7915 | sp=0.1515 | g_t=0.152 g_f=0.151 | Test: acc=0.8076
Ep 10/50 | Train: loss=0.4014 acc=0.8154 | Val: acc=0.7972 f1=0.7924 | sp=0.1547 | g_t=0.161 g_f=0.149 | Test: acc=0.8079
Ep 15/50 | Train: loss=0.3860 acc=0.8236 | Val: acc=0.7921 f1=0.7879 | sp=0.1554 | g_t=0.166 g_f=0.145 | Test: acc=0.8064
Ep 19/50 | Train: loss=0.3792 acc=0.8280 | Val: acc=0.8141 f1=0.8137 | sp=0.1589 | g_t=0.175 g_f=0.143 | Test: acc=0.8084  ← BEST
Ep 25/50 | Train: loss=0.3668 acc=0.8334 | Val: acc=0.8040 f1=0.8023 | sp=0.1587 | g_t=0.175 g_f=0.142 | Test: acc=0.8026
Ep 31/50 | Early stopping (best: epoch 19, metric=0.8141)
```

**逐 Epoch (seed=3407)**:
```
Ep  1/50 | Train: loss=0.5091 acc=0.7501 | Val: acc=0.7622 f1=0.7598 | sp=0.4897 | g_t=0.477 g_f=0.502 | Test: acc=0.7763
Ep  5/50 | Train: loss=0.4343 acc=0.7970 | Val: acc=0.7873 f1=0.7863 | sp=0.1602 | g_t=0.135 g_f=0.185 | Test: acc=0.7980
Ep 10/50 | Train: loss=0.4063 acc=0.8123 | Val: acc=0.8070 f1=0.8059 | sp=0.1670 | g_t=0.152 g_f=0.182 | Test: acc=0.8042
Ep 15/50 | Train: loss=0.3913 acc=0.8211 | Val: acc=0.8044 f1=0.8019 | sp=0.1843 | g_t=0.178 g_f=0.191 | Test: acc=0.7889
Ep 20/50 | Train: loss=0.3790 acc=0.8280 | Val: acc=0.8081 f1=0.8073 | sp=0.1848 | g_t=0.183 g_f=0.187 | Test: acc=0.7985
Ep 23/50 | Train: loss=0.3748 acc=0.8297 | Val: acc=0.8128 f1=0.8110 | sp=0.1918 | g_t=0.191 g_f=0.192 | Test: acc=0.8039  ← BEST
Ep 30/50 | Train: loss=0.3620 acc=0.8369 | Val: acc=0.7949 f1=0.7889 | sp=0.1846 | g_t=0.186 g_f=0.183 | Test: acc=0.7958
Ep 35/50 | Early stopping (best: epoch 23, metric=0.8128)
```

**最终测试结果 (λ=1e-3)**:

| Seed | Best Ep | Test Bal Acc | Macro F1 | W-F1 | Abn Recall | Abn F1 | Abn Prec | CE Loss | Sparsity@best |
|------|---------|-------------|----------|------|------------|--------|----------|---------|---------------|
| 42   | 20      | 0.8015      | 0.8031   | 0.8050 | 0.8589   | 0.8267 | 0.7968   | 0.4348  | 0.180 |
| 2025 | 19      | 0.8084      | 0.8089   | 0.8103 | 0.8348   | 0.8260 | 0.8173   | 0.4330  | 0.159 |
| 3407 | 23      | 0.8039      | 0.8060   | 0.8080 | 0.8759   | 0.8321 | 0.7924   | 0.4375  | 0.192 |
| **Mean±Std** | | **0.8046±0.0035** | **0.8060±0.0029** | **0.8078±0.0027** | **0.8565±0.0209** | **0.8283±0.0032** | **0.8022±0.0131** | | |

**Gate 统计 (test, seed=42, λ=1e-3)**:
- g_t_mean=0.1932, g_f_mean=0.1761 (gate 值较低, 说明稀疏性好)
- gate_coverage_0.5=0.0325 (仅 3.25% 的 gate 值>0.5)
- gate_entropy=0.4184 (较低熵, 说明 gate 分布集中)
- temporal gate coverage=0.0000, frequency gate coverage=0.0650
- gate_top10_ratio=0.1692, gate_top20_ratio=0.3661

---

### 5.2 Exp7A λ_sparse=1e-4 (3 seeds)

| Seed | Best Ep | Test Bal Acc | Macro F1 | W-F1 | Abn Recall | Abn F1 | Abn Prec | CE Loss |
|------|---------|-------------|----------|------|------------|--------|----------|---------|
| 42   | 20      | 0.7973      | 0.7987   | 0.8005 | 0.8488   | 0.8216 | 0.7960   | 0.4395 |
| 2025 | 19      | 0.8062      | 0.8068   | 0.8082 | 0.8352   | 0.8245 | 0.8141   | 0.4352 |
| 3407 | 27      | 0.8035      | 0.8053   | 0.8071 | 0.8649   | 0.8294 | 0.7967   | 0.4364 |
| **Mean±Std** | | **0.8023±0.0046** | **0.8036±0.0042** | **0.8053±0.0041** | **0.8496±0.0149** | **0.8252±0.0040** | **0.8023±0.0100** | |

**Sparsity 演变 (seed=2025)**: ep1(0.5248)→ep5(0.1757)→ep10(0.1712)→ep19(0.1797)→ep31(0.1721)

---

### 5.3 Exp7A λ_sparse=3e-4 (3 seeds)

| Seed | Best Ep | Test Bal Acc | Macro F1 | W-F1 | Abn Recall | Abn F1 | Abn Prec | CE Loss |
|------|---------|-------------|----------|------|------------|--------|----------|---------|
| 42   | 20      | 0.7992      | 0.8006   | 0.8023 | 0.8502   | 0.8231 | 0.7978   | 0.4409 |
| 2025 | 19      | 0.8073      | 0.8080   | 0.8094 | 0.8371   | 0.8257 | 0.8147   | 0.4323 |
| 3407 | 16      | 0.7996      | 0.8015   | 0.8035 | 0.8663   | 0.8270 | 0.7912   | 0.4314 |
| **Mean±Std** | | **0.8020±0.0046** | **0.8034±0.0040** | **0.8051±0.0038** | **0.8512±0.0147** | **0.8253±0.0020** | **0.8012±0.0121** | |

**Sparsity 演变 (seed=42)**: ep1(0.5964)→ep5(0.2393)→ep10(0.1995)→ep20(0.2058)→ep32(0.1977)

### Exp7A 三个 λ 对比汇总

| λ_sparse | Mean Bal Acc | Mean Macro F1 | Mean Abn Recall | Mean Abn F1 | Sparsity (converged) |
|----------|-------------|---------------|-----------------|-------------|---------------------|
| **1e-3** | **0.8046±0.0035** | **0.8060±0.0029** | **0.8565±0.0209** | **0.8283±0.0032** | ~0.17-0.19 |
| 1e-4     | 0.8023±0.0046 | 0.8036±0.0042 | 0.8496±0.0149 | 0.8252±0.0040 | ~0.17-0.19 |
| 3e-4     | 0.8020±0.0046 | 0.8034±0.0040 | 0.8512±0.0147 | 0.8253±0.0020 | ~0.18-0.20 |

**结论**: λ=1e-3 表现最好, 但三个 lambda 差异不大 (<0.3%). Sparsity 都收敛到 ~0.17-0.20 范围

---

## 6. Exp7B: Frozen Selector + Consistency (L2) 正则化

**配置**: 50 epochs, patience=12, 187,860 trainable params (1.23%), frozen backbone
- 正则化: L2 consistency loss (鼓励同一 sample 的 gate 输出稳定)
- 测试了三个 lambda: 1e-2, 1e-3, 3e-3

### 6.1 Exp7B λ_cons=1e-2 (3 seeds)

**逐 Epoch (seed=42)**:
```
Ep  1/50 | Train: loss=0.5067 acc=0.7516 | Val: acc=0.7663 f1=0.7627 | cons=0.0023 | g_t=0.576 g_f=0.628 | Test: acc=0.7671
Ep  2/50 | Train: loss=0.4688 acc=0.7756 | Val: acc=0.7995 f1=0.7975 | cons=0.0127 | g_t=0.466 g_f=0.450 | Test: acc=0.7925
Ep  5/50 | Train: loss=0.4339 acc=0.7970 | Val: acc=0.7899 f1=0.7867 | cons=0.0087 | g_t=0.224 g_f=0.213 | Test: acc=0.7891
Ep 10/50 | Train: loss=0.4076 acc=0.8105 | Val: acc=0.8018 f1=0.8007 | cons=0.0076 | g_t=0.203 g_f=0.192 | Test: acc=0.7997
Ep 15/50 | Train: loss=0.3929 acc=0.8196 | Val: acc=0.7902 f1=0.7848 | cons=0.0079 | g_t=0.208 g_f=0.195 | Test: acc=0.7914
Ep 20/50 | Train: loss=0.3820 acc=0.8264 | Val: acc=0.7966 f1=0.7944 | cons=0.0076 | g_t=0.205 g_f=0.192 | Test: acc=0.8108
Ep 21/50 | Train: loss=0.3790 acc=0.8264 | Val: acc=0.8090 f1=0.8067 | cons=0.0081 | g_t=0.215 g_f=0.197 | Test: acc=0.7999  ← BEST
Ep 25/50 | Train: loss=0.3718 acc=0.8319 | Val: acc=0.7934 f1=0.7904 | cons=0.0077 | g_t=0.213 g_f=0.196 | Test: acc=0.8021
Ep 30/50 | Train: loss=0.3640 acc=0.8360 | Val: acc=0.7980 f1=0.7962 | cons=0.0074 | g_t=0.213 g_f=0.191 | Test: acc=0.8070
Ep 33/50 | Early stopping (best: epoch 21, metric=0.8090)
```

**逐 Epoch (seed=2025)** — 注意: 此 run 在 epoch 35 被截断, 未完成:
```
Ep  1/50 | Train: loss=0.5082 acc=0.7494 | Val: acc=0.7598 f1=0.7548 | cons=0.0015 | g_t=0.576 g_f=0.487 | Test: acc=0.7605
Ep  5/50 | Train: loss=0.4270 acc=0.8007 | Val: acc=0.7918 f1=0.7902 | cons=0.0097 | g_t=0.259 g_f=0.210 | Test: acc=0.8105
Ep 10/50 | Train: loss=0.4019 acc=0.8153 | Val: acc=0.8023 f1=0.8010 | cons=0.0103 | g_t=0.249 g_f=0.194 | Test: acc=0.8144
Ep 15/50 | Train: loss=0.3880 acc=0.8224 | Val: acc=0.8068 f1=0.8055 | cons=0.0103 | g_t=0.252 g_f=0.190 | Test: acc=0.8106
Ep 20/50 | Train: loss=0.3778 acc=0.8275 | Val: acc=0.7969 f1=0.7951 | cons=0.0100 | g_t=0.252 g_f=0.186 | Test: acc=0.7950
Ep 25/50 | Train: loss=0.3694 acc=0.8321 | Val: acc=0.8047 f1=0.8035 | cons=0.0100 | g_t=0.256 g_f=0.187 | Test: acc=0.8118
Ep 30/50 | Train: loss=0.3602 acc=0.8373 | Val: acc=0.8027 f1=0.8006 | cons=0.0095 | g_t=0.258 g_f=0.188 | Test: acc=0.8091
Ep 34/50 | Train: loss=0.3541 acc=0.8404 | Val: acc=0.8102 f1=0.8082 | cons=0.0088 | Test: acc=0.8104
Ep 35/50 | (截断, 未完成)
```

**逐 Epoch (seed=3407)**:
```
Ep  1/50 | Train: loss=0.5081 acc=0.7498 | Val: acc=0.7611 f1=0.7568 | cons=0.0009 | g_t=0.489 g_f=0.516 | Test: acc=0.7714
Ep  5/50 | Train: loss=0.4334 acc=0.7971 | Val: acc=0.8051 f1=0.7996 | cons=0.0066 | g_t=0.159 g_f=0.198 | Test: acc=0.7846
Ep  6/50 | Train: loss=0.4257 acc=0.8014 | Val: acc=0.8097 f1=0.8062 | cons=0.0067 | g_t=0.160 g_f=0.200 | Test: acc=0.8014  ← BEST
Ep 10/50 | Train: loss=0.4037 acc=0.8145 | Val: acc=0.7944 f1=0.7908 | cons=0.0064 | g_t=0.167 g_f=0.184 | Test: acc=0.8075
Ep 15/50 | Train: loss=0.3897 acc=0.8216 | Val: acc=0.8010 f1=0.7996 | cons=0.0063 | g_t=0.173 g_f=0.175 | Test: acc=0.8071
Ep 18/50 | Early stopping (best: epoch 6, metric=0.8097)
```

**最终测试结果 (λ_cons=1e-2)**:

| Seed | Best Ep | Test Bal Acc | Macro F1 | W-F1 | Abn Recall | Abn F1 | Abn Prec | CE Loss | cons_l2 |
|------|---------|-------------|----------|------|------------|--------|----------|---------|---------|
| 42   | 21      | 0.7999      | 0.8021   | 0.8042 | 0.8779   | 0.8298 | 0.7867   | 0.4613 | 0.0011 |
| 2025 | —       | (截断, 未出 final) | — | — | — | — | — | — | — |
| 3407 | 6       | 0.8014      | 0.8039   | 0.8063 | 0.8937   | 0.8342 | 0.7822   | 0.4348 | 0.0005 |
| **Mean (42,3407)** | | **0.8007** | **0.8030** | **0.8053** | **0.8858** | **0.8320** | **0.7845** | | |

**Gate 统计 (test, seed=42, λ_cons=1e-2)**:
- g_t_mean=0.2302, g_f_mean=0.2124
- gate_consistency_l2=0.0011 (非常低, 说明一致性好)
- gate_coverage_0.5=0.0563
- gate_entropy=0.4542
- abnormal_only_consistency_l2=0.0009, normal_only_consistency_l2=0.0014

---

### 6.2 Exp7B λ_cons=1e-3 (3 seeds)

| Seed | Best Ep | Test Bal Acc | Macro F1 | W-F1 | Abn Recall | Abn F1 | Abn Prec | CE Loss | cons_l2 |
|------|---------|-------------|----------|------|------------|--------|----------|---------|---------|
| 42   | 21      | 0.7979      | 0.8001   | 0.8023 | 0.8792   | 0.8288 | 0.7838   | 0.4603 | 0.0015 |
| 2025 | 15      | 0.8092      | 0.8103   | 0.8118 | 0.8488   | 0.8297 | 0.8115   | 0.4264 | 0.0020 |
| 3407 | 6       | 0.8008      | 0.8033   | 0.8057 | 0.8950   | 0.8341 | 0.7809   | 0.4341 | 0.0007 |
| **Mean±Std** | | **0.8026±0.0059** | **0.8046±0.0052** | **0.8066±0.0048** | **0.8743±0.0237** | **0.8309±0.0028** | **0.7921±0.0166** | | |

---

### 6.3 Exp7B λ_cons=3e-3 (3 seeds)

| Seed | Best Ep | Test Bal Acc | Macro F1 | W-F1 | Abn Recall | Abn F1 | Abn Prec | CE Loss | cons_l2 |
|------|---------|-------------|----------|------|------------|--------|----------|---------|---------|
| 42   | 28      | 0.8073      | 0.8091   | 0.8110 | 0.8675   | 0.8326 | 0.8004   | 0.4419 | 0.0013 |
| 2025 | 15      | 0.8092      | 0.8105   | 0.8121 | 0.8529   | 0.8307 | 0.8096   | 0.4222 | 0.0023 |
| 3407 | 6       | 0.8000      | 0.8026   | 0.8050 | 0.8964   | 0.8339 | 0.7795   | 0.4347 | 0.0006 |
| **Mean±Std** | | **0.8055±0.0048** | **0.8074±0.0042** | **0.8094±0.0038** | **0.8723±0.0222** | **0.8324±0.0016** | **0.7965±0.0155** | | |

### Exp7B 三个 λ 对比汇总

| λ_cons | Mean Bal Acc | Mean Macro F1 | Mean Abn Recall | Mean Abn F1 | cons_l2 (test) |
|--------|-------------|---------------|-----------------|-------------|----------------|
| 1e-2   | 0.8007 (2 seeds) | 0.8030 | 0.8858 | 0.8320 | 0.0005-0.0011 |
| 1e-3   | 0.8026±0.0059 | 0.8046±0.0052 | 0.8743±0.0237 | 0.8309±0.0028 | 0.0007-0.0020 |
| **3e-3** | **0.8055±0.0048** | **0.8074±0.0042** | **0.8723±0.0222** | **0.8324±0.0016** | 0.0006-0.0023 |

**结论**: λ_cons=3e-3 表现最好 (0.8055 bal acc), 但差异也不大

---

## 7. 全局对比总结

### 按实验组的 Mean Test Balanced Accuracy 排名

| Rank | Experiment | Config | Mean Bal Acc | Mean Macro F1 | Mean Abn F1 | Seeds | Status |
|------|-----------|--------|-------------|---------------|-------------|-------|--------|
| 1 | **Exp6B P1 Baseline** | partial top-1, lr_bb=1e-5, 12ep | **0.8189±0.0021** | **0.8211±0.0019** | **0.8452±0.0018** | 3 | OK |
| 2 | Exp6B P3e2 Baseline | staged 2ep+20ep | 0.8139±0.0030 | 0.8160±0.0029 | 0.8405±0.0022 | 3 | OK |
| 3 | Exp6B P2 Baseline | partial top-1, lr_bb=1e-6, 12ep | 0.8086±0.0009 | 0.8105±0.0011 | 0.8348±0.0022 | 3 | OK |
| 4 | **Exp7B λ=3e-3** | frozen+consistency | **0.8055±0.0048** | **0.8074±0.0042** | 0.8324±0.0016 | 3 | OK |
| 5 | Exp7A λ=1e-3 | frozen+sparse | 0.8046±0.0035 | 0.8060±0.0029 | 0.8283±0.0032 | 3 | OK |
| 6 | Exp7B λ=1e-3 | frozen+consistency | 0.8026±0.0059 | 0.8046±0.0052 | 0.8309±0.0028 | 3 | OK |
| 7 | Exp7A λ=1e-4 | frozen+sparse | 0.8023±0.0046 | 0.8036±0.0042 | 0.8252±0.0040 | 3 | OK |
| 8 | Exp7A λ=3e-4 | frozen+sparse | 0.8020±0.0046 | 0.8034±0.0040 | 0.8253±0.0020 | 3 | OK |
| 9 | Exp7B λ=1e-2 | frozen+consistency | 0.8007 | 0.8030 | 0.8320 | 2* | OK |
| 10 | Exp6A Baseline | frozen, simple head | 0.7906±0.0018 | 0.7926±0.0021 | 0.8204±0.0027 | 5 | OK |
| 11 | Exp6B P3e1 Baseline | staged 1ep+20ep | 0.7895±0.0044 | 0.7910±0.0049 | 0.8161±0.0079 | 3 | OK |
| 12 | Exp6A Selector | frozen, no reg | 0.5000 | 0.3502 | 0.0000 | 5 | **FAILED** |
| 13 | Exp6B P1-P3e2 Selector | partial, no reg | 0.5000 | 0.3502 | 0.0000 | 12 | **FAILED** |

*Exp7B λ=1e-2 seed 2025 被截断

### 核心发现

1. **Selector 必须配合正则化**: 无正则化的 selector (Exp6A selector + Exp6B selector) 全部 17 个 run 100% 失败 (NaN loss)。加入 L1 sparse 或 L2 consistency 后, 训练完全恢复正常

2. **Partial fine-tuning > Frozen (在 baseline 上)**: 解冻 top-1 层的 baseline (Exp6B P1, 0.8189) 比冻结 baseline (Exp6A, 0.7906) 提升 ~2.8%

3. **Frozen Selector + 正则化 vs Frozen Baseline**: 最好的 selector (Exp7A λ=1e-3, 0.8046; Exp7B λ=3e-3, 0.8055) 比 frozen baseline (0.7906) 提升 ~1.4-1.5%, 但仍低于 partial baseline (0.8189) 约 1.3%

4. **L1 Sparse vs L2 Consistency**: 两种正则化效果相近, Consistency λ=3e-3 (0.8055) 略优于 Sparse λ=1e-3 (0.8046), 差异不显著

5. **Backbone LR 敏感性**: lr_bb=1e-5 (P1, 0.8189) > lr_bb=1e-6 (P2, 0.8086), 差 ~1%

6. **Staged vs Continuous**: 连续 12ep partial FT (P1, 0.8189) > 2ep+20ep staged (P3e2, 0.8139) > 1ep+20ep staged (P3e1, 0.7895)

7. **Gate 行为模式 (Exp7A/7B)**:
   - Sparsity 在前 5 个 epoch 快速下降 (0.5+ → 0.15-0.20), 之后稳定
   - Temporal gate 值略高于 frequency gate (g_t ~0.18-0.23 vs g_f ~0.14-0.20)
   - Gate coverage (>0.5) 极低 (<5%), 说明大部分 gate 值被压制到 0.5 以下
   - Temporal gate coverage 几乎为 0, frequency gate coverage 略高
   - Abnormal 样本的 gate 值略低于 normal 样本

---

## 8. 深入分析：7A Best vs 7B Best 的按类别 Gate 统计

### 选取的 Best Runs

| | Exp7A Best | Exp7B Best |
|---|---|---|
| 配置 | λ_sparse=1e-3, seed=2025 | λ_cons=3e-3, seed=2025 |
| Test Bal Acc | 0.8084 | 0.8092 |
| Best Epoch | 19 | 15 |

### 8.1 完整 Gate 统计对比

测试集构成: 19,907 normal (53.9%) + 17,038 abnormal (46.1%) = 36,945 total

#### Gate Mean 值 (越高 = gate 越 "打开")

| Metric | Exp7A (sparse) | Exp7B (consistency) | 差异 |
|--------|---------------|--------------------|----|
| **Overall g_t_mean** | 0.1758 | 0.3140 | 7B temporal gate 显著更高 (+79%) |
| **Overall g_f_mean** | 0.1353 | 0.1789 | 7B frequency gate 略高 (+32%) |
| **Abnormal-only g_t_mean** | 0.1767 | 0.3152 | 与 overall 一致 |
| **Abnormal-only g_f_mean** | 0.1108 | 0.1441 | Abnormal 的 freq gate 更低 |
| **Normal-only g_t_mean** (推算) | ~0.1750 | ~0.3130 | Normal ≈ Abnormal |
| **Normal-only g_f_mean** (推算) | ~0.1562 | ~0.2087 | **Normal 的 freq gate 明显更高** |

> Normal-only 由 overall = 0.539 × normal_only + 0.461 × abnormal_only 反推

#### Gate Coverage (gate值 > 0.5 的比例)

| Metric | Exp7A (sparse) | Exp7B (consistency) |
|--------|---------------|---------------------|
| **Overall gate_coverage** | 0.0162 | 0.0401 |
| **Abnormal-only coverage** | 0.0090 | 0.0244 |
| **Normal-only coverage** (推算) | ~0.0223 | ~0.0535 |
| Overall freq coverage | 0.0323 | 0.0779 |
| Abnormal freq coverage | 0.0180 | 0.0486 |
| Normal freq coverage (推算) | ~0.0445 | ~0.1030 |
| Overall temporal coverage | 0.0000 | 0.0023 |
| Abnormal temporal coverage | 0.0000 | 0.0003 |

#### Gate Entropy (越高 = gate 越不确定; 0=全关或全开, 0.693=均匀)

| Metric | Exp7A (sparse) | Exp7B (consistency) |
|--------|---------------|---------------------|
| **Overall gate_entropy** | 0.3902 | 0.4886 |
| **Abnormal-only entropy** | 0.3743 | 0.4693 |
| **Normal-only entropy** (推算) | ~0.4038 | ~0.5051 |
| Overall freq entropy | 0.3208 | 0.3639 |
| Abnormal freq entropy | 0.2867 | 0.3230 |
| Normal freq entropy (推算) | ~0.3500 | ~0.3989 |
| Overall temporal entropy | 0.4595 | 0.6132 |
| Abnormal temporal entropy | 0.4618 | 0.6157 |
| Normal temporal entropy (推算) | ~0.4575 | ~0.6111 |

#### Gate Top-K Concentration

| Metric | Exp7A (sparse) | Exp7B (consistency) |
|--------|---------------|---------------------|
| gate_top10_ratio | 0.1696 | 0.1656 |
| gate_top20_ratio | 0.3619 | 0.3613 |
| Abnormal freq top10 | 0.2235 | 0.2302 |
| Abnormal freq top20 | 0.4951 | 0.5178 |
| Abnormal temporal top10 | 0.1230 | 0.1180 |
| Abnormal temporal top20 | 0.2389 | 0.2305 |

#### Consistency (仅 Exp7B)

| Metric | Exp7B (consistency) |
|--------|---------------------|
| Overall consistency_l2 | 0.0023 |
| Abnormal-only consistency_l2 | 0.0018 |
| Normal-only consistency_l2 | 0.0028 |

### 8.2 关键发现与偏差分析

#### 发现 1: Frequency Gate 对 Abnormal 样本的系统性抑制 (⚠️ 警告)

**在两种正则化下一致出现:**

| | g_f_mean (abnormal) | g_f_mean (normal) | 差值 | 相对差 |
|---|---|---|---|---|
| Exp7A sparse | 0.1108 | ~0.1562 | -0.0454 | **-29%** |
| Exp7B consistency | 0.1441 | ~0.2087 | -0.0646 | **-31%** |

**这是系统性偏差, 不是轻微现象.** Abnormal 样本的 frequency gate 值比 normal 低约 30%. 意味着模型在频率维度上对异常 EEG 的特征赋予了更低的注意力权重.

同样反映在 coverage 上:
- Exp7A: abnormal freq coverage 0.0180 vs normal ~0.0445 (abnormal 低 60%)
- Exp7B: abnormal freq coverage 0.0486 vs normal ~0.1030 (abnormal 低 53%)

#### 发现 2: Temporal Gate 无明显类别偏差 (正常)

| | g_t_mean (abnormal) | g_t_mean (normal) | 差值 |
|---|---|---|---|
| Exp7A sparse | 0.1767 | ~0.1750 | +0.0017 (+1%) |
| Exp7B consistency | 0.3152 | ~0.3130 | +0.0022 (+0.7%) |

Temporal gate 在两类上几乎一致, 无偏差.

#### 发现 3: Exp7B 的 Gate 更 "活跃" 但更 "模糊"

- Exp7B g_t_mean (0.314) 远高于 Exp7A (0.176) → consistency 正则化让 temporal gate 更活跃
- Exp7B entropy 更高 (0.489 vs 0.390) → gate 决策更不确定
- Exp7B coverage 更高 (0.040 vs 0.016) → 更多 gate 值超过 0.5
- 但 Exp7B 的 top10/top20 ratio 与 Exp7A 几乎一致 → 信息集中度无差异

#### 发现 4: Abnormal 样本的 Entropy 低于 Normal

在两种正则化下:
- Exp7A: abnormal entropy 0.3743 < normal ~0.4038
- Exp7B: abnormal entropy 0.4693 < normal ~0.5051

**解读**: Abnormal 样本的 gate 决策更 "确定" (entropy 低), 但方向是 "更倾向关闭" (mean 更低). 模型对 abnormal 更果断地压低了 frequency gate.

### 8.3 对 Exp7C 方向的影响

**Frequency gate 对 abnormal 的系统性抑制是核心问题.**

1. **如果目标是 "让 gate 学到有临床意义的频率选择"**: 当前 gate 在做相反的事 — 对异常 EEG (通常含有 spike、slow wave 等异常频率模式) 关闭 frequency gate, 等于在丢弃最重要的诊断信息

2. **Sparse 正则化方向**: 会进一步加剧这个问题 (sparse 鼓励更多 gate 关闭, 而 abnormal 已经偏低)

3. **Consistency 正则化方向**: 至少保证了 gate 行为在相似输入下稳定, 但没有解决方向性偏差

4. **建议 Exp7C 考虑**:
   - 引入 **class-conditional gate balancing**: 让 abnormal 和 normal 的 gate mean 不要差太远
   - 或者 **反转 frequency gate 的梯度信号**: 鼓励 abnormal 样本有更高的 freq gate (因为异常频率模式正是诊断关键)
   - 或者 **用 abnormal-aware sparse**: 对 normal 样本 sparse, 对 abnormal 样本不做或反向

---

## 9. Case 可视化脚本

可视化脚本已创建: `experiments/deb/scripts/visualize_gate_cases.py`

**功能**: 对 10 个 abnormal 正确样本 + 10 个 abnormal 错误样本, 生成:
- Raw EEG 信号热力图 (channels × time), 叠加 temporal gate 高亮
- Frequency gate 柱状图 (每个通道的 gate 值)
- Temporal gate 柱状图 (每个时间窗的 gate 值)
- Top-4 gated channels 的原始波形
- 预测概率
- 汇总对比图 (correct vs incorrect 的 gate 分布差异)

**在 Jean Zay 上运行** (checkpoint 在远程集群):

```bash
# Exp7A best (sparse λ=1e-3, seed=2025)
python experiments/deb/scripts/visualize_gate_cases.py \
    --checkpoint checkpoints_selector/exp7a_sparse_l1e3_selector/best_TUAB_codebrain_selector_frozen_acc0.8084_s2025.pth \
    --dataset TUAB --model codebrain --seed 2025 \
    --n_correct 10 --n_incorrect 10 \
    --output_dir gate_visualizations/exp7a_sparse_l1e3_s2025

# Exp7B best (consistency λ=3e-3, seed=2025)
python experiments/deb/scripts/visualize_gate_cases.py \
    --checkpoint checkpoints_selector/exp7b_cons_l3e3_selector/best_TUAB_codebrain_selector_frozen_acc0.8092_s2025.pth \
    --dataset TUAB --model codebrain --seed 2025 \
    --n_correct 10 --n_incorrect 10 \
    --output_dir gate_visualizations/exp7b_cons_l3e3_s2025
```

**输出**:
- `correct_00.png` ~ `correct_09.png`: 10个正确预测的 abnormal 样本
- `incorrect_00.png` ~ `incorrect_09.png`: 10个错误预测的 abnormal 样本
- `comparison_summary.png`: 汇总对比图
- `comparison_stats.json`: 数值统计
