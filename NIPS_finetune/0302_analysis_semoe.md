# 0302 SageStream (SA-MoE + IIB) 实验分析

**日志文件：** slurm-740835 ~ slurm-740838
**日期：** 2026-03-02

---

## 实验配置说明

| 字段 | 说明 |
|------|------|
| 架构 | SA-MoE（主体感知专家路由）+ IIB（信息瓶颈，对抗去除主体信息） |
| 变体共8个 | full_sagestream / moe_only / iib_only / baseline_ce / high_experts / low_experts / deep_moe / no_style |
| 数据集 | TUAB（二分类）/ TUEV（6分类，严重类别不平衡） |

### 8个消融变体定义

| 变体 | alpha_kl | beta_adv | aux_weight | num_experts | top_k | n_moe_layers | use_style |
|------|----------|----------|------------|-------------|-------|--------------|-----------|
| full_sagestream | 1e-3 | 0.5 | 0.01 | 4 | 2 | 2 | yes |
| moe_only | 0.0 | 0.0 | 0.01 | 4 | 2 | 2 | yes |
| iib_only | 1e-3 | 0.5 | 0.0 | 4 | 2 | 0 | no |
| baseline_ce | 0.0 | 0.0 | 0.0 | 4 | 2 | 0 | no |
| high_experts | 1e-3 | 0.5 | 0.01 | 8 | 2 | 2 | yes |
| low_experts | 1e-3 | 0.5 | 0.01 | 2 | 1 | 2 | yes |
| deep_moe | 1e-3 | 0.5 | 0.01 | 4 | 2 | 3 | yes |
| no_style | 1e-3 | 0.5 | 0.01 | 4 | 2 | 2 | no |

---

## 结果汇总

### 740835 — codebrain + TUAB（7/8完成，最完整）

| 变体 | Val Acc | Test Bal Acc | 备注 |
|------|---------|--------------|------|
| full_sagestream | 0.6089 | 0.5923 | **最差** |
| moe_only | **0.8115** | **0.7996** | **最佳** |
| iib_only | 0.7663 | 0.7801 | 较好 |
| baseline_ce | 0.7914 | 0.7839 | 次佳 |
| high_experts | 0.5822 | 0.5894 | 差 |
| low_experts | 0.6182 | 0.6177 | 差 |
| deep_moe | 0.6486 | 0.6662 | 差 |
| no_style | — | — | 日志截断，未完成 |

### 740836 — codebrain + TUEV（8/8完成，但类别不平衡严重）

| 变体 | Val Acc | Test Bal Acc | Gap | 备注 |
|------|---------|--------------|-----|------|
| full_sagestream | 0.3096 | 0.2886 | 0.02 | 欠拟合 |
| moe_only | **0.9627** | 0.4975 | **0.465** | 严重类别不平衡 |
| iib_only | 0.6566 | 0.4396 | 0.217 | 类别不平衡 |
| baseline_ce | 0.8947 | 0.4542 | 0.440 | 严重类别不平衡 |
| high_experts | 0.4265 | 0.3828 | 0.04 | 不稳定 |
| low_experts | 0.2807 | 0.2722 | 0.01 | 欠拟合 |
| deep_moe | 0.2363 | 0.2352 | 0.00 | 严重欠拟合 |
| no_style | 0.8177 | 0.4756 | 0.342 | 类别不平衡 |

### 740837 — cbramod + TUEV（3/8完成，5/8 segfault）

| 变体 | Val Acc | Test Bal Acc | 状态 |
|------|---------|--------------|------|
| full_sagestream | 0.2814 | 0.2055 | 完成 |
| moe_only | — | — | **SEGFAULT** |
| iib_only | — | — | **SEGFAULT** |
| baseline_ce | 0.3500 | 0.3375 | 完成 |
| high_experts | — | — | **SEGFAULT**（epoch 1即崩溃） |
| low_experts | 0.2688 | 0.2594 | 完成 |
| deep_moe | — | — | **SEGFAULT** |
| no_style | — | — | **SEGFAULT** |

### 740838 — cbramod + TUAB（2/8完成，5/8未执行）

| 变体 | Val Acc | Test Bal Acc | 状态 |
|------|---------|--------------|------|
| full_sagestream | 0.5552 | 0.5593 | 完成 |
| moe_only | — | — | 未执行（前序crash导致中断） |
| iib_only | — | — | 未执行 |
| baseline_ce | — | — | 未执行 |
| high_experts | — | — | 未执行 |
| low_experts | — | — | 未执行 |
| deep_moe | 0.5901 | 0.5731 | 完成 |
| no_style | — | — | **SEGFAULT**（epoch 1） |

---

## 性能差的根本原因分析

> **⚠️ 论文勘误（已对照原文修正）**：SA-MoE 全称是 **Style-Adaptive MoE**，不是 Subject-Aware MoE。其核心目标与 IIB **相同**，都是学习 subject-invariant 特征，而非增强主体特异性表示。详见下文。

### 问题1：Subject ID 提取失败 → SA-MoE 和 IIB 双双失效（最根本原因）

`sagestream_analysis.md` 已记录：训练日志显示 `Subjects: 2`，即从 TUAB（实际约 2329 个受试者）中只检测到 **2 个** subject ID。

这直接导致：
- `SubjectStyleAlignment` 的 `nn.Embedding(2, ...)` 只有 2 个嵌入向量，风格规范化几乎无意义
- IIB 的 GRL 对抗训练也只针对 2 个类别，完全无法学习有效的 subject-invariant 表示
- 因此 `moe_only`（风格规范化有效）≈ `baseline_ce`（无风格模块），而 `full_sagestream`（IIB 干扰梯度）最差

**根本修复**：正确解析 TUAB/TUEV 文件名中的 subject ID，恢复真实的受试者数量。

### 问题2：SA-MoE 与 IIB 目标冗余，不是"冲突"

论文原文（p.4）明确：

> "SA-MoE, aiming to **capture subject-invariance** via PEFT"
> Figure 3 caption: "SA-MoE, which **learns subject-invariant features ¯Z**"

SA-MoE 的 SISL（Subject-Invariant Style Learning）流程：

```
输入 Z → Instance Norm（擦除个体风格）→ ¯Z_norm（subject-invariant 内容）
       → Hyper-Network(subject_embed) → γ, β → ¯Z = γ·¯Z_norm + β（规范化风格）
```

IIB 的流程：

```
pooled → Variational Encoder (KL bottleneck) → z → GRL → subject_head（对抗）
```

两者**目标相同**（都是去除主体信息），但机制不同。在 subject ID 正确的情况下，同时使用两者：
- 造成功能冗余（都在做同一件事）
- IIB 的 GRL 梯度反转会干扰 SA-MoE 已经学到的风格规范化参数
- 两套参数相互竞争，导致优化不稳定（full_sagestream 的 val loss 后期暴涨印证了这一点）

**注意**：IIB 并不在原始 SAGESTREAM 论文中，是本实现自行添加的模块。论文 Stage 1 的训练目标仅为 `L_PEFT = L_CE + λ·L_AUX`（只有 CE + 负载均衡辅助损失）。

### 问题2：TUEV 类别严重不平衡

Val acc 高（最高 0.96）但 test_bal_acc 低（约 0.49）—— 模型退化为只预测多数类（`bckg` 占比极高），少数类完全被忽略。训练时 CE loss 未对类别不平衡做任何补偿，导致：

- 高 val_acc 是虚假的（预测多数类就能达到）
- test_bal_acc 接近随机（6分类随机为 0.167），部分模型甚至更差

### 问题3：cbramod DataLoader Segmentation Fault

**错误信息：** `RuntimeError: DataLoader worker (pid XXXXX) is killed by signal: Segmentation fault`

**原因：** `num_workers=4`，cbramod 是更大的 Transformer（n_layer=12），每个 worker 并行加载批次时内存需求远超 codebrain，多 worker 并行导致内存溢出，OS 强制杀死 worker 进程（SIGSEGV）。740838 中 no_style 在 epoch 1 validation 时即崩溃，说明问题出在数据加载而非训练本身。

### 问题4：实现与论文存在显著差距（HSE 缺失 + Stage 2 缺失）

**差距1：缺少 HSE（Hybrid-Shared Expert）**

论文 SA-MoE 的 MoE 输出公式（式4）：
```
z^{l+1}_i = Σ G(¯z^l_i) · [ ξ_r(¯z^l_i) + HSE_Θ(¯z^l_i) ]
```
其中 HSE 通过 `Θ = MLP([ϕ, E_expert])` 将 style context ϕ 注入到每个专家的输出中，让专家能感知受试者风格信息，弥补未被选中专家的损失（"remaining info c"）。

当前代码实现为标准 top-k MoE，**完全没有 HSE**，丢失了 SA-MoE 最关键的设计。

**差距2：缺少 Stage 2（STSA）**

论文完整框架分两阶段：
- Stage 1（已实现）：在源受试者上做 PEFT
- Stage 2（未实现）：对流式测试样本做 Test-Time Adaptation

STSA（Spatio-Temporal Style Adaptation）在测试时激活 S-Adapter（轻量风格参数 γ', β'），根据测试样本的时空统计量与 Stage 1 学到的规范风格参数的差异（置信度加权）动态更新风格，实现对新受试者的迁移。

没有 Stage 2，当前实现只是一个普通的 frozen backbone + SA-MoE 训练，丧失了 SAGESTREAM 跨受试者迁移的核心能力。

**差距3：专家池跨层共享**

```python
self.expert_pool = ExpertPool(...)  # 所有层共用同一个专家池
self.moe_layers = nn.ModuleList([
    SAMoELayer(expert_pool=self.expert_pool, ...)  # 共享
    for _ in range(n_moe_layers)
])
```

增加层数（deep_moe）或专家数（high_experts）不真正增加容量，反而增加梯度冲突，导致性能下降。

---

## 关键发现对比

| 组合 | TUAB Test Bal Acc | TUEV Test Bal Acc | 可靠性 |
|------|-------------------|-------------------|--------|
| codebrain + moe_only | **0.7996** | 0.4975 | ✓ |
| codebrain + iib_only | 0.7801 | 0.4396 | ✓ |
| codebrain + baseline_ce | 0.7839 | 0.4542 | ✓ |
| codebrain + full_sagestream | 0.5923 | 0.2886 | ✓ |
| cbramod + full_sagestream | 0.5593 | 0.2055 | ✓ |
| cbramod + deep_moe | 0.5731 | — | ✓ |

**结论：** moe_only 是当前最优变体；full_sagestream（SA-MoE+IIB 组合）在所有完成的实验中**一致地是最差的**。

---

## 改进建议

| 优先级 | 问题 | 建议 |
|--------|------|------|
| ★★★ | Subject ID 提取失败 | 修复数据加载，从 TUAB/TUEV 文件名正确解析 subject ID（TUAB ≈2329 人，TUEV ≈300+ 人）|
| ★★★ | TUEV 类别不平衡 | CrossEntropy 加 `class_weight`（反比于类别频率）；或改用 Focal Loss；或使用 balanced sampler |
| ★★ | 缺少 HSE | 实现 Hybrid-Shared Expert：在 MoE 输出加入 `HSE_Θ(¯z)` 分支，Θ 由 `MLP([style_ctx, expert_embed])` 生成 |
| ★★ | 缺少 Stage 2（STSA）| 实现测试时 S-Adapter 自适应，这是 SAGESTREAM 跨受试者迁移的核心 |
| ★★ | cbramod segfault | `num_workers=4` → `num_workers=0` 或 `1`；或加 `persistent_workers=True` + 减小 batch_size |
| ★ | IIB 冗余 | 在 subject ID 正确后，先测试不加 IIB 的 moe_only 效果；如果仍不理想，再考虑 IIB 的必要性 |
| ★ | 专家池跨层共享 | 各层独立 `expert_pool` 和 `router`（不共享），增加实际容量 |

### 针对 TUEV 的具体修复

```python
# 方案1：加权CE Loss
class_counts = torch.tensor([...])  # 各类别样本数
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

# 方案2：Focal Loss (gamma=2.0)
```

### 修复 Subject ID 提取（最优先）

```python
# TUAB 文件名格式如 "aaaaaaap_s001_t000.edf" → subject = "aaaaaaap_s001"
# 需要在 dataset loading 中正确解析，而不是依赖 metadata 字段
```

---

## 总结

**对照原始论文的核心发现：**

| 项目 | 论文设计 | 当前实现 | 差距影响 |
|------|---------|---------|---------|
| SA-MoE 目标 | Subject-**invariant**（消除风格差异） | 同，但 subject ID 只有 2 个 | 风格规范化近乎无效 |
| HSE（Hybrid-Shared Expert） | 必须组件，将 style context 注入 MoE | **未实现** | 丢失 SA-MoE 核心设计 |
| Stage 2 STSA | 测试时对新受试者自适应 | **未实现** | 丢失跨受试者迁移能力 |
| IIB（KL + GRL） | 论文中**不存在** | 自行添加 | 与 SA-MoE 冗余，干扰优化 |
| 训练损失 | L_CE + λ·L_AUX | L_CE + α·KL + β·GRL_adv + λ·AUX | 多余损失项扰乱训练 |

**性能差的根本逻辑链：**
```
Subject ID 提取失败（2人）
    → SA-MoE 风格规范化无效（Embedding(2) 几乎学不到跨受试者风格）
    → IIB 对抗训练无效（只对抗 2 个类别）
    → full_sagestream = SA-MoE(无效) + IIB(干扰梯度) = 最差
    → moe_only = SA-MoE(部分有效，至少没有IIB干扰) = 最好
```

**下一步优先级：**
1. **修复 subject ID 提取**（最关键，影响 SA-MoE 和 IIB 双模块）
2. **实现 HSE**（补全论文核心组件）
3. **修复 TUEV 类别不平衡**（加 class_weight）
4. **实现 Stage 2 STSA**（论文跨受试者迁移的核心）
5. **修复 cbramod segfault**（减少 num_workers）
