# MSFT CBraMod Ablation Study框架 - 完整总结

## 🎉 创建的文件概览

本框架为MSFT CBraMod的ablation study提供了完整的实验基础设施，支持本地运行、WandB Sweep和Slurm集群。

### 📂 文件列表

```
eeg2025/NIPS/
├── 训练脚本
│   ├── finetune_msft_improved.py          # 改进的MSFT训练脚本
│   └── msft_modules_improved.py            # 改进的MSFT模块
│
├── WandB Sweep配置
│   ├── sweep_msft_cbramod.yaml             # Grid搜索配置
│   └── sweep_msft_cbramod_bayesian.yaml    # Bayesian优化配置
│
├── 本地运行脚本
│   └── run_ablation_study.sh               # 本地批量测试脚本
│
├── Slurm作业脚本
│   ├── slurm_single_experiment.sh          # 单个实验的Slurm脚本
│   ├── slurm_submit_ablation.sh            # 批量提交管理脚本
│   └── slurm_wandb_agent.sh                # WandB Sweep的agent脚本
│
└── 文档
    ├── README_ABLATION.md                  # 快速参考指南
    ├── ABLATION_STUDY_GUIDE.md             # 详细使用指南
    ├── SLURM_USAGE.md                      # Slurm使用指南
    └── ABLATION_FRAMEWORK_SUMMARY.md       # 本文件
```

## 🚀 三种运行模式

### 1️⃣ 本地运行模式

适合：快速测试、小规模实验

```bash
# 单个测试
./run_ablation_study.sh single

# 核心配置（24个实验）
./run_ablation_study.sh local

# 查看帮助
./run_ablation_study.sh help
```

**特点:**
- ✅ 简单易用，快速启动
- ✅ 支持本地顺序或并行运行
- ❌ 需要手动管理GPU资源
- ❌ 不适合大规模实验

### 2️⃣ Slurm集群模式 (推荐)

适合：大规模ablation study、生产环境

```bash
# 测试单个实验
sbatch slurm_single_experiment.sh TUEV full 3 64 1e-3 5e-2 0.1

# 批量提交核心配置（自动管理4张A100）
./slurm_submit_ablation.sh core

# 查看作业状态
./slurm_submit_ablation.sh status
```

**特点:**
- ✅ 自动资源管理和调度
- ✅ 支持最多4张A100并行
- ✅ 作业队列和优先级
- ✅ 详细的日志记录
- ✅ 适合长时间运行

### 3️⃣ WandB Sweep模式

适合：超参数搜索、自动化优化

```bash
# Grid搜索
wandb sweep sweep_msft_cbramod.yaml
sbatch --array=0-3 slurm_wandb_agent.sh <sweep_id>

# Bayesian优化
wandb sweep sweep_msft_cbramod_bayesian.yaml
sbatch --array=0-1 slurm_wandb_agent.sh <sweep_id>
```

**特点:**
- ✅ 智能超参数搜索
- ✅ 实时可视化
- ✅ 自动最优配置发现
- ✅ 支持提前停止
- ✅ 完整的实验追踪

## 🔬 Ablation Study设计

### 实验维度

| 维度 | 选项 | 数量 |
|------|------|------|
| **数据集** | TUEV, TUAB | 2 |
| **MSFT变体** | baseline, pos_refiner, criss_cross_agg, full | 4 |
| **Scale数量** | 2, 3, 4 | 3 |
| **Batch Size** | 32, 64 | 2 |
| **Learning Rate** | 5e-4, 1e-3, 2e-3 | 3 |
| **Weight Decay** | 5e-2, 1e-1 | 2 |
| **Dropout** | 0.1, 0.2 | 2 |

### 实验规模

| 配置 | 实验数 | 预计时间* | 模式 |
|------|--------|-----------|------|
| 核心配置 | 24 | ~12小时 | Slurm (4 GPU) |
| Grid Sweep | 576 | ~6天 | Slurm (4 GPU) |
| Bayesian | 50-100 | ~1-2天 | Slurm (2 GPU) |

*假设每个实验约2小时

## 📋 快速开始检查清单

### Slurm集群使用（推荐路径）

- [ ] **步骤1**: 配置环境加载
  ```bash
  vim slurm_single_experiment.sh
  # 修改module load和环境激活命令
  ```

- [ ] **步骤2**: 测试单个实验
  ```bash
  sbatch slurm_single_experiment.sh TUEV full 3 64 1e-3 5e-2 0.1
  squeue -u $USER
  ```

- [ ] **步骤3**: 确认WandB登录
  ```bash
  wandb login
  ```

- [ ] **步骤4**: 提交核心配置
  ```bash
  ./slurm_submit_ablation.sh core
  ```

- [ ] **步骤5**: 监控实验
  ```bash
  ./slurm_submit_ablation.sh status
  # 或访问WandB网页
  ```

### 本地快速测试

- [ ] **步骤1**: 单个测试
  ```bash
  ./run_ablation_study.sh single
  ```

- [ ] **步骤2**: 检查结果
  ```bash
  ls -lh logs/
  ls -lh checkpoints/
  ```

## 🎯 MSFT改进解释

### 核心问题

原始MSFT在CBraMod上的实现**破坏了Criss-Cross Attention结构**：
- ❌ 将不同scale的序列concat后一起通过transformer
- ❌ 导致T-Attention跨越不同时间分辨率的patches
- ❌ frozen ACPE无法适应不同scale的seq_len

### 我们的改进

#### 1. Independent Scale Processing ✅
```python
# 每个scale独立通过transformer
for layer_idx, layer in enumerate(self.backbone.encoder.layers):
    for h_k in scale_hiddens:
        h_k_out = layer(h_k)  # 保持Criss-Cross结构
```

#### 2. Scale-specific Positional Refinement ✅
```python
# 为每个scale添加可训练的位置refinement
self.scale_pos_refiners[k](emb_k)
```

#### 3. Criss-Cross-aware Aggregator ✅
```python
# 同时考虑temporal和spatial的跨scale聚合
temporal_agg = self._temporal_aggregation(scale_hiddens)
spatial_agg = self._spatial_fusion(temporal_agg)
```

### 4个实验变体

| 变体 | Pos Refiner | Criss-Cross Agg | 说明 |
|------|-------------|-----------------|------|
| **baseline** | ❌ | ❌ | 无改进（对照组） |
| **pos_refiner** | ✅ | ❌ | 只测试位置refinement效果 |
| **criss_cross_agg** | ❌ | ✅ | 只测试改进的aggregator |
| **full** | ✅ | ✅ | 完整改进（预期最佳） |

## 📊 预期结果分析

### 关键指标

1. **final_test/balanced_acc** - 主要优化目标
2. **scale_weight/*** - 分析哪个scale贡献最大
3. **final_test/kappa** - 类别平衡性能
4. **final_test/f1** - 精确率和召回率平衡

### 分析问题

1. **改进有效性**: full > criss_cross_agg/pos_refiner > baseline?
2. **单项贡献**: 哪个改进（pos_refiner vs criss_cross_agg）贡献更大？
3. **Scale分布**: 不同变体的scale权重分布有何不同？
4. **数据集差异**: TUEV vs TUAB表现是否一致？
5. **最优配置**: 哪个超参数组合最好？

### WandB可视化建议

```python
# 创建对比图表
# X轴: variant (baseline, pos_refiner, criss_cross_agg, full)
# Y轴: final_test/balanced_acc
# 分组: dataset (TUEV, TUAB)
# 颜色: num_scales (2, 3, 4)

# 分析scale权重
# 热力图: variant × scale → weight value
```

## 🛠️ 故障排除快速参考

| 问题 | 解决方案 |
|------|----------|
| CUDA OOM | 减小batch_size或num_scales |
| 作业PENDING | 检查GPU队列: `squeue -p gpu` |
| 环境加载失败 | 测试交互式会话: `srun --pty bash` |
| WandB登录 | 运行 `wandb login` |
| 时间超限 | 增加 `--time` 或减少epochs |

## 📚 文档导航

- **新手**: 从 `README_ABLATION.md` 开始
- **详细教程**: 阅读 `ABLATION_STUDY_GUIDE.md`
- **Slurm用户**: 参考 `SLURM_USAGE.md`
- **快速参考**: 本文件

## 🔮 未来扩展

### 添加CodeBrain支持

当前框架只支持CBraMod。要添加CodeBrain：

1. **实现模型** (`msft_modules_improved.py`)
   ```python
   class ImprovedMSFTCodeBrainModel(nn.Module):
       # 类似CBraMod实现
   ```

2. **更新训练脚本** (`finetune_msft_improved.py`)
   ```python
   parser.add_argument('--model', choices=['cbramod', 'codebrain'])
   ```

3. **更新配置**
   ```yaml
   # sweep配置
   parameters:
     model:
       values: [cbramod, codebrain]
   ```

详见 `ABLATION_STUDY_GUIDE.md` 的相关章节。

## 🎓 引用和致谢

如果这个框架对你的研究有帮助，请引用：
- CBraMod论文 (参考 `cbramod.pdf`)
- MSFT方法论文

## 📞 支持

- **问题报告**: 检查日志文件 `logs/`
- **实验监控**: WandB dashboard
- **作业状态**: `./slurm_submit_ablation.sh status`

---

**框架版本**: 1.0
**创建日期**: 2026-02-26
**支持的模型**: CBraMod (CodeBrain待添加)
**支持的数据集**: TUEV, TUAB
**测试状态**: ✅ 框架完整，待实验验证

**下一步**: 运行 `sbatch slurm_single_experiment.sh TUEV full 3 64 1e-3 5e-2 0.1` 开始第一个实验！
