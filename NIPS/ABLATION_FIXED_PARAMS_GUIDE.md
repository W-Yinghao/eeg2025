# Ablation Study - 固定参数版本

## 📋 概述

这个版本的ablation study脚本**只测试MSFT改进的效果**，所有超参数固定为最优默认值。

### 实验配置

- **数据集**: TUEV, TUAB (2个)
- **MSFT变体**: baseline, pos_refiner, criss_cross_agg, full (4个)
- **总实验数**: 2 × 4 = **8个实验**

### 固定参数

```bash
epochs=50
batch_size=64
learning_rate=1e-3
weight_decay=5e-2
dropout=0.1
num_scales=3
label_smoothing=0.1
clip_value=1.0
seed=3407
```

## 🚀 快速开始

### 方式1: 本地运行 (run_ablation_fixed_params.sh)

```bash
cd ~/eeg2025/NIPS

# 运行所有实验 (8个)
./run_ablation_fixed_params.sh

# 只运行TUEV数据集 (4个)
./run_ablation_fixed_params.sh TUEV

# 只运行TUAB数据集 (4个)
./run_ablation_fixed_params.sh TUAB

# 查看帮助
./run_ablation_fixed_params.sh --help
```

**特点:**
- ✅ 简单易用，立即开始
- ✅ 自动顺序运行所有实验
- ✅ 详细的日志记录
- ⚠️ 需要手动等待，不能并行

### 方式2: Slurm集群 (slurm_ablation_fixed_params.sh)

```bash
cd ~/eeg2025/NIPS

# 提交所有8个实验到Slurm队列
./slurm_ablation_fixed_params.sh submit

# 提交单个实验
./slurm_ablation_fixed_params.sh submit_one TUEV full
./slurm_ablation_fixed_params.sh submit_one TUAB baseline

# 查看作业状态
./slurm_ablation_fixed_params.sh status

# 取消所有作业
./slurm_ablation_fixed_params.sh cancel
```

**特点:**
- ✅ 自动并行运行（取决于可用GPU）
- ✅ 队列管理，资源调度
- ✅ 适合长时间运行
- ⚠️ 需要先配置环境加载

## 📊 实验变体说明

| 变体名称 | Positional Refiner | Criss-Cross Aggregator | 说明 |
|---------|-------------------|----------------------|------|
| **baseline** | ❌ | ❌ | 无改进（对照组） |
| **pos_refiner** | ✅ | ❌ | 只测试位置编码refinement |
| **criss_cross_agg** | ❌ | ✅ | 只测试Criss-Cross aggregator |
| **full** | ✅ | ✅ | 所有改进（预期最佳） |

### 预期结果

基于之前的分析，我们预期：
- **baseline** < **pos_refiner** / **criss_cross_agg** < **full**
- 单项改进应该各自带来1-2%的提升
- 组合效果可能是叠加的或超叠加的

## 📁 文件结构

运行后会生成：

```
eeg2025/NIPS/
├── logs_ablation/              # 日志目录
│   ├── MSFT_baseline_TUEV_s3_*.log
│   ├── MSFT_pos_refiner_TUEV_s3_*.log
│   ├── MSFT_criss_cross_agg_TUEV_s3_*.log
│   ├── MSFT_full_TUEV_s3_*.log
│   ├── MSFT_baseline_TUAB_s3_*.log
│   └── ... (共8个日志)
│
├── checkpoints_ablation/       # 模型检查点
│   ├── msft_improved_cbramod_baseline_tuev_*.pth
│   ├── msft_improved_cbramod_full_tuev_*.pth
│   └── ... (共8个模型)
│
└── WandB项目: msft-ablation-study
```

## 🔧 Slurm配置（如需要）

编辑 `slurm_ablation_fixed_params.sh` 中的环境加载部分：

```bash
# 在脚本的这一部分添加你的环境配置
# 加载环境 - TODO: 根据你的集群配置修改
# module load cuda/11.8
# module load python/3.9
# source /path/to/venv/bin/activate

# 或使用conda
# module load anaconda3
# conda activate eeg_env
```

## 📊 监控和分析

### 实时监控

```bash
# 本地运行 - 查看当前日志
tail -f logs_ablation/MSFT_full_TUEV_*.log

# Slurm运行 - 查看作业状态
./slurm_ablation_fixed_params.sh status
squeue -u $USER

# 查看Slurm作业日志
tail -f logs_ablation/msft_full_TUEV_*.out
```

### WandB查看

所有实验都会自动记录到WandB项目：`msft-ablation-study`

**关键指标:**
- `final_test/balanced_acc` - 主要对比指标
- `final_test/kappa` - 类别平衡性能
- `final_test/f1` - F1分数
- `scale_weight/*` - 各scale的权重分布

### 结果汇总

运行完成后，可以使用Python脚本提取结果：

```python
import wandb
import pandas as pd

api = wandb.Api()
runs = api.runs("your_entity/msft-ablation-study")

results = []
for run in runs:
    results.append({
        'dataset': run.config.get('dataset'),
        'variant': run.tags[3] if len(run.tags) > 3 else 'unknown',
        'test_acc': run.summary.get('final_test/balanced_acc'),
        'test_kappa': run.summary.get('final_test/kappa'),
        'test_f1': run.summary.get('final_test/f1')
    })

df = pd.DataFrame(results)

# 按变体和数据集分组
pivot = df.pivot_table(
    values='test_acc',
    index='variant',
    columns='dataset',
    aggfunc='mean'
)

print(pivot)
```

## 📈 预期时间估算

假设每个实验约2小时（取决于数据集大小和GPU性能）：

| 模式 | 实验数 | 顺序运行时间 | 并行运行时间* |
|------|--------|------------|--------------|
| 全部 (TUEV + TUAB) | 8 | ~16小时 | ~4-8小时 |
| 只TUEV | 4 | ~8小时 | ~2-4小时 |
| 只TUAB | 4 | ~8小时 | ~2-4小时 |

*并行时间取决于可用GPU数量

## 🎯 完成后的分析步骤

### 1. 对比不同变体

```bash
# 在WandB中创建对比图表
# X轴: variant (baseline, pos_refiner, criss_cross_agg, full)
# Y轴: final_test/balanced_acc
# 分组: dataset (TUEV, TUAB)
```

### 2. 分析改进贡献

- **Pos Refiner贡献**: pos_refiner - baseline
- **Criss-Cross Agg贡献**: criss_cross_agg - baseline
- **组合效果**: full - baseline
- **协同效应**: full - (pos_refiner + criss_cross_agg - baseline)

### 3. 数据集特异性

对比TUEV和TUAB上的表现：
- 哪个变体在哪个数据集上更好？
- 改进是否在两个数据集上都一致？

### 4. Scale权重分布

观察不同变体的scale权重：
- baseline vs. full的权重分布有何不同？
- 哪个scale通常获得最高权重？

## 🐛 故障排除

### 问题1: 脚本权限错误

```bash
chmod +x run_ablation_fixed_params.sh
chmod +x slurm_ablation_fixed_params.sh
```

### 问题2: CUDA Out of Memory

修改脚本中的 `BATCH_SIZE`:
```bash
BATCH_SIZE=32  # 从64减少到32
```

### 问题3: 数据集路径错误

在脚本命令中添加 `--datasets_dir`:
```bash
python finetune_msft_improved.py \
    ... \
    --datasets_dir /path/to/your/datasets
```

### 问题4: WandB登录问题

```bash
wandb login
# 输入你的API key
```

## 📝 与原始脚本的区别

| 特性 | 原始sweep脚本 | 新的固定参数脚本 |
|------|--------------|-----------------|
| **参数搜索** | Grid/Bayesian搜索 | 固定最优值 |
| **实验数量** | 576个 | 8个 |
| **运行时间** | ~2周 | ~16小时 |
| **目的** | 寻找最优超参数 | 验证ablation效果 |
| **复杂度** | 高（需要WandB sweep） | 低（直接运行） |
| **适用场景** | 超参数优化 | 快速验证改进 |

## ✅ 推荐工作流程

### 快速验证（1-2天）

```bash
# 1. 先在TUEV上测试
./run_ablation_fixed_params.sh TUEV

# 2. 如果结果好，再测TUAB
./run_ablation_fixed_params.sh TUAB
```

### 完整实验（Slurm集群）

```bash
# 1. 配置环境加载（编辑slurm_ablation_fixed_params.sh）

# 2. 提交所有实验
./slurm_ablation_fixed_params.sh submit

# 3. 定期检查状态
watch -n 60 './slurm_ablation_fixed_params.sh status'

# 4. 完成后分析WandB结果
```

## 📞 获取帮助

```bash
# 查看脚本帮助
./run_ablation_fixed_params.sh --help
./slurm_ablation_fixed_params.sh help

# 查看详细文档
cat ABLATION_STUDY_GUIDE.md
cat SLURM_USAGE.md
```

---

**创建时间**: 2026-02-26
**版本**: 1.0 - 简化版（固定参数）
**目的**: 快速验证MSFT改进的有效性

准备好开始了！运行 `./run_ablation_fixed_params.sh` 开始你的ablation study 🚀
