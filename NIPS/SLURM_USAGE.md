# Slurm使用指南 - MSFT CBraMod Ablation Study

本指南说明如何在Slurm集群上运行MSFT CBraMod的ablation study实验。

## 📋 前提条件

1. **Slurm集群访问**: 确保你有访问GPU分区的权限
2. **A100 GPU**: 脚本配置为使用A100，最多4张并行
3. **WandB账号**: 如果使用sweep功能
4. **环境配置**: Python环境已安装所需依赖

## 🚀 快速开始

### 1. 配置环境加载

首先，编辑Slurm脚本中的环境加载部分：

```bash
# 编辑 slurm_single_experiment.sh
vim slurm_single_experiment.sh

# 找到这一行并修改为你的环境
# 示例1: 使用module + virtualenv
module load cuda/11.8
module load python/3.9
source /path/to/your/venv/bin/activate

# 示例2: 使用conda
module load anaconda3
conda activate eeg_env
```

对 `slurm_wandb_agent.sh` 做同样的修改。

### 2. 测试单个实验

```bash
cd ~/eeg2025/NIPS

# 提交一个测试作业
sbatch slurm_single_experiment.sh TUEV full 3 64 1e-3 5e-2 0.1

# 查看作业状态
squeue -u $USER

# 查看日志（使用实际的Job ID）
tail -f logs/slurm_<job_id>_msft_full_TUEV_s3.out
```

### 3. 批量提交核心配置

```bash
# 提交24个核心实验（自动管理并行）
./slurm_submit_ablation.sh core

# 查看作业状态
./slurm_submit_ablation.sh status

# 或使用Slurm命令
squeue -u $USER
```

### 4. 使用WandB Sweep

```bash
# Step 1: 创建sweep
wandb sweep sweep_msft_cbramod.yaml
# 输出: wandb agent user/project/abc123def

# Step 2: 提交4个并行agent（使用4张A100）
sbatch --array=0-3 slurm_wandb_agent.sh user/project/abc123def

# 查看agent状态
squeue -u $USER
```

## 📁 Slurm脚本说明

### 1. `slurm_single_experiment.sh`

单个实验的Slurm作业脚本。

**参数:**
```bash
sbatch slurm_single_experiment.sh <dataset> <variant> <num_scales> <batch_size> <lr> <wd> <dropout>
```

**示例:**
```bash
# TUEV数据集, full变体, 3 scales
sbatch slurm_single_experiment.sh TUEV full 3 64 1e-3 5e-2 0.1

# TUAB数据集, baseline变体, 4 scales, 大batch
sbatch slurm_single_experiment.sh TUAB baseline 4 128 2e-3 1e-1 0.2
```

**资源配置:**
- 1个节点
- 1张A100 GPU
- 8个CPU核心
- 64GB内存
- 12小时时间限制

### 2. `slurm_submit_ablation.sh`

批量提交脚本，管理多个实验的提交和并行控制。

**命令:**

```bash
# 提交核心配置（24个实验）
./slurm_submit_ablation.sh core

# 提交完整grid搜索（576个实验，需要确认）
./slurm_submit_ablation.sh full

# 交互式提交自定义配置
./slurm_submit_ablation.sh custom

# 查看所有作业状态
./slurm_submit_ablation.sh status

# 取消所有提交的作业
./slurm_submit_ablation.sh cancel
```

**并行控制:**
- 脚本会自动限制最多4个作业同时运行
- 每次提交前检查队列，等待空闲槽位
- 所有作业ID记录在 `logs/submitted_job_ids.txt`

### 3. `slurm_wandb_agent.sh`

WandB Sweep的agent作业脚本，支持并行运行多个agent。

**使用方法:**

```bash
# 1个agent
sbatch slurm_wandb_agent.sh <sweep_id>

# 2个agent并行
sbatch --array=0-1 slurm_wandb_agent.sh <sweep_id>

# 4个agent并行（最大）
sbatch --array=0-3 slurm_wandb_agent.sh <sweep_id>
```

**资源配置:**
- 每个agent: 1张A100 GPU
- 48小时时间限制（适合长时间sweep）
- 支持job array自动分配GPU

## 📊 监控和管理

### 查看作业状态

```bash
# 查看所有作业
squeue -u $USER

# 查看特定作业详情
scontrol show job <job_id>

# 查看作业历史
sacct -u $USER --format=JobID,JobName,State,Elapsed,MaxRSS

# 使用脚本查看
./slurm_submit_ablation.sh status
```

### 查看日志

```bash
# 实时查看输出日志
tail -f logs/slurm_<job_id>_<job_name>.out

# 查看错误日志
tail -f logs/slurm_<job_id>_<job_name>.err

# 查看所有日志
ls -lth logs/
```

### 取消作业

```bash
# 取消单个作业
scancel <job_id>

# 取消所有自己的作业
scancel -u $USER

# 取消特定名称的作业
scancel -n msft_full_TUEV_s3

# 使用脚本取消所有记录的作业
./slurm_submit_ablation.sh cancel
```

### 作业资源统计

```bash
# 查看作业的资源使用情况
sstat -j <job_id> --format=JobID,MaxRSS,MaxVMSize,AveCPU

# 查看已完成作业的统计
sacct -j <job_id> --format=JobID,JobName,Elapsed,MaxRSS,MaxVMSize,State
```

## ⚙️ 配置调整

### 修改GPU类型

如果你的集群使用不同的GPU类型：

```bash
# 编辑slurm脚本，修改这一行
#SBATCH --gres=gpu:a100:1

# 改为
#SBATCH --gres=gpu:v100:1
# 或
#SBATCH --gres=gpu:1  # 不指定类型
```

### 修改分区名称

```bash
# 如果GPU分区名称不是"gpu"
#SBATCH --partition=gpu

# 改为你的分区名称
#SBATCH --partition=gpu_a100
```

### 修改时间限制

```bash
# 对于快速测试
#SBATCH --time=2:00:00  # 2小时

# 对于长时间训练
#SBATCH --time=24:00:00  # 24小时
```

### 修改并行数量

```bash
# 编辑 slurm_submit_ablation.sh
MAX_PARALLEL_JOBS=8  # 从4改为8（如果有更多GPU）
```

### 修改内存需求

```bash
# 如果遇到内存不足
#SBATCH --mem=128G  # 从64G增加到128G
```

## 🔬 实验配置示例

### 示例1: 快速测试所有变体

```bash
# 只在TUEV上测试所有4个变体
for variant in baseline pos_refiner criss_cross_agg full; do
    sbatch slurm_single_experiment.sh TUEV ${variant} 3 64 1e-3 5e-2 0.1
done
```

### 示例2: 对比不同scale数量

```bash
# 使用full变体对比2,3,4个scales
for scales in 2 3 4; do
    sbatch slurm_single_experiment.sh TUEV full ${scales} 64 1e-3 5e-2 0.1
done
```

### 示例3: 超参数搜索

```bash
# 在固定配置下搜索最佳学习率
for lr in 5e-4 1e-3 2e-3 5e-3; do
    sbatch slurm_single_experiment.sh TUEV full 3 64 ${lr} 5e-2 0.1
done
```

### 示例4: 完整的数据集对比

```bash
# 两个数据集上使用相同配置
for dataset in TUEV TUAB; do
    sbatch slurm_single_experiment.sh ${dataset} full 3 64 1e-3 5e-2 0.1
done
```

## 📈 WandB Sweep高级用法

### 创建并启动Grid Sweep

```bash
# 1. 创建sweep
wandb sweep sweep_msft_cbramod.yaml

# 2. 启动多个agent并行运行
SWEEP_ID="user/project/abc123def"
sbatch --array=0-3 slurm_wandb_agent.sh ${SWEEP_ID}

# 3. 监控sweep进度
# 访问WandB网页查看实时结果
```

### 创建并启动Bayesian Sweep

```bash
# 1. 创建Bayesian sweep（智能搜索）
wandb sweep sweep_msft_cbramod_bayesian.yaml

# 2. 启动2个agent（Bayesian不需要太多并行）
SWEEP_ID="user/project/xyz789ghi"
sbatch --array=0-1 slurm_wandb_agent.sh ${SWEEP_ID}
```

### 限制Agent运行次数

如果不想让agent一直运行到sweep结束：

```bash
# 编辑 slurm_wandb_agent.sh，修改最后的命令
wandb agent --count 10 ${SWEEP_ID}  # 每个agent只运行10个实验
```

## 🐛 常见问题

### Q1: 作业一直处于PENDING状态

**可能原因:**
- GPU资源不足，等待其他作业完成
- 分区配置错误
- 资源请求超出限制

**解决方案:**
```bash
# 查看作业为什么pending
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# 查看详细原因
scontrol show job <job_id> | grep Reason
```

### Q2: CUDA Out of Memory

**解决方案:**
```bash
# 减小batch size
sbatch slurm_single_experiment.sh TUEV full 3 32 1e-3 5e-2 0.1  # bs=32

# 或减少scale数量
sbatch slurm_single_experiment.sh TUEV full 2 64 1e-3 5e-2 0.1  # 2 scales
```

### Q3: 环境加载失败

**解决方案:**
```bash
# 测试环境加载
srun --pty --gres=gpu:1 bash
# 在交互式会话中测试你的环境加载命令
module load ...
source activate ...
python -c "import torch; print(torch.cuda.is_available())"
```

### Q4: WandB登录问题

**解决方案:**
```bash
# 在登录节点或交互式会话中登录
wandb login

# 或设置API key环境变量
export WANDB_API_KEY=your_api_key

# 在slurm脚本中添加
echo "export WANDB_API_KEY=your_api_key" >> ~/.bashrc
```

### Q5: 作业时间超限被终止

**解决方案:**
```bash
# 增加时间限制
sbatch --time=24:00:00 slurm_single_experiment.sh ...

# 或修改脚本中的默认值
#SBATCH --time=24:00:00
```

## 📊 结果收集

### 自动收集结果

所有实验结果会自动记录到：
1. **WandB**: 在线查看和分析
2. **模型文件**: `checkpoints/` 目录
3. **日志文件**: `logs/` 目录

### 导出WandB结果

```bash
# 安装wandb
pip install wandb

# 导出所有运行结果
wandb export runs \
  --project eeg-msft-improved-ablation \
  --format csv \
  --output results.csv

# 或使用Python API
python << EOF
import wandb
import pandas as pd

api = wandb.Api()
runs = api.runs("your_entity/eeg-msft-improved-ablation")

data = []
for run in runs:
    data.append({
        'name': run.name,
        'state': run.state,
        'dataset': run.config.get('dataset'),
        'variant': run.tags[3] if len(run.tags) > 3 else 'unknown',
        'num_scales': run.config.get('num_scales'),
        'test_acc': run.summary.get('final_test/balanced_acc'),
        'test_kappa': run.summary.get('final_test/kappa'),
    })

df = pd.DataFrame(data)
df.to_csv('wandb_results.csv', index=False)
print(df)
EOF
```

## 🎯 最佳实践

1. **先测试单个实验**: 确保环境配置正确
2. **使用core模式**: 先运行核心配置了解趋势
3. **监控资源使用**: 使用 `sstat` 检查GPU/内存利用率
4. **合理设置时间限制**: 避免作业因超时被终止
5. **定期检查日志**: 及早发现问题
6. **使用WandB**: 实时监控所有实验
7. **记录作业ID**: 便于后续追踪和管理

## 📚 相关文档

- [ABLATION_STUDY_GUIDE.md](ABLATION_STUDY_GUIDE.md) - 完整的ablation study指南
- [README_ABLATION.md](README_ABLATION.md) - 快速参考
- [Slurm官方文档](https://slurm.schedmd.com/)
- [WandB Sweeps文档](https://docs.wandb.ai/guides/sweeps)

---

**创建日期**: 2026-02-26
**版本**: 1.0
**维护者**: Claude + User
