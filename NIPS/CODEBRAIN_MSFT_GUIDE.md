# CodeBrain MSFT 使用指南

## 📋 概述

使用CodeBrain (SSSM)作为backbone进行Multi-Scale Fine-Tuning (MSFT)微调。

### CodeBrain架构特点

- **Backbone**: SSSM (Structured State Space Model)
- **输入格式**: 4D tensor (Batch, Channels, Seq_Len, Patch_Size)
- **层数**: 8层（默认）
- **特征维度**: 200
- **Codebook**: Temporal(512) + Frequency(512)

## 🎯 实验配置

### 测试维度

- **数据集**: TUEV, TUAB (2个)
- **Scale配置**: 1, 2, 3, 4 scales (4种)
- **总实验数**: 2 × 4 = **8个实验**

### Scale配置说明

| Scale数 | 配置 | 下采样因子 | 说明 |
|---------|------|-----------|------|
| 1 | Single | [1x] | Baseline（无多尺度） |
| 2 | Two | [1x, 2x] | 原始+2倍下采样 |
| 3 | Three | [1x, 2x, 4x] | 三个尺度 |
| 4 | Four | [1x, 2x, 4x, 8x] | 四个尺度 |

### 固定参数

```bash
epochs=50
batch_size=64
learning_rate=1e-3
weight_decay=5e-2
dropout=0.1
n_layer=8
codebook_size_t=512
codebook_size_f=512
seed=3407
```

## 🚀 快速开始

### 方式1: 本地运行

```bash
cd ~/eeg2025/NIPS

# 运行所有实验 (8个)
./run_codebrain_msft.sh

# 只运行TUEV数据集 (4个)
./run_codebrain_msft.sh TUEV

# 只运行TUAB数据集 (4个)
./run_codebrain_msft.sh TUAB

# 查看帮助
./run_codebrain_msft.sh --help
```

### 方式2: Slurm集群

```bash
cd ~/eeg2025/NIPS

# 提交所有8个实验
./slurm_codebrain_msft.sh submit

# 提交单个实验 (TUEV + 3 scales)
./slurm_codebrain_msft.sh submit_one TUEV 3

# 查看作业状态
./slurm_codebrain_msft.sh status

# 取消所有作业
./slurm_codebrain_msft.sh cancel
```

## 📁 文件结构

```
eeg2025/NIPS/
├── CodeBrain/
│   └── Checkpoints/
│       └── CodeBrain.pth           # 预训练权重（必需）
│
├── finetune_msft.py                # 训练脚本（已支持codebrain）
├── run_codebrain_msft.sh           # 本地运行脚本
├── slurm_codebrain_msft.sh         # Slurm提交脚本
│
├── logs_codebrain/                 # 日志目录（自动创建）
│   ├── CodeBrain_MSFT_TUEV_s1_*.log
│   ├── CodeBrain_MSFT_TUEV_s2_*.log
│   ├── CodeBrain_MSFT_TUEV_s3_*.log
│   ├── CodeBrain_MSFT_TUEV_s4_*.log
│   └── ... (TUAB日志)
│
└── checkpoints_codebrain/          # 模型检查点（自动创建）
    └── msft_codebrain_*.pth
```

## ⚙️ 预训练权重

### 检查权重文件

```bash
ls -lh CodeBrain/Checkpoints/CodeBrain.pth
```

如果权重文件不存在：
1. **下载**: 从CodeBrain项目获取预训练权重
2. **放置**: 将权重文件放到 `CodeBrain/Checkpoints/CodeBrain.pth`
3. **或使用随机初始化**: 脚本会自动使用随机初始化（性能会降低）

### 自定义权重路径

编辑脚本中的：
```bash
CODEBRAIN_WEIGHTS="/path/to/your/codebrain_weights.pth"
```

## 📊 监控和分析

### 实时监控

```bash
# 本地运行 - 查看当前日志
tail -f logs_codebrain/CodeBrain_MSFT_TUEV_s3_*.log

# Slurm运行 - 查看作业状态
./slurm_codebrain_msft.sh status
squeue -u $USER

# 查看Slurm作业日志
tail -f logs_codebrain/codebrain_s3_TUEV_*.out
```

### WandB监控

项目: `codebrain-msft-ablation`

**关键指标:**
- `final_test/balanced_acc` - 主要性能指标
- `final_test/kappa` - Cohen's Kappa
- `final_test/f1` - F1分数
- `scale_weight/*` - 各scale的混合权重

### 结果分析

```python
import wandb
import pandas as pd

api = wandb.Api()
runs = api.runs("your_entity/codebrain-msft-ablation")

results = []
for run in runs:
    results.append({
        'dataset': run.config.get('dataset'),
        'num_scales': run.config.get('num_scales'),
        'test_acc': run.summary.get('final_test/balanced_acc'),
        'test_kappa': run.summary.get('final_test/kappa'),
        'scale_weights': [
            run.summary.get(f'scale_weight/{i}', 0)
            for i in range(run.config.get('num_scales', 1))
        ]
    })

df = pd.DataFrame(results)

# 分析scale数量对性能的影响
pivot = df.pivot_table(
    values='test_acc',
    index='num_scales',
    columns='dataset',
    aggfunc='mean'
)

print("\nScale数量对性能的影响:")
print(pivot)
```

## 🔍 与CBraMod对比

| 特性 | CodeBrain (SSSM) | CBraMod |
|------|-----------------|---------|
| **架构** | State Space Model | Criss-Cross Transformer |
| **输入格式** | 4D (B, C, S, P) | 3D (B, C, T) |
| **层数** | 8 (default) | 12 (default) |
| **特征维度** | 200 | 200 |
| **多尺度方式** | Seq维度下采样 | Time维度下采样 |
| **特点** | 长序列建模 | 空间-时间分离注意力 |

### 对比实验

同时运行两种backbone的实验：

```bash
# CodeBrain MSFT
./run_codebrain_msft.sh TUEV

# CBraMod MSFT
./run_ablation_fixed_params.sh TUEV
```

然后在WandB中对比：
- 哪个backbone在MSFT下表现更好？
- Scale权重分布有何不同？
- 收敛速度对比

## 📈 预期结果

基于MSFT的理论，我们预期：

### Scale数量对性能的影响

| Scales | 预期性能 | 说明 |
|--------|---------|------|
| 1 | Baseline | 无多尺度，单一分辨率 |
| 2 | +1-2% | 初步引入多尺度 |
| 3 | +2-3% | **最优配置**（经验值） |
| 4 | +2-3% | 可能略有提升或持平 |

**注意**: 4 scales可能不一定优于3 scales，因为：
- 过多尺度可能引入噪声
- 最细尺度权重可能很小
- 计算开销增加

### Scale权重分布

预期观察：
- **中等尺度**（2x, 4x）通常获得较高权重
- **原始尺度**（1x）权重中等
- **最粗尺度**（8x）权重较低（如果使用4 scales）

## 🐛 故障排除

### 问题1: 预训练权重加载失败

```bash
# 检查文件是否存在
ls -lh CodeBrain/Checkpoints/CodeBrain.pth

# 检查权重格式
python << EOF
import torch
weights = torch.load('CodeBrain/Checkpoints/CodeBrain.pth')
print(list(weights.keys())[:5])
EOF
```

**解决方案**:
- 确保权重文件完整下载
- 检查权重是否匹配模型架构
- 或使用 `--no_pretrained` 从头训练

### 问题2: CUDA Out of Memory

**解决方案**:
```bash
# 减小batch size（编辑脚本）
BATCH_SIZE=32  # 从64减到32

# 或减少scale数量
# 只测试1-3 scales，跳过4 scales
```

### 问题3: CodeBrain模块导入失败

```bash
# 检查CodeBrain路径
ls -d CodeBrain/Models/

# 检查SSSM模块
python -c "from Models.SSSM import SSSM; print('OK')"
```

**解决方案**:
- 确保CodeBrain代码完整
- 检查Python路径配置
- 安装缺失依赖

### 问题4: 训练不收敛

**可能原因**:
- 学习率过高/过低
- 预训练权重未加载
- 数据格式不匹配

**解决方案**:
```bash
# 调整学习率（编辑脚本）
LEARNING_RATE=5e-4  # 从1e-3降低

# 确保使用预训练权重
# 检查日志中的"Pretrained weights loaded"
```

## 🔧 高级配置

### 修改CodeBrain架构

编辑脚本中的参数：

```bash
# 更多层（更强大但更慢）
N_LAYER=16

# 更大的codebook
CODEBOOK_SIZE_T=1024
CODEBOOK_SIZE_F=1024

# 更高的dropout（防止过拟合）
DROPOUT=0.2
```

### 自定义scale配置

编辑脚本中的 `SCALE_CONFIGS`:

```bash
# 测试更多scale配置
declare -a SCALE_CONFIGS=(
    "1|single_scale"
    "2|two_scales"
    "3|three_scales"
    "4|four_scales"
    "5|five_scales"     # 新增
)
```

### 添加学习率调度

修改 `finetune_msft.py` 的scheduler部分。

## 📊 完整分析流程

### 1. 运行实验

```bash
# 提交所有实验
./slurm_codebrain_msft.sh submit

# 监控进度
watch -n 60 './slurm_codebrain_msft.sh status'
```

### 2. 收集结果

```python
# 从WandB导出结果
import wandb
import pandas as pd

api = wandb.Api()
runs = api.runs("your_entity/codebrain-msft-ablation")

df = pd.DataFrame([{
    'name': r.name,
    'dataset': r.config['dataset'],
    'num_scales': r.config['num_scales'],
    'test_acc': r.summary.get('final_test/balanced_acc'),
    'test_kappa': r.summary.get('final_test/kappa'),
} for r in runs])

df.to_csv('codebrain_msft_results.csv', index=False)
```

### 3. 分析scale效果

```python
# 绘制scale数量 vs 性能
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

for i, dataset in enumerate(['TUEV', 'TUAB']):
    data = df[df['dataset'] == dataset]

    ax[i].plot(data['num_scales'], data['test_acc'], 'o-')
    ax[i].set_xlabel('Number of Scales')
    ax[i].set_ylabel('Test Balanced Accuracy')
    ax[i].set_title(f'{dataset} - Scale Effect')
    ax[i].grid(True)

plt.tight_layout()
plt.savefig('codebrain_scale_analysis.png', dpi=300)
```

### 4. 对比CodeBrain vs CBraMod

```python
# 加载两个实验的结果
codebrain_results = pd.read_csv('codebrain_msft_results.csv')
cbramod_results = pd.read_csv('cbramod_msft_results.csv')

# 对比最优配置（3 scales）
codebrain_best = codebrain_results[codebrain_results['num_scales'] == 3]
cbramod_best = cbramod_results[
    (cbramod_results['variant'] == 'full') &
    (cbramod_results['num_scales'] == 3)
]

print("CodeBrain MSFT (3 scales):")
print(codebrain_best[['dataset', 'test_acc', 'test_kappa']].to_string())

print("\nCBraMod MSFT (full, 3 scales):")
print(cbramod_best[['dataset', 'test_acc', 'test_kappa']].to_string())
```

## ✅ 完成清单

实验完成后，确保：

- [ ] 所有8个实验成功运行
- [ ] WandB记录完整（包括scale权重）
- [ ] 模型检查点已保存
- [ ] 日志文件已备份
- [ ] 结果已导出并分析
- [ ] 与baseline（single scale）对比
- [ ] （可选）与CBraMod MSFT对比

## 📚 参考

- **CodeBrain论文**: [添加链接]
- **MSFT方法**: Multi-Scale Fine-Tuning for EEG Foundation Models
- **WandB文档**: https://docs.wandb.ai/

## 🤝 支持

如有问题：
1. 查看日志文件: `logs_codebrain/*.log`
2. 检查WandB run页面
3. 参考 [SLURM_USAGE.md](SLURM_USAGE.md)

---

**创建时间**: 2026-02-26
**版本**: 1.0
**状态**: Ready to use ✅
