# MSFT CBraMod Ablation Study 使用指南

本指南说明如何使用创建的测试框架进行全面的ablation study。

## 📁 文件结构

```
eeg2025/NIPS/
├── finetune_msft_improved.py          # 改进的MSFT训练脚本
├── msft_modules_improved.py            # 改进的MSFT模块
├── run_ablation_study.sh               # 主测试脚本（本脚本）
├── sweep_msft_cbramod.yaml             # WandB Grid Sweep配置
├── sweep_msft_cbramod_bayesian.yaml    # WandB Bayesian Sweep配置
├── logs/                               # 实验日志目录（自动创建）
└── checkpoints/                        # 模型检查点目录（自动创建）
```

## 🚀 快速开始

### 1. 运行单个测试实验

```bash
cd ~/eeg2025/NIPS
./run_ablation_study.sh single
```

这会运行一个快速测试：TUEV数据集 + full variant + 3 scales

### 2. 本地顺序运行核心配置

```bash
./run_ablation_study.sh local
```

这会运行约24个实验（2数据集 × 4变体 × 3scale配置），使用默认超参数。

### 3. 使用WandB Sweep进行Grid搜索

```bash
# Step 1: 创建sweep
./run_ablation_study.sh sweep_grid

# Step 2: 启动agent（使用返回的sweep ID）
./run_ablation_study.sh agent <your_sweep_id>

# 或者直接使用wandb命令
wandb agent <your_sweep_id>
```

### 4. 使用WandB Sweep进行Bayesian优化

```bash
# Step 1: 创建sweep
./run_ablation_study.sh sweep_bayes

# Step 2: 启动agent
./run_ablation_study.sh agent <your_sweep_id>
```

## 🔬 Ablation Study设计

### 测试维度

#### 1. **数据集** (2个)
- TUEV: 癫痫分类
- TUAB: 异常检测

#### 2. **MSFT改进变体** (4个)
| 变体名称 | use_pos_refiner | use_criss_cross_agg | 说明 |
|---------|----------------|---------------------|------|
| baseline | ❌ | ❌ | 无改进（类似原始MSFT） |
| pos_refiner | ✅ | ❌ | 仅位置编码refinement |
| criss_cross_agg | ❌ | ✅ | 仅Criss-Cross感知的aggregator |
| full | ✅ | ✅ | 所有改进 |

#### 3. **Scale数量** (3个)
- 2 scales: [1x, 2x]
- 3 scales: [1x, 2x, 4x]
- 4 scales: [1x, 2x, 4x, 8x]

#### 4. **超参数**（Grid Sweep中）
- Batch size: [32, 64]
- Learning rate: [5e-4, 1e-3, 2e-3]
- Weight decay: [5e-2, 1e-1]
- Dropout: [0.1, 0.2]

### 总实验数量估算

- **核心配置** (local模式): 2 × 4 × 3 = **24个实验**
- **Grid Sweep**: 2 × 4 × 3 × 2 × 3 × 2 × 2 = **576个实验**
- **Bayesian Sweep**: 用户定义（建议50-100次）

## 📊 结果记录与分析

### WandB Metrics

所有实验都会记录以下指标：

**训练指标：**
- `train/loss`: 训练损失
- `train/acc`: 训练准确率
- `learning_rate`: 当前学习率

**验证指标：**
- `val/balanced_acc`: 验证集平衡准确率
- `val/kappa`: Cohen's Kappa
- `val/f1`: F1分数
- `val/roc_auc`: ROC-AUC（binary任务）
- `val/pr_auc`: PR-AUC（binary任务）

**测试指标：**
- `test/balanced_acc`: 测试集平衡准确率
- `test/kappa`: Cohen's Kappa
- `test/f1`: F1分数
- `test/roc_auc`: ROC-AUC（binary任务）
- `test/pr_auc`: PR-AUC（binary任务）

**最终指标：**
- `final_test/balanced_acc`: 最佳模型的测试准确率
- `final_test/kappa`: 最佳模型的Kappa
- `final_test/f1`: 最佳模型的F1
- `best_epoch`: 最佳epoch

**MSFT特定指标：**
- `scale_weight/0`, `scale_weight/1`, ...: 每个scale的混合权重

### 运行命名规则

```
MSFT_{variant}_{model}_{dataset}_s{num_scales}_bs{batch_size}_lr{lr}_wd{wd}_do{dropout}_{timestamp}
```

示例：
```
MSFT_full_cbramod_TUEV_s3_bs64_lr0.001_wd0.05_do0.1_20260226_143022
```

### 日志文件

每个实验的详细日志保存在 `logs/` 目录：
```bash
logs/
└── MSFT_full_cbramod_TUEV_s3_bs64_lr0.001_wd0.05_do0.1_20260226_143022.log
```

### 模型检查点

最佳模型保存在 `checkpoints/` 目录：
```bash
checkpoints/
└── msft_improved_cbramod_full_tuev_scales3_epoch15_bal_acc_0.85432_kappa_0.71234_f1_0.84321.pth
```

## ⚙️ 配置修改

### 修改GPU设置

编辑 `run_ablation_study.sh`:
```bash
# 单GPU
CUDA_DEVICE=0

# 并行运行（需要多GPU）
NUM_GPUS=4
```

### 修改超参数范围

编辑对应的YAML配置文件：

**Grid搜索** (`sweep_msft_cbramod.yaml`):
```yaml
parameters:
  lr:
    values: [5e-4, 1e-3, 2e-3, 5e-3]  # 添加更多值
```

**Bayesian搜索** (`sweep_msft_cbramod_bayesian.yaml`):
```yaml
parameters:
  lr:
    distribution: log_uniform_values
    min: 1e-4
    max: 1e-2  # 调整范围
```

### 修改训练epoch数

编辑 `run_ablation_study.sh`:
```bash
EPOCHS=100  # 从50改为100
```

或在YAML中：
```yaml
parameters:
  epochs:
    value: 100
```

## 🔍 分析建议

### 1. 对比不同变体的效果

在WandB中创建对比图：
- X轴: 变体 (baseline, pos_refiner, criss_cross_agg, full)
- Y轴: final_test/balanced_acc
- 分组: dataset, num_scales

### 2. 分析scale权重分布

观察 `scale_weight/*` 指标：
- 哪个scale获得最高权重？
- 不同变体的权重分布是否不同？
- 权重是否随训练变化？

### 3. 超参数敏感性分析

使用Bayesian sweep结果：
- 哪些超参数对性能影响最大？
- 是否存在超参数之间的交互作用？
- 最优超参数配置是什么？

### 4. 数据集特异性

对比TUEV vs TUAB：
- 哪个变体在哪个数据集上表现更好？
- 是否需要针对数据集调整配置？

## 📈 WandB使用技巧

### 创建自定义图表

```python
# 在WandB UI中：
# 1. 进入workspace
# 2. 点击 "Add visualization"
# 3. 选择 "Custom Chart"
# 4. 配置X/Y轴和分组
```

### 导出结果

```bash
# 使用WandB API导出结果
wandb export runs \
  --project eeg-msft-improved-ablation \
  --format csv \
  --output results.csv
```

### 对比运行

```python
import wandb

api = wandb.Api()
runs = api.runs("your_entity/eeg-msft-improved-ablation")

# 筛选特定变体
full_runs = [r for r in runs if r.config.get('use_pos_refiner') and r.config.get('use_criss_cross_agg')]
baseline_runs = [r for r in runs if not r.config.get('use_pos_refiner') and not r.config.get('use_criss_cross_agg')]

# 对比结果
import pandas as pd
df = pd.DataFrame([{
    'name': r.name,
    'variant': r.tags[3] if len(r.tags) > 3 else 'unknown',
    'dataset': r.config['dataset'],
    'test_acc': r.summary.get('final_test/balanced_acc', 0)
} for r in runs])

print(df.groupby(['variant', 'dataset'])['test_acc'].agg(['mean', 'std', 'max']))
```

## 🐛 故障排除

### 问题1: CUDA Out of Memory

**解决方案:**
- 减小batch_size
- 减少num_scales
- 使用更少的并行实验

### 问题2: WandB登录问题

```bash
wandb login
# 输入你的API key
```

### 问题3: 数据集路径错误

检查 `DATASET_CONFIGS` 中的 `data_dir` 是否正确：
```python
# 在finetune_msft_improved.py中
DATASET_CONFIGS = {
    'TUEV': {
        'data_dir': '/path/to/your/TUEV',  # 修改此处
        ...
    }
}
```

或使用命令行参数：
```bash
python finetune_msft_improved.py --dataset TUEV --datasets_dir /path/to/your/datasets
```

## 📝 下一步：添加CodeBrain支持

当前框架只支持CBraMod。如需添加CodeBrain支持：

### 1. 在 `msft_modules_improved.py` 中添加

```python
class ImprovedMSFTCodeBrainModel(nn.Module):
    """MSFT with CodeBrain backbone."""

    def __init__(self, backbone, num_classes, num_scales=3, ...):
        # 类似ImprovedMSFTCBraModModel的实现
        # 但需要适配CodeBrain的架构
        pass

def create_improved_msft_codebrain_model(...):
    # 创建CodeBrain + MSFT模型
    pass
```

### 2. 修改 `finetune_msft_improved.py`

```python
# 添加导入
from msft_modules_improved import create_improved_msft_codebrain_model

# 修改model choices
parser.add_argument('--model', type=str, default='cbramod',
                    choices=['cbramod', 'codebrain'],
                    help='backbone model')

# 在main()中添加分支
if params.model == 'codebrain':
    model = create_improved_msft_codebrain_model(...)
else:
    model = create_improved_msft_cbramod_model(...)
```

### 3. 更新sweep配置和bash脚本

```yaml
# sweep_msft_comprehensive.yaml
parameters:
  model:
    values: [cbramod, codebrain]
```

```bash
# run_ablation_study.sh
MODELS=("cbramod" "codebrain")
```

## 📚 参考文献

1. **CBraMod论文**: 参考 `cbramod.pdf`
2. **MSFT方法**: Multi-Scale Fine-Tuning
3. **WandB文档**: https://docs.wandb.ai/guides/sweeps

## 🤝 贡献

如有问题或改进建议，请联系项目维护者。

---

**创建日期**: 2026-02-26
**版本**: 1.0
**作者**: Claude + User
