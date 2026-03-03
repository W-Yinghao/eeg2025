# SageStream Refactored

重构后的 SageStream 代码，具有更清晰的代码结构和支持 ablation study。

## 目录结构

```
sagestream_refactored/
├── __init__.py              # 包初始化
├── main.py                  # 主入口文件
├── config.py                # 配置管理（已移至 configs/）
├── data.py                  # 数据加载和预处理
├── tta.py                   # Test-Time Adaptation 模块
├── ablation.py              # Ablation Study 工具
│
├── configs/
│   ├── __init__.py
│   └── config.py            # 配置类定义
│
├── trainers/
│   ├── __init__.py
│   ├── base_trainer.py      # 训练器基类
│   └── sagestream_trainer.py # SageStream 训练器实现
│
└── utils/
    ├── __init__.py
    └── metrics.py           # 指标计算工具
```

## 主要改进

### 1. 配置管理
- 使用 dataclass 进行类型安全的配置管理
- 支持从 JSON 文件加载配置
- 提供 `AblationConfig` 专门用于 ablation study

### 2. 模块化设计
- **数据模块** (`data.py`): 独立的数据加载和预处理
- **训练模块** (`trainers/`): 可扩展的训练器架构
- **TTA 模块** (`tta.py`): 独立的测试时适应模块
- **评估模块** (`utils/metrics.py`): 统一的指标计算

### 3. Ablation Study 支持
- 预定义的 ablation 实验配置
- 灵活的自定义实验支持
- 自动结果聚合和比较

## 使用方法

### 环境设置

使用 anaconda 的 sage 环境：

```bash
# 设置 Python 路径
export PYTHON=/home/infres/yinwang/anaconda3/envs/sage/bin/python

# 或使用便捷脚本
cd ~/eeg2025/NIPS/SageStream
./sagestream_refactored/run.sh --help
```

### 支持的数据集

| 数据集 | 描述 | 类别数 | 通道数 | 序列长度 |
|--------|------|--------|--------|----------|
| APAVA | 医疗时间序列分类 | 2 | 16 | 256 |
| TUAB | TUH EEG 异常检测 | 2 | 16 | 256 |

### 基本训练 (K-Fold Cross-Validation)

```bash
cd ~/eeg2025/NIPS/SageStream

# APAVA 数据集 (默认)
./sagestream_refactored/run.sh --mode kfold --k 5 --epochs 30

# TUAB 数据集
./sagestream_refactored/run.sh --dataset TUAB --mode kfold --epochs 10

# 使用 sage 环境
/home/infres/yinwang/anaconda3/envs/sage/bin/python -m sagestream_refactored.main --mode kfold --k 5 --epochs 30
```

### 运行 Ablation Study

```bash
# 查看可用的 ablation 实验
./sagestream_refactored/run.sh --list_experiments

# 运行特定的 ablation 实验
./sagestream_refactored/run.sh --mode ablation --experiments full,no_tta,no_subject_embedding

# 运行所有标准 ablation 实验
./sagestream_refactored/run.sh --mode ablation
```

### 禁用 TTA

```bash
./sagestream_refactored/run.sh --mode kfold --no_tta
```

### Wandb 日志记录

```bash
# 启用 wandb (默认启用)
./sagestream_refactored/run.sh --mode kfold --wandb_project my_project

# 禁用 wandb
./sagestream_refactored/run.sh --mode kfold --no_wandb

# 指定 wandb entity
./sagestream_refactored/run.sh --mode kfold --wandb_project my_project --wandb_entity my_team
```

Wandb 会自动记录：
- 训练/验证指标 (accuracy, f1, precision, recall 等)
- 学习率变化
- 模型配置
- 测试集最终结果

### 自定义参数

```bash
./sagestream_refactored/run.sh \
    --mode kfold \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4 \
    --seed 42 \
    --output_dir ./my_experiment
```

### 使用配置文件

```bash
# 首先创建配置文件
/home/infres/yinwang/anaconda3/envs/sage/bin/python -c "
from sagestream_refactored.configs import create_default_config
import json
config = create_default_config()
with open('my_config.json', 'w') as f:
    json.dump(config.to_dict(), f, indent=2)
"

# 使用配置文件运行
./sagestream_refactored/run.sh --config my_config.json
```

## Ablation Study 配置

### 预定义实验

| 实验名称 | 描述 |
|---------|------|
| `full` | 完整模型（基线） |
| `no_subject_embedding` | 无 subject embedding |
| `no_style_alignment` | 无 style alignment |
| `no_hypernetwork` | 无 hypernetwork |
| `no_aux_loss` | 无辅助损失 |
| `no_tta` | 无测试时适应 |
| `no_freq_learning` | 无频率感知学习 |
| `minimal` | 最小模型（大多数组件禁用） |
| `subject_only` | 仅 subject embedding |
| `tta_only` | 仅 TTA |

### 代码中使用

```python
from sagestream_refactored.configs import SageStreamConfig, get_ablation_preset
from sagestream_refactored.ablation import AblationStudy

# 创建基础配置
config = SageStreamConfig()

# 获取特定 ablation 配置
ablation_config = get_ablation_preset('no_tta')
config.ablation = ablation_config

# 或使用 AblationStudy 类
study = AblationStudy(config, output_dir='./ablation_results')
exp_config = study.get_experiment_config('no_subject_embedding')
```

## 配置说明

### AblationConfig 参数

```python
@dataclass
class AblationConfig:
    use_moe: bool = True                 # 是否使用 MoE
    use_subject_embedding: bool = True   # 是否使用 subject embedding
    use_style_alignment: bool = True     # 是否使用 style alignment
    use_hypernetwork: bool = True        # 是否使用 hypernetwork
    use_shared_router: bool = True       # 是否共享 router
    use_shared_experts: bool = True      # 是否共享 experts
    use_expert_conditioning: bool = True # 是否使用 expert conditioning
    use_aux_loss: bool = True            # 是否使用辅助损失
    use_tta: bool = True                 # 是否使用 TTA
    use_freq_learning: bool = True       # 是否使用频率学习
```

## 输出结果

训练结果保存在 `output_dir` 下：

```
outputs/
└── sagestream_20250220_120000/
    ├── config.json              # 使用的配置
    ├── results.json             # 训练结果
    ├── best_model_fold_1.pth    # 各折最佳模型
    ├── best_model_fold_2.pth
    └── ...
```

Ablation study 结果：

```
outputs/
└── ablation/
    ├── full_result.json
    ├── no_tta_result.json
    ├── comparison.txt           # 比较表格
    └── all_results.json         # 所有结果
```

## 依赖

- Python >= 3.8
- PyTorch >= 1.10
- transformers
- scikit-learn
- numpy
- torcheeg

## 与原代码的兼容性

重构后的代码复用了原始模型实现 (`MoE_moment/`)，仅重构了：
- 配置管理
- 训练流程
- 数据加载
- TTA 实现

原始模型权重可以直接加载使用。
