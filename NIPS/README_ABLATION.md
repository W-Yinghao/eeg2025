# MSFT CBraMod Ablation Study - 快速参考

## 🎯 目标

对MSFT在CBraMod上的改进进行全面的ablation study，涵盖：
- **2个数据集**: TUEV, TUAB
- **4个变体**: baseline, pos_refiner, criss_cross_agg, full
- **3个scale配置**: 2, 3, 4 scales
- **多种超参数组合**

## ⚡ 快速开始

```bash
cd ~/eeg2025/NIPS

# 1. 运行单个测试（推荐第一步）
./run_ablation_study.sh single

# 2. 本地运行核心配置（24个实验）
./run_ablation_study.sh local

# 3. WandB Grid Sweep（576个实验）
./run_ablation_study.sh sweep_grid
./run_ablation_study.sh agent <sweep_id>

# 4. WandB Bayesian Sweep（智能搜索）
./run_ablation_study.sh sweep_bayes
./run_ablation_study.sh agent <sweep_id>
```

## 📂 文件说明

| 文件 | 说明 |
|------|------|
| `run_ablation_study.sh` | 主测试脚本（可执行） |
| `sweep_msft_cbramod.yaml` | Grid搜索配置 |
| `sweep_msft_cbramod_bayesian.yaml` | Bayesian优化配置 |
| `ABLATION_STUDY_GUIDE.md` | 详细使用指南 |
| `finetune_msft_improved.py` | 改进的训练脚本 |
| `msft_modules_improved.py` | 改进的MSFT模块 |

## 🔬 MSFT改进变体

| 变体 | Positional Refiner | Criss-Cross Agg | 说明 |
|------|-------------------|-----------------|------|
| **baseline** | ❌ | ❌ | 无改进 |
| **pos_refiner** | ✅ | ❌ | 仅位置编码refinement |
| **criss_cross_agg** | ❌ | ✅ | 仅Criss-Cross aggregator |
| **full** | ✅ | ✅ | 所有改进 |

## 📊 关键指标

- `final_test/balanced_acc` - 主要优化目标
- `final_test/kappa` - Cohen's Kappa
- `final_test/f1` - F1 Score
- `scale_weight/*` - 各scale的权重分布

## 🔧 自定义配置

### 修改GPU
```bash
# 编辑 run_ablation_study.sh
CUDA_DEVICE=1  # 改为你的GPU编号
```

### 修改超参数
```bash
# 编辑 sweep_msft_cbramod.yaml
parameters:
  lr:
    values: [1e-4, 5e-4, 1e-3]  # 添加或修改
```

### 单独运行某个配置
```bash
python finetune_msft_improved.py \
  --model cbramod \
  --dataset TUEV \
  --cuda 0 \
  --use_pos_refiner \
  --use_criss_cross_agg \
  --num_scales 3 \
  --batch_size 64 \
  --lr 1e-3 \
  --epochs 50
```

## 📈 实验规模估算

| 模式 | 实验数 | 预计时间* | 推荐场景 |
|------|--------|-----------|----------|
| single | 1 | ~2小时 | 快速测试 |
| local | 24 | ~2天 | 核心配置对比 |
| Grid Sweep | 576 | ~2周 | 全面搜索 |
| Bayesian Sweep | 50-100 | ~1周 | 智能优化 |

*基于单个实验约2小时估算，实际取决于硬件和数据集大小

## 🎓 重要改进

### 1. Scale-specific Positional Refinement
适应不同scale的序列长度，refinement frozen ACPE

### 2. Criss-Cross-aware Aggregator
- **Temporal aggregation**: C2F + F2C跨scale信息流
- **Spatial fusion**: 跨scale的通道交互

### 3. Independent Scale Processing
每个scale独立通过transformer，保持Criss-Cross结构

## 🚨 常见问题

**Q: CUDA Out of Memory?**
A: 减小batch_size或num_scales

**Q: WandB不工作?**
A: 运行 `wandb login` 并输入API key

**Q: 数据集路径错误?**
A: 使用 `--datasets_dir` 参数指定路径

**Q: 如何停止sweep?**
A: 在WandB UI中点击"Stop sweep"

## 📖 详细文档

完整文档请参考: [ABLATION_STUDY_GUIDE.md](ABLATION_STUDY_GUIDE.md)

## 🔮 下一步：CodeBrain支持

当前只支持CBraMod。添加CodeBrain支持需要：
1. 实现 `ImprovedMSFTCodeBrainModel`
2. 修改训练脚本支持codebrain选项
3. 更新sweep配置包含codebrain

详见 `ABLATION_STUDY_GUIDE.md` 的相关章节。

---

**需要帮助?** 查看 `ABLATION_STUDY_GUIDE.md` 或运行 `./run_ablation_study.sh help`
