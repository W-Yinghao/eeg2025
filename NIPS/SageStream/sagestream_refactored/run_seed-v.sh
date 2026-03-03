#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=H100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --job-name=iib_compare_seedv
#SBATCH --output=logs/iib_compare_seedv_%j.out
#SBATCH --error=logs/iib_compare_seedv_%j.err

# ============================================================================
# IIB Variant Comparison Script for SEED-V 5-Fold Cross-Validation
# ============================================================================
# This script runs 5-fold CV experiments comparing three configurations:
#   1. No IIB: Baseline without Information Bottleneck
#   2. NIPS variant: GRL + Subject Discriminator
#   3. ICML variant: Dual-head + Conditional Independence (CI) loss
#
# Usage:
#   sbatch run_seed-v.sh
#
# Or run directly (without SLURM):
#   bash run_seed-v.sh
# ============================================================================

# Set the conda environment
CONDA_ENV="/home/infres/yinwang/anaconda3/envs/sage/bin/python"

# Change to project directory
cd ~/eeg2025/NIPS/SageStream/sagestream_refactored

# Create logs directory if it doesn't exist
mkdir -p logs

# Common parameters
DATASET="SEEDV"
K_FOLDS=5
EPOCHS=30
BATCH_SIZE=32
LR=5e-5
SEED=2025

# IIB hidden/latent dimensions
IIB_HIDDEN_DIM=256
IIB_LATENT_DIM=256

# ============================================================================
# Experiment 1: No IIB (Baseline)
# ============================================================================

echo "=========================================="
echo "Running No IIB Baseline - 5-Fold CV"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "K-Folds: ${K_FOLDS}"
echo "Epochs: ${EPOCHS}"
echo "IIB: Disabled"
echo "=========================================="

${CONDA_ENV} main.py \
    --mode kfold \
    --dataset ${DATASET} \
    --k ${K_FOLDS} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --seed ${SEED} \
    --disable_iib \
    --wandb_project sagestream_seed \
    --wandb_run_name SEEDV_NoIIB_5fold

NO_IIB_EXIT_CODE=$?

if [ ${NO_IIB_EXIT_CODE} -eq 0 ]; then
    echo "No IIB Baseline experiment completed successfully!"
else
    echo "No IIB Baseline experiment failed with exit code: ${NO_IIB_EXIT_CODE}"
fi

# ============================================================================
# Experiment 2: NIPS IIB Variant (GRL + Discriminator)
# ============================================================================
# Loss: L_total = L_task + α * L_KL + β * L_adv
# Default: α=0.1 (kl_weight), β=0.1 (adv_weight)

echo ""
echo "=========================================="
echo "Running NIPS IIB Variant - 5-Fold CV"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "K-Folds: ${K_FOLDS}"
echo "Epochs: ${EPOCHS}"
echo "IIB Variant: nips"
echo "KL Weight (α): 0.1"
echo "Adv Weight (β): 0.1"
echo "=========================================="

${CONDA_ENV} main.py \
    --mode kfold \
    --dataset ${DATASET} \
    --k ${K_FOLDS} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --seed ${SEED} \
    --enable_iib \
    --iib_variant nips \
    --iib_hidden_dim ${IIB_HIDDEN_DIM} \
    --iib_latent_dim ${IIB_LATENT_DIM} \
    --iib_kl_weight 0.1 \
    --iib_adv_weight 0.1 \
    --iib_grl_alpha 1.0 \
    --wandb_project sagestream_seed \
    --wandb_run_name SEEDV_IIB_NIPS_5fold

NIPS_EXIT_CODE=$?

if [ ${NIPS_EXIT_CODE} -eq 0 ]; then
    echo "NIPS IIB experiment completed successfully!"
else
    echo "NIPS IIB experiment failed with exit code: ${NIPS_EXIT_CODE}"
fi

# ============================================================================
# Experiment 3: ICML IIB Variant (Dual-head + CI Loss)
# ============================================================================
# Loss: L_total = L_inv + L_env + λ * L_IB + β * L_CI
# Default: λ=0.1 (lambda_ib), β=10.0 (beta_ci)

echo ""
echo "=========================================="
echo "Running ICML IIB Variant - 5-Fold CV"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "K-Folds: ${K_FOLDS}"
echo "Epochs: ${EPOCHS}"
echo "IIB Variant: icml"
echo "Lambda IB (λ): 0.1"
echo "Beta CI (β): 10.0"
echo "=========================================="

${CONDA_ENV} main.py \
    --mode kfold \
    --dataset ${DATASET} \
    --k ${K_FOLDS} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --seed ${SEED} \
    --enable_iib \
    --iib_variant icml \
    --iib_hidden_dim ${IIB_HIDDEN_DIM} \
    --iib_latent_dim ${IIB_LATENT_DIM} \
    --icml_lambda_ib 0.1 \
    --icml_beta_ci 10.0 \
    --wandb_project sagestream_seed \
    --wandb_run_name SEEDV_IIB_ICML_5fold

ICML_EXIT_CODE=$?

if [ ${ICML_EXIT_CODE} -eq 0 ]; then
    echo "ICML IIB experiment completed successfully!"
else
    echo "ICML IIB experiment failed with exit code: ${ICML_EXIT_CODE}"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=========================================="
echo "IIB Comparison Experiments Summary (SEED-V)"
echo "=========================================="
echo "No IIB Exit Code:   ${NO_IIB_EXIT_CODE}"
echo "NIPS IIB Exit Code: ${NIPS_EXIT_CODE}"
echo "ICML IIB Exit Code: ${ICML_EXIT_CODE}"
echo "=========================================="

if [ ${NO_IIB_EXIT_CODE} -eq 0 ] && [ ${NIPS_EXIT_CODE} -eq 0 ] && [ ${ICML_EXIT_CODE} -eq 0 ]; then
    echo "All IIB comparison experiments completed successfully!"
    exit 0
else
    echo "Some experiments failed. Check logs for details."
    exit 1
fi
