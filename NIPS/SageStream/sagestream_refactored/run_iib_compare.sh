#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --job-name=iib_compare_apava
#SBATCH --output=logs/iib_compare_%j.out
#SBATCH --error=logs/iib_compare_%j.err

# ============================================================================
# IIB Variant Comparison Script for APAVA 5-Fold Cross-Validation
# ============================================================================
# This script runs 5-fold CV experiments comparing two IIB variants:
#   1. NIPS variant: GRL + Subject Discriminator
#   2. ICML variant: Dual-head + Conditional Independence (CI) loss
#
# Usage:
#   sbatch run_iib_compare.sh
#
# Or run directly (without SLURM):
#   bash run_iib_compare.sh
# ============================================================================

# Set the conda environment
CONDA_ENV="/home/infres/yinwang/anaconda3/envs/sage/bin/python"

# Change to project directory
cd ~/eeg2025/NIPS/SageStream/sagestream_refactored

# Create logs directory if it doesn't exist
mkdir -p logs

# Common parameters
DATASET="PTB"
K_FOLDS=5
EPOCHS=30
BATCH_SIZE=32
LR=5e-5
SEED=2025

# IIB hidden/latent dimensions
IIB_HIDDEN_DIM=256
IIB_LATENT_DIM=256

# ============================================================================
# Experiment 1: NIPS IIB Variant (GRL + Discriminator)
# ============================================================================
# Loss: L_total = L_task + α * L_KL + β * L_adv
# Default: α=0.1 (kl_weight), β=0.1 (adv_weight)

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
    --wandb_project sagestream \
    --wandb_run_name PTB_IIB_NIPS_5fold

NIPS_EXIT_CODE=$?

if [ ${NIPS_EXIT_CODE} -eq 0 ]; then
    echo "NIPS IIB experiment completed successfully!"
else
    echo "NIPS IIB experiment failed with exit code: ${NIPS_EXIT_CODE}"
fi

# ============================================================================
# Experiment 2: ICML IIB Variant (Dual-head + CI Loss)
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
    --wandb_project sagestream \
    --wandb_run_name PTB_IIB_ICML_5fold

ICML_EXIT_CODE=$?

if [ ${ICML_EXIT_CODE} -eq 0 ]; then
    echo "ICML IIB experiment completed successfully!"
else
    echo "ICML IIB experiment failed with exit code: ${ICML_EXIT_CODE}"
fi

# ============================================================================
# Experiment 3: NIPS IIB Only (No SA-MoE)
# ============================================================================
# Same as Experiment 1, but with SA-MoE disabled (plain encoder + IIB only)

echo ""
echo "=========================================="
echo "Running NIPS IIB Only (No SA-MoE) - 5-Fold CV"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "K-Folds: ${K_FOLDS}"
echo "Epochs: ${EPOCHS}"
echo "IIB Variant: nips"
echo "SA-MoE: DISABLED"
echo "=========================================="

${CONDA_ENV} main.py \
    --mode kfold \
    --dataset ${DATASET} \
    --k ${K_FOLDS} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --seed ${SEED} \
    --disable_moe \
    --enable_iib \
    --iib_variant nips \
    --iib_hidden_dim ${IIB_HIDDEN_DIM} \
    --iib_latent_dim ${IIB_LATENT_DIM} \
    --iib_kl_weight 0.1 \
    --iib_adv_weight 0.1 \
    --iib_grl_alpha 1.0 \
    --wandb_project sagestream \
    --wandb_run_name PTB_IIB_NIPS_noMoE_5fold

NIPS_NOMOE_EXIT_CODE=$?

if [ ${NIPS_NOMOE_EXIT_CODE} -eq 0 ]; then
    echo "NIPS IIB (No SA-MoE) experiment completed successfully!"
else
    echo "NIPS IIB (No SA-MoE) experiment failed with exit code: ${NIPS_NOMOE_EXIT_CODE}"
fi

# ============================================================================
# Experiment 4: ICML IIB Only (No SA-MoE)
# ============================================================================
# Same as Experiment 2, but with SA-MoE disabled (plain encoder + IIB only)

echo ""
echo "=========================================="
echo "Running ICML IIB Only (No SA-MoE) - 5-Fold CV"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "K-Folds: ${K_FOLDS}"
echo "Epochs: ${EPOCHS}"
echo "IIB Variant: icml"
echo "SA-MoE: DISABLED"
echo "=========================================="

${CONDA_ENV} main.py \
    --mode kfold \
    --dataset ${DATASET} \
    --k ${K_FOLDS} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --seed ${SEED} \
    --disable_moe \
    --enable_iib \
    --iib_variant icml \
    --iib_hidden_dim ${IIB_HIDDEN_DIM} \
    --iib_latent_dim ${IIB_LATENT_DIM} \
    --icml_lambda_ib 0.1 \
    --icml_beta_ci 10.0 \
    --wandb_project sagestream \
    --wandb_run_name PTB_IIB_ICML_noMoE_5fold

ICML_NOMOE_EXIT_CODE=$?

if [ ${ICML_NOMOE_EXIT_CODE} -eq 0 ]; then
    echo "ICML IIB (No SA-MoE) experiment completed successfully!"
else
    echo "ICML IIB (No SA-MoE) experiment failed with exit code: ${ICML_NOMOE_EXIT_CODE}"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=========================================="
echo "IIB Comparison Experiments Summary"
echo "=========================================="
echo "NIPS IIB (with SA-MoE) Exit Code: ${NIPS_EXIT_CODE}"
echo "ICML IIB (with SA-MoE) Exit Code: ${ICML_EXIT_CODE}"
echo "NIPS IIB (no SA-MoE)   Exit Code: ${NIPS_NOMOE_EXIT_CODE}"
echo "ICML IIB (no SA-MoE)   Exit Code: ${ICML_NOMOE_EXIT_CODE}"
echo "=========================================="

if [ ${NIPS_EXIT_CODE} -eq 0 ] && [ ${ICML_EXIT_CODE} -eq 0 ] && [ ${NIPS_NOMOE_EXIT_CODE} -eq 0 ] && [ ${ICML_NOMOE_EXIT_CODE} -eq 0 ]; then
    echo "All IIB comparison experiments completed successfully!"
    exit 0
else
    echo "Some experiments failed. Check logs for details."
    exit 1
fi
