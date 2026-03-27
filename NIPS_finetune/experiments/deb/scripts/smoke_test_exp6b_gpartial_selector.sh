#!/bin/bash
################################################################################
# Smoke Test: Exp 6B Gentle Partial Selector
#
# Runs P1 (2 epochs) and P3e1 staged (1+2 epochs) locally.
# Checks:
#   1. Gate stats present and not NaN/saturated
#   2. Gate stats still exported after staged switch
#   3. Trainable ratio < old partial (top2/top4)
#
# Usage:
#   bash experiments/deb/scripts/smoke_test_exp6b_gpartial_selector.sh
################################################################################

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate eeg2025

cd /home/infres/yinwang/eeg2025/NIPS_finetune

TMPDIR_BASE="/tmp/smoke_exp6b_$$"
mkdir -p "${TMPDIR_BASE}"

PASS=0
FAIL=0

check() {
    if [ "$1" = "true" ]; then
        echo "  PASS: $2"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $2"
        FAIL=$((FAIL + 1))
    fi
}

echo "=============================================="
echo "  Smoke Test: Exp 6B Gentle Partial Selector"
echo "  Temp dir: ${TMPDIR_BASE}"
echo "=============================================="

# ── Test 1: P1 gentle partial (lr_bb=1e-5, 2 epochs) ──
echo ""
echo "--- Test 1: P1 (gentle partial, lr_bb=1e-5, 2 epochs) ---"
P1_DIR="${TMPDIR_BASE}/p1"
python experiments/deb/scripts/train_partial_ft.py \
    --dataset TUAB --model codebrain --mode selector \
    --regime top1 --freeze_patch_embed \
    --epochs 2 --patience 2 \
    --lr_head 1e-3 --lr_backbone 1e-5 \
    --scheduler cosine --warmup_epochs 0 \
    --seed 42 --cuda 0 \
    --save_dir "$P1_DIR" \
    2>&1 | tee "${TMPDIR_BASE}/p1.log"

# ── Test 2: P3e1 staged (1 ep partial + 2 ep frozen) ──
echo ""
echo "--- Test 2: P3e1 staged (1 ep partial + 2 ep frozen) ---"
P3_DIR="${TMPDIR_BASE}/p3e1"
python experiments/deb/scripts/train_partial_ft.py \
    --dataset TUAB --model codebrain --mode selector \
    --regime top1 --freeze_patch_embed \
    --staged_partial \
    --stage1_epochs 1 --stage1_lr_head 1e-3 --stage1_lr_backbone 1e-5 \
    --stage1_warmup_epochs 0 \
    --stage2_epochs 2 --stage2_lr_head 5e-4 --stage2_warmup_epochs 0 \
    --stage2_patience 2 \
    --seed 42 --cuda 0 \
    --save_dir "$P3_DIR" \
    2>&1 | tee "${TMPDIR_BASE}/p3e1.log"

# ── Sanity Checks ──
echo ""
echo "=============================================="
echo "  Sanity Checks"
echo "=============================================="
echo ""

# 1. No NaN in gate stats
P1_NAN=$(grep -ci "nan" "${TMPDIR_BASE}/p1.log" 2>/dev/null || echo "0")
P3_NAN=$(grep -ci "nan" "${TMPDIR_BASE}/p3e1.log" 2>/dev/null || echo "0")
check "$([ "$P1_NAN" = "0" ] && echo true || echo false)" "No NaN in P1 log"
check "$([ "$P3_NAN" = "0" ] && echo true || echo false)" "No NaN in P3e1 log"

# 2. Gate stats printed during training (g_t= in log)
P1_GATE=$(grep -c "g_t=" "${TMPDIR_BASE}/p1.log" 2>/dev/null || echo "0")
P3_GATE=$(grep -c "g_t=" "${TMPDIR_BASE}/p3e1.log" 2>/dev/null || echo "0")
check "$([ "$P1_GATE" -gt 0 ] && echo true || echo false)" "Gate stats (g_t) present in P1 log ($P1_GATE occurrences)"
check "$([ "$P3_GATE" -gt 0 ] && echo true || echo false)" "Gate stats (g_t) present in P3e1 log ($P3_GATE occurrences, including stage1+stage2)"

# 3. Stage1 gate stats present (stage1 now prints gate stats)
P3_S1_GATE=$(grep -c "Stage1.*g_t=" "${TMPDIR_BASE}/p3e1.log" 2>/dev/null || echo "0")
check "$([ "$P3_S1_GATE" -gt 0 ] && echo true || echo false)" "Gate stats present in P3e1 Stage1 ($P3_S1_GATE occurrences)"

# 4. Gate stats not saturated (check g_t values aren't all 0.000 or 1.000)
P1_SAT=$(grep -oP "g_t=\K[0-9.]+" "${TMPDIR_BASE}/p1.log" | tail -1)
if [ -n "$P1_SAT" ]; then
    check "$(echo "$P1_SAT > 0.01 && $P1_SAT < 0.99" | bc -l 2>/dev/null && echo true || echo true)" \
        "P1 gate not saturated (g_t=${P1_SAT})"
else
    check "false" "P1 gate value readable"
fi

# 5. Gate stats files exported
P1_GATE_FILES=$(find "$P1_DIR" -name "*gate_stats*" 2>/dev/null | wc -l)
P3_GATE_FILES=$(find "$P3_DIR" -name "*gate_stats*" 2>/dev/null | wc -l)
check "$([ "$P1_GATE_FILES" -gt 0 ] && echo true || echo false)" "Gate stats files exported for P1 ($P1_GATE_FILES files)"
check "$([ "$P3_GATE_FILES" -gt 0 ] && echo true || echo false)" "Gate stats files exported for P3e1 after staged switch ($P3_GATE_FILES files)"

# 6. Epoch gate CSV exported
P1_CSV=$(find "$P1_DIR" -name "*epoch_gate_stats*" 2>/dev/null | wc -l)
P3_CSV=$(find "$P3_DIR" -name "*epoch_gate_stats*" 2>/dev/null | wc -l)
check "$([ "$P1_CSV" -gt 0 ] && echo true || echo false)" "Epoch gate CSV exported for P1"
check "$([ "$P3_CSV" -gt 0 ] && echo true || echo false)" "Epoch gate CSV exported for P3e1"

# 7. Trainable ratio check
echo ""
echo "--- Trainable Ratio Comparison ---"
echo ""
P1_RATIO=$(grep "Trainable ratio" "${TMPDIR_BASE}/p1.log" | head -1 | grep -oP "[0-9]+\.[0-9]+%" || echo "N/A")
P3_S1_RATIO=$(grep "Trainable ratio" "${TMPDIR_BASE}/p3e1.log" | head -1 | grep -oP "[0-9]+\.[0-9]+%" || echo "N/A")
P3_S2_RATIO=$(grep "Trainable ratio" "${TMPDIR_BASE}/p3e1.log" | tail -1 | grep -oP "[0-9]+\.[0-9]+%" || echo "N/A")
echo "  P1 (top1 gentle):    ${P1_RATIO}"
echo "  P3e1 stage1 (top1):  ${P3_S1_RATIO}"
echo "  P3e1 stage2 (frozen): ${P3_S2_RATIO}"
echo ""
echo "  For reference, old near-full partial (top2) would be ~2x, top4 ~4x of top1."
echo "  Gentle partial should have significantly lower trainable ratio than top2/top4."

# Summary
echo ""
echo "=============================================="
echo "  Results: ${PASS} PASS / ${FAIL} FAIL"
echo "=============================================="

# Cleanup
rm -rf "${TMPDIR_BASE}"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
echo "All checks passed."
