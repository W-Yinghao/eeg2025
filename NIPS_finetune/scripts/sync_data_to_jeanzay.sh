#!/bin/bash
################################################################################
# Sync datasets and pretrained weights to Jean Zay
#
# Usage:
#   ./sync_data_to_jeanzay.sh              # sync all (datasets + weights)
#   ./sync_data_to_jeanzay.sh datasets     # sync datasets only
#   ./sync_data_to_jeanzay.sh weights      # sync weights only
#   ./sync_data_to_jeanzay.sh TUEV TUSZ    # sync specific datasets only
#
# Add --dry-run as the LAST argument to preview without transferring.
################################################################################

set -u

# ============================================================================
# Remote config
# ============================================================================
REMOTE_HOST="jeanzay"
REMOTE_BASE="/lustre/fswork/projects/rech/ifd/uti56bh/yinghao/data"

# ============================================================================
# Local paths
# ============================================================================
LOCAL_DATA_ROOT="/projects/EEG-foundation-model"

# Dataset name -> local directory mapping
declare -A DATASETS=(
    [TUEV]="${LOCAL_DATA_ROOT}/diagnosis_data/tuev_preprocessed"
    [TUAB]="${LOCAL_DATA_ROOT}/diagnosis_data/tuab_preprocessed"
    [CHB-MIT]="${LOCAL_DATA_ROOT}/diagnosis_data/CHB-MIT_preprocessed"
    [TUSZ]="${LOCAL_DATA_ROOT}/diagnosis_data/tusz_preprocessed"
    [DIAGNOSIS]="${LOCAL_DATA_ROOT}/diagnosis_data_lmdb——5s"
    [DEPRESSION]="${LOCAL_DATA_ROOT}/diagnosis_data/depression_normal_preprocessed_CBramod"
    [CVD]="${LOCAL_DATA_ROOT}/diagnosis_data/cvd_normal_preprocessed_CBramod"
    [CVD_DEPRESSION_NORMAL]="${LOCAL_DATA_ROOT}/diagnosis_data/cvd_normal_depression_preprocessed"
    [UNIFIED_DIAGNOSIS]="${LOCAL_DATA_ROOT}/diagnosis_data/unified_diagnosis_preprocessed"
    [AD_DIAGNOSIS]="${LOCAL_DATA_ROOT}/diagnosis_data/ad_diagnosis_preprocessed"
)

# Pretrained weights (local_path -> remote_relative_path)
declare -A WEIGHTS=(
    ["/home/infres/yinwang/eeg2025/NIPS/Cbramod_pretrained_weights.pth"]="weights/Cbramod_pretrained_weights.pth"
    ["/home/infres/yinwang/eeg2025/NIPS/CodeBrain/Checkpoints/CodeBrain.pth"]="weights/CodeBrain/Checkpoints/CodeBrain.pth"
    ["/home/infres/yinwang/eeg2025/NIPS_finetune/BioFoundation/checkpoints/LUNA/LUNA_base.safetensors"]="weights/LUNA/LUNA_base.safetensors"
    ["/home/infres/yinwang/eeg2025/NIPS_finetune/BioFoundation/checkpoints/LUNA/LUNA_large.safetensors"]="weights/LUNA/LUNA_large.safetensors"
    ["/home/infres/yinwang/eeg2025/NIPS_finetune/BioFoundation/checkpoints/LUNA/LUNA_huge.safetensors"]="weights/LUNA/LUNA_huge.safetensors"
)

# rsync common options
RSYNC_OPTS="-avhP --compress"
MAX_RETRIES=3
RETRY_DELAY=10

# ============================================================================
# Functions
# ============================================================================

sync_dataset() {
    local name=$1
    local src="${DATASETS[$name]}"
    local dst="${REMOTE_HOST}:${REMOTE_BASE}/datasets/${name}/"

    if [ ! -d "${src}" ]; then
        echo "[SKIP] ${name}: local path not found: ${src}"
        return 1
    fi

    echo ""
    echo "======================================================================"
    echo "Syncing dataset: ${name}"
    echo "  ${src}/ -> ${dst}"
    echo "======================================================================"
    local attempt=0
    while [ $attempt -lt $MAX_RETRIES ]; do
        attempt=$((attempt + 1))
        if rsync ${RSYNC_OPTS} ${DRY_RUN} "${src}/" "${dst}"; then
            return 0
        fi
        echo "[RETRY] Attempt ${attempt}/${MAX_RETRIES} failed, waiting ${RETRY_DELAY}s..."
        sleep $RETRY_DELAY
    done
    echo "[FAILED] ${name} after ${MAX_RETRIES} attempts"
    return 1
}

sync_weights() {
    echo ""
    echo "======================================================================"
    echo "Syncing pretrained weights"
    echo "======================================================================"

    for local_path in "${!WEIGHTS[@]}"; do
        local remote_rel="${WEIGHTS[$local_path]}"
        local remote_dir="${REMOTE_HOST}:${REMOTE_BASE}/${remote_rel%/*}/"

        if [ ! -f "${local_path}" ]; then
            echo "[SKIP] Weight not found: ${local_path}"
            continue
        fi

        echo "  ${local_path} -> ${REMOTE_BASE}/${remote_rel}"
        # Ensure remote directory exists, then sync the file
        ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_BASE}/${remote_rel%/*}"
        local attempt=0
        while [ $attempt -lt $MAX_RETRIES ]; do
            attempt=$((attempt + 1))
            if rsync ${RSYNC_OPTS} ${DRY_RUN} "${local_path}" "${REMOTE_HOST}:${REMOTE_BASE}/${remote_rel}"; then
                break
            fi
            echo "  [RETRY] Attempt ${attempt}/${MAX_RETRIES}, waiting ${RETRY_DELAY}s..."
            sleep $RETRY_DELAY
        done
    done
}

show_usage() {
    echo "Usage: $0 [TARGET...] [--dry-run]"
    echo ""
    echo "  TARGET: all (default) | datasets | weights | TUEV | TUAB | TUSZ | ..."
    echo ""
    echo "Available datasets: ${!DATASETS[*]}"
    echo ""
    echo "Remote: ${REMOTE_HOST}:${REMOTE_BASE}/"
    echo ""
    echo "Remote structure:"
    echo "  ${REMOTE_BASE}/"
    echo "    datasets/"
    echo "      TUEV/           <- tuev_preprocessed"
    echo "      TUAB/           <- tuab_preprocessed"
    echo "      TUSZ/           <- tusz_preprocessed"
    echo "      DIAGNOSIS/      <- diagnosis_data_lmdb——5s"
    echo "      ..."
    echo "    weights/"
    echo "      Cbramod_pretrained_weights.pth"
    echo "      CodeBrain/Checkpoints/CodeBrain.pth"
    echo "      LUNA/LUNA_base.safetensors"
    echo "      LUNA/LUNA_large.safetensors"
    echo "      LUNA/LUNA_huge.safetensors"
}

# ============================================================================
# Main
# ============================================================================

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    show_usage
    exit 0
fi

# Check for --dry-run (scan all arguments)
DRY_RUN=""
args=()
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        DRY_RUN="--dry-run"
        echo "[DRY RUN] No files will be transferred."
    else
        args+=("$arg")
    fi
done

# Default to "all" if no arguments
if [ ${#args[@]} -eq 0 ]; then
    args=("all")
fi

# Ensure remote base directory exists
echo "Ensuring remote directory: ${REMOTE_BASE}"
ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_BASE}/{datasets,weights}"

success=0
fail=0

for target in "${args[@]}"; do
    case "${target}" in
        all)
            for ds in "${!DATASETS[@]}"; do
                if sync_dataset "${ds}"; then
                    ((success++))
                else
                    ((fail++))
                fi
            done
            sync_weights
            ;;
        datasets)
            for ds in "${!DATASETS[@]}"; do
                if sync_dataset "${ds}"; then
                    ((success++))
                else
                    ((fail++))
                fi
            done
            ;;
        weights)
            sync_weights
            ;;
        *)
            # Treat as dataset name
            if [ -n "${DATASETS[$target]+x}" ]; then
                if sync_dataset "${target}"; then
                    ((success++))
                else
                    ((fail++))
                fi
            else
                echo "[ERROR] Unknown target: ${target}"
                echo "Available: all | datasets | weights | ${!DATASETS[*]}"
                exit 1
            fi
            ;;
    esac
done

echo ""
echo "======================================================================"
echo "Sync complete. ${success} succeeded, ${fail} skipped."
echo "Remote: ${REMOTE_HOST}:${REMOTE_BASE}/"
echo "======================================================================"
