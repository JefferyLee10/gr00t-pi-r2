#!/usr/bin/env bash
set -euo pipefail

cd /path/to/lab-home/pi-r2-flow/learning/Isaac-GR00T

source /path/to/lab-home/pi-r2-flow/learning/Isaac-GR00T/.venv/bin/activate

# ============================================================
# GPUs
# ============================================================
export CUDA_VISIBLE_DEVICES=2,3

# ============================================================
# NCCL
# ============================================================
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# ============================================================
# Offline
# ============================================================
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

export PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1
export GR00T_IMAGE_DELAY_MAX=5

# ============================================================
# Resolve cached HF snapshots
# ============================================================
HFROOT="${HF_HOME:-$HOME/.cache/huggingface}"

resolve_snapshot() {
    local root="$1"

    [ -d "$root/snapshots" ] || return 1

    if [ -f "$root/refs/main" ]; then
        local rev
        rev="$(tr -d '\r\n' < "$root/refs/main")"

        if [ -d "$root/snapshots/$rev" ]; then
            printf '%s\n' "$root/snapshots/$rev"
            return 0
        fi
    fi

    find "$root/snapshots" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        -print \
        | sort \
        | tail -n 1
}

GR_ROOT="$HFROOT/hub/models--nvidia--GR00T-N1.7-3B"
COS_ROOT="$HFROOT/hub/models--nvidia--Cosmos-Reason2-2B"

if ! GR00T_BASE_LOCAL_PATH="$(resolve_snapshot "$GR_ROOT")"; then
    echo "ERROR: local GR00T-N1.7-3B cache not found"
    exit 1
fi

if ! GR00T_COSMOS_LOCAL_PATH="$(resolve_snapshot "$COS_ROOT")"; then
    echo "ERROR: local Cosmos-Reason2-2B cache not found"
    exit 1
fi

export GR00T_BASE_LOCAL_PATH
export GR00T_COSMOS_LOCAL_PATH

if [ ! -f "$GR00T_BASE_LOCAL_PATH/config.json" ]; then
    echo "ERROR: GR00T local snapshot is incomplete:"
    echo "$GR00T_BASE_LOCAL_PATH"
    exit 1
fi

if [ ! -f "$GR00T_COSMOS_LOCAL_PATH/config.json" ]; then
    echo "ERROR: Cosmos local snapshot is incomplete:"
    echo "$GR00T_COSMOS_LOCAL_PATH"
    exit 1
fi

# ============================================================
# Paths
# ============================================================
DATASET=/path/to/lab-home/pi-r2-flow/experiments/bag_transfer_22d_rgb_abs_pir2_exact/train_dataset_view_pir2

CONFIG=/path/to/lab-home/pi-r2-flow/experiments/bag_transfer_22d_rgb_abs_pir2_exact/config/bag_transfer_22d_rgb_abs.py

OUT=/path/to/lab-home/pi-r2-flow/experiments/bag_transfer_22d_rgb_abs_pir2_exact/train_2gpu23_effbs128_50k_pir2_v3_offline_local

LOGDIR=/path/to/lab-home/pi-r2-flow/experiments/bag_transfer_22d_rgb_abs_pir2_exact/logs

LOG=$LOGDIR/train_2gpu23_effbs128_50k_pir2_v3_offline_local.log

mkdir -p "$LOGDIR"

if [ -e "$OUT" ]; then
    echo "ERROR: output directory already exists:"
    echo "$OUT"
    exit 1
fi

echo "============================================================"
echo "πR² GR00T-N1.7 — 50K FULL LOCAL/OFFLINE"
echo "============================================================"
echo "Physical GPUs = 2,3"
echo "GR00T local   = $GR00T_BASE_LOCAL_PATH"
echo "Cosmos local  = $GR00T_COSMOS_LOCAL_PATH"
echo "Dataset       = $DATASET"
echo "Output        = $OUT"
echo
echo "Max steps     = 50000"
echo "Global batch  = 64"
echo "Grad accum    = 2"
echo "Effective BS  = 128"
echo "Peak LR       = 1e-4"
echo "Warmup ratio  = 0.05"
echo "Save steps    = 1000"
echo "Save limit    = 15"
echo "============================================================"

torchrun \
    --nproc_per_node=2 \
    --master_port=29529 \
    gr00t/experiment/launch_finetune_offline_local.py \
    --base-model-path "$GR00T_BASE_LOCAL_PATH" \
    --dataset-path "$DATASET" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "$CONFIG" \
    --num-gpus 2 \
    --output-dir "$OUT" \
    --max-steps 50000 \
    --global-batch-size 64 \
    --gradient-accumulation-steps 2 \
    --dataloader-num-workers 2 \
    --learning-rate 1e-4 \
    --weight-decay 1e-5 \
    --warmup-ratio 0.05 \
    --state-dropout-prob 0.0 \
    --save-steps 1000 \
    --save-total-limit 15 \
    --streaming \
    --streaming-constant-weight 0.2 \
    --streaming-chunk-wise-weight 0.8 \
    --streaming-schedule-mode pir2 \
    --streaming-chunk-size-max 5 \
    --streaming-mask-clean-end \
    --image-delay-max 5 \
    --image-delay-embed-dim 64 \
    2>&1 | tee "$LOG"
