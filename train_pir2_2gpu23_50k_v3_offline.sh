#!/usr/bin/env bash
set -euo pipefail

cd /path/to/lab-home/pi-r2-flow/learning/Isaac-GR00T

source /path/to/lab-home/pi-r2-flow/learning/Isaac-GR00T/.venv/bin/activate

# ============================================================
# Physical GPUs 2,3
# ============================================================
export CUDA_VISIBLE_DEVICES=2,3

# ============================================================
# NCCL workaround
# ============================================================
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1

# ============================================================
# Offline mode: use existing local Hugging Face cache only
# ============================================================
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export GR00T_IMAGE_DELAY_MAX=5

DATASET=/path/to/lab-home/pi-r2-flow/experiments/bag_transfer_22d_rgb_abs_pir2_exact/train_dataset_view_pir2

CONFIG=/path/to/lab-home/pi-r2-flow/experiments/bag_transfer_22d_rgb_abs_pir2_exact/config/bag_transfer_22d_rgb_abs.py

OUT=/path/to/lab-home/pi-r2-flow/experiments/bag_transfer_22d_rgb_abs_pir2_exact/train_2gpu23_effbs128_50k_pir2_v3_offline

LOGDIR=/path/to/lab-home/pi-r2-flow/experiments/bag_transfer_22d_rgb_abs_pir2_exact/logs

LOG=$LOGDIR/train_2gpu23_effbs128_50k_pir2_v3_offline.log

mkdir -p "$LOGDIR"

if [ -e "$OUT" ]; then
    echo "ERROR: output directory already exists:"
    echo "$OUT"
    exit 1
fi

echo "============================================================"
echo "πR² GR00T-N1.7 FORMAL TRAINING — 50K"
echo "============================================================"
echo "Physical GPUs = 2,3"
echo "Dataset       = $DATASET"
echo "Config        = $CONFIG"
echo "Output        = $OUT"
echo "Log           = $LOG"
echo
echo "Max steps     = 50000"
echo "Global batch  = 64"
echo "Grad accum    = 2"
echo "Effective BS  = 128"
echo "Peak LR       = 1e-4"
echo "Warmup        = 5% = 2500 steps"
echo "Save steps    = 1000"
echo "Save limit    = 15"
echo "============================================================"

torchrun \
    --nproc_per_node=2 \
    --master_port=29529 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
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
