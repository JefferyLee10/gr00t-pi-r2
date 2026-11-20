#!/usr/bin/env bash

set -uo pipefail

cd /path/to/lab-home/pi-r2-flow/learning/Isaac-GR00T
source .venv/bin/activate

# ============================================================
# Required NCCL workaround on this server
# ============================================================
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Physical GPU 2,3 only
export CUDA_VISIBLE_DEVICES=2,3

EXP=/path/to/lab-home/pi-r2-flow/experiments/bag_transfer_16h_22d_headrgb_pir2_exact

OUT=$EXP/train_2gpu23_effbs128_240k_pir2_headrgb_v1
LOG=$EXP/logs/train_2gpu23_effbs128_240k_pir2_headrgb_v1.log

DATASET=/path/to/lab-home/pi-r2-flow/datasets/bag_transfer_16h/train_dataset_view_22d_headrgb
MODALITY=$EXP/config/bag_transfer_16h_22d_headrgb_abs.py

mkdir -p "$EXP/logs"

# ============================================================
# First-launch safety
# ============================================================
if [ -e "$OUT" ]; then
    echo "ERROR: output directory already exists:"
    echo "$OUT"
    echo
    echo "Refusing to start a second independent run."
    exit 2
fi

: > "$LOG"

echo "====================================================================================================" | tee -a "$LOG"
echo "FORMAL UNATTENDED PI-R2 TRAINING" | tee -a "$LOG"
echo "====================================================================================================" | tee -a "$LOG"
echo "DATE                 : $(date)" | tee -a "$LOG"
echo "HOST                 : $(hostname)" | tee -a "$LOG"
echo "CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES" | tee -a "$LOG"
echo "NCCL_P2P_DISABLE     : $NCCL_P2P_DISABLE" | tee -a "$LOG"
echo "NCCL_IB_DISABLE      : $NCCL_IB_DISABLE" | tee -a "$LOG"
echo "DATASET              : $DATASET" | tee -a "$LOG"
echo "MODALITY             : $MODALITY" | tee -a "$LOG"
echo "OUTPUT               : $OUT" | tee -a "$LOG"
echo | tee -a "$LOG"
echo "MAX STEPS            : 240000" | tee -a "$LOG"
echo "GLOBAL BATCH         : 64" | tee -a "$LOG"
echo "GRAD ACCUM           : 2" | tee -a "$LOG"
echo "EFFECTIVE BATCH      : 128" | tee -a "$LOG"
echo "WARMUP STEPS         : ~2500" | tee -a "$LOG"
echo "SAVE EVERY           : 20000" | tee -a "$LOG"
echo "SAVE LIMIT           : 10" | tee -a "$LOG"
echo "ACTION HORIZON       : 16" | tee -a "$LOG"
echo "VIDEO                : head_rgb only" | tee -a "$LOG"
echo "====================================================================================================" | tee -a "$LOG"


# ============================================================
# Unattended retry policy
#
# Initial attempt + at most 3 automatic retries.
# Same OUT directory is reused so GR00T can recover from
# the latest valid checkpoint if one exists.
# ============================================================

MAX_ATTEMPTS=4
ATTEMPT=1

while [ "$ATTEMPT" -le "$MAX_ATTEMPTS" ]; do

    echo | tee -a "$LOG"
    echo "====================================================================================================" | tee -a "$LOG"
    echo "TRAIN ATTEMPT $ATTEMPT / $MAX_ATTEMPTS" | tee -a "$LOG"
    echo "START: $(date)" | tee -a "$LOG"
    echo "====================================================================================================" | tee -a "$LOG"

    df -h /path/to/lab-home | tee -a "$LOG"

    torchrun \
      --nproc_per_node=2 \
      --master_port=29533 \
      gr00t/experiment/launch_finetune.py \
      --base-model-path nvidia/GR00T-N1.7-3B \
      --dataset-path "$DATASET" \
      --embodiment-tag NEW_EMBODIMENT \
      --modality-config-path "$MODALITY" \
      --num-gpus 2 \
      --output-dir "$OUT" \
      --max-steps 240000 \
      --global-batch-size 64 \
      --gradient-accumulation-steps 2 \
      --dataloader-num-workers 2 \
      --learning-rate 1e-4 \
      --weight-decay 1e-5 \
      --warmup-ratio 0.0104167 \
      --state-dropout-prob 0.0 \
      --save-steps 20000 \
      --save-total-limit 10 \
      --streaming \
      --streaming-constant-weight 0.2 \
      --streaming-chunk-wise-weight 0.8 \
      --streaming-schedule-mode pir2 \
      --streaming-chunk-size-max 5 \
      --streaming-mask-clean-end \
      --image-delay-max 5 \
      --image-delay-embed-dim 64 \
      2>&1 | tee -a "$LOG"

    RC=${PIPESTATUS[0]}

    if [ "$RC" -eq 0 ]; then
        echo | tee -a "$LOG"
        echo "====================================================================================================" | tee -a "$LOG"
        echo "TRAINING COMPLETED SUCCESSFULLY" | tee -a "$LOG"
        echo "DATE: $(date)" | tee -a "$LOG"
        echo "====================================================================================================" | tee -a "$LOG"
        exit 0
    fi

    echo | tee -a "$LOG"
    echo "Training exited with code: $RC" | tee -a "$LOG"
    echo "DATE: $(date)" | tee -a "$LOG"

    if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then
        echo "Maximum retry count reached." | tee -a "$LOG"
        exit "$RC"
    fi

    echo "Waiting 120 seconds before automatic restart..." | tee -a "$LOG"

    sleep 120

    ATTEMPT=$((ATTEMPT + 1))
done
