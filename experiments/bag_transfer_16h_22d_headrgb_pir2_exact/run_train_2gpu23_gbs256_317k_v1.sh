#!/usr/bin/env bash

set -u
set -o pipefail

ROOT=/path/to/lab-home/pi-r2-flow/learning/Isaac-GR00T
EXP=/path/to/lab-home/pi-r2-flow/experiments/bag_transfer_16h_22d_headrgb_pir2_exact
OUT=$EXP/train_2gpu23_gbs256_acc1_317k_pir2_headrgb_v1
LOG=$EXP/logs/train_2gpu23_gbs256_acc1_317k_pir2_headrgb_v1.log

cd "$ROOT"
source .venv/bin/activate

# ============================================================
# GPU / NCCL
# ============================================================
export CUDA_VISIBLE_DEVICES=2,3
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 禁止 albumentations 每次启动联网检查版本
export NO_ALBUMENTATIONS_UPDATE=1

mkdir -p "$EXP/logs"

# ============================================================
# 安全检查：绝不覆盖已有正式实验
# ============================================================
if [ -e "$OUT" ]; then
    echo "ERROR: output directory already exists:"
    echo "$OUT"
    echo "Refusing to overwrite it."
    exit 2
fi

# ============================================================
# 磁盘检查
# 10 个完整 checkpoint + final model 建议至少留 450 GB
# ============================================================
FREE_GB=$(df -BG --output=avail "$EXP" | tail -n 1 | tr -dc '0-9')

echo "Available disk space: ${FREE_GB} GB"

if [ "$FREE_GB" -lt 450 ]; then
    echo "ERROR: Less than 450 GB free."
    echo "Do NOT start the 317k run yet."
    exit 3
fi

MAX_ATTEMPTS=4

for ((ATTEMPT=1; ATTEMPT<=MAX_ATTEMPTS; ATTEMPT++)); do

    PORT=$((29538 + ATTEMPT - 1))

    RESUME_ARGS=()

    if compgen -G "$OUT/checkpoint-*" > /dev/null; then
        echo "Found checkpoint. Resuming latest checkpoint..."
        RESUME_ARGS+=(--resume-from-checkpoint)
    else
        echo "No checkpoint found. Starting from base model..."
    fi

    echo
    echo "============================================================"
    echo "Training attempt: $ATTEMPT / $MAX_ATTEMPTS"
    echo "Master port:      $PORT"
    echo "Output:           $OUT"
    echo "============================================================"
    echo

    torchrun \
      --nproc_per_node=2 \
      --master_port="$PORT" \
      gr00t/experiment/launch_finetune.py \
      --base-model-path nvidia/GR00T-N1.7-3B \
      --dataset-path /path/to/lab-home/pi-r2-flow/datasets/bag_transfer_16h/train_dataset_view_22d_headrgb \
      --embodiment-tag NEW_EMBODIMENT \
      --modality-config-path /path/to/lab-home/pi-r2-flow/experiments/bag_transfer_16h_22d_headrgb_pir2_exact/config/bag_transfer_16h_22d_headrgb_abs.py \
      --num-gpus 2 \
      --output-dir "$OUT" \
      --max-steps 317000 \
      --global-batch-size 256 \
      --gradient-accumulation-steps 1 \
      --dataloader-num-workers 4 \
      --learning-rate 1e-4 \
      --weight-decay 1e-5 \
      --warmup-ratio 0.0078864 \
      --state-dropout-prob 0.0 \
      --save-steps 15000 \
      --save-total-limit 10 \
      --streaming \
      --streaming-constant-weight 0.2 \
      --streaming-chunk-wise-weight 0.8 \
      --streaming-schedule-mode pir2 \
      --streaming-chunk-size-max 5 \
      --streaming-mask-clean-end \
      --image-delay-max 5 \
      --image-delay-embed-dim 64 \
      "${RESUME_ARGS[@]}" \
      2>&1 | tee -a "$LOG"

    RC=${PIPESTATUS[0]}

    if [ "$RC" -eq 0 ]; then
        echo "Training completed successfully."
        exit 0
    fi

    echo
    echo "Training attempt $ATTEMPT failed with exit code $RC."

    if [ "$ATTEMPT" -eq "$MAX_ATTEMPTS" ]; then
        echo "Maximum retry count reached."
        exit "$RC"
    fi

    echo "Waiting 120 seconds before retry..."
    sleep 120
done
