#!/usr/bin/env bash
# Launch a single-node run. All path overrides are relative to the calling shell.
set -euo pipefail
usage() {
    cat <<'HELP'
Usage: bash training/train.sh {headrgb|rgb3} [pir2|plain_flow]

Defaults reproduce the 317k head-camera or 50k three-camera training recipe.
DATASET, BASE_MODEL, GR00T_COSMOS_LOCAL_PATH and OUTPUT_DIR override paths.
NUM_GPUS, MAX_STEPS, GLOBAL_BATCH_SIZE, ACCUMULATION_STEPS, WORKERS,
WARMUP_RATIO, SAVE_STEPS, SAVE_TOTAL_LIMIT, SHARD_SIZE, NUM_SHARDS_PER_EPOCH
override training parameters. GR00T_IMAGE_DELAY_MAX sets loader delay for pir2.
Set RESUME=1 to continue an existing run from its latest training checkpoint.
HELP
}
if [[ ${1:-} == -h || ${1:-} == --help ]]; then usage; exit 0; fi
if (( $# < 1 || $# > 2 )); then usage >&2; exit 2; fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source_dir="$root/third_party/groot_train"
export BAG_CAMERA_PROFILE=$1
variant=${2:-pir2}
case "$BAG_CAMERA_PROFILE" in
    headrgb)
        steps=317000; batch=256; accumulation=1; warmup=0.0078864
        save=15000; keep=10; workers=4 ;;
    rgb3)
        steps=50000; batch=64; accumulation=2; warmup=0.05
        save=1000; keep=15; workers=2 ;;
    *) usage >&2; exit 2 ;;
esac
case "$variant" in
    pir2)
        export GR00T_IMAGE_DELAY_MAX=${GR00T_IMAGE_DELAY_MAX:-5}
        variant_args=(--streaming --streaming-constant-weight 0.2
            --streaming-chunk-wise-weight 0.8 --streaming-schedule-mode pir2
            --streaming-chunk-size-max 5 --streaming-mask-clean-end
            --image-delay-max 5 --image-delay-embed-dim 64) ;;
    plain_flow)
        export GR00T_IMAGE_DELAY_MAX=0
        variant_args=() ;;
    *) usage >&2; exit 2 ;;
esac
export GR00T_COSMOS_LOCAL_PATH
GR00T_COSMOS_LOCAL_PATH=$(realpath -m "${GR00T_COSMOS_LOCAL_PATH:-$root/weights/backbone}")
base=$(realpath -m "${BASE_MODEL:-$root/weights/base}")
dataset=$(realpath -m "${DATASET:-$root/data/processed/train_view}")
out=$(realpath -m "${OUTPUT_DIR:-$root/outputs/${BAG_CAMERA_PROFILE}_${variant}}")
num_gpus=${NUM_GPUS:-2}
global_batch=${GLOBAL_BATCH_SIZE:-$batch}
if ! [[ $num_gpus =~ ^[1-9][0-9]*$ && $global_batch =~ ^[1-9][0-9]*$ ]] ||
        (( global_batch % num_gpus != 0 )); then
    echo 'Use positive NUM_GPUS and GLOBAL_BATCH_SIZE divisible by NUM_GPUS.' >&2; exit 2
fi
if ! [[ "$GR00T_IMAGE_DELAY_MAX" =~ ^[0-5]$ ]]; then
    echo 'Use GR00T_IMAGE_DELAY_MAX in 0..5 for the released model.' >&2; exit 2
fi
launcher="$source_dir/gr00t/experiment/launch_finetune.py"
for required in "$launcher" "$base/config.json" "$GR00T_COSMOS_LOCAL_PATH/config.json" \
                "$dataset/meta/info.json" "$dataset/meta/modality.json" "$dataset/meta/tasks.jsonl"; do
    [[ -f "$required" ]] || { echo "Missing required file: $required" >&2; exit 1; }
done
if [[ ${RESUME:-0} == 1 ]]; then
    compgen -G "$out/checkpoint-*/trainer_state.json" >/dev/null || {
        echo "No training checkpoint to resume in $out" >&2; exit 1; }
else
    mkdir -p "$(dirname "$out")"
    mkdir "$out" || { echo 'Use a fresh OUTPUT_DIR or set RESUME=1 to continue.' >&2; exit 1; }
fi

export PYTHONPATH="$source_dir${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1
cd "$source_dir"
args=(--base-model-path "$base" --dataset-path "$dataset" --embodiment-tag NEW_EMBODIMENT
    --modality-config-path "$root/training/configs/bag_transfer_22d.py"
    --num-gpus "$num_gpus" --output-dir "$out" --max-steps "${MAX_STEPS:-$steps}"
    --global-batch-size "$global_batch" --gradient-accumulation-steps "${ACCUMULATION_STEPS:-$accumulation}"
    --dataloader-num-workers "${WORKERS:-$workers}" --learning-rate 1e-4 --weight-decay 1e-5
    --warmup-ratio "${WARMUP_RATIO:-$warmup}" --state-dropout-prob 0.0
    --save-steps "${SAVE_STEPS:-$save}" --save-total-limit "${SAVE_TOTAL_LIMIT:-$keep}"
    --shard-size "${SHARD_SIZE:-1024}" --num-shards-per-epoch "${NUM_SHARDS_PER_EPOCH:-100000}"
    "${variant_args[@]}")
{
    printf 'Camera profile: %s\nPolicy: %s\nGR00T_IMAGE_DELAY_MAX=%s\n' "$BAG_CAMERA_PROFILE" "$variant" "$GR00T_IMAGE_DELAY_MAX"
    printf '%q ' python -m torch.distributed.run --standalone --nproc-per-node="$num_gpus" "$launcher" "${args[@]}"
    printf '\n'
    python -m torch.distributed.run --standalone --nproc-per-node="$num_gpus" "$launcher" "${args[@]}"
} 2>&1 | tee -a "$out/launch.log"
