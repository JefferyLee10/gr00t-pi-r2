#!/usr/bin/env bash
# Run one decoupled server per GPU, typically VLM on 5555 and DiT on 5556.
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: bash deployment/serve.sh CHECKPOINT GPU PORT [BIND_HOST]" >&2
    exit 2
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
checkpoint=$(realpath -e -- "$1")
python "$root/scripts/prepare_weights.py" check checkpoint --path "$checkpoint"
export GR00T_COSMOS_LOCAL_PATH="${GR00T_COSMOS_LOCAL_PATH:-$root/weights/backbone}"
python "$root/scripts/prepare_weights.py" check backbone --path "$GR00T_COSMOS_LOCAL_PATH"
export CUDA_VISIBLE_DEVICES="$2" PYTHONUNBUFFERED=1
export PYTHONPATH="$root/third_party/groot_deploy${PYTHONPATH:+:$PYTHONPATH}"
exec python -m gr00t.eval.run_gr00t_server --model-path "$checkpoint" --port "$3" \
    --host "${4:-127.0.0.1}" --embodiment-tag NEW_EMBODIMENT --decoupled
