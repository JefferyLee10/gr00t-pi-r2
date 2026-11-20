#!/usr/bin/env bash

set -Eeuo pipefail

# ==============================================================================
# 固定路径
# ==============================================================================

DEPLOY_ROOT="/path/to/lab-home/pi-r2-flow/learning/Isaac-GR00T-pir2-deploy"
TRAIN_ENV="/path/to/lab-home/pi-r2-flow/learning/Isaac-GR00T/.venv"

# 317k 正式训练输出目录
RUN_DIR="/path/to/lab-home/pi-r2-flow/experiments/bag_transfer_16h_22d_headrgb_pir2_exact/train_2gpu23_gbs256_acc1_317k_pir2_headrgb_v1"

# 之前已经在真机上验证过的 πR² V2 controller
BASE_CTRL="/path/to/lab-home/pi-r2-flow/deploy_pir2_robot_22d_rgb3_continuous_v2.py"

# 本次 head-RGB-only controller
CTRL="/path/to/lab-home/pi-r2-flow/deploy_pir2_robot_22d_headrgb_continuous_v1.py"

LOG_DIR="/path/to/lab-home/pi-r2-flow/deploy_logs/headrgb_317k"

mkdir -p "$LOG_DIR"


# ==============================================================================
# 0. 基础文件检查
# ==============================================================================

if [[ ! -d "$DEPLOY_ROOT" ]]; then
    echo "ERROR: deploy worktree does not exist:"
    echo "  $DEPLOY_ROOT"
    exit 1
fi

if [[ ! -x "$TRAIN_ENV/bin/python" ]]; then
    echo "ERROR: Python env does not exist:"
    echo "  $TRAIN_ENV"
    exit 1
fi

if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: training run directory does not exist:"
    echo "  $RUN_DIR"
    exit 1
fi

if [[ ! -f "$BASE_CTRL" ]]; then
    echo "ERROR: trusted V2 controller does not exist:"
    echo "  $BASE_CTRL"
    exit 1
fi


# ==============================================================================
# 1. 第一次运行时：
#    从已经验证过的 RGB3 V2 controller 生成 HEAD-RGB-only 版本
#
#    不修改原 V2。
# ==============================================================================

if [[ ! -f "$CTRL" ]]; then

    echo
    echo "Creating HEAD-RGB controller from trusted V2..."
    echo

    TMP="${CTRL}.tmp.$$"

    cp "$BASE_CTRL" "$TMP"

    "$TRAIN_ENV/bin/python" - "$TMP" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")

# ----------------------------------------------------------------------
# 1. 模型现在只接受 head_rgb
# ----------------------------------------------------------------------

old_video = 'expected_video = ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"]'
new_video = 'expected_video = ["head_rgb"]'

if old_video not in s:
    raise SystemExit(
        "ERROR: could not find expected RGB3 validation line in V2 controller.\n"
        "Refusing to patch automatically."
    )

s = s.replace(old_video, new_video, 1)


# ----------------------------------------------------------------------
# 2. make_snapshot 不再依赖左右腕部相机
# ----------------------------------------------------------------------

old_rgbs = '''    rgbs = {
        "head_rgb": np.ascontiguousarray(obs["head_rgb"][None]),
        "left_wrist_rgb": np.ascontiguousarray(obs["left_wrist_rgb"][None]),
        "right_wrist_rgb": np.ascontiguousarray(obs["right_wrist_rgb"][None]),
    }'''

new_rgbs = '''    rgbs = {
        "head_rgb": np.ascontiguousarray(obs["head_rgb"][None]),
    }'''

if old_rgbs not in s:
    raise SystemExit(
        "ERROR: could not find RGB3 make_snapshot block in V2 controller.\n"
        "Refusing to patch automatically."
    )

s = s.replace(old_rgbs, new_rgbs, 1)


# ----------------------------------------------------------------------
# 3. 仅修改终端显示文字
# ----------------------------------------------------------------------

s = s.replace(
    "πR² GR00T-N1.7 | RGB3 | 22D ABSOLUTE | DECOUPLED CONTINUOUS",
    "πR² GR00T-N1.7 | HEAD_RGB | 22D ABSOLUTE | DECOUPLED CONTINUOUS",
    1,
)

p.write_text(s, encoding="utf-8")

print("HEAD-RGB controller patch: OK")
PY

    mv "$TMP" "$CTRL"
    chmod +x "$CTRL"

    echo "Created:"
    echo "  $CTRL"
    echo
fi


# ==============================================================================
# 2. 激活部署环境
# ==============================================================================

cd "$DEPLOY_ROOT"

source "$TRAIN_ENV/bin/activate"

# 保持你的服务器上一直使用的 NCCL 规避设置
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1

export PYTHONPATH="/path/to/lab-home/pi-r2-flow/deployment:${DEPLOY_ROOT}:/path/to/lab-home/Isaac-GR00T:${PYTHONPATH:-}"


# ==============================================================================
# 3. 用户输入 checkpoint step
# ==============================================================================

echo
echo "=============================================================================="
echo "πR² GR00T N1.7 | HEAD RGB | 22D | CHECKPOINT SELECTOR"
echo "=============================================================================="
echo
echo "Training run:"
echo "  $RUN_DIR"
echo

while true; do
    read -r -p "请输入 checkpoint step（例如 19500）: " STEP

    if [[ "$STEP" =~ ^[0-9]+$ ]] && [[ "$STEP" -gt 0 ]]; then
        break
    fi

    echo "请输入纯数字，例如：19500"
done

CKPT="${RUN_DIR}/checkpoint-${STEP}"

echo
echo "Requested checkpoint:"
echo "  $CKPT"
echo


# ==============================================================================
# 4. checkpoint 必须精确存在
# ==============================================================================

if [[ ! -d "$CKPT" ]]; then

    echo "ERROR: checkpoint-${STEP} 不存在。"
    echo
    echo "当前实际存在的 checkpoint："
    echo "----------------------------------------------------------------------"

    find "$RUN_DIR" \
        -maxdepth 1 \
        -type d \
        -name 'checkpoint-*' \
        -printf '%f\n' \
        | sed 's/^checkpoint-//' \
        | sort -n \
        | tail -n 100

    echo "----------------------------------------------------------------------"
    echo
    echo "不会自动选择其它 checkpoint。"
    exit 2
fi


# 基本完整性检查
REQUIRED_FILES=(
    "config.json"
    "processor_config.json"
    "statistics.json"
    "embodiment_id.json"
    "model.safetensors.index.json"
)

for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$CKPT/$f" ]]; then
        echo "ERROR: checkpoint missing required file:"
        echo "  $CKPT/$f"
        exit 2
    fi
done

if ! compgen -G "$CKPT/model-*.safetensors" > /dev/null; then
    echo "ERROR: checkpoint has no model-*.safetensors"
    exit 2
fi

echo "[OK] checkpoint inference files complete"


# ==============================================================================
# 5. 选择 DRY RUN / 真机发送
# ==============================================================================

echo
echo "运行模式："
echo "  d = DRY RUN，不发送任何机器人动作"
echo "  r = REAL SEND，真实控制机器人"
echo

read -r -p "请选择 [d/r]（默认 d）: " MODE
MODE="${MODE:-d}"

SEND_ARGS=()

case "$MODE" in
    d|D)
        echo
        echo "Mode = DRY RUN"
        ;;
    r|R)
        echo
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "REAL ROBOT SEND"
        echo "Checkpoint = checkpoint-${STEP}"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo

        read -r -p "如果确认真实下发动作，请输入 SEND: " CONFIRM

        if [[ "$CONFIRM" != "SEND" ]]; then
            echo "Cancelled."
            exit 0
        fi

        SEND_ARGS=(--send)
        ;;
    *)
        echo "ERROR: only d or r is accepted"
        exit 2
        ;;
esac


# ==============================================================================
# 6. 运行时间
# ==============================================================================

echo
read -r -p "运行秒数（默认 10；输入 0 表示一直运行到 Ctrl+C）: " RUN_SECONDS
RUN_SECONDS="${RUN_SECONDS:-10}"

if [[ ! "$RUN_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "ERROR: invalid seconds: $RUN_SECONDS"
    exit 2
fi


# ==============================================================================
# 7. 确保 5555 / 5556 没有旧服务器
# ==============================================================================

echo
echo "Checking inference ports..."

PORT_BUSY=0

for PORT in 5555 5556; do
    if ss -ltn | awk '{print $4}' | grep -Eq ":${PORT}$"; then
        echo
        echo "ERROR: port ${PORT} is already occupied."
        PORT_BUSY=1
    fi
done

if [[ "$PORT_BUSY" -ne 0 ]]; then
    echo
    echo "Existing listeners:"
    ss -ltnp | grep -E ':(5555|5556)\b' || true
    echo
    echo "为避免误杀其它进程，本脚本不会自动 kill 已存在的 server。"
    exit 3
fi


# ==============================================================================
# 8. 日志
# ==============================================================================

STAMP="$(date +%Y%m%d_%H%M%S)"

VLM_LOG="${LOG_DIR}/checkpoint-${STEP}_${STAMP}_vlm_gpu2_5555.log"
DIT_LOG="${LOG_DIR}/checkpoint-${STEP}_${STAMP}_dit_gpu3_5556.log"

echo
echo "=============================================================================="
echo "Deployment configuration"
echo "=============================================================================="
echo "checkpoint      = checkpoint-${STEP}"
echo "checkpoint path = $CKPT"
echo "VLM             = physical GPU 0 / port 5555"
echo "DiT             = physical GPU 2 / port 5556"
echo "controller      = $CTRL"
echo "NFE             = 24"
echo "initial d       = 2"
echo "control         = 20 Hz"
echo "runtime         = ${RUN_SECONDS} s"
echo "mode            = $([[ ${#SEND_ARGS[@]} -gt 0 ]] && echo REAL_SEND || echo DRY_RUN)"
echo "=============================================================================="
echo


# ==============================================================================
# 9. 清理函数
#
# 只关闭本脚本自己启动的服务器，不碰其它进程。
# ==============================================================================

VLM_PID=""
DIT_PID=""

cleanup() {
    RC=$?

    trap - EXIT INT TERM

    echo
    echo "Stopping inference servers..."

    if [[ -n "${VLM_PID:-}" ]]; then
        kill -TERM -- "-${VLM_PID}" 2>/dev/null || true
    fi

    if [[ -n "${DIT_PID:-}" ]]; then
        kill -TERM -- "-${DIT_PID}" 2>/dev/null || true
    fi

    sleep 1

    if [[ -n "${VLM_PID:-}" ]]; then
        kill -KILL -- "-${VLM_PID}" 2>/dev/null || true
    fi

    if [[ -n "${DIT_PID:-}" ]]; then
        kill -KILL -- "-${DIT_PID}" 2>/dev/null || true
    fi

    wait "${VLM_PID:-}" 2>/dev/null || true
    wait "${DIT_PID:-}" 2>/dev/null || true

    echo "[DONE] inference servers stopped"

    exit "$RC"
}

trap cleanup EXIT INT TERM


# ==============================================================================
# 10. 启动 VLM server
# ==============================================================================

echo "[1/4] Starting VLM server..."
echo "      GPU0 -> 127.0.0.1:5555"
echo "      log: $VLM_LOG"

setsid env \
    MODEL_PATH="$CKPT" \
    GPU=0 \
    PORT=5555 \
    DECOUPLED=1 \
    NCCL_P2P_DISABLE=1 \
    NCCL_IB_DISABLE=1 \
    NO_ALBUMENTATIONS_UPDATE=1 \
    bash scripts/gr00t_inference.sh \
    > "$VLM_LOG" 2>&1 &

VLM_PID=$!


# ==============================================================================
# 11. 启动 DiT server
# ==============================================================================

echo
echo "[2/4] Starting DiT server..."
echo "      GPU2 -> 127.0.0.1:5556"
echo "      log: $DIT_LOG"

setsid env \
    MODEL_PATH="$CKPT" \
    GPU=2 \
    PORT=5556 \
    DECOUPLED=1 \
    NCCL_P2P_DISABLE=1 \
    NCCL_IB_DISABLE=1 \
    NO_ALBUMENTATIONS_UPDATE=1 \
    bash scripts/gr00t_inference.sh \
    > "$DIT_LOG" 2>&1 &

DIT_PID=$!


# ==============================================================================
# 12. 等待 server 真正监听
# ==============================================================================

port_open() {
    local PORT="$1"

    "$TRAIN_ENV/bin/python" - "$PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])

try:
    s = socket.create_connection(("127.0.0.1", port), timeout=0.5)
    s.close()
except OSError:
    raise SystemExit(1)

raise SystemExit(0)
PY
}


wait_server() {
    local NAME="$1"
    local PID="$2"
    local PORT="$3"
    local LOG="$4"

    echo
    echo "Waiting for ${NAME} on port ${PORT}..."

    for _ in $(seq 1 240); do

        if ! kill -0 "$PID" 2>/dev/null; then
            echo
            echo "ERROR: ${NAME} server exited before becoming ready."
            echo
            echo "Last 120 lines:"
            tail -n 120 "$LOG" || true
            return 1
        fi

        if port_open "$PORT"; then
            echo "[OK] ${NAME} port ${PORT} ready"
            return 0
        fi

        sleep 1
    done

    echo
    echo "ERROR: timeout waiting for ${NAME} server."
    tail -n 120 "$LOG" || true
    return 1
}


wait_server "VLM" "$VLM_PID" 5555 "$VLM_LOG"
wait_server "DiT" "$DIT_PID" 5556 "$DIT_LOG"


# ==============================================================================
# 13. 启动真机 controller
# ==============================================================================

echo
echo "[3/4] Servers ready"
echo
echo "[4/4] Starting πR² real-robot controller..."
echo
echo "Checkpoint = checkpoint-${STEP}"
echo

"$TRAIN_ENV/bin/python" \
    "$CTRL" \
    "${SEND_ARGS[@]}" \
    --seconds "$RUN_SECONDS" \
    --nfe 24 \
    --initial-d 2

