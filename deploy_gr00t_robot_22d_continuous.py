#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import time
import uuid
import traceback

import cv2
import numpy as np
import Pyro5.api
import torch

from Pyro5 import serializers
from Pyro5.serializers import SerializerBase
from gr00t.policy.gr00t_policy import Gr00tPolicy


# ============================================================
# Configuration
# ============================================================

ROBOT_IP = "10.0.0.11"
ROBOT_PORT = 7788

MODEL_PATH = (
    "/path/to/lab-home/Isaac-GR00T/"
    "experiments/bag_transfer_22d_rgb_abs/"
    "train_bs128_20k_v1/"
    "checkpoint-7000"
)

TASK_TEXT = "快递分拣"
EMBODIMENT_TAG = "NEW_EMBODIMENT"

DEVICE = "cuda:0"
DENOISING_STEPS = 4

# ------------------------------------------------------------
# Safety
# ------------------------------------------------------------

# 你的“15°以内”：
# raw model target 任一 arm joint 偏离当前状态超过 15°，
# 本轮直接不发送。
RAW_ARM_GATE_DEG = 15.0
RAW_ARM_GATE_RAD = np.deg2rad(RAW_ARM_GATE_DEG)

# 真正每轮发送给机械臂的最大变化仍保持 2°。
MAX_ARM_DELTA_DEG = 2.0
MAX_ARM_DELTA_RAD = np.deg2rad(MAX_ARM_DELTA_DEG)

# 每个真实 hand joint 每轮最多 0.05 rad。
MAX_HAND_DELTA_RAD = 0.05

# 连续多少次超过 15° 就自动停止整个程序。
MAX_CONSECUTIVE_REJECTS = 3

# 保险：避免无上限疯狂循环。
# 0 = 不限制，由 Ctrl+C 停止。
MAX_ITERATIONS = 0


# ============================================================
# Pyro pickle wire protocol
# ============================================================

class PickleSerializer(SerializerBase):
    serializer_id = 5

    def dumpsCall(self, obj, method, vargs, kwargs):
        return pickle.dumps(
            (obj, method, vargs, kwargs),
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    def dumps(self, data):
        return pickle.dumps(
            data,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    def loadsCall(self, data):
        data = self._convertToBytes(data)
        return pickle.loads(data)

    def loads(self, data):
        data = self._convertToBytes(data)
        return pickle.loads(data)

    @classmethod
    def register_type_replacement(
        cls,
        object_type,
        replacement_function,
    ):
        pass


def configure_pyro():

    if "pickle" not in serializers.serializers:

        s = PickleSerializer()

        serializers.serializers[
            "pickle"
        ] = s

        serializers.serializers_by_id[
            s.serializer_id
        ] = s

    Pyro5.api.config.SERIALIZER = "pickle"


# ============================================================
# Observation
# ============================================================

def decode_rgb(jpeg_bytes):

    data = np.frombuffer(
        jpeg_bytes,
        dtype=np.uint8,
    )

    bgr = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR,
    )

    if bgr is None:
        raise RuntimeError(
            "JPEG decode failed"
        )

    return cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )


def decode_observation(wire):

    if wire is None:
        raise RuntimeError(
            "observation not ready"
        )

    wire = dict(wire)

    obs = {
        "timestamp_ns":
            int(wire["timestamp_ns"]),

        "head_rgb":
            decode_rgb(
                wire["head_rgb_jpeg"]
            ),

        "left_wrist_rgb":
            decode_rgb(
                wire["left_wrist_rgb_jpeg"]
            ),

        "right_wrist_rgb":
            decode_rgb(
                wire["right_wrist_rgb_jpeg"]
            ),

        "left_arm":
            np.asarray(
                wire["left_arm"],
                dtype=np.float64,
            ).reshape(-1),

        "right_arm":
            np.asarray(
                wire["right_arm"],
                dtype=np.float64,
            ).reshape(-1),

        "left_hand":
            np.asarray(
                wire["left_hand"],
                dtype=np.float64,
            ).reshape(-1),

        "right_hand":
            np.asarray(
                wire["right_hand"],
                dtype=np.float64,
            ).reshape(-1),
    }

    expected = {
        "left_arm": 7,
        "right_arm": 7,
        "left_hand": 6,
        "right_hand": 6,
    }

    for key, dim in expected.items():

        if obs[key].shape != (dim,):
            raise RuntimeError(
                f"{key}: expected ({dim},), "
                f"got {obs[key].shape}"
            )

        if not np.all(np.isfinite(obs[key])):
            raise RuntimeError(
                f"{key}: NaN/Inf"
            )

    return obs


# ============================================================
# Hand 6D <-> 4D
# ============================================================

def compress_hand_6_to_4(hand6):

    h = np.asarray(
        hand6,
        dtype=np.float64,
    ).reshape(6)

    return np.array(
        [
            h[0],
            h[1],
            h[2],
            np.mean(h[3:6]),
        ],
        dtype=np.float64,
    )


def expand_hand_4_to_6(hand4):

    h = np.asarray(
        hand4,
        dtype=np.float64,
    ).reshape(4)

    o = h[3]

    return np.array(
        [
            h[0],
            h[1],
            h[2],
            o,
            o,
            o,
        ],
        dtype=np.float64,
    )


# ============================================================
# GR00T input
# ============================================================

def make_video(x):

    x = np.asarray(x)

    if (
        x.ndim != 3
        or x.shape[-1] != 3
        or x.dtype != np.uint8
    ):
        raise ValueError(
            f"bad video: "
            f"{x.shape}, {x.dtype}"
        )

    return np.ascontiguousarray(
        x[None, None, ...]
    )


def make_state(x, dim):

    x = np.asarray(
        x,
        dtype=np.float32,
    ).reshape(-1)

    if x.shape != (dim,):
        raise ValueError(
            f"expected ({dim},), "
            f"got {x.shape}"
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "state contains NaN/Inf"
        )

    return np.ascontiguousarray(
        x[None, None, :]
    )


def build_model_obs(obs, policy):

    lh4 = compress_hand_6_to_4(
        obs["left_hand"]
    )

    rh4 = compress_hand_6_to_4(
        obs["right_hand"]
    )

    return {
        "video": {
            "head_rgb":
                make_video(
                    obs["head_rgb"]
                ),

            "left_wrist_rgb":
                make_video(
                    obs["left_wrist_rgb"]
                ),

            "right_wrist_rgb":
                make_video(
                    obs["right_wrist_rgb"]
                ),
        },

        "state": {
            "left_arm":
                make_state(
                    obs["left_arm"],
                    7,
                ),

            "right_arm":
                make_state(
                    obs["right_arm"],
                    7,
                ),

            "left_hand":
                make_state(
                    lh4,
                    4,
                ),

            "right_hand":
                make_state(
                    rh4,
                    4,
                ),
        },

        "language": {
            policy.language_key:
                [[TASK_TEXT]]
        },
    }


# ============================================================
# Action
# ============================================================

def first_action(
    action,
    key,
    dim,
):

    x = np.asarray(
        action[key],
        dtype=np.float64,
    )

    if x.ndim == 3:

        if x.shape[0] != 1:
            raise RuntimeError(
                f"{key}: bad batch "
                f"{x.shape}"
            )

        x = x[0]

    if (
        x.ndim != 2
        or x.shape[1] != dim
    ):
        raise RuntimeError(
            f"{key}: bad action shape "
            f"{x.shape}"
        )

    x = x[0].copy()

    if not np.all(np.isfinite(x)):
        raise RuntimeError(
            f"{key}: NaN/Inf"
        )

    return x


def rate_limit(
    target,
    current,
    max_delta,
):

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    current = np.asarray(
        current,
        dtype=np.float64,
    )

    if target.shape != current.shape:
        raise RuntimeError(
            f"shape mismatch: "
            f"{target.shape} vs {current.shape}"
        )

    raw_delta = (
        target
        - current
    )

    safe_delta = np.clip(
        raw_delta,
        -max_delta,
        max_delta,
    )

    command = (
        current
        + safe_delta
    )

    return (
        command,
        raw_delta,
        safe_delta,
    )


# ============================================================
# Send 26D robot target
# ============================================================

def send_targets(
    proxy,
    client_uuid,
    left_arm,
    right_arm,
    left_hand,
    right_hand,
):

    command = {
        "client_uuid":
            client_uuid,

        "left_arm": [
            float(v)
            for v in np.asarray(
                left_arm
            ).reshape(7)
        ],

        "right_arm": [
            float(v)
            for v in np.asarray(
                right_arm
            ).reshape(7)
        ],

        "left_hand": [
            float(v)
            for v in np.asarray(
                left_hand
            ).reshape(6)
        ],

        "right_hand": [
            float(v)
            for v in np.asarray(
                right_hand
            ).reshape(6)
        ],
    }

    return proxy.submit_command(
        command
    )


# ============================================================
# Main
# ============================================================

def main():

    np.set_printoptions(
        precision=4,
        suppress=True,
    )

    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("=" * 90)
    print(
        "GR00T N1.7 22D CONTINUOUS REAL-ROBOT DEPLOYMENT"
    )
    print("=" * 90)

    print(
        "robot             =",
        f"{ROBOT_IP}:{ROBOT_PORT}"
    )

    print(
        "model             =",
        MODEL_PATH
    )

    print(
        "raw arm gate      =",
        RAW_ARM_GATE_DEG,
        "deg"
    )

    print(
        "arm step limiter  =",
        MAX_ARM_DELTA_DEG,
        "deg / inference"
    )

    print(
        "hand step limiter =",
        MAX_HAND_DELTA_RAD,
        "rad / inference"
    )

    print(
        "reject-stop       =",
        MAX_CONSECUTIVE_REJECTS
    )

    print(
        "Ctrl+C to stop."
    )

    print("=" * 90)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\n[1] Loading checkpoint-7000..."
    )

    policy = Gr00tPolicy(
        embodiment_tag=
            EMBODIMENT_TAG,

        model_path=
            MODEL_PATH,

        device=
            DEVICE,
    )

    policy.model.action_head.num_inference_timesteps = (
        DENOISING_STEPS
    )

    print(
        "[OK] model loaded"
    )

    # --------------------------------------------------------
    # Pyro
    # --------------------------------------------------------

    configure_pyro()

    uri = (
        f"PYRO:model_inference_bridge@"
        f"{ROBOT_IP}:{ROBOT_PORT}"
    )

    stream_proxy = Pyro5.api.Proxy(
        uri
    )

    cmd_proxy = Pyro5.api.Proxy(
        uri
    )

    query_proxy = Pyro5.api.Proxy(
        uri
    )

    for p in (
        stream_proxy,
        cmd_proxy,
        query_proxy,
    ):
        p._pyroTimeout = 10.0

    client_uuid = str(
        uuid.uuid4()
    )

    print(
        "\n[2] Connecting bridge..."
    )

    print(
        "ping =",
        query_proxy.ping()
    )

    print(
        "obs status =",
        query_proxy.get_obs_status()
    )

    iteration = 0
    consecutive_rejects = 0

    try:

        while True:

            if (
                MAX_ITERATIONS > 0
                and iteration >= MAX_ITERATIONS
            ):
                print(
                    "[STOP] maximum iterations reached"
                )
                break

            wire = (
                stream_proxy
                .get_observation()
            )

            if wire is None:

                print(
                    "[WAIT] observation not ready"
                )

                time.sleep(0.05)
                continue

            obs = decode_observation(
                wire
            )

            iteration += 1

            # -----------------------------------------------
            # Model input
            # -----------------------------------------------

            model_obs = (
                build_model_obs(
                    obs,
                    policy,
                )
            )

            # -----------------------------------------------
            # Inference
            # -----------------------------------------------

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t0 = time.perf_counter()

            with torch.inference_mode():

                action, _ = (
                    policy.get_action(
                        model_obs
                    )
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            infer_s = (
                time.perf_counter()
                - t0
            )

            # -----------------------------------------------
            # Native 22D
            # -----------------------------------------------

            pred_la = first_action(
                action,
                "left_arm",
                7,
            )

            pred_ra = first_action(
                action,
                "right_arm",
                7,
            )

            pred_lh4 = first_action(
                action,
                "left_hand",
                4,
            )

            pred_rh4 = first_action(
                action,
                "right_hand",
                4,
            )

            # -----------------------------------------------
            # Expand hand 4D -> physical 6D
            # -----------------------------------------------

            pred_lh6 = (
                expand_hand_4_to_6(
                    pred_lh4
                )
            )

            pred_rh6 = (
                expand_hand_4_to_6(
                    pred_rh4
                )
            )

            # -----------------------------------------------
            # Raw arm gate
            # -----------------------------------------------

            raw_dla = (
                pred_la
                - obs["left_arm"]
            )

            raw_dra = (
                pred_ra
                - obs["right_arm"]
            )

            raw_arm_deg = max(
                np.max(
                    np.abs(
                        np.rad2deg(
                            raw_dla
                        )
                    )
                ),
                np.max(
                    np.abs(
                        np.rad2deg(
                            raw_dra
                        )
                    )
                ),
            )

            if (
                raw_arm_deg
                > RAW_ARM_GATE_DEG
            ):

                consecutive_rejects += 1

                print(
                    f"[{iteration:05d}] "
                    f"REJECT | "
                    f"infer={infer_s:.3f}s | "
                    f"raw_arm={raw_arm_deg:.2f}° | "
                    f"rejects="
                    f"{consecutive_rejects}/"
                    f"{MAX_CONSECUTIVE_REJECTS}"
                )

                if (
                    consecutive_rejects
                    >= MAX_CONSECUTIVE_REJECTS
                ):
                    raise RuntimeError(
                        "Safety stop: "
                        f"{MAX_CONSECUTIVE_REJECTS} "
                        "consecutive raw-arm "
                        "gate violations"
                    )

                continue

            consecutive_rejects = 0

            # -----------------------------------------------
            # Per-step rate limiting
            # -----------------------------------------------

            cmd_la, _, safe_dla = (
                rate_limit(
                    pred_la,
                    obs["left_arm"],
                    MAX_ARM_DELTA_RAD,
                )
            )

            cmd_ra, _, safe_dra = (
                rate_limit(
                    pred_ra,
                    obs["right_arm"],
                    MAX_ARM_DELTA_RAD,
                )
            )

            cmd_lh6, raw_dlh6, safe_dlh6 = (
                rate_limit(
                    pred_lh6,
                    obs["left_hand"],
                    MAX_HAND_DELTA_RAD,
                )
            )

            cmd_rh6, raw_drh6, safe_drh6 = (
                rate_limit(
                    pred_rh6,
                    obs["right_hand"],
                    MAX_HAND_DELTA_RAD,
                )
            )

            # -----------------------------------------------
            # Final shape checks
            # -----------------------------------------------

            native22 = np.concatenate(
                [
                    pred_la,
                    pred_ra,
                    pred_lh4,
                    pred_rh4,
                ]
            )

            robot26 = np.concatenate(
                [
                    cmd_la,
                    cmd_ra,
                    cmd_lh6,
                    cmd_rh6,
                ]
            )

            if native22.shape != (22,):
                raise RuntimeError(
                    f"native22={native22.shape}"
                )

            if robot26.shape != (26,):
                raise RuntimeError(
                    f"robot26={robot26.shape}"
                )

            # -----------------------------------------------
            # Send exactly one latest target this cycle
            # -----------------------------------------------

            result = send_targets(
                cmd_proxy,
                client_uuid,

                left_arm=
                    cmd_la,

                right_arm=
                    cmd_ra,

                left_hand=
                    cmd_lh6,

                right_hand=
                    cmd_rh6,
            )

            cmd_arm_deg = max(
                np.max(
                    np.abs(
                        np.rad2deg(
                            safe_dla
                        )
                    )
                ),
                np.max(
                    np.abs(
                        np.rad2deg(
                            safe_dra
                        )
                    )
                ),
            )

            raw_hand = max(
                np.max(
                    np.abs(
                        raw_dlh6
                    )
                ),
                np.max(
                    np.abs(
                        raw_drh6
                    )
                ),
            )

            cmd_hand = max(
                np.max(
                    np.abs(
                        safe_dlh6
                    )
                ),
                np.max(
                    np.abs(
                        safe_drh6
                    )
                ),
            )

            print(
                f"[{iteration:05d}] "
                f"SEND | "
                f"infer={infer_s:.3f}s | "
                f"raw_arm={raw_arm_deg:6.2f}° | "
                f"cmd_arm={cmd_arm_deg:4.2f}° | "
                f"raw_hand={raw_hand:.4f} | "
                f"cmd_hand={cmd_hand:.4f} | "
                f"rpc={result}"
            )

            if (
                iteration % 10
                == 0
            ):

                print(
                    "  current LA deg:",
                    np.round(
                        np.rad2deg(
                            obs["left_arm"]
                        ),
                        2,
                    )
                )

                print(
                    "  pred    LA deg:",
                    np.round(
                        np.rad2deg(
                            pred_la
                        ),
                        2,
                    )
                )

                print(
                    "  current RA deg:",
                    np.round(
                        np.rad2deg(
                            obs["right_arm"]
                        ),
                        2,
                    )
                )

                print(
                    "  pred    RA deg:",
                    np.round(
                        np.rad2deg(
                            pred_ra
                        ),
                        2,
                    )
                )

    except KeyboardInterrupt:

        print(
            "\n[STOP] Ctrl+C"
        )

    except Exception:

        print(
            "\n[ERROR]"
        )

        traceback.print_exc()

    finally:

        print(
            "\nReleasing control lease..."
        )

        try:
            cmd_proxy.release_control_lease(
                client_uuid
            )
        except Exception as e:
            print(
                "release lease:",
                repr(e)
            )

        for p in (
            stream_proxy,
            cmd_proxy,
            query_proxy,
        ):

            try:
                p._pyroRelease()
            except Exception:
                pass

        print(
            "[DONE]"
        )


if __name__ == "__main__":
    main()
