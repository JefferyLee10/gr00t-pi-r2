#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
πR² / GR00T-N1.7 deployment for this project's dual-arm + dual-hand robot.

Model contract
--------------
- RGB only: head_rgb, left_wrist_rgb, right_wrist_rgb
- State/action: 22D absolute joint space
    left_arm 7 + right_arm 7 + left_hand 4 + right_hand 4
- Policy inference: official πR² decoupled 2-server path
    VLM server: localhost:5555
    DiT server: localhost:5556
- Robot bridge: existing Pyro bridge @ 10.0.0.11:7788
- Control loop: 20 Hz

Safety
------
- DRY RUN by default. Real commands require --send.
- Retains the already-validated observation decode, hand compression,
  rate limit and send interface from deploy_gr00t_robot_22d_continuous.py.
- Retains raw-arm / adjacent-action / stale-observation safety gates.
- Emits only πR²'s freshly denoised window [d, 2d).
"""

from __future__ import annotations

import argparse
import collections
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import Pyro5.api

# -----------------------------------------------------------------------------
# Import local project code
# -----------------------------------------------------------------------------

PI_R2_ROOT = Path("/path/to/lab-home/pi-r2-flow")
OLD_GROOT_ROOT = Path("/path/to/lab-home/Isaac-GR00T")

sys.path.insert(0, str(PI_R2_ROOT / "deployment"))
sys.path.insert(0, str(OLD_GROOT_ROOT))

from mindex.policy.groot_client import GR00TClient  # noqa: E402
import deploy_gr00t_robot_22d_continuous as base   # noqa: E402


# =============================================================================
# Project constants
# =============================================================================

ROBOT_IP = "10.0.0.11"
ROBOT_PORT = 7788
TASK_TEXT = "快递分拣"

DIT_HOST = "127.0.0.1"
DIT_PORT = 5556
VLM_HOST = "127.0.0.1"
VLM_PORT = 5555

CONTROL_HZ = 20.0
ACTION_DT = 1.0 / CONTROL_HZ
OBS_HZ = 20.0

# Official πR² deployment recipe uses NFE=24 and starts with d/chunk_len=2.
DEFAULT_NFE = 24
DEFAULT_INITIAL_D = 2

# Training used --streaming-chunk-size-max 5 and --image-delay-max 5.
TRAINED_D_MAX = 5
TRAINED_IMAGE_DELAY_MAX = 5

# Safety gates inherited from the trusted RGB controller.
RAW_ARM_GATE_DEG = 30.0
MAX_PRED_STEP_JUMP_DEG = 8.0
MAX_ARM_STEP_DEG = 15.0
MAX_ARM_STEP_RAD = np.deg2rad(MAX_ARM_STEP_DEG)
MAX_HAND_STEP_RAD = 1.0
MAX_CONSECUTIVE_REJECTS = 3
MAX_OBS_AGE_SEC = 0.15


# =============================================================================
# Custom πR² client adapter: official client -> our 3-RGB / 22D embodiment
# =============================================================================

class GR00T22DRGB3Client(GR00TClient):
    """Adapt the official πR² GR00TClient to this checkpoint's modalities."""

    _STATE_KEY_DIMS = {
        **GR00TClient._STATE_KEY_DIMS,
        "left_arm": 7,
        "right_arm": 7,
        "left_hand": 4,
        "right_hand": 4,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        cfg = self._call("get_modality_config", requires_input=False)
        action_cfg = cfg.get("action", {})
        self._action_keys = list(
            action_cfg.get(
                "modality_keys",
                ["left_arm", "right_arm", "left_hand", "right_hand"],
            )
        )

        expected_video = ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"]
        expected_state = ["left_arm", "right_arm", "left_hand", "right_hand"]
        expected_action = ["left_arm", "right_arm", "left_hand", "right_hand"]

        if list(self._video_keys) != expected_video:
            raise RuntimeError(
                f"server video keys={self._video_keys}, expected={expected_video}"
            )
        if list(self._state_keys) != expected_state:
            raise RuntimeError(
                f"server state keys={self._state_keys}, expected={expected_state}"
            )
        if list(self._action_keys) != expected_action:
            raise RuntimeError(
                f"server action keys={self._action_keys}, expected={expected_action}"
            )
        if self.expected_state_dim != 22:
            raise RuntimeError(
                f"server expected_state_dim={self.expected_state_dim}, expected=22"
            )

    def _build_obs(
        self,
        rgb_history: dict[str, np.ndarray],
        state_history: np.ndarray,
        language: str | None = None,
    ) -> dict:
        """
        Unlike upstream's one-camera helper, preserve all three camera streams.

        rgb_history[key]: (T,H,W,3) uint8
        state_history:     (T,22) float32
        """
        if not isinstance(rgb_history, dict):
            raise TypeError("rgb_history must be a dict of the 3 RGB histories")

        video_dict = {}
        for key in self._video_keys:
            if key not in rgb_history:
                raise RuntimeError(f"missing RGB history: {key}")

            arr = np.asarray(rgb_history[key])
            if arr.ndim != 4 or arr.shape[-1] != 3:
                raise RuntimeError(f"{key}: expected (T,H,W,3), got {arr.shape}")
            if arr.dtype != np.uint8:
                raise RuntimeError(f"{key}: expected uint8, got {arr.dtype}")
            if arr.shape[0] != self.video_T:
                raise RuntimeError(
                    f"{key}: T={arr.shape[0]}, model expects T={self.video_T}"
                )

            video_dict[key] = np.ascontiguousarray(arr[None])

        state_history = np.asarray(state_history, dtype=np.float32)
        if state_history.shape != (self.state_T, 22):
            raise RuntimeError(
                f"state_history expected ({self.state_T},22), got {state_history.shape}"
            )

        state_dict = self._build_state_dict(state_history)
        lang = language if language is not None else self._task
        lang_dict = {k: [[lang]] for k in self._lang_keys}

        return {
            "video": video_dict,
            "state": state_dict,
            "language": lang_dict,
        }

    def _parse_chunk_response(self, response) -> tuple[np.ndarray, dict]:
        """Convert four custom action keys into a flat (T,22) chunk."""
        if isinstance(response, (list, tuple)) and len(response) >= 2:
            action_dict, info = response[0], response[1]
        elif isinstance(response, (list, tuple)) and len(response) >= 1:
            action_dict, info = response[0], {}
        else:
            action_dict, info = response, {}

        if not isinstance(action_dict, dict):
            raise RuntimeError(
                f"server action response must be dict, got {type(action_dict)}"
            )

        parts = []
        T = None
        expected_dims = {
            "left_arm": 7,
            "right_arm": 7,
            "left_hand": 4,
            "right_hand": 4,
        }

        for key in self._action_keys:
            if key not in action_dict:
                raise RuntimeError(
                    f"action response missing {key}; got={list(action_dict)}"
                )

            arr = np.asarray(action_dict[key])
            if arr.ndim == 3:
                if arr.shape[0] != 1:
                    raise RuntimeError(f"{key}: bad batch shape {arr.shape}")
                arr = arr[0]

            if arr.ndim != 2 or arr.shape[1] != expected_dims[key]:
                raise RuntimeError(
                    f"{key}: expected (T,{expected_dims[key]}), got {arr.shape}"
                )

            if T is None:
                T = arr.shape[0]
            elif arr.shape[0] != T:
                raise RuntimeError(
                    f"action horizons differ: {key} has {arr.shape[0]}, expected {T}"
                )

            parts.append(arr.astype(np.float32))

        chunk = np.concatenate(parts, axis=-1)
        if chunk.ndim != 2 or chunk.shape[1] != 22:
            raise RuntimeError(f"bad flat action chunk shape: {chunk.shape}")
        if not np.all(np.isfinite(chunk)):
            raise RuntimeError("action chunk contains NaN/Inf")

        return np.ascontiguousarray(chunk), info if isinstance(info, dict) else {}


# =============================================================================
# Robot observation/action conversion
# =============================================================================

def obs_to_state22(obs: dict) -> np.ndarray:
    lh4 = base.compress_hand_6_to_4(obs["left_hand"])
    rh4 = base.compress_hand_6_to_4(obs["right_hand"])

    state = np.concatenate(
        [
            np.asarray(obs["left_arm"], dtype=np.float32).reshape(7),
            np.asarray(obs["right_arm"], dtype=np.float32).reshape(7),
            np.asarray(lh4, dtype=np.float32).reshape(4),
            np.asarray(rh4, dtype=np.float32).reshape(4),
        ],
        axis=0,
    )

    if state.shape != (22,) or not np.all(np.isfinite(state)):
        raise RuntimeError(f"invalid state22: shape={state.shape}")
    return state


def split_action22(action22: np.ndarray):
    x = np.asarray(action22, dtype=np.float64).reshape(22)
    if not np.all(np.isfinite(x)):
        raise RuntimeError("action22 contains NaN/Inf")
    return x[0:7], x[7:14], x[14:18], x[18:22]


def hand4_to_gripper_t(hand4):
    h = np.asarray(hand4, dtype=np.float64).reshape(4)
    y = np.array([h[1], h[2], h[3]], dtype=np.float64)
    coeff = np.array([0.6, 0.8, 0.8], dtype=np.float64)
    t = float(np.dot(coeff, y) / np.dot(coeff, coeff))
    return float(np.clip(t, 0.0, 1.0))


def hand4_gripper_t_to_hand6(hand4, t):
    h = np.asarray(hand4, dtype=np.float64).reshape(4)
    t = float(np.clip(t, 0.0, 1.0))
    return np.array(
        [h[0], 0.6 * t, 0.8 * t, 0.8 * t, 0.8 * t, 0.8 * t],
        dtype=np.float64,
    )


def make_snapshot(client: GR00T22DRGB3Client, obs: dict):
    # Current training config uses delta_indices=[0] for RGB and state.
    if client.video_T != 1 or client.state_T != 1:
        raise RuntimeError(
            "this controller expects video_T=state_T=1, but server reports "
            f"video_T={client.video_T}, state_T={client.state_T}"
        )

    rgbs = {
        "head_rgb": np.ascontiguousarray(obs["head_rgb"][None]),
        "left_wrist_rgb": np.ascontiguousarray(obs["left_wrist_rgb"][None]),
        "right_wrist_rgb": np.ascontiguousarray(obs["right_wrist_rgb"][None]),
    }
    state = obs_to_state22(obs)[None].astype(np.float32)
    return rgbs, state


# =============================================================================
# Safety conversion from model 22D -> physical robot targets
# =============================================================================

def safe_physical_action(raw22, prev_raw22, obs):
    pred_la, pred_ra, pred_lh4, pred_rh4 = split_action22(raw22)

    if prev_raw22 is not None:
        prev_la, prev_ra, _, _ = split_action22(prev_raw22)
        step_jump_deg = float(
            max(
                np.max(np.abs(np.rad2deg(pred_la - prev_la))),
                np.max(np.abs(np.rad2deg(pred_ra - prev_ra))),
            )
        )
    else:
        step_jump_deg = 0.0

    if step_jump_deg > MAX_PRED_STEP_JUMP_DEG:
        raise ValueError(
            f"step_jump={step_jump_deg:.2f}deg > {MAX_PRED_STEP_JUMP_DEG:.2f}deg"
        )

    raw_arm_deg = float(
        max(
            np.max(
                np.abs(
                    np.rad2deg(pred_la - np.asarray(obs["left_arm"], dtype=np.float64))
                )
            ),
            np.max(
                np.abs(
                    np.rad2deg(pred_ra - np.asarray(obs["right_arm"], dtype=np.float64))
                )
            ),
        )
    )

    if raw_arm_deg > RAW_ARM_GATE_DEG:
        raise ValueError(
            f"raw_arm={raw_arm_deg:.2f}deg > {RAW_ARM_GATE_DEG:.2f}deg"
        )

    lt = hand4_to_gripper_t(pred_lh4)
    rt = hand4_to_gripper_t(pred_rh4)
    pred_lh6 = hand4_gripper_t_to_hand6(pred_lh4, lt)
    pred_rh6 = hand4_gripper_t_to_hand6(pred_rh4, rt)

    cmd_la, _, safe_dla = base.rate_limit(
        pred_la, obs["left_arm"], MAX_ARM_STEP_RAD
    )
    cmd_ra, _, safe_dra = base.rate_limit(
        pred_ra, obs["right_arm"], MAX_ARM_STEP_RAD
    )
    cmd_lh6, _, safe_dlh6 = base.rate_limit(
        pred_lh6, obs["left_hand"], MAX_HAND_STEP_RAD
    )
    cmd_rh6, _, safe_drh6 = base.rate_limit(
        pred_rh6, obs["right_hand"], MAX_HAND_STEP_RAD
    )

    metrics = {
        "raw_arm_deg": raw_arm_deg,
        "step_jump_deg": step_jump_deg,
        "cmd_arm_deg": float(
            max(
                np.max(np.abs(np.rad2deg(safe_dla))),
                np.max(np.abs(np.rad2deg(safe_dra))),
            )
        ),
        "cmd_hand": float(
            max(
                np.max(np.abs(safe_dlh6)),
                np.max(np.abs(safe_drh6)),
            )
        ),
        "lt": lt,
        "rt": rt,
    }

    return cmd_la, cmd_ra, cmd_lh6, cmd_rh6, metrics


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--send",
        action="store_true",
        help="Actually send robot commands. Default is DRY RUN.",
    )
    ap.add_argument(
        "--seconds",
        type=float,
        default=20.0,
        help="Runtime in seconds. 0 = unlimited.",
    )
    ap.add_argument("--nfe", type=int, default=DEFAULT_NFE)
    ap.add_argument("--initial-d", type=int, default=DEFAULT_INITIAL_D)
    args = ap.parse_args()

    if not (1 <= args.initial_d <= TRAINED_D_MAX):
        raise SystemExit(
            f"--initial-d must be within training support 1..{TRAINED_D_MAX}"
        )

    np.set_printoptions(precision=3, suppress=True)

    print("=" * 100)
    print("πR² GR00T-N1.7 | RGB3 | 22D ABSOLUTE | DECOUPLED CONTINUOUS")
    print("=" * 100)
    print(f"robot        = {ROBOT_IP}:{ROBOT_PORT}")
    print(f"DiT server   = {DIT_HOST}:{DIT_PORT}")
    print(f"VLM server   = {VLM_HOST}:{VLM_PORT}")
    print(f"control      = {CONTROL_HZ:.1f} Hz")
    print(f"NFE          = {args.nfe}")
    print(f"initial d    = {args.initial_d}")
    print(f"trained dmax = {TRAINED_D_MAX}")
    print(f"mode         = {'REAL SEND' if args.send else 'DRY RUN'}")
    print("=" * 100)

    # Shared state.
    lock = threading.Lock()
    stop_event = threading.Event()
    latest_obs = {"value": None}
    latest_result = {"value": None, "seq": 0}
    fatal_error = {"value": None}
    query_ms = collections.deque(maxlen=20)

    def measured_d():
        if not query_ms:
            return args.initial_d, args.initial_d
        mean_ms = float(np.mean(query_ms))
        raw = max(1, int(np.round(mean_ms / (ACTION_DT * 1000.0))))
        return raw, min(TRAINED_D_MAX, raw)

    # -------------------------------------------------------------------------
    # Existing Pyro robot bridge
    # -------------------------------------------------------------------------

    base.configure_pyro()
    uri = f"PYRO:model_inference_bridge@{ROBOT_IP}:{ROBOT_PORT}"

    query_proxy = Pyro5.api.Proxy(uri)
    query_proxy._pyroTimeout = 10.0

    print("\n[1] robot bridge")
    print("ping       =", query_proxy.ping())
    print("obs status =", query_proxy.get_obs_status())

    cmd_proxy = None
    client_uuid = None

    if args.send:
        cmd_proxy = Pyro5.api.Proxy(uri)
        cmd_proxy._pyroTimeout = 10.0
        client_uuid = str(uuid.uuid4())
        print("\n" + "!" * 100)
        print("REAL ROBOT SEND ENABLED")
        print("This is a newly trained πR² checkpoint: start with a short --seconds run.")
        print("!" * 100)
    else:
        print("\n[DRY RUN] no command proxy created")

    # -------------------------------------------------------------------------
    # Official πR² 2-server client
    # -------------------------------------------------------------------------

    print("\n[2] connect πR² servers")
    client = GR00T22DRGB3Client(
        host=DIT_HOST,
        port=DIT_PORT,
        task=TASK_TEXT,
        vlm_host=VLM_HOST,
        vlm_port=VLM_PORT,
        control_period_ms=1000.0 / CONTROL_HZ,
    )

    if not client.ping():
        raise RuntimeError("DiT server ping failed")

    print("DiT ping    = OK")
    print("video keys  =", client._video_keys)
    print("state keys  =", client.state_keys)
    print("action keys =", client._action_keys)
    print("video_T     =", client.video_T)
    print("state_T     =", client.state_T)

    client.reset_streaming_buffer()
    print("streaming buffer reset = OK")

    # -------------------------------------------------------------------------
    # Observation worker
    # -------------------------------------------------------------------------

    def observation_worker():
        proxy = None
        try:
            proxy = Pyro5.api.Proxy(uri)
            proxy._pyroTimeout = 10.0
            period = 1.0 / OBS_HZ
            next_t = time.monotonic()

            while not stop_event.is_set():
                wire = proxy.get_observation()
                if wire is not None:
                    obs = base.decode_observation(wire)
                    obs["_recv_mono"] = time.monotonic()
                    obs["_recv_wall"] = time.time()
                    with lock:
                        latest_obs["value"] = obs

                next_t += period
                delay = next_t - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.monotonic()

        except Exception as e:
            with lock:
                fatal_error["value"] = (
                    "observation_worker",
                    repr(e),
                    traceback.format_exc(),
                )
            stop_event.set()
        finally:
            if proxy is not None:
                try:
                    proxy._pyroRelease()
                except Exception:
                    pass

    obs_thread = threading.Thread(
        target=observation_worker,
        daemon=True,
        name="robot-observation",
    )
    obs_thread.start()

    t_wait = time.monotonic()
    while True:
        with lock:
            obs0 = latest_obs["value"]
            err = fatal_error["value"]
        if err is not None:
            name, repr_e, tb = err
            raise RuntimeError(f"{name}: {repr_e}\n{tb}")
        if obs0 is not None:
            break
        if time.monotonic() - t_wait > 10.0:
            raise RuntimeError("timed out waiting for robot observation")
        time.sleep(0.01)

    # -------------------------------------------------------------------------
    # πR² warm start + client-side VLM feature cache prime
    # -------------------------------------------------------------------------

    print("\n[3] πR² warm-start")
    rgbs0, state0 = make_snapshot(client, obs0)

    t0 = time.perf_counter()
    first_chunk, seed_info = client.seed_streaming_from_obs(
        rgbs0,
        state0,
        TASK_TEXT,
        t_image_capture=float(obs0["_recv_wall"]),
        slide_steps=args.initial_d,
    )
    seed_ms = (time.perf_counter() - t0) * 1000.0

    if first_chunk.ndim != 2 or first_chunk.shape[1] != 22:
        raise RuntimeError(f"bad warm-start chunk shape: {first_chunk.shape}")

    print(f"warm-start = OK | chunk={first_chunk.shape} | {seed_ms:.1f} ms")

    print("\n[4] prime 2-server VLM cache")
    client.update_vlm_cache(
        rgbs0,
        state0,
        TASK_TEXT,
        t_image_capture=float(obs0["_recv_wall"]),
    )
    print("VLM cache prime = OK")

    # -------------------------------------------------------------------------
    # Async VLM worker
    # -------------------------------------------------------------------------

    def vlm_worker():
        last_ts = None
        try:
            while not stop_event.is_set():
                with lock:
                    obs = latest_obs["value"]
                if obs is None:
                    time.sleep(0.005)
                    continue

                ts = int(obs["timestamp_ns"])
                if ts == last_ts:
                    time.sleep(0.002)
                    continue
                last_ts = ts

                rgbs, state = make_snapshot(client, obs)
                client.update_vlm_cache(
                    rgbs,
                    state,
                    TASK_TEXT,
                    t_image_capture=float(obs["_recv_wall"]),
                )

        except Exception as e:
            with lock:
                fatal_error["value"] = (
                    "vlm_worker",
                    repr(e),
                    traceback.format_exc(),
                )
            stop_event.set()

    # -------------------------------------------------------------------------
    # Continuous DiT worker
    # -------------------------------------------------------------------------

    def dit_worker():
        try:
            while not stop_event.is_set():
                with lock:
                    obs = latest_obs["value"]
                if obs is None:
                    time.sleep(0.005)
                    continue

                raw_d, d_eff = measured_d()
                rgbs, state = make_snapshot(client, obs)

                q0 = time.perf_counter()
                chunk, info = client.get_action_chunk_cached(
                    rgbs,
                    state,
                    num_inference_timesteps=args.nfe,
                    slide_steps=d_eff,
                    inpaint_actions=None,
                    force_nonstreaming=False,
                    t_state_capture=float(obs["_recv_wall"]),
                )
                elapsed_ms = (time.perf_counter() - q0) * 1000.0
                query_ms.append(elapsed_ms)

                if chunk.ndim != 2 or chunk.shape[1] != 22:
                    raise RuntimeError(
                        f"bad cached πR² chunk shape {chunk.shape}; expected (T,22)"
                    )

                raw_after, clipped_after = measured_d()

                result = {
                    "chunk": np.ascontiguousarray(chunk),
                    # IMPORTANT: the server slid by THIS d for this result.
                    "d": int(d_eff),
                    "query_ms": float(elapsed_ms),
                    "raw_d_after": int(raw_after),
                    "clipped_d_after": int(clipped_after),
                    "info": info,
                }

                with lock:
                    latest_result["value"] = result
                    latest_result["seq"] += 1

                # One server slide per approximately d control ticks.
                target_s = d_eff * ACTION_DT
                spent_s = elapsed_ms / 1000.0
                if spent_s < target_s:
                    time.sleep(target_s - spent_s)

        except Exception as e:
            with lock:
                fatal_error["value"] = (
                    "dit_worker",
                    repr(e),
                    traceback.format_exc(),
                )
            stop_event.set()

    vlm_thread = threading.Thread(
        target=vlm_worker,
        daemon=True,
        name="pir2-vlm-cache",
    )
    dit_thread = threading.Thread(
        target=dit_worker,
        daemon=True,
        name="pir2-dit",
    )

    vlm_thread.start()
    dit_thread.start()

    # πR² streaming calls produce a d-wide freshly cleaned region [d, 2d).
    current_chunk = first_chunk
    current_d = int(args.initial_d)
    chunk_idx = min(current_d, len(current_chunk) - 1)
    swap_target = min(chunk_idx + current_d, len(current_chunk))

    prev_raw22 = None
    last_result_seq = 0
    consecutive_rejects = 0

    print("\n[5] workers started")
    print(f"initial emit window = [{chunk_idx}, {swap_target})")
    print("\n[6] 20 Hz controller started")
    print("[LIVE] commands WILL be sent" if args.send else "[DRY RUN] NOTHING will be sent")

    start_mono = time.monotonic()
    next_tick = start_mono
    step = 0

    try:
        while not stop_event.is_set():
            now = time.monotonic()
            if args.seconds > 0 and now - start_mono >= args.seconds:
                print("\n[STOP] requested runtime reached")
                break

            if now < next_tick:
                time.sleep(next_tick - now)
            now = time.monotonic()
            next_tick += ACTION_DT
            step += 1

            with lock:
                obs = latest_obs["value"]
                pending = latest_result["value"]
                pending_seq = latest_result["seq"]
                err = fatal_error["value"]

            if err is not None:
                name, repr_e, tb = err
                raise RuntimeError(f"{name}: {repr_e}\n{tb}")

            if obs is None:
                print(f"[{step:05d}] WAIT obs")
                continue

            obs_age = now - float(obs["_recv_mono"])
            if obs_age > MAX_OBS_AGE_SEC:
                consecutive_rejects += 1
                print(
                    f"[{step:05d}] REJECT stale_obs age={obs_age:.3f}s | "
                    f"reject={consecutive_rejects}/{MAX_CONSECUTIVE_REJECTS}"
                )
                if consecutive_rejects >= MAX_CONSECUTIVE_REJECTS:
                    raise RuntimeError("Safety stop: observation stale")
                continue

            # Swap only after the current d-wide fresh region has been consumed.
            if chunk_idx >= swap_target:
                if pending is not None and pending_seq > last_result_seq:
                    current_chunk = pending["chunk"]
                    current_d = int(pending["d"])
                    chunk_idx = min(current_d, len(current_chunk) - 1)
                    swap_target = min(chunk_idx + current_d, len(current_chunk))
                    last_result_seq = pending_seq

                    print(
                        f"[SWAP] d={current_d} emit=[{chunk_idx},{swap_target}) | "
                        f"query={pending['query_ms']:.1f}ms | "
                        f"raw_d_now={pending['raw_d_after']}",
                        flush=True,
                    )
                else:
                    # Never advance into the under-denoised tail. Hold the last
                    # already-accepted model target until a fresh result arrives.
                    if prev_raw22 is None:
                        print(f"[{step:05d}] WAIT new πR² chunk")
                        continue

                    raw22 = prev_raw22.copy()
                    try:
                        cmd_la, cmd_ra, cmd_lh6, cmd_rh6, metrics = safe_physical_action(
                            raw22, None, obs
                        )
                    except ValueError as e:
                        consecutive_rejects += 1
                        print(
                            f"[{step:05d}] REJECT hold: {e} | "
                            f"reject={consecutive_rejects}/{MAX_CONSECUTIVE_REJECTS}"
                        )
                        if consecutive_rejects >= MAX_CONSECUTIVE_REJECTS:
                            raise RuntimeError("Safety stop during hold")
                        continue

                    consecutive_rejects = 0
                    if args.send:
                        rpc = base.send_targets(
                            cmd_proxy,
                            client_uuid,
                            left_arm=cmd_la,
                            right_arm=cmd_ra,
                            left_hand=cmd_lh6,
                            right_hand=cmd_rh6,
                        )
                    else:
                        rpc = "DRY"

                    if step % 20 == 0:
                        print(
                            f"[{step:05d}] {'SEND' if args.send else 'DRY '} HOLD | "
                            f"d={current_d} | rpc={rpc}"
                        )
                    continue

            raw22 = np.asarray(current_chunk[chunk_idx], dtype=np.float64)

            try:
                cmd_la, cmd_ra, cmd_lh6, cmd_rh6, metrics = safe_physical_action(
                    raw22, prev_raw22, obs
                )
            except ValueError as e:
                consecutive_rejects += 1
                print(
                    f"[{step:05d}] REJECT {e} | d={current_d} "
                    f"idx={chunk_idx}/{len(current_chunk)} | "
                    f"reject={consecutive_rejects}/{MAX_CONSECUTIVE_REJECTS}"
                )
                if consecutive_rejects >= MAX_CONSECUTIVE_REJECTS:
                    raise RuntimeError("Safety stop: repeated action rejection")
                continue

            consecutive_rejects = 0

            if args.send:
                rpc = base.send_targets(
                    cmd_proxy,
                    client_uuid,
                    left_arm=cmd_la,
                    right_arm=cmd_ra,
                    left_hand=cmd_lh6,
                    right_hand=cmd_rh6,
                )
            else:
                rpc = "DRY"

            prev_raw22 = raw22.copy()
            emitted_idx = chunk_idx
            chunk_idx += 1

            raw_d, clipped_d = measured_d()
            info = pending["info"] if pending is not None else {}
            img_delay_ticks = info.get("image_delay_ticks")
            img_delay_ms = info.get("image_delay_ms")

            if step % 20 == 0 or chunk_idx >= swap_target:
                print(
                    f"[{step:05d}] {'SEND' if args.send else 'DRY '} | "
                    f"idx={emitted_idx:02d}/{len(current_chunk):02d} "
                    f"emit_end={swap_target:02d} | "
                    f"d={current_d} measured={raw_d} clipped={clipped_d} | "
                    f"qmean={float(np.mean(query_ms)) if query_ms else 0.0:.1f}ms | "
                    f"img_delay_ticks={img_delay_ticks} img_delay_ms={img_delay_ms} | "
                    f"raw_arm={metrics['raw_arm_deg']:.2f}deg | "
                    f"jump={metrics['step_jump_deg']:.2f}deg | "
                    f"cmd_arm={metrics['cmd_arm_deg']:.2f}deg | "
                    f"cmd_hand={metrics['cmd_hand']:.3f} | "
                    f"LT={metrics['lt']:.3f} RT={metrics['rt']:.3f} | "
                    f"rpc={rpc}",
                    flush=True,
                )

            if raw_d > TRAINED_D_MAX:
                msg = (
                    f"measured latency d={raw_d} exceeds training support "
                    f"dmax={TRAINED_D_MAX}"
                )
                if args.send:
                    raise RuntimeError("Safety stop: " + msg)
                print("[WARN] " + msg, flush=True)

            if img_delay_ticks is not None and float(img_delay_ticks) > TRAINED_IMAGE_DELAY_MAX:
                print(
                    f"[WARN] image_delay_ticks={img_delay_ticks} exceeds "
                    f"training support {TRAINED_IMAGE_DELAY_MAX}",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C")

    except Exception:
        print("\n[ERROR]")
        traceback.print_exc()

    finally:
        stop_event.set()
        print("\nStopping workers...")

        obs_thread.join(timeout=2.0)
        vlm_thread.join(timeout=2.0)
        dit_thread.join(timeout=3.0)

        if args.send and cmd_proxy is not None and client_uuid is not None:
            print("Releasing robot control lease...")
            try:
                cmd_proxy.release_control_lease(client_uuid)
            except Exception as e:
                print("release_control_lease:", repr(e))
            try:
                cmd_proxy._pyroRelease()
            except Exception:
                pass

        try:
            query_proxy._pyroRelease()
        except Exception:
            pass

        try:
            client.close()
        except Exception:
            pass

        print("[DONE]")


if __name__ == "__main__":
    main()
