"""Synchronous GR00T plain-flow baseline for the same dual-arm bridge."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import Pyro5.api

import robot_bridge as bridge
from runtime_paths import ROOT

MODEL_DIMS = {"left_arm": 7, "right_arm": 7, "left_hand": 4, "right_hand": 4}


def model_obs(obs, language_key, task, cameras):
    states = {}
    for name, dim in MODEL_DIMS.items():
        values = obs[name] if dim == 7 else bridge.compress_hand_6_to_4(obs[name])
        states[name] = np.asarray(values, dtype=np.float32)[None, None, :]
    return {
        "video": {key: obs[key][None, None] for key in cameras},
        "state": states,
        "language": {language_key: [[task]]},
    }


def targets_for_first_action(action, obs):
    targets = {}
    for name, dim in MODEL_DIMS.items():
        values = np.asarray(action[name])
        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]
        if values.ndim != 2 or values.shape[1] != dim or len(values) == 0:
            raise ValueError(f"Unexpected action chunk shape for {name}: {values.shape}")
        target = values[0].astype(np.float64)
        if not np.isfinite(target).all():
            raise ValueError(f"Non-finite model output for {name}")
        targets[name] = target if dim == 7 else bridge.expand_hand_4_to_6(target)
    arm_error = max(np.max(np.abs(np.rad2deg(targets[k] - obs[k]))) for k in ("left_arm", "right_arm"))
    if arm_error > 15.0:
        return None, arm_error
    for name, target in targets.items():
        limit = np.deg2rad(2) if MODEL_DIMS[name] == 7 else 0.05
        targets[name], _, _ = bridge.rate_limit(target, obs[name], limit)
    return targets, arm_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--robot-host", required=True, help="Bridge address, e.g. 10.0.0.11")
    parser.add_argument("--robot-port", type=int, default=7788)
    parser.add_argument("--task", default="快递分拣")
    parser.add_argument("--cameras", choices=("rgb3", "head"), default="rgb3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nfe", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=20, help="0 = run until interrupted")
    parser.add_argument("--send", action="store_true", help="Transmit robot commands")
    args = parser.parse_args()
    if args.nfe < 1 or not np.isfinite(args.seconds) or args.seconds < 0:
        parser.error("--nfe must be positive and --seconds finite and nonnegative")
    sys.path.insert(0, str(ROOT / "third_party" / "groot_deploy"))
    os.environ.setdefault("GR00T_COSMOS_LOCAL_PATH", str(ROOT / "weights" / "backbone"))
    import torch
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    np.random.seed(42)
    torch.manual_seed(42)
    policy = Gr00tPolicy(embodiment_tag="NEW_EMBODIMENT", model_path=str(args.checkpoint.resolve()),
                         device=args.device)
    cameras = bridge.CAMERAS if args.cameras == "rgb3" else ("head_rgb",)
    for modality, expected in (("video", cameras), ("state", MODEL_DIMS), ("action", MODEL_DIMS)):
        actual = list(policy.modality_configs[modality].modality_keys)
        if actual != list(expected):
            raise ValueError(f"Checkpoint {modality} keys {actual} differ from {list(expected)}")
    policy.model.action_head.num_inference_timesteps = args.nfe
    bridge.configure_pyro()
    uri = f"PYRO:model_inference_bridge@{args.robot_host}:{args.robot_port}"
    client_id = str(uuid.uuid4())
    print(f"GR00T {args.cameras}", "LIVE" if args.send else "DRY RUN", uri, flush=True)
    with ExitStack() as stack:
        stream = stack.enter_context(Pyro5.api.Proxy(uri))
        stream._pyroTimeout = 10
        print("Bridge", stream.ping(), stream.get_obs_status(), flush=True)
        command = None
        if args.send:
            command = stack.enter_context(Pyro5.api.Proxy(uri))
            command._pyroTimeout = 10
            stack.callback(command.release_control_lease, client_id)
        rejects = 0
        previous_timestamp = None
        start = time.monotonic()
        while not args.seconds or time.monotonic() - start < args.seconds:
            wire = stream.get_observation()
            if wire is None or (previous_timestamp is not None and int(wire["timestamp_ns"]) <= previous_timestamp):
                time.sleep(0.05)
                continue
            obs = bridge.decode_observation(wire, cameras)
            previous_timestamp = obs["timestamp_ns"]
            t0 = time.perf_counter()
            with torch.inference_mode():
                action, _ = policy.get_action(model_obs(obs, policy.language_key, args.task, cameras))
            targets, error = targets_for_first_action(action, obs)
            if targets is None:
                rejects += 1
                print(f"REJECT raw arm deviation {error:.2f}° ({rejects}/3)", flush=True)
                if rejects >= 3:
                    raise RuntimeError("Three consecutive arm gate violations")
                continue
            rejects = 0
            if command is not None:
                bridge.send_targets(command, client_id, **targets)
            print(f"{'SEND' if args.send else 'DRY'} infer+limit={time.perf_counter()-t0:.3f}s arm={error:.2f}°",
                  flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
