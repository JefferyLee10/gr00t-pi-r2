#!/usr/bin/env python3
"""Score independent 16-step action predictions against recorded 22D targets."""

import argparse
import json
import os
from pathlib import Path
import random


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path, help="Complete 22D v2.1 dataset")
    parser.add_argument("--split-json", type=Path, help="Defaults to DATASET/meta/split.json")
    parser.add_argument("--episode", action="append", type=int, help="Explicit episode ID; repeatable. Overrides validation split.")
    parser.add_argument("--max-windows", type=positive_int, help="Maximum number of prediction windows across all selected episodes")
    parser.add_argument("--stride", type=positive_int, default=16, help="Frames between window starts")
    parser.add_argument("--denoise", type=positive_int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("GR00T_COSMOS_LOCAL_PATH", str(root / "weights/backbone"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    import numpy as np
    import torch
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    episodes = args.episode
    if episodes is None:
        split = args.split_json or args.dataset / "meta/split.json"
        episodes = json.loads(split.read_text())["val_episode_indices"]
    if not episodes or len(set(episodes)) != len(episodes):
        raise ValueError("Select at least one episode and provide unique IDs.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy = Gr00tPolicy("NEW_EMBODIMENT", str(args.checkpoint), device=args.device)
    # The upstream policy does not forward per-call options. Use the same full
    # denoising path as warm-start to evaluate each window independently.
    policy.model.action_head.config.streaming = False
    policy.model.action_head.num_inference_timesteps = args.denoise
    modalities = policy.get_modality_config()
    groups = {"left_arm": 7, "right_arm": 7, "left_hand": 4, "right_hand": 4}
    if modalities["action"].modality_keys != list(groups) or list(modalities["action"].delta_indices) != list(range(16)):
        raise ValueError("Expected the 16-step absolute 22D policy for this robot.")
    loader = LeRobotEpisodeLoader(args.dataset, modalities, video_backend="torchcodec")
    missing_cameras = set(modalities["video"].modality_keys) - loader.modality_meta["video"].keys()
    if missing_cameras:
        raise ValueError(f"Missing checkpoint camera names in dataset: {sorted(missing_cameras)}")
    lengths = {row["episode_index"]: row["length"] for row in loader.episodes_metadata}
    missing = set(episodes) - lengths.keys()
    if missing:
        raise ValueError(f"Episode IDs missing from dataset: {sorted(missing)}")
    language_key = modalities["language"].modality_keys[0]
    sums_abs = {g: 0.0 for g in groups}
    sums_sq = {g: 0.0 for g in groups}
    windows = 0
    per_episode = {}
    for episode in episodes:
        # Read joint groups and task annotations through the pinned training loader.
        # Decode only the selected observation frames, in bounded-size batches.
        frame_data = loader._load_parquet_data(episode)
        if len(frame_data) != lengths[episode]:
            raise ValueError(f"Episode {episode}: parquet length disagrees with metadata")
        starts = range(0, len(frame_data) - 15, args.stride)
        if args.max_windows:
            starts = starts[:args.max_windows - windows]
        per_episode[episode] = len(starts)
        for offset in range(0, len(starts), 16):
            batch = starts[offset:offset + 16]
            videos = loader._load_video_data(episode, np.array(batch))
            for i, start in enumerate(batch):
                state = {g: np.asarray(frame_data[f"state.{g}"].iloc[start], dtype=np.float32)[None, None] for g in groups}
                observation = {
                    "state": state,
                    "video": {key: frames[i][None, None] for key, frames in videos.items()},
                    "language": {language_key: [[frame_data[f"language.{language_key}"].iloc[start]]]},
                }
                prediction, _ = policy.get_action(observation)
                for group, width in groups.items():
                    truth = np.stack(frame_data[f"action.{group}"].iloc[start:start + 16])
                    target_shape = (16, width)
                    if truth.shape != target_shape or prediction[group].shape != (1, *target_shape):
                        raise ValueError(f"Unexpected {group} shape for episode {episode} at {start}")
                    diff = prediction[group][0].astype(np.float64) - truth
                    if not np.all(np.isfinite(diff)):
                        raise ValueError(f"Non-finite {group} error for episode {episode} at {start}")
                    sums_abs[group] += np.abs(diff).sum()
                    sums_sq[group] += np.square(diff).sum()
                windows += 1
        if args.max_windows and windows >= args.max_windows:
            break
    if not windows:
        raise ValueError("Selected episodes contain no complete 16-frame window.")
    total_frames = windows * 16
    result = {
        "protocol": "open_loop_independent_full_horizon",
        "checkpoint": args.checkpoint.name,
        "selection": "explicit_episode_ids" if args.episode else "validation_split",
        "windows_per_episode": per_episode, "windows": windows, "horizon": 16,
        "stride": args.stride, "denoising_steps": args.denoise, "seed": args.seed,
        "mae": sum(sums_abs.values()) / (total_frames * 22),
        "rmse": (sum(sums_sq.values()) / (total_frames * 22)) ** 0.5,
        "by_group": {g: {"mae": sums_abs[g] / (total_frames * width),
                         "rmse": (sums_sq[g] / (total_frames * width)) ** 0.5}
                     for g, width in groups.items()},
    }
    report = json.dumps(result, indent=2, allow_nan=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
