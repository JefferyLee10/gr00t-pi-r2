#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 3RGB + 28D LeRobot v2.1 数据集转换为 3RGB + 22D exact 数据集。

28D:
    [0:7]    left_arm
    [7]      left_gripper        -> 删除
    [8:15]   right_arm
    [15]     right_gripper       -> 删除
    [16:22]  left_hand 6D
    [22:28]  right_hand 6D

每只灵巧手：
    [j0, j1, j2, j3, j4, j5]
        ->
    [j0, j1, j2, mean(j3,j4,j5)]

最终22D：
    [0:7]    left_arm 7D
    [7:14]   right_arm 7D
    [14:18]  left_hand 4D
    [18:22]  right_hand 4D

安全原则：
- source 永远只读
- dst 已存在则直接退出
- 不提供 overwrite
- RGB 不转码
- RGB 使用 hardlink，几乎不增加磁盘占用
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


RGB_KEYS = [
    "observation.images.head_rgb",
    "observation.images.left_wrist_rgb",
    "observation.images.right_wrist_rgb",
]

STATE_KEY = "observation.state"
ACTION_KEY = "action"

NEW_NAMES = [
    *[f"left_arm_j{i}" for i in range(7)],
    *[f"right_arm_j{i}" for i in range(7)],
    "left_hand_j0",
    "left_hand_j1",
    "left_hand_j2",
    "left_hand_j345_mean",
    "right_hand_j0",
    "right_hand_j1",
    "right_hand_j2",
    "right_hand_j345_mean",
]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--src",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--dst",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--limit-episodes",
        type=int,
        default=None,
        help="仅用于 smoke test，取前 N 个 episode。",
    )

    return p.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")


def load_jsonl(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def map_28_to_22(x: np.ndarray) -> np.ndarray:
    """
    x: [N, 28]
    return: [N, 22]
    """

    if x.ndim != 2 or x.shape[1] != 28:
        raise RuntimeError(
            f"Expected [N,28], got {x.shape}"
        )

    left_arm = x[:, 0:7]
    right_arm = x[:, 8:15]

    left_hand_first3 = x[:, 16:19]
    left_hand_other = np.mean(
        x[:, 19:22],
        axis=1,
        keepdims=True,
        dtype=np.float32,
    )

    right_hand_first3 = x[:, 22:25]
    right_hand_other = np.mean(
        x[:, 25:28],
        axis=1,
        keepdims=True,
        dtype=np.float32,
    )

    y = np.concatenate(
        [
            left_arm,
            right_arm,
            left_hand_first3,
            left_hand_other,
            right_hand_first3,
            right_hand_other,
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    if y.shape[1] != 22:
        raise RuntimeError(
            f"Internal mapping error: {y.shape}"
        )

    return y


def make_arrow_22(x: np.ndarray):
    flat = pa.array(
        x.reshape(-1),
        type=pa.float32(),
    )

    return pa.FixedSizeListArray.from_arrays(
        flat,
        22,
    )


def numeric_stats(x: np.ndarray):
    if x.ndim != 2 or x.shape[1] != 22:
        raise RuntimeError(
            f"stats input must be [N,22], got {x.shape}"
        )

    if not np.all(np.isfinite(x)):
        raise RuntimeError(
            "NaN/Inf detected while computing stats"
        )

    q = np.quantile(
        x,
        [0.01, 0.10, 0.50, 0.90, 0.99],
        axis=0,
    )

    return {
        "min": x.min(axis=0).astype(float).tolist(),
        "max": x.max(axis=0).astype(float).tolist(),
        "mean": x.mean(axis=0).astype(float).tolist(),
        "std": x.std(axis=0).astype(float).tolist(),
        "count": [int(x.shape[0])],
        "q01": q[0].astype(float).tolist(),
        "q10": q[1].astype(float).tolist(),
        "q50": q[2].astype(float).tolist(),
        "q90": q[3].astype(float).tolist(),
        "q99": q[4].astype(float).tolist(),
    }


def validate_source(src: Path):
    required = [
        src / "meta/info.json",
        src / "meta/modality.json",
        src / "meta/tasks.jsonl",
        src / "meta/episodes.jsonl",
        src / "meta/episodes_stats.jsonl",
        src / "meta/stats.json",
        src / "meta/split.json",
        src / "data",
        src / "videos",
    ]

    for p in required:
        if not p.exists():
            raise FileNotFoundError(
                f"Required source path missing: {p}"
            )

    info = load_json(src / "meta/info.json")

    if info["codebase_version"] != "v2.1":
        raise RuntimeError(
            "Source must be LeRobot v2.1"
        )

    if info["features"][STATE_KEY]["shape"] != [28]:
        raise RuntimeError(
            "Source state is not 28D"
        )

    if info["features"][ACTION_KEY]["shape"] != [28]:
        raise RuntimeError(
            "Source action is not 28D"
        )

    video_keys = {
        k
        for k, v in info["features"].items()
        if v.get("dtype") == "video"
    }

    if video_keys != set(RGB_KEYS):
        raise RuntimeError(
            f"Expected exactly 3 RGB videos, got {video_keys}"
        )

    return info


def prepare_destination(dst: Path):
    if dst.exists():
        raise RuntimeError(
            f"Destination already exists; refusing to overwrite: {dst}"
        )

    dst.mkdir(
        parents=True,
        exist_ok=False,
    )


def select_episodes(src: Path, limit):
    rows = load_jsonl(
        src / "meta/episodes.jsonl"
    )

    rows.sort(
        key=lambda x: int(x["episode_index"])
    )

    if limit is not None:
        if limit <= 0:
            raise ValueError(
                "--limit-episodes must be > 0"
            )

        rows = rows[:limit]

    indices = [
        int(x["episode_index"])
        for x in rows
    ]

    expected = list(range(len(rows)))

    if indices != expected:
        raise RuntimeError(
            "Selected episodes are not contiguous from index 0"
        )

    return rows


def build_info(
    src: Path,
    dst: Path,
    src_info,
    episodes,
):
    info = copy.deepcopy(src_info)

    n_ep = len(episodes)
    n_frames = sum(
        int(x["length"])
        for x in episodes
    )

    info["total_episodes"] = n_ep
    info["total_frames"] = n_frames

    chunks_size = int(
        info.get("chunks_size", 1000)
    )

    info["total_chunks"] = math.ceil(
        n_ep / chunks_size
    )

    info["total_videos"] = (
        n_ep * len(RGB_KEYS)
    )

    info["splits"] = {
        "train": f"0:{n_ep}"
    }

    for key in [STATE_KEY, ACTION_KEY]:
        info["features"][key]["shape"] = [22]
        info["features"][key]["names"] = NEW_NAMES

    dump_json(
        dst / "meta/info.json",
        info,
    )

    return chunks_size


def build_modality(src: Path, dst: Path):
    old = load_json(
        src / "meta/modality.json"
    )

    def group_from(name):
        x = copy.deepcopy(
            old[name]
        )
        return x

    new = {
        "state": {
            "left_arm": group_from(
                "state"
            )["left_arm"],
            "right_arm": group_from(
                "state"
            )["right_arm"],
            "left_hand": group_from(
                "state"
            )["left_hand"],
            "right_hand": group_from(
                "state"
            )["right_hand"],
        },
        "action": {
            "left_arm": group_from(
                "action"
            )["left_arm"],
            "right_arm": group_from(
                "action"
            )["right_arm"],
            "left_hand": group_from(
                "action"
            )["left_hand"],
            "right_hand": group_from(
                "action"
            )["right_hand"],
        },
        "video": copy.deepcopy(
            old["video"]
        ),
        "annotation": copy.deepcopy(
            old["annotation"]
        ),
    }

    for category in ["state", "action"]:
        new[category]["left_arm"]["start"] = 0
        new[category]["left_arm"]["end"] = 7

        new[category]["right_arm"]["start"] = 7
        new[category]["right_arm"]["end"] = 14

        new[category]["left_hand"]["start"] = 14
        new[category]["left_hand"]["end"] = 18

        new[category]["right_hand"]["start"] = 18
        new[category]["right_hand"]["end"] = 22

    dump_json(
        dst / "meta/modality.json",
        new,
    )


def build_split(src: Path, dst: Path, selected):
    selected_ids = {
        int(x["episode_index"])
        for x in selected
    }

    old = load_json(
        src / "meta/split.json"
    )

    episode_rows = [
        x
        for x in old["episodes"]
        if int(x["episode_index"])
        in selected_ids
    ]

    episode_rows.sort(
        key=lambda x: int(x["episode_index"])
    )

    train = [
        int(x["episode_index"])
        for x in episode_rows
        if x["split"] == "train"
    ]

    val = [
        int(x["episode_index"])
        for x in episode_rows
        if x["split"] == "val"
    ]

    val_names = [
        x["episode_name"]
        for x in episode_rows
        if x["split"] == "val"
    ]

    new = {
        "val_episodes": val_names,
        "val_episode_indices": val,
        "train_episode_indices": train,
        "episodes": episode_rows,
    }

    dump_json(
        dst / "meta/split.json",
        new,
    )

    print(
        f"[OK] split: "
        f"train={len(train)}, val={len(val)}"
    )


def transform_one_table(table):
    state = np.asarray(
        table[STATE_KEY].to_pylist(),
        dtype=np.float32,
    )

    action = np.asarray(
        table[ACTION_KEY].to_pylist(),
        dtype=np.float32,
    )

    if state.shape[1] != 28:
        raise RuntimeError(
            f"Unexpected state shape {state.shape}"
        )

    if action.shape[1] != 28:
        raise RuntimeError(
            f"Unexpected action shape {action.shape}"
        )

    if not np.all(np.isfinite(state)):
        raise RuntimeError(
            "NaN/Inf found in state"
        )

    if not np.all(np.isfinite(action)):
        raise RuntimeError(
            "NaN/Inf found in action"
        )

    state22 = map_28_to_22(state)
    action22 = map_28_to_22(action)

    state_idx = table.schema.get_field_index(
        STATE_KEY
    )

    action_idx = table.schema.get_field_index(
        ACTION_KEY
    )

    table = table.set_column(
        state_idx,
        STATE_KEY,
        make_arrow_22(state22),
    )

    table = table.set_column(
        action_idx,
        ACTION_KEY,
        make_arrow_22(action22),
    )

    return table, state22, action22


def load_source_episode_stats(src: Path):
    rows = load_jsonl(
        src / "meta/episodes_stats.jsonl"
    )

    return {
        int(x["episode_index"]): x["stats"]
        for x in rows
    }


def convert_parquets(
    src,
    dst,
    episodes,
    chunks_size,
):
    src_ep_stats = load_source_episode_stats(
        src
    )

    out_ep_stats = []

    all_state = []
    all_action = []

    for n, episode in enumerate(episodes):
        idx = int(
            episode["episode_index"]
        )

        chunk = idx // chunks_size

        src_file = (
            src
            / "data"
            / f"chunk-{chunk:03d}"
            / f"episode_{idx:06d}.parquet"
        )

        dst_file = (
            dst
            / "data"
            / f"chunk-{chunk:03d}"
            / f"episode_{idx:06d}.parquet"
        )

        if not src_file.exists():
            raise FileNotFoundError(
                src_file
            )

        table = pq.read_table(
            src_file
        )

        table22, state22, action22 = (
            transform_one_table(table)
        )

        if table22.num_rows != int(
            episode["length"]
        ):
            raise RuntimeError(
                f"Episode {idx}: row count mismatch"
            )

        dst_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pq.write_table(
            table22,
            dst_file,
        )

        all_state.append(
            state22
        )

        all_action.append(
            action22
        )

        old_stats = src_ep_stats.get(
            idx,
            {}
        )

        ep_stats = {}

        for key in RGB_KEYS:
            if key in old_stats:
                ep_stats[key] = copy.deepcopy(
                    old_stats[key]
                )

        ep_stats[STATE_KEY] = numeric_stats(
            state22
        )

        ep_stats[ACTION_KEY] = numeric_stats(
            action22
        )

        out_ep_stats.append(
            {
                "episode_index": idx,
                "stats": ep_stats,
            }
        )

        if (
            (n + 1) % 50 == 0
            or n + 1 == len(episodes)
        ):
            print(
                f"[DATA] {n+1}/{len(episodes)}"
            )

    write_jsonl(
        dst / "meta/episodes_stats.jsonl",
        out_ep_stats,
    )

    global_state = np.concatenate(
        all_state,
        axis=0,
    )

    global_action = np.concatenate(
        all_action,
        axis=0,
    )

    return (
        numeric_stats(global_state),
        numeric_stats(global_action),
    )


def build_global_stats(
    src,
    dst,
    state_stats,
    action_stats,
    full_dataset,
):
    out = {
        STATE_KEY: state_stats,
        ACTION_KEY: action_stats,
    }

    # 全量转换时 RGB 完全没变，
    # 可以安全沿用 Stage-1 的 RGB stats。
    if full_dataset:
        old = load_json(
            src / "meta/stats.json"
        )

        for key in RGB_KEYS:
            if key in old:
                out[key] = old[key]

    dump_json(
        dst / "meta/stats.json",
        out,
    )

    warning = (
        dst
        / "meta/STAGE2_STATS_WARNING.txt"
    )

    warning.write_text(
        "State/action are now correct 22D stats, "
        "but they were computed over all selected episodes.\n"
        "They are NOT the final training normalization stats.\n"
        "Before piR2 training, recompute stats using TRAIN episodes only.\n",
        encoding="utf-8",
    )


def link_videos(
    src,
    dst,
    episodes,
    chunks_size,
):
    count = 0

    for episode in episodes:
        idx = int(
            episode["episode_index"]
        )

        chunk = idx // chunks_size

        for key in RGB_KEYS:
            src_file = (
                src
                / "videos"
                / f"chunk-{chunk:03d}"
                / key
                / f"episode_{idx:06d}.mp4"
            )

            dst_file = (
                dst
                / "videos"
                / f"chunk-{chunk:03d}"
                / key
                / f"episode_{idx:06d}.mp4"
            )

            if not src_file.exists():
                raise FileNotFoundError(
                    src_file
                )

            dst_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            # hardlink:
            # - 不复制视频内容
            # - 不重新编码
            # - 几乎不额外占磁盘
            # - 删除 source 的目录项也不会让 dst 失效
            os.link(
                src_file,
                dst_file,
            )

            count += 1

    print(
        f"[OK] RGB hardlinks created: {count}"
    )


def main():
    args = parse_args()

    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().absolute()

    info = validate_source(
        src
    )

    prepare_destination(
        dst
    )

    episodes = select_episodes(
        src,
        args.limit_episodes,
    )

    print(
        f"[INFO] episodes selected: {len(episodes)}"
    )

    # tasks 不变
    shutil.copy2(
        src / "meta/tasks.jsonl",
        dst / "meta/tasks.jsonl",
    )

    # episodes 的 length/task 不变
    write_jsonl(
        dst / "meta/episodes.jsonl",
        episodes,
    )

    chunks_size = build_info(
        src,
        dst,
        info,
        episodes,
    )

    build_modality(
        src,
        dst,
    )

    build_split(
        src,
        dst,
        episodes,
    )

    state_stats, action_stats = (
        convert_parquets(
            src,
            dst,
            episodes,
            chunks_size,
        )
    )

    full_dataset = (
        len(episodes)
        == int(info["total_episodes"])
    )

    build_global_stats(
        src,
        dst,
        state_stats,
        action_stats,
        full_dataset,
    )

    link_videos(
        src,
        dst,
        episodes,
        chunks_size,
    )

    print()
    print("=" * 68)
    print("STAGE-2 28D -> 22D COMPLETE")
    print("=" * 68)
    print("src          :", src)
    print("dst          :", dst)
    print("episodes     :", len(episodes))
    print(
        "frames       :",
        sum(int(x["length"]) for x in episodes),
    )
    print("RGB cameras  : 3 (hardlink, no re-encode)")
    print("state/action : 22D / 22D")
    print("grippers     : removed")
    print(
        "hand mapping : "
        "[j0,j1,j2,mean(j3,j4,j5)]"
    )
    print()
    print(
        "IMPORTANT: final TRAIN-ONLY stats "
        "must still be recomputed."
    )
    print("=" * 68)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print(
            "\n[ABORTED] KeyboardInterrupt",
            file=sys.stderr,
        )
        sys.exit(130)

    except Exception as exc:
        print(
            f"\n[FATAL] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise
