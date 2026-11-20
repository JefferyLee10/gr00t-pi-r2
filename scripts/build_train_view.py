#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从完整 LeRobot v2.1 22D 数据集构建 train-only dataset view。

输入：
    full 22D dataset
    1315 episodes
    split.json 中：
        1183 train
        132 val

输出：
    只包含 train episodes 的独立 LeRobot v2.1 dataset

核心行为：
1. 只读取 split.json 中 train_episode_indices。
2. train episodes 连续重新编号：
       old episode_index -> new episode_index
3. parquet 中：
       episode_index 重新编号
       index 重新从 0 连续编号
       frame_index 保持 episode 内 0...length-1
       timestamp/state/action/task_index 不修改
4. 22D state/action 数值完全不修改。
5. RGB 使用 hardlink：
       不复制
       不重新编码
6. 不复制 full-dataset stats.json。
   后续必须单独计算 TRAIN-ONLY stats。
7. 不删除、不覆盖任何已有数据。
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


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--src",
        type=Path,
        required=True,
        help="完整1315-episode 22D数据集",
    )

    p.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="train-only输出数据集",
    )

    p.add_argument(
        "--limit-train-episodes",
        type=int,
        default=None,
        help="仅用于smoke test，取train列表前N个episode",
    )

    return p.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
                rows.append(
                    json.loads(line)
                )

    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def validate_source(src: Path):
    required = [
        src / "meta/info.json",
        src / "meta/modality.json",
        src / "meta/tasks.jsonl",
        src / "meta/episodes.jsonl",
        src / "meta/episodes_stats.jsonl",
        src / "meta/split.json",
        src / "data",
        src / "videos",
    ]

    for p in required:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing source path: {p}"
            )

    info = load_json(
        src / "meta/info.json"
    )

    if info["codebase_version"] != "v2.1":
        raise RuntimeError(
            "Source must be LeRobot v2.1"
        )

    if info["features"][STATE_KEY]["shape"] != [22]:
        raise RuntimeError(
            "Source state is not 22D"
        )

    if info["features"][ACTION_KEY]["shape"] != [22]:
        raise RuntimeError(
            "Source action is not 22D"
        )

    video_keys = {
        k
        for k, v in info["features"].items()
        if v.get("dtype") == "video"
    }

    if video_keys != set(RGB_KEYS):
        raise RuntimeError(
            f"Expected exactly 3RGB, got {video_keys}"
        )

    return info


def prepare_destination(dst: Path):
    if dst.exists():
        raise RuntimeError(
            "Destination already exists; "
            f"refusing to overwrite: {dst}"
        )

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dst.mkdir(
        parents=False,
        exist_ok=False,
    )

    (dst / "meta").mkdir(
        exist_ok=False,
    )


def load_train_selection(
    src: Path,
    limit: int | None,
):
    split = load_json(
        src / "meta/split.json"
    )

    train_ids = [
        int(x)
        for x in split["train_episode_indices"]
    ]

    val_ids = [
        int(x)
        for x in split["val_episode_indices"]
    ]

    if set(train_ids) & set(val_ids):
        raise RuntimeError(
            "Train/val leakage already exists in source split"
        )

    if len(train_ids) != 1183:
        raise RuntimeError(
            f"Expected 1183 train episodes, got {len(train_ids)}"
        )

    if len(val_ids) != 132:
        raise RuntimeError(
            f"Expected 132 val episodes, got {len(val_ids)}"
        )

    if limit is not None:
        if limit <= 0:
            raise ValueError(
                "--limit-train-episodes must be > 0"
            )

        train_ids = train_ids[:limit]

    return train_ids


def load_episode_metadata(src: Path):
    rows = load_jsonl(
        src / "meta/episodes.jsonl"
    )

    by_id = {
        int(row["episode_index"]): row
        for row in rows
    }

    return by_id


def load_episode_stats(src: Path):
    rows = load_jsonl(
        src / "meta/episodes_stats.jsonl"
    )

    return {
        int(row["episode_index"]): row
        for row in rows
    }


def build_mapping(train_old_ids):
    """
    old episode index -> new continuous index
    """

    return {
        old_idx: new_idx
        for new_idx, old_idx
        in enumerate(train_old_ids)
    }


def build_info(
    src_info,
    train_old_ids,
    episode_meta,
):
    info = copy.deepcopy(
        src_info
    )

    total_frames = sum(
        int(
            episode_meta[old]["length"]
        )
        for old in train_old_ids
    )

    n_ep = len(train_old_ids)

    chunks_size = int(
        info.get("chunks_size", 1000)
    )

    info["total_episodes"] = n_ep
    info["total_frames"] = total_frames
    info["total_chunks"] = math.ceil(
        n_ep / chunks_size
    )
    info["total_videos"] = (
        n_ep * len(RGB_KEYS)
    )

    # train view 中所有episode都是train
    info["splits"] = {
        "train": f"0:{n_ep}"
    }

    return info, chunks_size, total_frames


def build_episodes_jsonl(
    dst,
    train_old_ids,
    mapping,
    episode_meta,
):
    rows = []

    for old_idx in train_old_ids:
        old = episode_meta[old_idx]

        rows.append(
            {
                "episode_index": mapping[old_idx],
                "tasks": copy.deepcopy(
                    old["tasks"]
                ),
                "length": int(
                    old["length"]
                ),
            }
        )

    write_jsonl(
        dst / "meta/episodes.jsonl",
        rows,
    )


def build_episode_stats_jsonl(
    src,
    dst,
    train_old_ids,
    mapping,
):
    old_stats = load_episode_stats(
        src
    )

    rows = []

    for old_idx in train_old_ids:

        if old_idx not in old_stats:
            raise RuntimeError(
                f"Missing episode stats: {old_idx}"
            )

        row = copy.deepcopy(
            old_stats[old_idx]
        )

        row["episode_index"] = (
            mapping[old_idx]
        )

        rows.append(row)

    write_jsonl(
        dst / "meta/episodes_stats.jsonl",
        rows,
    )


def build_split_json(
    src,
    dst,
    train_old_ids,
    mapping,
):
    old_split = load_json(
        src / "meta/split.json"
    )

    old_episode_map = {
        int(x["episode_index"]): x
        for x in old_split["episodes"]
    }

    rows = []

    for old_idx in train_old_ids:
        src_row = old_episode_map[old_idx]

        if src_row["split"] != "train":
            raise RuntimeError(
                f"Episode {old_idx} is not train"
            )

        rows.append(
            {
                "episode_index": mapping[old_idx],
                # 保留原始episode_name，便于追踪数据来源
                "episode_name": src_row["episode_name"],
                "split": "train",
            }
        )

    n = len(rows)

    out = {
        "val_episodes": [],
        "val_episode_indices": [],
        "train_episode_indices": list(range(n)),
        "episodes": rows,
    }

    dump_json(
        dst / "meta/split.json",
        out,
    )


def replace_int64_column(
    table,
    key,
    values,
):
    idx = table.schema.get_field_index(
        key
    )

    if idx < 0:
        raise RuntimeError(
            f"Required column missing: {key}"
        )

    arr = pa.array(
        values,
        type=pa.int64(),
    )

    return table.set_column(
        idx,
        key,
        arr,
    )


def validate_22d_table(
    table,
    old_episode_index,
    expected_length,
):
    if table.num_rows != expected_length:
        raise RuntimeError(
            f"Episode {old_episode_index}: "
            f"rows={table.num_rows}, "
            f"expected={expected_length}"
        )

    required = [
        STATE_KEY,
        ACTION_KEY,
        "episode_index",
        "frame_index",
        "index",
        "task_index",
    ]

    for key in required:
        if key not in table.column_names:
            raise RuntimeError(
                f"Missing column {key}"
            )

    ep_values = set(
        table["episode_index"].to_pylist()
    )

    if ep_values != {old_episode_index}:
        raise RuntimeError(
            f"Source episode_index mismatch: "
            f"expected={old_episode_index}, "
            f"got={ep_values}"
        )

    frame_index = np.asarray(
        table["frame_index"].to_pylist(),
        dtype=np.int64,
    )

    expected_frame_index = np.arange(
        expected_length,
        dtype=np.int64,
    )

    if not np.array_equal(
        frame_index,
        expected_frame_index,
    ):
        raise RuntimeError(
            f"Episode {old_episode_index}: "
            "frame_index is not contiguous 0..length-1"
        )

    # 抽首尾两帧确认22D
    check_rows = [0]

    if expected_length > 1:
        check_rows.append(
            expected_length - 1
        )

    for i in check_rows:
        row = table.slice(
            i,
            1,
        ).to_pylist()[0]

        if len(row[STATE_KEY]) != 22:
            raise RuntimeError(
                f"Episode {old_episode_index}: state not 22D"
            )

        if len(row[ACTION_KEY]) != 22:
            raise RuntimeError(
                f"Episode {old_episode_index}: action not 22D"
            )


def convert_parquets(
    src,
    dst,
    train_old_ids,
    mapping,
    episode_meta,
    chunks_size,
):
    global_index = 0

    for n, old_idx in enumerate(
        train_old_ids
    ):
        new_idx = mapping[old_idx]

        old_chunk = old_idx // chunks_size
        new_chunk = new_idx // chunks_size

        src_file = (
            src
            / "data"
            / f"chunk-{old_chunk:03d}"
            / f"episode_{old_idx:06d}.parquet"
        )

        dst_file = (
            dst
            / "data"
            / f"chunk-{new_chunk:03d}"
            / f"episode_{new_idx:06d}.parquet"
        )

        if not src_file.exists():
            raise FileNotFoundError(
                src_file
            )

        table = pq.read_table(
            src_file
        )

        length = int(
            episode_meta[old_idx]["length"]
        )

        validate_22d_table(
            table,
            old_idx,
            length,
        )

        # episode_index:
        # old discontinuous index -> new contiguous index
        table = replace_int64_column(
            table,
            "episode_index",
            np.full(
                length,
                new_idx,
                dtype=np.int64,
            ),
        )

        # index:
        # 重建为整个train dataset连续全局index
        new_global_indices = np.arange(
            global_index,
            global_index + length,
            dtype=np.int64,
        )

        table = replace_int64_column(
            table,
            "index",
            new_global_indices,
        )

        global_index += length

        dst_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pq.write_table(
            table,
            dst_file,
        )

        if (
            (n + 1) % 50 == 0
            or n + 1 == len(train_old_ids)
        ):
            print(
                f"[DATA] "
                f"{n+1}/{len(train_old_ids)} "
                f"old_ep={old_idx} "
                f"new_ep={new_idx} "
                f"global_frames={global_index}"
            )

    return global_index


def link_rgb(
    src,
    dst,
    train_old_ids,
    mapping,
    chunks_size,
):
    count = 0

    for old_idx in train_old_ids:
        new_idx = mapping[old_idx]

        old_chunk = old_idx // chunks_size
        new_chunk = new_idx // chunks_size

        for key in RGB_KEYS:

            src_v = (
                src
                / "videos"
                / f"chunk-{old_chunk:03d}"
                / key
                / f"episode_{old_idx:06d}.mp4"
            )

            dst_v = (
                dst
                / "videos"
                / f"chunk-{new_chunk:03d}"
                / key
                / f"episode_{new_idx:06d}.mp4"
            )

            if not src_v.exists():
                raise FileNotFoundError(
                    src_v
                )

            dst_v.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            try:
                os.link(
                    src_v,
                    dst_v,
                )

            except OSError as exc:
                raise RuntimeError(
                    "RGB hardlink failed. "
                    "Refusing to silently copy video.\n"
                    f"source={src_v}\n"
                    f"dest={dst_v}\n"
                    f"error={exc}"
                ) from exc

            count += 1

    return count


def write_mapping(
    dst,
    train_old_ids,
    mapping,
):
    rows = [
        {
            "new_episode_index": mapping[old],
            "source_episode_index": old,
        }
        for old in train_old_ids
    ]

    write_jsonl(
        dst / "meta/source_episode_mapping.jsonl",
        rows,
    )


def main():
    args = parse_args()

    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().absolute()

    src_info = validate_source(
        src
    )

    train_old_ids = load_train_selection(
        src,
        args.limit_train_episodes,
    )

    episode_meta = load_episode_metadata(
        src
    )

    for old_idx in train_old_ids:
        if old_idx not in episode_meta:
            raise RuntimeError(
                f"Missing episode metadata: {old_idx}"
            )

    mapping = build_mapping(
        train_old_ids
    )

    info, chunks_size, expected_frames = build_info(
        src_info,
        train_old_ids,
        episode_meta,
    )

    print(
        f"[INFO] selected train episodes: "
        f"{len(train_old_ids)}"
    )

    print(
        f"[INFO] expected train frames: "
        f"{expected_frames}"
    )

    print(
        "[INFO] first mappings:",
        [
            (old, mapping[old])
            for old in train_old_ids[:10]
        ],
    )

    # 所有source验证完成后才创建dst
    prepare_destination(
        dst
    )

    dump_json(
        dst / "meta/info.json",
        info,
    )

    shutil.copy2(
        src / "meta/modality.json",
        dst / "meta/modality.json",
    )

    shutil.copy2(
        src / "meta/tasks.jsonl",
        dst / "meta/tasks.jsonl",
    )

    build_episodes_jsonl(
        dst,
        train_old_ids,
        mapping,
        episode_meta,
    )

    build_episode_stats_jsonl(
        src,
        dst,
        train_old_ids,
        mapping,
    )

    build_split_json(
        src,
        dst,
        train_old_ids,
        mapping,
    )

    write_mapping(
        dst,
        train_old_ids,
        mapping,
    )

    total_frames = convert_parquets(
        src,
        dst,
        train_old_ids,
        mapping,
        episode_meta,
        chunks_size,
    )

    if total_frames != expected_frames:
        raise RuntimeError(
            f"Final frame mismatch: "
            f"{total_frames} != {expected_frames}"
        )

    rgb_count = link_rgb(
        src,
        dst,
        train_old_ids,
        mapping,
        chunks_size,
    )

    expected_rgb = (
        len(train_old_ids)
        * len(RGB_KEYS)
    )

    if rgb_count != expected_rgb:
        raise RuntimeError(
            f"RGB count mismatch: "
            f"{rgb_count} != {expected_rgb}"
        )

    # 故意不复制 full-dataset stats.json
    warning = (
        dst
        / "meta/TRAIN_STATS_REQUIRED.txt"
    )

    warning.write_text(
        "This is a TRAIN-ONLY LeRobot v2.1 dataset view.\n"
        "Global stats.json has intentionally NOT been copied.\n"
        "Compute fresh normalization statistics using this "
        "train-only dataset before piR2 training.\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("TRAIN-ONLY DATASET VIEW COMPLETE")
    print("=" * 72)
    print("src              :", src)
    print("dst              :", dst)
    print(
        "train episodes   :",
        len(train_old_ids),
    )
    print(
        "train frames     :",
        total_frames,
    )
    print(
        "RGB hardlinks    :",
        rgb_count,
    )
    print(
        "state/action     : 22D / 22D"
    )
    print(
        "stats.json       : NOT CREATED"
    )
    print(
        "val leakage      : excluded by construction"
    )
    print("=" * 72)


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
            f"\n[FATAL] "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise
