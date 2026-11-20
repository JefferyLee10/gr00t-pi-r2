#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LeRobot v3.0 -> v2.1 exact converter for the 17h bag-transfer dataset.

本脚本只做 Stage-1：
    v3.0 consolidated dataset
        ->
    v2.1 episode-per-file dataset

特点：
1. 原始数据只读，不做任何原地修改。
2. 只保留三路 RGB：
   - observation.images.head_rgb
   - observation.images.left_wrist_rgb
   - observation.images.right_wrist_rgb
3. depth 完全不进入输出数据集。
4. state/action 暂时仍保持原始 28D。
5. 原样继承现有 train/val split.json。
6. task 正确读取 tasks.parquet 中的 row["task"]。
7. 视频不用 -c copy；使用 decode + libx264 re-encode。
8. 每个 episode 视频显式限制为 metadata 中的 episode length 帧。
9. modality.json 中：
      human.action.task_description.original_key = task_index
10. 支持 --resume 和小规模 --limit-episodes smoke test。

注意：
本阶段输出的 stats.json 仍然是“原始全部 episode 的 28D stats”。
后续完成 28D -> 22D、train-view 构建以后，必须重新计算 train-only stats。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


RGB_KEYS = [
    "observation.images.head_rgb",
    "observation.images.left_wrist_rgb",
    "observation.images.right_wrist_rgb",
]

RGB_ALIASES = {
    "head_rgb": "observation.images.head_rgb",
    "left_wrist_rgb": "observation.images.left_wrist_rgb",
    "right_wrist_rgb": "observation.images.right_wrist_rgb",
}

KEEP_STATS_KEYS = {
    "observation.state",
    "action",
    *RGB_KEYS,
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="LeRobot v3.0 dataset root",
    )

    parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="Output LeRobot v2.1 dataset root",
    )

    parser.add_argument(
        "--limit-episodes",
        type=int,
        default=None,
        help="Only convert the first N episodes. Intended for smoke test.",
    )

    parser.add_argument(
        "--video-workers",
        type=int,
        default=3,
        help="Number of concurrent ffmpeg processes.",
    )

    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=2,
        help="Threads used by each ffmpeg encoder.",
    )

    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="libx264 CRF.",
    )

    parser.add_argument(
        "--preset",
        type=str,
        default="fast",
        help="libx264 preset.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already existing non-empty output files.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete dst before conversion.",
    )

    parser.add_argument(
        "--verify-video-count",
        action="store_true",
        help="Use ffprobe packet counting to verify every output video's frame count.",
    )

    return parser.parse_args()


def jsonable(x: Any):
    """Convert Arrow/Numpy-like values into plain Python JSON objects."""

    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]

    if hasattr(x, "tolist"):
        return x.tolist()

    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass

    return x


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            jsonable(obj),
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    jsonable(row),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def command_exists(name: str):
    return shutil.which(name) is not None


def validate_environment(args):
    src = args.src.resolve()

    if not src.exists():
        raise FileNotFoundError(f"src does not exist: {src}")

    required = [
        src / "meta/info.json",
        src / "meta/modality.json",
        src / "meta/tasks.parquet",
        src / "meta/split.json",
        src / "meta/episodes",
        src / "data",
        src / "videos",
    ]

    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"required path missing: {p}")

    info = load_json(src / "meta/info.json")

    if info.get("codebase_version") != "v3.0":
        raise RuntimeError(
            f"Expected v3.0 source, got: {info.get('codebase_version')}"
        )

    for key in RGB_KEYS:
        if key not in info["features"]:
            raise RuntimeError(f"RGB feature missing: {key}")

    state_shape = info["features"]["observation.state"]["shape"]
    action_shape = info["features"]["action"]["shape"]

    if state_shape != [28]:
        raise RuntimeError(f"Expected 28D state, got {state_shape}")

    if action_shape != [28]:
        raise RuntimeError(f"Expected 28D action, got {action_shape}")

    if not command_exists("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH")

    if args.verify_video_count and not command_exists("ffprobe"):
        raise RuntimeError("ffprobe not found in PATH")

    src_resolved = src
    dst_resolved = args.dst.expanduser().absolute()

    if src_resolved == dst_resolved:
        raise RuntimeError("dst must not equal src")

    if src_resolved in dst_resolved.parents:
        raise RuntimeError(
            "dst must not be created inside the source dataset"
        )


def prepare_destination(args):
    dst = args.dst

    if dst.exists():
        if args.overwrite:
            print(f"[INFO] removing old dst: {dst}")
            shutil.rmtree(dst)

        elif not args.resume:
            raise RuntimeError(
                f"dst already exists: {dst}\n"
                f"Use --resume or --overwrite."
            )

    dst.mkdir(parents=True, exist_ok=True)


def load_episode_records(src: Path):
    files = sorted(
        (src / "meta/episodes").glob("chunk-*/file-*.parquet")
    )

    if not files:
        raise RuntimeError("No episode metadata parquet found.")

    rows = []

    for p in files:
        table = pq.read_table(p)
        rows.extend(table.to_pylist())

    rows.sort(key=lambda x: int(x["episode_index"]))

    return rows


def select_episode_records(records, limit):
    if limit is None:
        return records

    if limit <= 0:
        raise ValueError("--limit-episodes must be > 0")

    return records[: min(limit, len(records))]


def validate_episode_indices(records):
    expected = list(range(len(records)))
    actual = [int(r["episode_index"]) for r in records]

    if actual != expected:
        raise RuntimeError(
            "Selected episode indices are not contiguous from zero.\n"
            f"first actual indices: {actual[:20]}"
        )


def convert_tasks(src: Path, dst: Path):
    table = pq.read_table(src / "meta/tasks.parquet")
    rows = table.to_pylist()

    output = []

    for row in rows:
        if "task_index" not in row or "task" not in row:
            raise RuntimeError(
                f"Invalid tasks.parquet row: {row}"
            )

        output.append(
            {
                "task_index": int(row["task_index"]),
                # 关键：必须读取 row["task"]
                "task": str(row["task"]),
            }
        )

    write_jsonl(dst / "meta/tasks.jsonl", output)

    print(f"[OK] tasks.jsonl: {output}")


def convert_modality(src: Path, dst: Path):
    old = load_json(src / "meta/modality.json")

    new = {
        "state": copy.deepcopy(old["state"]),
        "action": copy.deepcopy(old["action"]),
        "video": {},
        "annotation": {
            "human.action.task_description": {
                "original_key": "task_index"
            }
        },
    }

    for alias, original_key in RGB_ALIASES.items():
        new["video"][alias] = {
            "original_key": original_key
        }

    dump_json(dst / "meta/modality.json", new)

    print("[OK] modality.json: 3 RGB only")


def convert_info(src: Path, dst: Path, records):
    info = load_json(src / "meta/info.json")
    info = copy.deepcopy(info)

    keep_features = {}

    for key, value in info["features"].items():
        dtype = value.get("dtype")

        if dtype == "video":
            if key in RGB_KEYS:
                keep_features[key] = value
        else:
            keep_features[key] = value

    total_episodes = len(records)
    total_frames = sum(int(r["length"]) for r in records)

    chunks_size = int(info.get("chunks_size", 1000))

    info["codebase_version"] = "v2.1"
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = 1

    # Stage-1 intermediate dataset:
    # 真正 train/val 仍以 meta/split.json 为准。
    info["splits"] = {
        "train": f"0:{total_episodes}"
    }

    info["features"] = keep_features

    info["data_path"] = (
        "data/chunk-{episode_chunk:03d}/"
        "episode_{episode_index:06d}.parquet"
    )

    info["video_path"] = (
        "videos/chunk-{episode_chunk:03d}/"
        "{video_key}/episode_{episode_index:06d}.mp4"
    )

    info["total_chunks"] = math.ceil(
        total_episodes / chunks_size
    )

    info["total_videos"] = (
        total_episodes * len(RGB_KEYS)
    )

    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)

    dump_json(dst / "meta/info.json", info)

    return chunks_size


def convert_split(src: Path, dst: Path, records):
    old_split = load_json(src / "meta/split.json")

    selected_indices = {
        int(r["episode_index"])
        for r in records
    }

    # 优先使用 split.json 中逐 episode 映射。
    episode_entries = []

    for row in old_split.get("episodes", []):
        idx = int(row["episode_index"])

        if idx in selected_indices:
            episode_entries.append(
                {
                    "episode_index": idx,
                    "episode_name": row["episode_name"],
                    "split": row["split"],
                }
            )

    if len(episode_entries) != len(records):
        raise RuntimeError(
            "split.json episode mapping count does not match "
            "selected episode count."
        )

    train_indices = [
        e["episode_index"]
        for e in episode_entries
        if e["split"] == "train"
    ]

    val_indices = [
        e["episode_index"]
        for e in episode_entries
        if e["split"] == "val"
    ]

    val_names = [
        e["episode_name"]
        for e in episode_entries
        if e["split"] == "val"
    ]

    new_split = {
        "val_episodes": val_names,
        "val_episode_indices": val_indices,
        "train_episode_indices": train_indices,
        "episodes": episode_entries,
    }

    dump_json(dst / "meta/split.json", new_split)

    print(
        "[OK] split.json: "
        f"train={len(train_indices)}, "
        f"val={len(val_indices)}"
    )


def convert_global_stats(src: Path, dst: Path):
    src_stats = src / "meta/stats.json"

    if not src_stats.exists():
        print("[WARN] source stats.json not found")
        return

    stats = load_json(src_stats)

    filtered = {
        key: value
        for key, value in stats.items()
        if key in KEEP_STATS_KEYS
    }

    dump_json(dst / "meta/stats.json", filtered)

    warning = dst / "meta/STAGE1_STATS_WARNING.txt"

    warning.write_text(
        "This stats.json still comes from the complete original "
        "28D dataset.\n"
        "DO NOT use it as final training normalization statistics.\n"
        "After 28D->22D and train-view creation, recompute "
        "TRAIN-ONLY stats.\n",
        encoding="utf-8",
    )

    print(
        "[WARN] copied filtered source stats.json; "
        "NOT final train-only stats"
    )


def collect_episode_stats(record):
    stats = {}

    prefix = "stats/"

    for key, value in record.items():
        if not key.startswith(prefix):
            continue

        rest = key[len(prefix):]

        if "/" not in rest:
            continue

        feature, metric = rest.rsplit("/", 1)

        if feature not in KEEP_STATS_KEYS:
            continue

        stats.setdefault(feature, {})[metric] = jsonable(value)

    return stats


def convert_episode_metadata(dst: Path, records):
    episodes_rows = []
    stats_rows = []

    for record in records:
        idx = int(record["episode_index"])

        length = int(record["length"])

        tasks = record.get("tasks", ["快递分拣"])

        episodes_rows.append(
            {
                "episode_index": idx,
                "tasks": jsonable(tasks),
                "length": length,
            }
        )

        stats_rows.append(
            {
                "episode_index": idx,
                "stats": collect_episode_stats(record),
            }
        )

    write_jsonl(
        dst / "meta/episodes.jsonl",
        episodes_rows,
    )

    write_jsonl(
        dst / "meta/episodes_stats.jsonl",
        stats_rows,
    )

    print(
        f"[OK] episodes.jsonl: {len(episodes_rows)} episodes"
    )


def group_by_data_file(records):
    groups = defaultdict(list)

    for r in records:
        key = (
            int(r["data/chunk_index"]),
            int(r["data/file_index"]),
        )

        groups[key].append(r)

    return groups


def convert_data(src: Path, dst: Path, records, chunks_size, resume):
    groups = group_by_data_file(records)

    converted = 0

    for (chunk_idx, file_idx), rs in sorted(groups.items()):
        source = (
            src
            / "data"
            / f"chunk-{chunk_idx:03d}"
            / f"file-{file_idx:03d}.parquet"
        )

        if not source.exists():
            raise FileNotFoundError(source)

        print(f"[DATA] reading {source}")

        table = pq.read_table(source)

        rs = sorted(
            rs,
            key=lambda x: int(x["dataset_from_index"])
        )

        file_offset = int(rs[0]["dataset_from_index"])

        for record in rs:
            episode_index = int(record["episode_index"])
            expected_length = int(record["length"])

            start = (
                int(record["dataset_from_index"])
                - file_offset
            )

            stop = (
                int(record["dataset_to_index"])
                - file_offset
            )

            length = stop - start

            if length != expected_length:
                raise RuntimeError(
                    f"episode {episode_index}: "
                    f"metadata length={expected_length}, "
                    f"slice length={length}"
                )

            dest_chunk = episode_index // chunks_size

            output = (
                dst
                / "data"
                / f"chunk-{dest_chunk:03d}"
                / f"episode_{episode_index:06d}.parquet"
            )

            if (
                resume
                and output.exists()
                and output.stat().st_size > 0
            ):
                converted += 1
                continue

            episode_table = table.slice(
                start,
                length,
            )

            if episode_table.num_rows != expected_length:
                raise RuntimeError(
                    f"episode {episode_index}: "
                    f"Arrow rows={episode_table.num_rows}, "
                    f"expected={expected_length}"
                )

            if "episode_index" in episode_table.column_names:
                ep_values = set(
                    episode_table["episode_index"].to_pylist()
                )

                if ep_values != {episode_index}:
                    raise RuntimeError(
                        f"episode_index mismatch in episode "
                        f"{episode_index}: {ep_values}"
                    )

            if expected_length > 0:
                row0 = episode_table.slice(0, 1).to_pylist()[0]

                if len(row0["observation.state"]) != 28:
                    raise RuntimeError(
                        f"episode {episode_index}: "
                        "state is not 28D"
                    )

                if len(row0["action"]) != 28:
                    raise RuntimeError(
                        f"episode {episode_index}: "
                        "action is not 28D"
                    )

            output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            pq.write_table(
                episode_table,
                output,
            )

            converted += 1

            if converted % 50 == 0:
                print(
                    f"[DATA] {converted}/{len(records)} episodes"
                )

    print(
        f"[OK] parquet conversion complete: "
        f"{converted}/{len(records)}"
    )


def build_video_jobs(src: Path, dst: Path, records, chunks_size):
    jobs = []

    for video_key in RGB_KEYS:
        chunk_col = f"videos/{video_key}/chunk_index"
        file_col = f"videos/{video_key}/file_index"
        from_col = f"videos/{video_key}/from_timestamp"
        to_col = f"videos/{video_key}/to_timestamp"

        for record in records:
            episode_index = int(record["episode_index"])

            for required in [
                chunk_col,
                file_col,
                from_col,
                to_col,
            ]:
                if required not in record:
                    raise RuntimeError(
                        f"Episode {episode_index}: "
                        f"missing video metadata {required}"
                    )

            chunk_idx = int(record[chunk_col])
            file_idx = int(record[file_col])

            start = float(record[from_col])
            end = float(record[to_col])

            if not (0.0 <= start < end):
                raise RuntimeError(
                    f"Episode {episode_index}, {video_key}: "
                    f"invalid timestamps {start} -> {end}"
                )

            source = (
                src
                / "videos"
                / video_key
                / f"chunk-{chunk_idx:03d}"
                / f"file-{file_idx:03d}.mp4"
            )

            dest_chunk = episode_index // chunks_size

            output = (
                dst
                / "videos"
                / f"chunk-{dest_chunk:03d}"
                / video_key
                / f"episode_{episode_index:06d}.mp4"
            )

            jobs.append(
                {
                    "episode_index": episode_index,
                    "video_key": video_key,
                    "source": source,
                    "output": output,
                    "start": start,
                    "end": end,
                    "expected_frames": int(record["length"]),
                }
            )

    return jobs


def ffprobe_frame_count(path: Path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    value = result.stdout.strip()

    return int(value)


def convert_one_video(
    job,
    fps,
    crf,
    preset,
    ffmpeg_threads,
    resume,
    verify,
):
    source = job["source"]
    output = job["output"]

    expected_frames = job["expected_frames"]

    if not source.exists():
        raise FileNotFoundError(source)

    if (
        resume
        and output.exists()
        and output.stat().st_size > 1024
    ):
        if verify:
            count = ffprobe_frame_count(output)

            if count != expected_frames:
                raise RuntimeError(
                    f"existing video frame mismatch: "
                    f"{output}: {count} != {expected_frames}"
                )

        return "skip"

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = output.with_suffix(".tmp.mp4")

    if temp.exists():
        temp.unlink()

    # 关键：
    # 这里绝不使用 -c copy。
    #
    # -ss 在 input 前：
    #   ffmpeg 会 seek 到前一关键帧并解码，
    #   在 transcoding 模式下执行 accurate seek。
    #
    # -frames:v：
    #   强制最终 episode 视频帧数与 parquet length 一致。
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{job['start']:.9f}",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-frames:v",
        str(expected_frames),
        "-fps_mode",
        "cfr",
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-threads",
        str(ffmpeg_threads),
        "-movflags",
        "+faststart",
        "-y",
        str(temp),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        if temp.exists():
            temp.unlink()

        raise RuntimeError(
            f"ffmpeg failed\n"
            f"episode={job['episode_index']}\n"
            f"video={job['video_key']}\n"
            f"source={source}\n"
            f"stderr={result.stderr}"
        )

    if not temp.exists() or temp.stat().st_size == 0:
        raise RuntimeError(
            f"ffmpeg produced empty output: {temp}"
        )

    temp.replace(output)

    if verify:
        count = ffprobe_frame_count(output)

        if count != expected_frames:
            raise RuntimeError(
                f"video frame mismatch:\n"
                f"{output}\n"
                f"expected={expected_frames}, actual={count}"
            )

    return "done"


def convert_videos(args, src, dst, records, chunks_size, fps):
    jobs = build_video_jobs(
        src,
        dst,
        records,
        chunks_size,
    )

    total = len(jobs)

    print(
        f"[VIDEO] total jobs={total}, "
        f"workers={args.video_workers}"
    )

    done = 0
    skipped = 0

    with ThreadPoolExecutor(
        max_workers=args.video_workers
    ) as pool:

        futures = [
            pool.submit(
                convert_one_video,
                job,
                fps,
                args.crf,
                args.preset,
                args.ffmpeg_threads,
                args.resume,
                args.verify_video_count,
            )
            for job in jobs
        ]

        for future in as_completed(futures):
            status = future.result()

            if status == "skip":
                skipped += 1
            else:
                done += 1

            finished = done + skipped

            if finished % 25 == 0 or finished == total:
                print(
                    f"[VIDEO] "
                    f"{finished}/{total}, "
                    f"done={done}, skipped={skipped}"
                )

    print(
        f"[OK] video conversion complete: "
        f"done={done}, skipped={skipped}"
    )


def print_summary(src, dst, records):
    total_frames = sum(
        int(r["length"])
        for r in records
    )

    print()
    print("=" * 72)
    print("STAGE-1 CONVERSION COMPLETE")
    print("=" * 72)
    print(f"src          : {src}")
    print(f"dst          : {dst}")
    print(f"episodes     : {len(records)}")
    print(f"frames       : {total_frames}")
    print(f"RGB cameras  : {len(RGB_KEYS)}")
    print("state/action : 28D / 28D")
    print("depth        : NOT COPIED")
    print()
    print(
        "IMPORTANT: meta/stats.json is still source full-dataset "
        "28D stats."
    )
    print(
        "Next stage must perform 28D -> 22D and recompute "
        "TRAIN-ONLY stats."
    )
    print("=" * 72)


def main():
    args = parse_args()

    args.src = args.src.expanduser()
    args.dst = args.dst.expanduser()

    validate_environment(args)

    src = args.src.resolve()
    dst = args.dst.absolute()

    prepare_destination(args)

    info = load_json(src / "meta/info.json")

    fps = int(info["fps"])

    records = load_episode_records(src)

    records = select_episode_records(
        records,
        args.limit_episodes,
    )

    validate_episode_indices(records)

    print(
        f"[INFO] source episodes selected: {len(records)}"
    )

    convert_tasks(src, dst)

    convert_modality(src, dst)

    chunks_size = convert_info(
        src,
        dst,
        records,
    )

    convert_split(
        src,
        dst,
        records,
    )

    convert_global_stats(
        src,
        dst,
    )

    convert_episode_metadata(
        dst,
        records,
    )

    convert_data(
        src,
        dst,
        records,
        chunks_size,
        args.resume,
    )

    convert_videos(
        args,
        src,
        dst,
        records,
        chunks_size,
        fps,
    )

    print_summary(
        src,
        dst,
        records,
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\n[ABORTED] KeyboardInterrupt")
        sys.exit(130)

    except Exception as exc:
        print(
            f"\n[FATAL] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise
