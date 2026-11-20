#!/usr/bin/env python3
"""Build episode-level LeRobot v2.1 views from the bag-transfer v3.0 export."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


NUMERIC = ("observation.state", "action")
PROFILES = {"headrgb": ("head_rgb",), "rgb3": ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")}
INPUT_GROUPS = {
    "left_arm": (0, 7), "left_gripper": (7, 8), "right_arm": (8, 15),
    "right_gripper": (15, 16), "left_hand": (16, 22), "right_hand": (22, 28),
}
OUTPUT_GROUPS = {"left_arm": (0, 7), "right_arm": (7, 14), "left_hand": (14, 18), "right_hand": (18, 22)}
OUTPUT_NAMES = [f"{side}_arm_j{i}" for side in ("left", "right") for i in range(7)] + [
    f"{side}_hand_{joint}" for side in ("left", "right") for joint in ("j0", "j1", "j2", "j345_mean")
]
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def dump(path, value, *, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as stream:
        if lines:
            for row in value:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        else:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")


def stats(values, episode=False):
    if not np.isfinite(values).all():
        raise ValueError("Encountered non-finite state, action or timestamp")
    result = {name: getattr(values, name)(axis=0).tolist() for name in ("mean", "std", "min", "max")}
    percentiles = [1, 10, 50, 90, 99] if episode else [1, 99]
    for p, q in zip(percentiles, np.percentile(values, percentiles, axis=0)):
        result[f"q{p:02d}"] = q.tolist()
    if episode:
        result["count"] = [len(values)]
    return result


def project(values):
    """Absolute 28D -> absolute 22D, identical for state and action."""
    return np.concatenate((
        values[:, :7], values[:, 8:15], values[:, 16:19], values[:, 19:22].mean(1, keepdims=True),
        values[:, 22:25], values[:, 25:28].mean(1, keepdims=True),
    ), axis=1)


def load_metadata(src, profile, limit):
    meta = src / "meta"
    info = json.loads((meta / "info.json").read_text())
    modality = json.loads((meta / "modality.json").read_text())
    if info["codebase_version"] != "v3.0" or info["fps"] <= 0 or info.get("chunks_size", 1000) <= 0:
        raise ValueError("Expected LeRobot v3.0 and positive FPS/chunk size")
    for key, category in zip(NUMERIC, ("state", "action")):
        if info["features"][key]["shape"] != [28]:
            raise ValueError(f"Expected {key} width 28")
        for name, (start, end) in INPUT_GROUPS.items():
            group = modality[category][name]
            if (group["start"], group["end"]) != (start, end) or group.get("absolute") is False:
                raise ValueError(f"Unsupported {category}.{name} layout")
    cameras = [f"observation.images.{name}" for name in PROFILES[profile]]
    for camera in cameras:
        if info["features"].get(camera, {}).get("dtype") != "video":
            raise ValueError(f"Source is missing {camera} for profile {profile}")

    rows = []
    for path in sorted((meta / "episodes").glob("chunk-*/file-*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda row: int(row["episode_index"]))
    if [row["episode_index"] for row in rows] != list(range(info["total_episodes"])):
        raise ValueError("Expected one source metadata record for every episode 0..N-1")
    if not rows or sum(row["length"] for row in rows) != info["total_frames"]:
        raise ValueError("Episode lengths and total_frames disagree")
    rows = rows[:limit]
    split_rows = json.loads((meta / "split.json").read_text())["episodes"]
    splits = {r["episode_index"]: r for r in split_rows}
    if len(splits) != len(split_rows) or set(splits) != set(range(info["total_episodes"])):
        raise ValueError("Expected one train/val split entry per source episode")
    if any(r["split"] not in ("train", "val") for r in split_rows):
        raise ValueError("Expected train or val in split.json")
    if not any(splits[r["episode_index"]]["split"] == "train" for r in rows):
        raise ValueError("Selected prefix contains no training episodes")
    tasks = pq.read_table(meta / "tasks.parquet", columns=["task_index", "task"]).to_pylist()
    if not tasks or [r["task_index"] for r in sorted(tasks, key=lambda r: r["task_index"])] != list(range(len(tasks))):
        raise ValueError("Expected a task vocabulary indexed from zero")
    return info, modality, cameras, rows, splits, tasks


def encode_video(job):
    source, dest, start, end, frames, fps = job
    if start < 0 or abs((end - start) * fps - frames) >= 0.51:
        raise ValueError(f"Video interval and row count disagree for {dest.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", f"{start:.9f}", "-i", str(source),
        "-map", "0:v:0", "-an", "-frames:v", str(frames), "-vsync", "cfr", "-r", str(fps),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "2",
        "-movflags", "+faststart", str(dest),
    ], check=True)
    count = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
        "stream=nb_read_packets", "-of", "default=noprint_wrappers=1:nokey=1", str(dest),
    ], text=True)
    if int(count) != frames:
        raise ValueError(f"{dest} has {count.strip()} frames, expected {frames}")


def video_jobs(src, dest, cameras, rows, chunk_size, fps):
    for row in rows:
        ep = row["episode_index"]
        for camera in cameras:
            prefix = f"videos/{camera}/"
            chunk = row[prefix + "chunk_index"]
            file_id = row[prefix + "file_index"]
            source = src / "videos" / camera / f"chunk-{chunk:03d}" / f"file-{file_id:03d}.mp4"
            path = VIDEO_PATH.format(episode_chunk=ep // chunk_size, video_key=camera, episode_index=ep)
            yield (source, dest / path, row[prefix + "from_timestamp"], row[prefix + "to_timestamp"], row["length"], fps)


def read_episodes(src, rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["data/chunk_index"], row["data/file_index"])].append(row)
    for (chunk, file_id), group in sorted(groups.items()):
        path = src / "data" / f"chunk-{chunk:03d}" / f"file-{file_id:03d}.parquet"
        table = pq.read_table(path)
        offset = table["index"][0].as_py()
        for row in group:
            start, stop = row["dataset_from_index"], row["dataset_to_index"]
            ep, size = row["episode_index"], row["length"]
            if size <= 0 or stop - start != size or start < offset:
                raise ValueError(f"Invalid episode range for {ep}")
            episode = table.slice(start - offset, size)
            if len(episode) != size or not np.all(episode["episode_index"].to_numpy() == ep):
                raise ValueError(f"Incorrect episode slice for {ep}")
            if not np.array_equal(episode["frame_index"].to_numpy(), np.arange(size)):
                raise ValueError(f"Frame index mismatch for episode {ep}")
            if not np.array_equal(episode["index"].to_numpy(), np.arange(start, stop)):
                raise ValueError(f"Global index mismatch for episode {ep}")
            yield row, episode


def write_table(root, table, ep, chunk_size):
    path = root / DATA_PATH.format(episode_index=ep, episode_chunk=ep // chunk_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def write_metadata(root, source_info, source_modality, cameras, splits, episodes, ep_stats, tasks, width):
    info, modality = deepcopy(source_info), deepcopy(source_modality)
    count = len(episodes)
    info.update(codebase_version="v2.1", total_episodes=count, total_frames=sum(r["length"] for r in episodes),
                total_tasks=len(tasks), total_videos=count * len(cameras), data_path=DATA_PATH, video_path=VIDEO_PATH,
                total_chunks=math.ceil(count / info.get("chunks_size", 1000)), splits={"train": f"0:{count}"})
    for key in ("data_files_size_in_mb", "video_files_size_in_mb"):
        info.pop(key, None)
    info["features"] = {k: v for k, v in info["features"].items() if v["dtype"] != "video" or k in cameras}
    for camera in cameras:
        info["features"][camera].setdefault("info", {}).update({"video.codec": "h264", "video.pix_fmt": "yuv420p"})
    modality["video"] = {k.rsplit(".", 1)[-1]: {"original_key": k} for k in cameras}
    modality["annotation"] = {"human.action.task_description": {"original_key": "task_index"}}
    if width == 22:
        for key, category in zip(NUMERIC, ("state", "action")):
            info["features"][key].update(shape=[22], names=OUTPUT_NAMES)
            modality[category] = {k: dict(modality[category][k], start=a, end=b) for k, (a, b) in OUTPUT_GROUPS.items()}
    splits = sorted(splits, key=lambda r: r["episode_index"])
    split = {"episodes": splits, "val_episodes": [r["episode_name"] for r in splits if r["split"] == "val"]}
    for label in ("train", "val"):
        split[label + "_episode_indices"] = [r["episode_index"] for r in splits if r["split"] == label]
    for name, value in (("info", info), ("modality", modality), ("split", split)):
        dump(root / "meta" / f"{name}.json", value)
    dump(root / "meta/tasks.jsonl", sorted(tasks, key=lambda r: r["task_index"]), lines=True)
    dump(root / "meta/episodes.jsonl", sorted(episodes, key=lambda r: r["episode_index"]), lines=True)
    dump(root / "meta/episodes_stats.jsonl", sorted(ep_stats, key=lambda r: r["episode_index"]), lines=True)


def build(src, output, profile, limit, workers):
    info, modality, cameras, rows, splits, tasks = load_metadata(src, profile, limit)
    chunk_size = info.get("chunks_size", 1000)
    train_ids = [r["episode_index"] for r in rows if splits[r["episode_index"]]["split"] == "train"]
    # Training IDs and global indices become contiguous even when the split interleaves episodes.
    train_mapping = {ep: new for new, ep in enumerate(train_ids)}
    train_offsets = {}
    next_row = 0
    for row in rows:
        ep = row["episode_index"]
        if ep in train_mapping:
            train_offsets[ep] = next_row
            next_row += row["length"]
    roots = {name: output / name for name in ("28d", "22d", "train_view")}
    datasets = {name: [] for name in roots}
    episode_stats = {name: [] for name in roots}
    float_keys = [key for key, feature in info["features"].items() if "float" in feature["dtype"]]
    for row, table in read_episodes(src, rows):
        ep, length = row["episode_index"], row["length"]
        unique_tasks = np.unique(table["task_index"].to_numpy())
        if any(i < 0 or i >= len(tasks) for i in unique_tasks):
            raise ValueError(f"Unknown task index in episode {ep}")
        task_names = {r["task_index"]: r["task"] for r in tasks}
        metadata = {"episode_index": ep, "length": length, "tasks": [task_names[i] for i in unique_tasks]}
        projected = table
        source_stats = {}
        for col in row:
            if col.startswith("stats/") and row[col] is not None:
                feature, metric = col[6:].rsplit("/", 1)
                if feature in cameras or feature in NUMERIC:
                    source_stats.setdefault(feature, {})[metric] = row[col]
        projected_stats = dict(source_stats)
        for key in NUMERIC:
            matrix = np.asarray(table[key].to_pylist(), dtype=np.float32)
            if matrix.shape != (length, 28) or not np.isfinite(matrix).all():
                raise ValueError(f"Invalid {key} for episode {ep}")
            values = project(matrix)
            array = pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), 22)
            projected = projected.set_column(table.schema.get_field_index(key), key, array)
            projected_stats[key] = stats(values, episode=True)
        for name, part, stat in (("28d", table, source_stats), ("22d", projected, projected_stats)):
            write_table(roots[name], part, ep, chunk_size)
            datasets[name].append(metadata)
            episode_stats[name].append({"episode_index": ep, "stats": stat})
        if ep in train_mapping:
            new = train_mapping[ep]
            train = projected.set_column(projected.schema.get_field_index("episode_index"), "episode_index",
                                         pa.array(np.full(length, new, dtype=np.int64)))
            indices = np.arange(train_offsets[ep], train_offsets[ep] + length, dtype=np.int64)
            train = train.set_column(train.schema.get_field_index("index"), "index", pa.array(indices))
            write_table(roots["train_view"], train, new, chunk_size)
            datasets["train_view"].append(dict(metadata, episode_index=new))
            episode_stats["train_view"].append({"episode_index": new, "stats": projected_stats})

    for name in roots:
        if name == "train_view":
            split = [dict(splits[ep], episode_index=train_mapping[ep]) for ep in train_ids]
        else:
            split = [splits[row["episode_index"]] for row in rows]
        write_metadata(roots[name], info, modality, cameras, split, datasets[name], episode_stats[name], tasks,
                       28 if name == "28d" else 22)
    mapping = [{"source_episode_index": ep, "new_episode_index": train_mapping[ep]} for ep in train_ids]
    dump(roots["train_view"] / "meta/source_episode_mapping.jsonl", mapping, lines=True)

    # Match the GR00T preprocessor's float32 mean/std and exact, frame-weighted quantiles.
    # Reading one feature at a time bounds memory and makes the train-only boundary explicit.
    train_files = sorted((roots["train_view"] / "data").glob("*/*.parquet"))
    train_stats = {}
    for feature in float_keys:
        parts = []
        for path in train_files:
            array = pq.read_table(path, columns=[feature])[feature]
            values = np.asarray(array.to_pylist(), dtype=np.float32).reshape(len(array), -1)
            parts.append(values)
        train_stats[feature] = stats(np.concatenate(parts))
    dump(roots["train_view"] / "meta/stats.json", train_stats)
    # The evaluation loader opens stats.json on the full 22D view as well.
    # Reuse the training split's normalization so validation frames never enter the estimates.
    dump(roots["22d"] / "meta/stats.json", train_stats)

    jobs = list(video_jobs(src, roots["28d"], cameras, rows, chunk_size, info["fps"]))
    print(f"Encoding and checking {len(jobs)} episode videos", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, _ in enumerate(pool.map(encode_video, jobs), 1):
            if n % 50 == 0:
                print(f"Videos {n}/{len(jobs)}", flush=True)
    for row in rows:
        ep = row["episode_index"]
        for camera in cameras:
            relative = VIDEO_PATH.format(episode_chunk=ep // chunk_size, video_key=camera, episode_index=ep)
            dests = [roots["22d"] / relative]
            if ep in train_mapping:
                new = train_mapping[ep]
                dests.append(roots["train_view"] / VIDEO_PATH.format(
                    episode_chunk=new // chunk_size, video_key=camera, episode_index=new))
            for path in dests:
                path.parent.mkdir(parents=True, exist_ok=True)
                os.link(roots["28d"] / relative, path)
    print(f"{len(rows)} episodes, {len(train_ids)} train, {next_row} train frames, {len(cameras)} RGB streams")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="LeRobot v3.0 dataset root")
    parser.add_argument("output", type=Path, help="Parent for 28d, 22d and train_view")
    parser.add_argument("profile", choices=PROFILES)
    parser.add_argument("--limit-episodes", type=int, help="Use the first N source episodes for a smoke run")
    parser.add_argument("--video-workers", type=int, default=3, help="Concurrent FFmpeg encoders (2 threads each)")
    args = parser.parse_args()
    if args.video_workers < 1 or (args.limit_episodes is not None and args.limit_episodes < 1):
        parser.error("workers and episode limit must be positive")
    src, dest = args.source.resolve(), args.output.resolve()
    if dest == src or src in dest.parents:
        parser.error("Choose an output outside the source dataset")
    dest.mkdir(parents=True, exist_ok=True)
    # A directory lock prevents overlapping builds from publishing mixed outputs.
    dir_fd = os.open(dest, os.O_RDONLY)
    try:
        fcntl.flock(dir_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        names = ("28d", "22d", "train_view")
        for name in names:
            if os.path.lexists(dest / name):
                parser.error(f"{dest / name} already exists. Choose a fresh output parent.")
        with TemporaryDirectory(prefix=".prepare-", dir=dest) as tmp:
            build(src, Path(tmp), args.profile, args.limit_episodes, args.video_workers)
            for name in names:
                (Path(tmp) / name).rename(dest / name)
    finally:
        os.close(dir_fd)
    print(f"Training dataset: {dest / 'train_view'}")


if __name__ == "__main__":
    main()
