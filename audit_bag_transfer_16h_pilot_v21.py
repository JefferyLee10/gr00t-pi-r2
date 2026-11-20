#!/usr/bin/env python3

from pathlib import Path
import json
import subprocess

import pyarrow.parquet as pq


ROOT = Path(
    "/path/to/lab-home/pi-r2-flow/datasets/"
    "bag_transfer_16h/pilot_workspace"
)

V21 = ROOT / "bag_transfer_pilot_v1"
V30 = ROOT / "bag_transfer_pilot_v1_v3.0"

mapping = json.loads(
    (
        V30
        / "meta/pilot_mapping.json"
    ).read_text(
        encoding="utf-8"
    )
)


def count_frames(path: Path) -> int:

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]

    out = subprocess.check_output(
        cmd,
        text=True,
    ).strip()

    return int(out)


print("=" * 120)
print("PILOT V2.1 FRAME AUDIT")
print("=" * 120)

all_ok = True

for item in mapping:

    ep = int(item["pilot_episode_index"])
    src_ep = int(item["source_episode_index"])
    expected = int(item["length"])

    chunk = ep // 1000

    parquet = (
        V21
        / "data"
        / f"chunk-{chunk:03d}"
        / f"episode_{ep:06d}.parquet"
    )

    rgb = (
        V21
        / "videos"
        / f"chunk-{chunk:03d}"
        / "observation.images.head_rgb"
        / f"episode_{ep:06d}.mp4"
    )

    depth = (
        V21
        / "videos"
        / f"chunk-{chunk:03d}"
        / "observation.images.head_depth"
        / f"episode_{ep:06d}.mp4"
    )

    rows = pq.ParquetFile(
        parquet
    ).metadata.num_rows

    rgb_frames = count_frames(rgb)
    depth_frames = count_frames(depth)

    ok = (
        rows == expected
        and rgb_frames == expected
        and depth_frames == expected
    )

    if not ok:
        all_ok = False

    print()
    print(
        f"pilot_ep={ep:02d} "
        f"source_ep={src_ep:04d}"
    )

    print(
        f"  expected = {expected}"
    )

    print(
        f"  parquet  = {rows}"
    )

    print(
        f"  RGB      = {rgb_frames}"
    )

    print(
        f"  Depth    = {depth_frames}"
    )

    print(
        "  RESULT   =",
        "PASS" if ok else "FAIL",
    )


print()
print("=" * 120)

if all_ok:
    print("ALL PILOT EPISODES PASS")
else:
    print("PILOT FAILED")
    raise SystemExit(1)
