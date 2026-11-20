#!/usr/bin/env python3

from pathlib import Path
import subprocess
import shutil

import pyarrow.parquet as pq


ROOT = Path(
    "/path/to/lab-home/pi-r2-flow/datasets/"
    "bag_transfer_16h/pilot_workspace"
)

# converter 已经把原 Pilot V3 移到了这里
V3 = ROOT / "bag_transfer_pilot_v1_v3.0"

# 当前转换后的 V2.1
V21 = ROOT / "bag_transfer_pilot_v1"

VIDEO_KEYS = [
    "observation.images.head_rgb",
    "observation.images.head_depth",
]

FPS = 20.0


def frame_count(path: Path) -> int:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]

    out = subprocess.check_output(
        cmd,
        text=True,
    ).strip()

    return int(out)


# ============================================================
# Load V3 episode metadata
# ============================================================

meta_files = sorted(
    (V3 / "meta" / "episodes").rglob("*.parquet")
)

if not meta_files:
    raise RuntimeError(
        f"No episode metadata found in {V3}"
    )

records = []

for p in meta_files:
    records.extend(
        pq.read_table(p).to_pylist()
    )

records.sort(
    key=lambda r: int(r["episode_index"])
)

print("=" * 110)
print("EXACT-FRAME VIDEO REPAIR")
print("=" * 110)
print("V3 source =", V3)
print("V2.1      =", V21)
print("episodes  =", len(records))
print()


for record in records:

    ep = int(record["episode_index"])
    length = int(record["length"])

    dest_chunk = ep // 1000

    print()
    print("=" * 110)
    print(
        f"EPISODE {ep:06d} | "
        f"expected frames = {length}"
    )
    print("=" * 110)

    for key in VIDEO_KEYS:

        chunk_idx = int(
            record[
                f"videos/{key}/chunk_index"
            ]
        )

        file_idx = int(
            record[
                f"videos/{key}/file_index"
            ]
        )

        start = float(
            record[
                f"videos/{key}/from_timestamp"
            ]
        )

        end = float(
            record[
                f"videos/{key}/to_timestamp"
            ]
        )

        src = (
            V3
            / "videos"
            / key
            / f"chunk-{chunk_idx:03d}"
            / f"file-{file_idx:03d}.mp4"
        )

        dst = (
            V21
            / "videos"
            / f"chunk-{dest_chunk:03d}"
            / key
            / f"episode_{ep:06d}.mp4"
        )

        tmp = dst.with_name(
            dst.stem + ".exact_tmp.mp4"
        )

        backup = dst.with_name(
            dst.stem + ".streamcopy_bad.mp4"
        )

        if not src.is_file():
            raise RuntimeError(
                f"Missing source video: {src}"
            )

        if not dst.is_file():
            raise RuntimeError(
                f"Missing existing V2.1 video: {dst}"
            )

        if backup.exists():
            raise RuntimeError(
                f"Backup already exists, refusing overwrite: "
                f"{backup}"
            )

        if tmp.exists():
            raise RuntimeError(
                f"Temporary file already exists: {tmp}"
            )

        print()
        print("key   =", key)
        print("src   =", src)
        print("start =", start)
        print("end   =", end)
        print("len   =", length)

        # ----------------------------------------------------
        # Keep the bad stream-copy result as a backup.
        # No deletion.
        # ----------------------------------------------------

        shutil.move(
            str(dst),
            str(backup),
        )

        dst.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # This is the same frame-exact strategy that fixed the
        # previous dataset:
        #
        #   -i SOURCE
        #   -ss START
        #   decode
        #   force 20 fps
        #   force EXACT number of frames
        #   re-encode
        #
        # DO NOT use -c copy.
        # ----------------------------------------------------

        if key == "observation.images.head_rgb":

            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-y",

                # Input FIRST, then accurate output seeking
                "-i", str(src),

                "-ss", f"{start:.9f}",

                "-vf",
                f"fps={FPS},setpts=PTS-STARTPTS",

                "-frames:v", str(length),

                "-an",

                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "16",
                "-pix_fmt", "yuv420p",

                str(tmp),
            ]

        else:

            # Depth is gray8_scaled packed in video.
            # Use lossless H.264 instead of CRF encoding.
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-y",

                "-i", str(src),

                "-ss", f"{start:.9f}",

                "-vf",
                f"fps={FPS},setpts=PTS-STARTPTS",

                "-frames:v", str(length),

                "-an",

                "-c:v", "libx264",
                "-preset", "fast",

                # lossless H.264
                "-qp", "0",

                "-pix_fmt", "yuv420p",

                str(tmp),
            ]

        print(
            "$",
            " ".join(cmd),
        )

        subprocess.run(
            cmd,
            check=True,
        )

        n = frame_count(tmp)

        print(
            f"exact output frames = {n}"
        )

        if n != length:
            raise RuntimeError(
                f"{key} ep={ep}: "
                f"expected={length}, actual={n}"
            )

        # Only after exact frame count passes
        # do we promote tmp to final output.
        shutil.move(
            str(tmp),
            str(dst),
        )

        print("PASS")


print()
print("=" * 110)
print("ALL EXACT-FRAME PILOT VIDEOS GENERATED")
print("=" * 110)
