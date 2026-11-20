#!/usr/bin/env python3

from pathlib import Path
import copy
import json

import pyarrow as pa
import pyarrow.parquet as pq


SRC = Path(
    "/path/to/lab-home/pi-r2-flow/datasets/"
    "bag_transfer_16h/source_v3_view"
)

PARENT = Path(
    "/path/to/lab-home/pi-r2-flow/datasets/"
    "bag_transfer_16h/pilot_workspace"
)

DST = PARENT / "bag_transfer_pilot_v1"


# ============================================================
# Safety
# ============================================================

for p in [
    DST,
    PARENT / "bag_transfer_pilot_v1_v3.0",
    PARENT / "bag_transfer_pilot_v1_v2.1",
]:
    if p.exists() or p.is_symlink():
        raise RuntimeError(
            f"Refusing to overwrite existing path: {p}"
        )


# ============================================================
# Load source episode metadata
# ============================================================

ep_files = sorted(
    (SRC / "meta" / "episodes").rglob("*.parquet")
)

if not ep_files:
    raise RuntimeError("No source episode metadata")

records = []

for p in ep_files:
    records.extend(
        pq.read_table(p).to_pylist()
    )

records.sort(
    key=lambda x: int(x["episode_index"])
)

if len(records) != 1293:
    raise RuntimeError(
        f"Expected 1293 episodes, got {len(records)}"
    )

by_ep = {
    int(r["episode_index"]): r
    for r in records
}


# ============================================================
# Find representative episodes
# ============================================================

selected = []


def add(ep):
    ep = int(ep)
    if ep not in selected:
        selected.append(ep)


# Start
add(0)

# Known validation episode
add(6)

# First RGB source-video transition
prev = int(
    records[0][
        "videos/observation.images.head_rgb/file_index"
    ]
)

for r in records[1:]:
    cur = int(
        r[
            "videos/observation.images.head_rgb/file_index"
        ]
    )
    if cur != prev:
        add(r["episode_index"])
        break
    prev = cur


# First Depth source-video transition
prev = int(
    records[0][
        "videos/observation.images.head_depth/file_index"
    ]
)

for r in records[1:]:
    cur = int(
        r[
            "videos/observation.images.head_depth/file_index"
        ]
    )
    if cur != prev:
        add(r["episode_index"])
        break
    prev = cur


# First episode of data file 1 and 2
for target_file in [1, 2]:
    for r in records:
        if int(r["data/file_index"]) == target_file:
            add(r["episode_index"])
            break


# Middle and last
add(len(records) // 2)
add(len(records) - 1)


# Fill if duplicates reduced count
for ep in [
    100, 250, 400, 700,
    900, 1100, 1200,
]:
    if len(selected) >= 8:
        break
    add(ep)

selected = selected[:8]

print("Selected source episodes:")
print(selected)


# ============================================================
# Determine global starting row of each source data file
# ============================================================

data_file_offsets = {}

for r in records:
    key = (
        int(r["data/chunk_index"]),
        int(r["data/file_index"]),
    )

    start = int(r["dataset_from_index"])

    if key not in data_file_offsets:
        data_file_offsets[key] = start
    else:
        data_file_offsets[key] = min(
            data_file_offsets[key],
            start,
        )


# ============================================================
# Build one consolidated pilot data parquet
# ============================================================

DST.mkdir(parents=True)
(DST / "data/chunk-000").mkdir(parents=True)
(DST / "meta/episodes/chunk-000").mkdir(
    parents=True
)

tables = []
pilot_records = []
mapping = []

pilot_global = 0

table_cache = {}

for new_ep, old_ep in enumerate(selected):

    r = copy.deepcopy(by_ep[old_ep])

    key = (
        int(r["data/chunk_index"]),
        int(r["data/file_index"]),
    )

    src_parquet = (
        SRC
        / "data"
        / f"chunk-{key[0]:03d}"
        / f"file-{key[1]:03d}.parquet"
    )

    if key not in table_cache:
        table_cache[key] = pq.read_table(
            src_parquet
        )

    source_table = table_cache[key]

    global_start = int(
        r["dataset_from_index"]
    )
    global_stop = int(
        r["dataset_to_index"]
    )

    length = global_stop - global_start

    local_start = (
        global_start
        - data_file_offsets[key]
    )

    episode_table = source_table.slice(
        local_start,
        length,
    )

    if episode_table.num_rows != length:
        raise RuntimeError(
            f"Episode {old_ep}: expected "
            f"{length} rows, got "
            f"{episode_table.num_rows}"
        )

    # Rewrite episode_index inside data rows
    if "episode_index" in episode_table.column_names:
        idx = episode_table.schema.get_field_index(
            "episode_index"
        )

        new_col = pa.array(
            [new_ep] * length,
            type=episode_table[
                "episode_index"
            ].type,
        )

        episode_table = (
            episode_table.set_column(
                idx,
                "episode_index",
                new_col,
            )
        )

    # If a global index column exists, rewrite it
    if "index" in episode_table.column_names:
        idx = episode_table.schema.get_field_index(
            "index"
        )

        new_col = pa.array(
            range(
                pilot_global,
                pilot_global + length,
            ),
            type=episode_table["index"].type,
        )

        episode_table = (
            episode_table.set_column(
                idx,
                "index",
                new_col,
            )
        )

    tables.append(episode_table)

    # Rewrite V3 episode metadata for pilot
    r["episode_index"] = new_ep

    r["data/chunk_index"] = 0
    r["data/file_index"] = 0

    r["dataset_from_index"] = pilot_global
    r["dataset_to_index"] = (
        pilot_global + length
    )

    if "meta/episodes/chunk_index" in r:
        r["meta/episodes/chunk_index"] = 0

    if "meta/episodes/file_index" in r:
        r["meta/episodes/file_index"] = 0

    pilot_records.append(r)

    mapping.append(
        {
            "pilot_episode_index": new_ep,
            "source_episode_index": old_ep,
            "source_episode_name": (
                next(
                    (
                        x["episode_name"]
                        for x in json.loads(
                            (
                                SRC
                                / "meta"
                                / "split.json"
                            ).read_text(
                                encoding="utf-8"
                            )
                        )["episodes"]
                        if int(
                            x["episode_index"]
                        ) == old_ep
                    ),
                    None,
                )
            ),
            "length": length,
        }
    )

    pilot_global += length


pilot_table = pa.concat_tables(
    tables,
    promote_options="default",
)

if pilot_table.num_rows != pilot_global:
    raise RuntimeError(
        "Pilot row-count mismatch"
    )

pq.write_table(
    pilot_table,
    DST
    / "data/chunk-000/file-000.parquet",
)


# ============================================================
# Write episode metadata
# ============================================================

ep_table = pa.Table.from_pylist(
    pilot_records
)

pq.write_table(
    ep_table,
    DST
    / "meta/episodes/chunk-000/file-000.parquet",
)


# ============================================================
# info.json
# ============================================================

source_info = json.loads(
    (SRC / "meta/info.json").read_text(
        encoding="utf-8"
    )
)

info = copy.deepcopy(source_info)

info["total_episodes"] = len(selected)
info["total_frames"] = pilot_global

if "splits" in info:
    info["splits"] = {
        "train": f"0:{len(selected)}"
    }

# Keep v3 marker for converter
info["codebase_version"] = "v3.0"

(DST / "meta/info.json").write_text(
    json.dumps(
        info,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


# ============================================================
# tasks.parquet
# ============================================================

tasks = SRC / "meta/tasks.parquet"

if not tasks.exists():
    raise RuntimeError(
        f"Missing {tasks}"
    )

import shutil

shutil.copy2(
    tasks,
    DST / "meta/tasks.parquet",
)


# ============================================================
# Pilot mapping
# ============================================================

(DST / "meta/pilot_mapping.json").write_text(
    json.dumps(
        mapping,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


# ============================================================
# Video link
# ============================================================

video_link = DST / "videos"

video_link.symlink_to(
    SRC / "videos",
    target_is_directory=True,
)


# ============================================================
# Final verification
# ============================================================

print()
print("=" * 100)
print("PILOT V3 CREATED")
print("=" * 100)

print("DST =", DST)
print("episodes =", len(selected))
print("rows =", pilot_global)

print()
print("Mapping:")

for x in mapping:
    print(x)

print()
print("videos ->", video_link.resolve())

print()
print("PASS")
