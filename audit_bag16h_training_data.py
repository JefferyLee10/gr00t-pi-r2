#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import gc
import json
import re
import sys

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

try:
    import cv2
except Exception:
    cv2 = None


# ==============================================================================
# Paths
# ==============================================================================

SRC = Path(
    "/path/to/lab-home/pi-r2-flow/datasets/"
    "bag_transfer_16h/source_v3_view"
)

V28 = Path(
    "/path/to/lab-home/pi-r2-flow/datasets/"
    "bag_transfer_16h/v2_1_full_28d_exact"
)

V22 = Path(
    "/path/to/lab-home/pi-r2-flow/datasets/"
    "bag_transfer_16h/v2_1_full_22d_exact"
)

TRAIN = Path(
    "/path/to/lab-home/pi-r2-flow/datasets/"
    "bag_transfer_16h/train_dataset_view_22d_headrgb"
)

CKPT = Path(
    "/path/to/lab-home/pi-r2-flow/experiments/"
    "bag_transfer_16h_22d_headrgb_pir2_exact/"
    "train_2gpu23_gbs256_acc1_317k_pir2_headrgb_v1/"
    "checkpoint-317000"
)

EXPECTED_FULL_EPISODES = 1293
EXPECTED_FULL_ROWS = 1189157
EXPECTED_TRAIN_EPISODES = 1164
EXPECTED_TRAIN_ROWS = 1059331
EXPECTED_FPS = 20.0

FAILS = []
WARNS = []


def title(s):
    print()
    print("=" * 120)
    print(s)
    print("=" * 120)


def fail(msg):
    FAILS.append(str(msg))
    print("[FAIL]", msg)


def warn(msg):
    WARNS.append(str(msg))
    print("[WARN]", msg)


def ok(msg):
    print("[ OK ]", msg)


def dataset(root):
    p = root / "data"
    if not p.exists():
        raise RuntimeError(f"missing data/: {p}")
    return ds.dataset(str(p), format="parquet")


def scalar_np(table, name, dtype=None):
    arr = table.column(name).combine_chunks().to_numpy(zero_copy_only=False)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return np.asarray(arr)


def list2d_np(table, name):
    arr = table.column(name).combine_chunks()

    if pa.types.is_fixed_size_list(arr.type):
        width = arr.type.list_size
        values = arr.values.to_numpy(zero_copy_only=False)
        return np.asarray(values, dtype=np.float32).reshape(len(arr), width)

    # fallback
    return np.asarray(arr.to_pylist(), dtype=np.float32)


def transform_28_to_22(x):
    x = np.asarray(x, dtype=np.float32)

    if x.ndim != 2 or x.shape[1] != 28:
        raise RuntimeError(f"28D expected, got {x.shape}")

    left_arm = x[:, 0:7]
    right_arm = x[:, 8:15]

    left_j012 = x[:, 16:19]
    left_other3 = x[:, 19:22].mean(
        axis=1, keepdims=True, dtype=np.float32
    )

    right_j012 = x[:, 22:25]
    right_other3 = x[:, 25:28].mean(
        axis=1, keepdims=True, dtype=np.float32
    )

    return np.concatenate(
        [
            left_arm,
            right_arm,
            left_j012,
            left_other3,
            right_j012,
            right_other3,
        ],
        axis=1,
    ).astype(np.float32)


def load_numeric(root):
    d = dataset(root)

    required = [
        "episode_index",
        "frame_index",
        "timestamp",
        "observation.state",
        "action",
    ]

    missing = [x for x in required if x not in d.schema.names]
    if missing:
        raise RuntimeError(f"{root}: missing columns {missing}")

    print(f"loading: {root}")
    t = d.to_table(columns=required)

    ep = scalar_np(t, "episode_index", np.int64)
    fr = scalar_np(t, "frame_index", np.int64)
    ts = scalar_np(t, "timestamp", np.float64)
    state = list2d_np(t, "observation.state")
    action = list2d_np(t, "action")

    # Do not trust physical parquet order. Canonicalize using episode/frame.
    order = np.lexsort((fr, ep))

    return {
        "episode": ep[order],
        "frame": fr[order],
        "timestamp": ts[order],
        "state": state[order],
        "action": action[order],
    }


def maxabs(a, b):
    if a.shape != b.shape:
        return np.inf
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def compare_keys(a, b, label):
    if not np.array_equal(a["episode"], b["episode"]):
        fail(f"{label}: episode_index mismatch")
        return False

    if not np.array_equal(a["frame"], b["frame"]):
        fail(f"{label}: frame_index mismatch")
        return False

    dt = maxabs(a["timestamp"], b["timestamp"])
    print(f"{label}: timestamp max abs diff = {dt:.12g}")

    if dt > 1e-6:
        fail(f"{label}: timestamps changed")

    return True


# ==============================================================================
# 1. Basic filesystem
# ==============================================================================

title("1. BASIC PATH CHECK")

for p in [SRC, V28, V22, TRAIN, CKPT]:
    if p.exists():
        ok(str(p))
    else:
        fail(f"missing path: {p}")

print()
print("TRAIN/videos:")
vp = TRAIN / "videos"

if vp.exists():
    print(" path    =", vp)
    print(" symlink =", vp.is_symlink())
    try:
        print(" resolves =", vp.resolve())
    except Exception as e:
        warn(f"cannot resolve TRAIN/videos: {e}")
else:
    fail("TRAIN/videos does not exist")


# ==============================================================================
# 2. Dataset row / episode counts
# ==============================================================================

title("2. GLOBAL ROW / EPISODE COUNTS")

roots = {
    "SOURCE V3": SRC,
    "V2.1 28D": V28,
    "V2.1 22D": V22,
    "TRAIN VIEW": TRAIN,
}

episode_sets = {}

for name, root in roots.items():
    d = dataset(root)

    n = d.count_rows()

    ept = d.to_table(columns=["episode_index"])
    eps = scalar_np(ept, "episode_index", np.int64)

    uniq, counts = np.unique(eps, return_counts=True)
    episode_sets[name] = set(map(int, uniq))

    print(
        f"{name:12s}: "
        f"rows={n:,} "
        f"episodes={len(uniq):,} "
        f"min_ep={uniq.min()} "
        f"max_ep={uniq.max()}"
    )

    if name != "TRAIN VIEW":
        if n != EXPECTED_FULL_ROWS:
            fail(f"{name}: rows {n} != {EXPECTED_FULL_ROWS}")
        if len(uniq) != EXPECTED_FULL_EPISODES:
            fail(
                f"{name}: episodes {len(uniq)} "
                f"!= {EXPECTED_FULL_EPISODES}"
            )
    else:
        if n != EXPECTED_TRAIN_ROWS:
            fail(f"TRAIN rows {n} != {EXPECTED_TRAIN_ROWS}")
        if len(uniq) != EXPECTED_TRAIN_EPISODES:
            fail(
                f"TRAIN episodes {len(uniq)} "
                f"!= {EXPECTED_TRAIN_EPISODES}"
            )


# ==============================================================================
# 3. Source V3 episode metadata vs actual rows
# ==============================================================================

title("3. SOURCE V3 EPISODE METADATA")

meta_files = sorted((SRC / "meta" / "episodes").rglob("*.parquet"))

if not meta_files:
    fail("source V3 meta/episodes parquet missing")
    src_meta = None
else:
    tables = [pq.read_table(p) for p in meta_files]
    src_meta = pa.concat_tables(tables).to_pandas()

    print("episode metadata rows =", len(src_meta))

    required_meta = [
        "episode_index",
        "length",
        "dataset_from_index",
        "dataset_to_index",
        "videos/observation.images.head_rgb/chunk_index",
        "videos/observation.images.head_rgb/file_index",
        "videos/observation.images.head_rgb/from_timestamp",
        "videos/observation.images.head_rgb/to_timestamp",
    ]

    for k in required_meta:
        if k not in src_meta.columns:
            fail(f"source episode metadata missing: {k}")

    if len(src_meta) == EXPECTED_FULL_EPISODES:
        ok("source V3 metadata episode count = 1293")

    bad = src_meta[
        (
            src_meta["dataset_to_index"]
            - src_meta["dataset_from_index"]
        )
        != src_meta["length"]
    ]

    print("bad dataset index-length rows =", len(bad))

    if len(bad):
        fail("source V3 dataset_from/to index does not match length")
    else:
        ok("all source metadata lengths internally consistent")


# ==============================================================================
# 4. Full numeric comparison: source V3 -> exact V2.1 28D
# ==============================================================================

title("4. FULL NUMERIC CHECK: SOURCE V3 -> V2.1 28D")

src = load_numeric(SRC)
v28 = load_numeric(V28)

print("SOURCE state/action =", src["state"].shape, src["action"].shape)
print("V28    state/action =", v28["state"].shape, v28["action"].shape)

compare_keys(src, v28, "SRC -> V28")

if src["state"].shape[1] != 28:
    fail(f"SOURCE state dimension = {src['state'].shape[1]}, expected 28")

if src["action"].shape[1] != 28:
    fail(f"SOURCE action dimension = {src['action'].shape[1]}, expected 28")

d_state = maxabs(src["state"], v28["state"])
d_action = maxabs(src["action"], v28["action"])

print("state max abs diff =", d_state)
print("action max abs diff =", d_action)

if d_state > 1e-6:
    fail("V3 -> V2.1 changed state numeric values")
else:
    ok("V3 -> V2.1 state values exact")

if d_action > 1e-6:
    fail("V3 -> V2.1 changed action numeric values")
else:
    ok("V3 -> V2.1 action values exact")

del src
gc.collect()


# ==============================================================================
# 5. Full numeric comparison: 28D -> 22D
# ==============================================================================

title("5. FULL NUMERIC CHECK: 28D -> 22D")

v22 = load_numeric(V22)

compare_keys(v28, v22, "V28 -> V22")

print("V22 state/action =", v22["state"].shape, v22["action"].shape)

expected_state22 = transform_28_to_22(v28["state"])
expected_action22 = transform_28_to_22(v28["action"])

d_state = maxabs(expected_state22, v22["state"])
d_action = maxabs(expected_action22, v22["action"])

print("22D state mapping max abs diff =", d_state)
print("22D action mapping max abs diff =", d_action)

if d_state > 1e-6:
    fail("28D -> 22D STATE mapping is wrong")
else:
    ok("28D -> 22D STATE mapping exact")

if d_action > 1e-6:
    fail("28D -> 22D ACTION mapping is wrong")
else:
    ok("28D -> 22D ACTION mapping exact")

del v28, expected_state22, expected_action22
gc.collect()


# ==============================================================================
# 6. Full train-view comparison
# ==============================================================================

title("6. FULL NUMERIC CHECK: FULL 22D -> TRAIN VIEW")

train = load_numeric(TRAIN)

train_eps = np.unique(train["episode"])
mask = np.isin(v22["episode"], train_eps)

v22_train = {
    k: v[mask]
    for k, v in v22.items()
}

compare_keys(v22_train, train, "V22 -> TRAIN")

d_state = maxabs(v22_train["state"], train["state"])
d_action = maxabs(v22_train["action"], train["action"])

print("train state max abs diff =", d_state)
print("train action max abs diff =", d_action)

if d_state > 1e-6:
    fail("TRAIN VIEW state differs from full V22")
else:
    ok("TRAIN VIEW state exact")

if d_action > 1e-6:
    fail("TRAIN VIEW action differs from full V22")
else:
    ok("TRAIN VIEW action exact")

val_eps = episode_sets["V2.1 22D"] - episode_sets["TRAIN VIEW"]

print("train episodes =", len(train_eps))
print("held-out episodes =", len(val_eps))
print("overlap =", len(set(train_eps) & val_eps))

if len(val_eps) != 129:
    fail(f"expected 129 held-out episodes, got {len(val_eps)}")
else:
    ok("1164 train / 129 held-out split consistent")


# ==============================================================================
# 7. Language / modality metadata
# ==============================================================================

title("7. LANGUAGE + MODALITY METADATA")

tasks_path = TRAIN / "meta" / "tasks.jsonl"
mod_path = TRAIN / "meta" / "modality.json"

if not tasks_path.exists():
    fail("tasks.jsonl missing")
else:
    tasks = []
    for line in tasks_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            tasks.append(json.loads(line))

    print("tasks =", tasks)

    if not tasks:
        fail("tasks.jsonl empty")
    else:
        non_strings = [
            x for x in tasks
            if not isinstance(x.get("task"), str)
        ]

        if non_strings:
            fail(f"non-string task entries: {non_strings[:5]}")
        else:
            ok("all task values are strings")

        values = sorted(set(x.get("task") for x in tasks))
        print("unique task strings =", values)

        if values != ["快递分拣"]:
            fail(f"unexpected task strings: {values}")
        else:
            ok('task mapping = "快递分拣"')

if not mod_path.exists():
    fail("modality.json missing")
else:
    modality = json.loads(mod_path.read_text(encoding="utf-8"))

    ann = (
        modality
        .get("annotation", {})
        .get("human.action.task_description", {})
    )

    print("language modality =", ann)

    if ann.get("original_key") != "task_index":
        fail("language original_key is not task_index")
    else:
        ok("language original_key = task_index")


# ==============================================================================
# 8. Train stats
# ==============================================================================

title("8. TRAIN-ONLY STATS")

stats_path = TRAIN / "meta" / "stats.json"

if not stats_path.exists():
    fail("TRAIN meta/stats.json missing")
else:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    for key in ["observation.state", "action"]:
        if key not in stats:
            fail(f"stats missing {key}")
            continue

        print()
        print(key)

        for statname in [
            "mean", "std", "min", "max", "q01", "q99"
        ]:
            if statname not in stats[key]:
                fail(f"{key}: missing {statname}")
                continue

            x = np.asarray(stats[key][statname], dtype=np.float64).reshape(-1)

            print(
                f" {statname:5s}: "
                f"shape={x.shape} "
                f"finite={np.isfinite(x).all()}"
            )

            if x.size != 22:
                fail(f"{key}/{statname}: dimension {x.size} != 22")

            if not np.isfinite(x).all():
                fail(f"{key}/{statname}: NaN/Inf")

    if "relative_stats" in stats:
        fail("ABS dataset unexpectedly contains relative_stats at top level")
    else:
        ok("no relative_stats in ABS train stats")


# ==============================================================================
# 9. Checkpoint processor - what did the ACTUAL model train on?
# ==============================================================================

title("9. CHECKPOINT PROCESSOR / ACTUAL MODEL INPUT")

proc_path = CKPT / "processor_config.json"

if not proc_path.exists():
    fail("checkpoint processor_config.json missing")
else:
    proc = json.loads(proc_path.read_text(encoding="utf-8"))

    modality_lists = []

    def walk(x, path="root"):
        if isinstance(x, dict):
            for k, v in x.items():
                p = f"{path}.{k}"

                if k == "modality_keys" and isinstance(v, list):
                    modality_lists.append((p, v))

                walk(v, p)

        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")

    walk(proc)

    print("modality_keys found:")

    camera_bad = False
    head_seen = False

    for p, v in modality_lists:
        print(" ", p, "=", v)

        camera_values = [
            str(z) for z in v
            if isinstance(z, str)
            and (
                "rgb" in z.lower()
                or "depth" in z.lower()
            )
        ]

        if camera_values:
            if any("head_rgb" in z for z in camera_values):
                head_seen = True

            if any(
                ("depth" in z.lower())
                or ("wrist" in z.lower())
                for z in camera_values
            ):
                camera_bad = True

    if not head_seen:
        warn("could not positively identify head_rgb in modality_keys")

    if camera_bad:
        fail("checkpoint processor appears to contain depth/wrist camera input")
    else:
        ok("no depth/wrist input found in checkpoint modality_keys")


# ==============================================================================
# 10. Critical temporal-label check
#
# Compare action[t] to state[t+k].
#
# Dataset is supposed to use absolute_state_t_plus_1 semantics.
# If action[t] matches state[t] much better than state[t+1], that is suspicious.
# ==============================================================================

title("10. ACTION / STATE TEMPORAL ALIGNMENT")

ep_unique = np.unique(train["episode"])

# deterministic spread across whole training set
pick_idx = np.linspace(
    0, len(ep_unique) - 1, min(100, len(ep_unique))
).round().astype(int)

selected_eps = ep_unique[pick_idx]

lags = range(-4, 7)

sums_all = {k: 0.0 for k in lags}
count_all = {k: 0 for k in lags}

sums_arm = {k: 0.0 for k in lags}
count_arm = {k: 0 for k in lags}

sums_hand = {k: 0.0 for k in lags}
count_hand = {k: 0 for k in lags}

for ep in selected_eps:
    m = train["episode"] == ep
    S = train["state"][m]
    A = train["action"][m]

    n = len(S)

    if n < 20:
        continue

    for lag in lags:
        if lag >= 0:
            if lag == 0:
                aa = A
                ss = S
            else:
                aa = A[:-lag]
                ss = S[lag:]
        else:
            d = -lag
            aa = A[d:]
            ss = S[:-d]

        if len(aa) == 0:
            continue

        diff = np.abs(
            aa.astype(np.float64)
            - ss.astype(np.float64)
        )

        sums_all[lag] += diff.sum()
        count_all[lag] += diff.size

        arm = diff[:, :14]
        hand = diff[:, 14:22]

        sums_arm[lag] += arm.sum()
        count_arm[lag] += arm.size

        sums_hand[lag] += hand.sum()
        count_hand[lag] += hand.size


mae_all = {
    k: sums_all[k] / count_all[k]
    for k in lags
}

mae_arm = {
    k: sums_arm[k] / count_arm[k]
    for k in lags
}

mae_hand = {
    k: sums_hand[k] / count_hand[k]
    for k in lags
}

print()
print("lag definition: compare action[t] with state[t + lag]")
print()

for k in lags:
    print(
        f"lag {k:+d}: "
        f"all={mae_all[k]:.8f} "
        f"arm={mae_arm[k]:.8f} "
        f"hand={mae_hand[k]:.8f}"
    )

best_all = min(mae_all, key=mae_all.get)
best_arm = min(mae_arm, key=mae_arm.get)

print()
print("BEST lag all =", best_all)
print("BEST lag arm =", best_arm)

print()
print("action[t] vs state[t]   =", mae_all[0])
print("action[t] vs state[t+1] =", mae_all[1])

if best_all == 1 or best_arm == 1:
    ok("temporal relation is consistent with t+1 target")
else:
    warn(
        "best action/state lag is NOT +1. "
        "This needs careful investigation."
    )

ratio = mae_all[0] / max(mae_all[1], 1e-12)

print("MAE(t) / MAE(t+1) =", ratio)

if mae_all[0] < mae_all[1] * 0.8:
    fail(
        "action[t] is substantially closer to state[t] "
        "than state[t+1]; possible temporal-label problem"
    )


# ==============================================================================
# 11. Video frame-count / FPS audit
# ==============================================================================

title("11. FINAL HEAD_RGB VIDEO FRAME COUNT / FPS")

if cv2 is None:
    warn("opencv not available; skipping video audit")
else:
    video_root = V22 / "videos"

    vmap = {}

    for p in video_root.rglob("*.mp4"):
        sp = p.as_posix()

        if "head_rgb" not in sp:
            continue

        m = re.search(
            r"episode[_-](\d+)",
            p.stem,
            flags=re.IGNORECASE,
        )

        if m:
            vmap[int(m.group(1))] = p

    print("head_rgb episode videos found =", len(vmap))

    if len(vmap) < EXPECTED_FULL_EPISODES:
        warn(
            "could not map all V2 episode videos by filename; "
            "showing examples:"
        )

        for p in list(video_root.rglob("*.mp4"))[:10]:
            print(" ", p)

    # expected data row count per episode
    uniq, cnt = np.unique(
        v22["episode"],
        return_counts=True,
    )

    expected_counts = dict(zip(map(int, uniq), map(int, cnt)))

    bad_video = []
    checked = 0

    for ep, expected_n in expected_counts.items():
        p = vmap.get(ep)

        if p is None:
            bad_video.append((ep, "missing video", expected_n))
            continue

        cap = cv2.VideoCapture(str(p))

        if not cap.isOpened():
            bad_video.append((ep, "cannot open", expected_n))
            continue

        frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))

        cap.release()

        checked += 1

        if frames != expected_n:
            bad_video.append(
                (
                    ep,
                    f"frames={frames}",
                    f"expected={expected_n}",
                )
            )

        if not np.isfinite(fps) or abs(fps - EXPECTED_FPS) > 0.05:
            bad_video.append(
                (
                    ep,
                    f"fps={fps}",
                    f"expected={EXPECTED_FPS}",
                )
            )

    print("videos checked =", checked)
    print("video mismatches =", len(bad_video))

    if bad_video:
        for x in bad_video[:50]:
            print(" BAD:", x)

        fail("head_rgb video frame count/FPS mismatches detected")
    elif checked == EXPECTED_FULL_EPISODES:
        ok("all 1293 head_rgb videos match parquet episode length and 20 Hz")
    else:
        warn("video filename mapping incomplete")


# ==============================================================================
# 12. Source-V3 -> final video frame-position audit
# ==============================================================================

title("12. SOURCE V3 -> FINAL VIDEO TEMPORAL OFFSET TEST")

if cv2 is None or src_meta is None:
    warn("cannot perform source/final pixel temporal test")
else:

    # Find final episode videos
    final_map = {}

    for p in (V22 / "videos").rglob("*.mp4"):
        if "head_rgb" not in p.as_posix():
            continue

        m = re.search(
            r"episode[_-](\d+)",
            p.stem,
            flags=re.IGNORECASE,
        )

        if m:
            final_map[int(m.group(1))] = p


    def find_source_video(chunk_idx, file_idx):
        candidates = []

        for p in (SRC / "videos").rglob("*.mp4"):
            s = p.as_posix()

            if "head_rgb" not in s:
                continue

            if (
                f"chunk-{chunk_idx:03d}" not in s
                and f"chunk_{chunk_idx:03d}" not in s
            ):
                continue

            nums = [
                f"file-{file_idx:03d}",
                f"file_{file_idx:03d}",
                f"file-{file_idx:06d}",
                f"file_{file_idx:06d}",
            ]

            if any(x in p.stem for x in nums):
                candidates.append(p)

        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) > 1:
            return candidates[0]

        return None


    # Use episodes spread across full dataset
    eps_all = sorted(map(int, src_meta["episode_index"].tolist()))

    idxs = np.linspace(
        0, len(eps_all) - 1, 12
    ).round().astype(int)

    sampled_eps = [eps_all[i] for i in idxs]

    best_deltas = []
    pixel_tested = 0

    for ep in sampled_eps:

        row = src_meta[
            src_meta["episode_index"] == ep
        ]

        if len(row) != 1:
            warn(f"ep {ep}: source metadata row count={len(row)}")
            continue

        row = row.iloc[0]

        chunk_idx = int(
            row[
                "videos/observation.images.head_rgb/chunk_index"
            ]
        )

        file_idx = int(
            row[
                "videos/observation.images.head_rgb/file_index"
            ]
        )

        from_ts = float(
            row[
                "videos/observation.images.head_rgb/from_timestamp"
            ]
        )

        src_video = find_source_video(chunk_idx, file_idx)
        dst_video = final_map.get(ep)

        if src_video is None or dst_video is None:
            warn(
                f"ep {ep}: cannot resolve video "
                f"src={src_video}, dst={dst_video}"
            )
            continue

        cap_src = cv2.VideoCapture(str(src_video))
        cap_dst = cv2.VideoCapture(str(dst_video))

        if not cap_src.isOpened() or not cap_dst.isOpened():
            warn(f"ep {ep}: video open failed")
            cap_src.release()
            cap_dst.release()
            continue

        src_fps = float(cap_src.get(cv2.CAP_PROP_FPS))
        dst_frames = int(
            round(cap_dst.get(cv2.CAP_PROP_FRAME_COUNT))
        )

        source_start = int(round(from_ts * src_fps))

        # Prefer interior frames with more useful motion than first/last.
        fractions = [0.20, 0.40, 0.60, 0.80]

        ep_deltas = []

        for frac in fractions:
            fi = int(round((dst_frames - 1) * frac))

            cap_dst.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok_d, dst_frame = cap_dst.read()

            if not ok_d:
                continue

            expected_src = source_start + fi

            best = None

            # Search ±4 frames around the expected position.
            for delta in range(-4, 5):
                src_idx = expected_src + delta

                if src_idx < 0:
                    continue

                cap_src.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
                ok_s, src_frame = cap_src.read()

                if not ok_s:
                    continue

                if src_frame.shape != dst_frame.shape:
                    continue

                # downsample for speed
                a = src_frame[::8, ::8].astype(np.float32)
                b = dst_frame[::8, ::8].astype(np.float32)

                mad = float(np.mean(np.abs(a - b)))

                if best is None or mad < best[0]:
                    best = (mad, delta)

            if best is not None:
                mad, delta = best
                ep_deltas.append(delta)
                best_deltas.append(delta)
                pixel_tested += 1

                print(
                    f"ep={ep:4d} "
                    f"frame={fi:5d} "
                    f"best_delta={delta:+d} "
                    f"MAD={mad:.3f}"
                )

        cap_src.release()
        cap_dst.release()

        if ep_deltas:
            print(
                f" ep {ep} deltas = {ep_deltas}"
            )

    if pixel_tested == 0:
        warn("no source/final frames could be compared")
    else:
        vals = np.asarray(best_deltas)

        print()
        print("frame comparisons =", pixel_tested)
        print("best delta distribution:")

        for x in sorted(set(vals.tolist())):
            print(
                f" delta {x:+d}: "
                f"{int(np.sum(vals == x))}"
            )

        frac_zero = float(np.mean(vals == 0))

        print("fraction best_delta == 0 =", frac_zero)

        # Do not use one static frame as hard proof,
        # but systematic non-zero offset is a major red flag.
        if frac_zero >= 0.75:
            ok(
                "source/final video temporal positions strongly "
                "support zero-frame conversion offset"
            )
        else:
            warn(
                "less than 75% sampled frames prefer zero offset. "
                "Possible video conversion shift; inspect manually."
            )


# ==============================================================================
# Final summary
# ==============================================================================

title("FINAL AUDIT SUMMARY")

print("FAIL count =", len(FAILS))
print("WARN count =", len(WARNS))

if FAILS:
    print()
    print("FAILURES:")
    for i, x in enumerate(FAILS, 1):
        print(f"{i:02d}. {x}")

if WARNS:
    print()
    print("WARNINGS:")
    for i, x in enumerate(WARNS, 1):
        print(f"{i:02d}. {x}")

print()

if not FAILS:
    print("CORE NUMERIC/METADATA AUDIT: PASS")
else:
    print("CORE NUMERIC/METADATA AUDIT: FAIL")

print("=" * 120)
