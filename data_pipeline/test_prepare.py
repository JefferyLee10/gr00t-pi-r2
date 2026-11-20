"""Small offline regression: interleaved splits, nonzero video offsets and train-only normalization."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from prepare import INPUT_GROUPS, NUMERIC, PROFILES, dump


def write_parquet(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def make_v3(root):
    # Three episodes share one RGB video and two Parquet shards. Episode 1 is held out.
    fps, frames = 20, 4
    meta = root / "meta"
    cameras = ["observation.images." + key for key in PROFILES["rgb3"]]
    features = {key: {"dtype": "float32", "shape": [28]} for key in NUMERIC}
    features["timestamp"] = {"dtype": "float32", "shape": [1]}
    for camera in cameras:
        features[camera] = {"dtype": "video", "shape": [3, 16, 16]}
        dest = root / "videos" / camera / "chunk-000/file-000.mp4"
        dest.parent.mkdir(parents=True)
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c=blue:s=16x16:r={fps}",
            "-frames:v", str(3 * frames), "-threads", "1", "-c:v", "libx264", str(dest),
        ], check=True)
    dump(meta / "info.json", {"codebase_version": "v3.0", "fps": fps, "chunks_size": 2, "features": features,
                             "total_episodes": 3, "total_frames": 3 * frames})
    groups = {key: {"start": start, "end": end, "absolute": True} for key, (start, end) in INPUT_GROUPS.items()}
    dump(meta / "modality.json", {"state": groups, "action": groups})
    split = [{"episode_index": i, "episode_name": f"example{i}", "split": "val" if i == 1 else "train"}
             for i in range(3)]
    dump(meta / "split.json", {"episodes": split})
    write_parquet(meta / "tasks.parquet", pa.table({"task_index": [0], "task": ["move a cube"]}))
    records, tables = [], []
    for ep, value in enumerate((0, 999, 10)):
        start = ep * frames
        record = {"episode_index": ep, "length": frames, "data/chunk_index": 0, "data/file_index": ep // 2,
                  "dataset_from_index": start, "dataset_to_index": start + frames}
        for camera in cameras:
            prefix = "videos/" + camera + "/"
            record.update({prefix + "chunk_index": 0, prefix + "file_index": 0,
                           prefix + "from_timestamp": start / fps, prefix + "to_timestamp": (start + frames) / fps})
        records.append(record)
        vector = np.arange(value, value + 28, dtype=np.float32)
        tables.append(pa.table({
            "episode_index": [ep] * frames, "index": range(start, start + frames), "frame_index": range(frames),
            "timestamp": np.arange(frames, dtype=np.float32) / fps, "task_index": [0] * frames,
            "observation.state": [vector] * frames, "action": [vector + 1] * frames,
        }))
    write_parquet(meta / "episodes/chunk-000/file-000.parquet", pa.Table.from_pylist(records))
    write_parquet(root / "data/chunk-000/file-000.parquet", pa.concat_tables(tables[:2]))
    write_parquet(root / "data/chunk-000/file-001.parquet", tables[2])


class DataPipelineTest(unittest.TestCase):
    def test_train_projection_and_eval_statistics(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / "raw", root / "processed"
            make_v3(source)
            subprocess.run([sys.executable, str(Path(__file__).with_name("prepare.py")), str(source), str(output),
                            "rgb3", "--video-workers", "1"], check=True)
            full, train = output / "22d", output / "train_view"
            source_ids = (train / "meta/source_episode_mapping.jsonl").read_text().splitlines()
            self.assertEqual([json.loads(line)["source_episode_index"] for line in source_ids], [0, 2])
            stats = json.loads((train / "meta/stats.json").read_text())
            self.assertEqual(stats, json.loads((full / "meta/stats.json").read_text()))
            self.assertEqual(stats["observation.state"]["mean"][0], 5.0)
            self.assertEqual(stats["observation.state"]["max"][0], 10.0)
            table = pq.read_table(train / "data/chunk-000/episode_000001.parquet")
            self.assertEqual(table["index"].to_pylist(), [4, 5, 6, 7])
            self.assertEqual(table["episode_index"].to_pylist(), [1] * 4)
            self.assertEqual(table["observation.state"][0].as_py(),
                             [10,11,12,13,14,15,16,18,19,20,21,22,23,24,26,27,28,30,32,33,34,36])
            video = "observation.images.right_wrist_rgb"
            self.assertTrue((train / "videos/chunk-000" / video / "episode_000001.mp4").samefile(
                full / "videos/chunk-001" / video / "episode_000002.mp4"))


if __name__ == "__main__":
    unittest.main()
