"""Serve a recorded v2.1 28D frame to the controller's dry-run path on localhost."""

import argparse
import io
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import Pyro5.api
from Pyro5 import serializers


class PrimitiveUnpickler(pickle.Unpickler):
    """The client sends primitive RPC arguments; class reconstruction is unnecessary."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError("Only primitive replay RPC arguments are accepted")


class ReplaySerializer(serializers.SerializerBase):
    serializer_id = 5  # Wire identifier used by the original robot bridge.

    @classmethod
    def register_type_replacement(cls, object_type, replacement_function):
        # The replay service returns value objects and never serializes Pyro proxies.
        pass

    def loads(self, data):
        return PrimitiveUnpickler(io.BytesIO(data)).load()

    loadsCall = loads

    def dumps(self, data):
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

    def dumpsCall(self, obj, method, vargs, kwargs):
        return self.dumps((obj, method, vargs, kwargs))


def read_sample(root, episode, frame):
    info = json.loads((root / "meta/info.json").read_text())
    if info["codebase_version"] != "v2.1" or info["features"]["observation.state"]["shape"] != [28]:
        raise ValueError("Expected the stage-1 v2.1 28D dataset")
    chunk = episode // int(info["chunks_size"])
    fields = {"episode_chunk": chunk, "episode_index": episode}
    table = pq.read_table(root / info["data_path"].format(**fields), columns=["observation.state"])
    if not 0 <= frame < len(table):
        raise ValueError(f"frame must be in [0, {len(table)})")
    state = np.asarray(table["observation.state"][frame].as_py())
    if state.shape != (28,) or not np.isfinite(state).all():
        raise ValueError("Invalid 28D state")
    sample = {name: state[a:b].tolist() for name, a, b in [
        ("left_arm", 0, 7), ("right_arm", 8, 15), ("left_hand", 16, 22), ("right_hand", 22, 28)
    ]}
    for camera in ("head_rgb", "left_wrist_rgb", "right_wrist_rgb"):
        key = "observation.images." + camera
        if key not in info["features"] and camera != "head_rgb":
            continue
        path = root / info["video_path"].format(video_key=key, **fields)
        capture = cv2.VideoCapture(str(path))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Unreadable frame {frame} in {path}")
            ok, jpeg = cv2.imencode(".jpg", image)
            if not ok:
                raise ValueError(f"JPEG encoding failed for {camera}")
            sample[camera + "_jpeg"] = jpeg.tobytes()
        finally:
            capture.release()
    return sample


@Pyro5.api.expose
class ReplayBridge:
    def __init__(self, sample):
        self.sample = sample

    def ping(self):
        return True

    def get_obs_status(self):
        return {"mode": "recorded_frame", "ready": True}

    def get_observation(self):
        return dict(self.sample, timestamp_ns=time.time_ns())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/28d"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--port", type=int, default=17788)
    args = parser.parse_args()
    sample = read_sample(args.dataset, args.episode, args.frame)
    wire = ReplaySerializer()
    serializers.serializers_by_id[wire.serializer_id] = wire
    serializers.serializers["pickle"] = wire
    with Pyro5.api.Daemon(host="127.0.0.1", port=args.port) as daemon:
        uri = daemon.register(ReplayBridge(sample), objectId="model_inference_bridge")
        print(f"Recorded episode {args.episode}, frame {args.frame}: {uri}", flush=True)
        daemon.requestLoop()


if __name__ == "__main__":
    main()
