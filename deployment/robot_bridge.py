"""Observation and command contract shared by the two controllers."""
import io
import pickle

import cv2
import numpy as np
import Pyro5.api
from Pyro5 import serializers

CAMERAS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
JOINT_DIMS = {"left_arm": 7, "right_arm": 7, "left_hand": 6, "right_hand": 6}


class WireUnpickler(pickle.Unpickler):
    """Accept primitives and the numeric NumPy arrays used by the lab bridge."""

    def find_class(self, module, name):
        if module == "numpy" and name in {"dtype", "ndarray"}:
            return getattr(np, name)
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"} and name in {"_reconstruct", "scalar"}:
            return getattr(np.core.multiarray, name)
        if module in {"numpy.core.numeric", "numpy._core.numeric"} and name == "_frombuffer":
            return np.core.numeric._frombuffer
        raise pickle.UnpicklingError(f"Unsupported class in robot bridge response: {module}.{name}")


class BridgeSerializer(serializers.SerializerBase):
    serializer_id = 5

    def dumps(self, data):
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

    def dumpsCall(self, obj, method, vargs, kwargs):
        return self.dumps((obj, method, vargs, kwargs))

    def loads(self, data):
        return WireUnpickler(io.BytesIO(data)).load()

    loadsCall = loads


def configure_pyro():
    wire = BridgeSerializer()
    serializers.serializers["pickle"] = wire
    serializers.serializers_by_id[wire.serializer_id] = wire
    Pyro5.api.config.SERIALIZER = "pickle"


def decode_observation(wire, cameras=CAMERAS):
    obs = {"timestamp_ns": int(wire["timestamp_ns"])}
    for name in cameras:
        encoded = np.frombuffer(wire[name + "_jpeg"], dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Invalid JPEG for {name}")
        obs[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    for name, dim in JOINT_DIMS.items():
        values = np.asarray(wire[name], dtype=np.float64).reshape(-1)
        if values.shape != (dim,) or not np.isfinite(values).all():
            raise ValueError(f"Expected {dim} finite joint values for {name}")
        obs[name] = values
    return obs


def compress_hand_6_to_4(hand):
    h = np.asarray(hand).reshape(6)
    return np.array([h[0], h[1], h[2], h[3:].mean()])


def expand_hand_4_to_6(hand):
    return np.asarray(hand).reshape(4)[[0, 1, 2, 3, 3, 3]]


def rate_limit(target, current, max_delta):
    target, current = np.asarray(target), np.asarray(current)
    if target.shape != current.shape:
        raise ValueError("Joint target and current state have different shapes")
    delta = target - current
    limited = np.clip(delta, -max_delta, max_delta)
    return current + limited, delta, limited


def send_targets(proxy, client_uuid, left_arm, right_arm, left_hand, right_hand):
    command = {"client_uuid": client_uuid}
    for (name, dim), target in zip(JOINT_DIMS.items(), (left_arm, right_arm, left_hand, right_hand)):
        values = np.asarray(target, dtype=np.float64).reshape(dim)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite target for {name}")
        command[name] = values.tolist()
    return proxy.submit_command(command)
