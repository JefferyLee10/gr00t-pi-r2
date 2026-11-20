"""Offline checks for the wire observation and two model-to-robot conversions."""
import pickle
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

import gr00t as baseline
import pir2
import robot_bridge as bridge


class RobotContractTest(unittest.TestCase):
    def setUp(self):
        image = np.full((8, 12, 3), (12, 32, 200), dtype=np.uint8)
        ok, jpeg = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        self.wire = {name: [0.] * size for name, size in bridge.JOINT_DIMS.items()}
        self.wire.update(timestamp_ns=567, head_rgb_jpeg=jpeg.tobytes())
        self.wire["left_hand"] = np.linspace(0, .1, 6, dtype=np.float32)

    def test_observation(self):
        for protocol in (4, 5):
            decoded = bridge.BridgeSerializer().loads(pickle.dumps(self.wire, protocol=protocol))
            np.testing.assert_array_equal(decoded["left_hand"], self.wire["left_hand"])

        for cameras in (("head_rgb",), bridge.CAMERAS):
            wire = dict(self.wire)
            for camera in cameras:
                wire[camera + "_jpeg"] = wire["head_rgb_jpeg"]
            obs = bridge.decode_observation(wire, cameras)
            np.testing.assert_allclose(obs["head_rgb"][0, 0], [200, 32, 12], atol=4)
            client = SimpleNamespace(video_T=1, state_T=1, _video_keys=cameras)
            images, state = pir2.make_snapshot(client, obs)
            self.assertEqual(tuple(images), cameras)
            self.assertEqual(state.shape, (1, 22))
            np.testing.assert_allclose(state[0, 14:18], [0, .02, .04, .08])
            inputs = baseline.model_obs(obs, "task", "example", cameras)
            self.assertEqual(tuple(inputs["video"]), cameras)
            self.assertEqual(inputs["language"]["task"], [["example"]])

    def test_commands(self):
        obs = bridge.decode_observation(self.wire, ("head_rgb",))
        state = baseline.model_obs(obs, "task", "", ("head_rgb",))["state"]
        action = {k: np.repeat(v, 4, axis=1) for k, v in state.items()}
        action["left_arm"][0, 0, 0] = np.deg2rad(10)
        targets, _ = baseline.targets_for_first_action(action, obs)
        self.assertAlmostEqual(targets["left_arm"][0], np.deg2rad(2))
        self.assertEqual([v.shape for v in targets.values()], [(7,), (7,), (6,), (6,)])
        action["left_arm"][0, 0, 0] = np.deg2rad(16)
        self.assertIsNone(baseline.targets_for_first_action(action, obs)[0])

        previous = np.zeros(22)
        proposed = previous.copy()
        proposed[0] = np.deg2rad(10)
        cmd = pir2.safe_physical_action(proposed, previous, obs)
        self.assertEqual([v.shape for v in cmd[:-1]], [(7,), (7,), (6,), (6,)])
        with self.assertRaisesRegex(ValueError, "step_jump"):
            pir2.safe_physical_action(proposed, previous, obs, max_jump=8.0)


if __name__ == "__main__":
    unittest.main()
