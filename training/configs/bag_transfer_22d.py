"""Register the 22D dual-arm modality for launch_finetune.py."""
import os

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig,
)

CAMERAS = {
    "headrgb": ["head_rgb"],
    "rgb3": ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"],
}
JOINT_GROUPS = ["left_arm", "right_arm", "left_hand", "right_hand"]

register_modality_config({
    # The πR² loader samples the frame delay; each camera has one input slot.
    "video": ModalityConfig(
        delta_indices=[0], modality_keys=CAMERAS[os.environ["BAG_CAMERA_PROFILE"]]
    ),
    "state": ModalityConfig(delta_indices=[0], modality_keys=JOINT_GROUPS),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=JOINT_GROUPS,
        action_configs=[
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF,
                         format=ActionFormat.DEFAULT) for _ in JOINT_GROUPS
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0], modality_keys=["annotation.human.action.task_description"]
    ),
}, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
