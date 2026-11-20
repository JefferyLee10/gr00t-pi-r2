# -*- coding: utf-8 -*-

"""
Bag-transfer 17h / 3RGB / 22D / piR2 modality config.

Dataset:
- 3 RGB:
    head_rgb
    left_wrist_rgb
    right_wrist_rgb
- state/action:
    left_arm  7D
    right_arm 7D
    left_hand 4D
    right_hand 4D
  total = 22D
- absolute joint-space actions
- action horizon = 16

Important:
Current piR2 loader implements async image delay by shifting
the video step_index by -d, where d ~ U[0, image_delay_max].
Therefore video.delta_indices must remain [0].
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


bag_transfer_17h_3rgb_22d_pir2_config = {

    # --------------------------------------------------------
    # Video
    #
    # Current frame slot only.
    # image_delay_max is handled dynamically by the piR2
    # ShardedSingleStepDataset:
    #
    #     step_index_for_video = step_index - image_delay
    #
    # so DO NOT put [-5,-4,-3,-2,-1,0] here.
    # --------------------------------------------------------
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "head_rgb",
            "left_wrist_rgb",
            "right_wrist_rgb",
        ],
    ),

    # --------------------------------------------------------
    # State: current proprioception only
    # --------------------------------------------------------
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
    ),

    # --------------------------------------------------------
    # Action: 16-step horizon
    #
    # All four groups are ABSOLUTE joint-space targets.
    # --------------------------------------------------------
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),

    # --------------------------------------------------------
    # Language
    #
    # meta/modality.json:
    # human.action.task_description
    #     original_key = task_index
    # --------------------------------------------------------
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "annotation.human.action.task_description",
        ],
    ),
}


register_modality_config(
    bag_transfer_17h_3rgb_22d_pir2_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
