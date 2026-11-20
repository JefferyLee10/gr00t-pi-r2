from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


bag_transfer_16h_22d_headrgb_abs = {

    # ============================================================
    # VIDEO
    # Only monocular head RGB.
    # Depth is preserved in dataset but NOT used in this training.
    # ============================================================
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "head_rgb",
        ],
    ),

    # ============================================================
    # STATE = 22D
    #
    # left_arm   : 7D
    # right_arm  : 7D
    # left_hand  : 4D
    # right_hand : 4D
    # ============================================================
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
    ),

    # ============================================================
    # ACTION = 22D
    #
    # H = 16
    # Absolute joint targets.
    # ============================================================
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

    # ============================================================
    # LANGUAGE
    # ============================================================
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "annotation.human.action.task_description",
        ],
    ),
}


register_modality_config(
    bag_transfer_16h_22d_headrgb_abs,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
