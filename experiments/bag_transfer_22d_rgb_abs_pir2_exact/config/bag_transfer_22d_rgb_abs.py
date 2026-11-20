from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


bag_transfer_22d_rgb_abs = {

    # ============================================================
    # VIDEO
    # 三路 RGB，不使用 depth
    # ============================================================
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "head_rgb",
            "left_wrist_rgb",
            "right_wrist_rgb",
        ],
    ),

    # ============================================================
    # STATE = 22D
    #
    # left_arm   : 7D
    # right_arm  : 7D
    # left_hand  : 4D
    # right_hand : 4D
    #
    # 每只手4D:
    #   j0
    #   j1
    #   j2
    #   other3 = mean(j3,j4,j5)
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
    # 16-step action chunk
    # 保持 absolute joint target
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
    bag_transfer_22d_rgb_abs,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
