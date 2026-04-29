from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig

from .prismatic.vla.constants import NUM_ACTIONS_CHUNK


@PreTrainedConfig.register_subclass("mergevla")
@dataclass
class MergeVLAConfig(PreTrainedConfig):
    """MergeVLA policy (Prismatic / OpenVLA-style stack with LoRA) for LeRobot."""

    pretrained_checkpoint: str = ""
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    use_proprio: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    lora_rank: int = 64
    num_open_loop_steps: int = 20
    num_images_in_input: int = 2
    center_crop: bool = True
    unnorm_key: str = "pushcube_so101"
    chunk_size: int = NUM_ACTIONS_CHUNK
    n_action_steps: int = NUM_ACTIONS_CHUNK
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        return None

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None


@PreTrainedConfig.register_subclass("adapter")
@dataclass
class AdapterConfig(MergeVLAConfig):
    """Legacy config type string (`adapter`) for older checkpoints; prefer `mergevla` for new runs."""

    pass
