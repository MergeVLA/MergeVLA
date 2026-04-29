from collections import deque

import torch
from torch import Tensor

from lerobot.policies.mergevla.configuration_mergevla import AdapterConfig, MergeVLAConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction

from .experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
)
from .prismatic.vla.constants import PROPRIO_DIM


def encode_obs(obs: dict) -> dict:
    return {
        "full_image": obs["observation.images.top"].squeeze(0),
        "wrist_image": obs["observation.images.wrist"].squeeze(0),
        "state": obs["observation.state"].squeeze(0),
        "instruction": obs["task"],
    }


class Model:
    def __init__(self, cfg: MergeVLAConfig):
        self.cfg = cfg
        self.cfg.num_open_loop_steps = 20
        self.vla = get_vla(cfg)
        self.processor = get_processor(cfg)
        self.action_head = None
        if cfg.use_l1_regression:
            self.action_head = get_action_head(cfg, self.vla.llm_dim)
        self.proprio_projector = None
        if cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(cfg, self.vla.llm_dim, PROPRIO_DIM)

    def get_action(self, observation: dict):
        obs = encode_obs(observation)
        actions = get_vla_action(
            cfg=self.cfg,
            vla=self.vla,
            processor=self.processor,
            obs=obs,
            task_label=obs["instruction"],
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            use_film=self.cfg.use_film,
        )
        return actions


class MergeVLAPolicy(PreTrainedPolicy):
    """MergeVLA / improved adapter-style VLA for LeRobot (Prismatic + LoRA)."""

    name = "mergevla"
    config_class = MergeVLAConfig

    def __init__(self, config: MergeVLAConfig):
        super().__init__(config)

        self.cfg = config
        self.vla = get_vla(config)
        self.processor = get_processor(config)
        self.action_head = None
        if config.use_l1_regression:
            self.action_head = get_action_head(config, self.vla.llm_dim)
        self.proprio_projector = None
        if config.use_proprio:
            self.proprio_projector = get_proprio_projector(config, self.vla.llm_dim, PROPRIO_DIM)
        self.reset()

    def _create_adapter_model(self):
        config_args = {
            "pretrained_checkpoint": self.cfg.pretrained_checkpoint,
            "use_l1_regression": self.cfg.use_l1_regression,
            "use_diffusion": self.cfg.use_diffusion,
            "use_film": self.cfg.use_film,
            "use_proprio": self.cfg.use_proprio,
            "load_in_8bit": self.cfg.load_in_8bit,
            "load_in_4bit": self.cfg.load_in_4bit,
            "num_images_in_input": self.cfg.num_images_in_input,
            "center_crop": self.cfg.center_crop,
            "unnorm_key": self.cfg.unnorm_key,
            "chunk_size": self.cfg.chunk_size,
            "n_action_steps": self.cfg.n_action_steps,
            "lora_rank": self.cfg.lora_rank,
        }

        cfg = MergeVLAConfig(**config_args)
        return Model(cfg)

    def reset(self):
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        obs = encode_obs(batch)
        actions = get_vla_action(
            cfg=self.cfg,
            vla=self.vla,
            processor=self.processor,
            obs=obs,
            task_label=obs["instruction"],
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            use_film=self.cfg.use_film,
        )
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            self._action_queue.extend(actions)
        return PolicyAction(torch.tensor(self._action_queue.popleft()))

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        return None

    def get_optim_params(self) -> dict:
        return self.parameters()


class AdapterPolicy(MergeVLAPolicy):
    """Loads configs with legacy JSON `"type": "adapter"`; identical weights and behavior to MergeVLAPolicy."""

    name = "adapter"
    config_class = AdapterConfig
