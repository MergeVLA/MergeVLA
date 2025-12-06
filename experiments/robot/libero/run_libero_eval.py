"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

from copy import deepcopy
import json
import logging
import os
import sys
import copy
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import time
from typing import Optional, Union, OrderedDict, Dict

import draccus
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_moe_action_head,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    DATE,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK

def get_current_time():
    return time.strftime("%Y_%m_%d-%H_%M_%S")

# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)



@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "mergevla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    pretrained_vlm_checkpoint: Union[str, Path] = "" # Pretrained VLM checkpoint path for MoE model merging
    use_minivlm: bool = True                         # If True, uses minivlm
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    load_moe: bool = False                           # Whether to load Mixture-of-Experts version model
    k_gate: int = 8                                  # k_gate
    action_head_layer_num: int = 1                   # The layer number of action head

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy
    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on
    phase: str = "Inference"



def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"



def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    proprio_projector = get_proprio_projector(
        cfg,
        model.llm_dim,
        proprio_dim=8,  # 8-dimensional proprio for LIBERO
    )

    # Load action head if needed
    action_head = get_action_head(cfg, model.llm_dim)

    # Get processor if needed
    processor = None
    if cfg.model_family == "mergevla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, processor



def initialize_moe_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load pretrained vlm model
    cfg_vlm = deepcopy(cfg)
    cfg_vlm.pretrained_checkpoint = cfg.pretrained_vlm_checkpoint
    vlm = get_model(cfg_vlm)

    # Load model
    model = get_model(cfg)

    if os.path.exists(cfg.pretrained_checkpoint + "/tall_masks.pt"):
        masks = torch.load(cfg.pretrained_checkpoint + "/tall_masks.pt")
        vla_sd = model.state_dict()
        vlm_sd = vlm.state_dict()
        mask = masks[cfg.expert_name]

        final_sd = OrderedDict()
        if 'rescaler' in mask:
            for key, tensor in mask.items():
                if key == 'rescaler':
                    continue
                mask[key] = tensor.to(vla_sd[key].device, dtype=vla_sd[key].dtype)
            for key in vla_sd:
                if key == "action_queries.weight":
                    continue
                if key in mask:
                    multi_task_vector = vla_sd[key] * mask[key] * mask['rescaler']
                    final_sd[key] =  vlm_sd[key] + multi_task_vector
                else:
                    final_sd[key] = vlm_sd[key] + vla_sd[key]
        else:
            for key, tensor in mask.items():
                mask[key] = tensor.to(vla_sd[key].device, dtype=vla_sd[key].dtype)
            for key in vla_sd:
                if key == "action_queries.weight":
                    continue
                if key in mask:
                    final_sd[key] = vlm_sd[key] + vla_sd[key] * mask[key]
                else:
                    final_sd[key] = vlm_sd[key] + vla_sd[key]
        model.load_state_dict(final_sd, strict=False)

    assert "TallMask" not in cfg.pretrained_checkpoint or "EMR" not in cfg.pretrained_checkpoint, \
        "Path with TallMask or EMR found. Please use the merged checkpoint without masks."


    # Load proprio projector if needed
    proprio_projector = get_proprio_projector(
        cfg,
        model.llm_dim,
        proprio_dim=8,  # 8-dimensional proprio for LIBERO
    )

    # Load action head if needed
    action_head = get_moe_action_head(cfg, model.llm_dim)

    # Get processor if needed
    processor = None
    if cfg.model_family == "mergevla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, processor

def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.expert_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key



def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"{DATE_TIME}-{cfg.model_family}-EVAL-{cfg.task_suite_name}-{cfg.num_trials_per_task}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    cfg.local_log_dir = os.path.join(cfg.local_log_dir, DATE)

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Log config
    log_file.write(f"{get_current_time()} ---------------> python file: run_libero_eval.py\n")
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"{get_current_time()} ---------------> Task suite: {cfg.task_suite_name}\n")
    if cfg.load_moe:
        print(f"Task suite identity: {cfg.expert_name}")
        log_file.write(f"{get_current_time()} ---------------> Task suite identity: {cfg.expert_name}\n")
        print(f"Expert idx: {cfg.expert_idx}")
        log_file.write(f"{get_current_time()} ---------------> Expert idx: {cfg.expert_idx}\n")
    print(f"num_images_in_input: {cfg.num_images_in_input}")
    log_file.write(f"{get_current_time()} ---------------> num_images_in_input: {cfg.num_images_in_input}\n")
    print(f"Use MoE Version: {cfg.load_moe}")
    log_file.write(f"{get_current_time()} ---------------> Use MoE Version: {cfg.load_moe}\n")
    print(f"Num trials per task: {cfg.num_trials_per_task}")
    log_file.write(f"{get_current_time()} ---------------> Num trials per task: {cfg.num_trials_per_task}\n")
    print(f"Pretrained VLM checkpoint path: {cfg.pretrained_vlm_checkpoint}")
    log_file.write(f"{get_current_time()} ---------------> Pretrained VLM checkpoint path: {cfg.pretrained_vlm_checkpoint}\n")
    print(f"Checkpoint path: {cfg.pretrained_checkpoint}\n")
    log_file.write(f"{get_current_time()} ---------------> Checkpoint path: {cfg.pretrained_checkpoint}\n\n{'-'*120}\n\n")

    return log_file, local_log_filepath, run_id



def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()



def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"{get_current_time()} ---------------> Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message(f"{get_current_time()} ---------------> Using default initial states", log_file)
        return initial_states, None



def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay



def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family == "mergevla":
        action = invert_gripper_action(action)

    return action



def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    initial_state=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # Initialize action queue
    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
               "{NUM_ACTIONS_CHUNK} constant defined in prismatic.vla.constants! For best performance (in terms of "
               "both speed and success rate), we recommend executing the full action chunk.")
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Run episode
    success = False
    while t < max_steps + cfg.num_steps_wait:
        # Do nothing for the first few timesteps to let objects stabilize
        if t < cfg.num_steps_wait:
            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue

        # Prepare observation
        observation, img = prepare_observation(obs, resize_size)
        replay_images.append(img)

        # If action queue is empty, requery model
        if len(action_queue) == 0:
            # Query model to get action
            actions = get_action(
                cfg,
                model,
                observation,
                task_description,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                # use_film=cfg.use_film,
                use_minivlm=cfg.use_minivlm
            )

            action_queue.extend(actions) 

        # Get action from queue
        action = action_queue.popleft()

        # Process action
        action = process_action(action, cfg.model_family)

        # Execute action in environment
        obs, reward, done, info = env.step(action.tolist())
        if done:
            success = True
            break
        t += 1

    return success, replay_images




def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"{get_current_time()} ---------------> Task: {task_description} [{task_id+1} / {task_suite.n_tasks}]", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"{get_current_time()} ---------------> Starting episode {episode_idx+1}... ({cfg.num_trials_per_task} in total)", log_file)
        

        # Run episode
        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            initial_state,
        )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video(
            replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file, save_version=cfg.model_family,
            rollout_name=f'{cfg.task_suite_name}_{DATE_TIME}', task_idx=task_id
        )

        # Log results
        log_message(f"{get_current_time()} ---------------> Success: {success}", log_file)
        log_message(f"{get_current_time()} ---------------> # episodes completed so far: {total_episodes}", log_file)
        log_message(f"{get_current_time()} ---------------> # successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"{get_current_time()} ---------------> Current task success rate: {float(task_successes) / float(task_episodes)}", log_file)
    log_message(f"{get_current_time()} ---------------> Current total success rate: {float(total_successes) / float(total_episodes)}\n\n{'-'*120}\n", log_file)

    # close env
    env.close()
    del env

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes



@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    
    # Task identify
    if cfg.load_moe:
        # load the expert info
        task2idx_path = os.path.join(cfg.pretrained_checkpoint, 'task2idx.pt')
        assert os.path.exists(task2idx_path), f"task2idx file not found at {task2idx_path}"
        task2idx = torch.load(task2idx_path)
        
        cfg.num_experts = len(task2idx)
        
        cfg_for_identity = copy.deepcopy(cfg)
        cfg_for_identity.is_task_identity = True
        cfg_for_identity.task2idx = task2idx
        expert_name, expert_idx = task_identity(cfg_for_identity)
        
        cfg.expert_name = expert_name
        cfg.expert_idx = expert_idx
    else:
        cfg.expert_name = cfg.task_suite_name
    cfg.is_task_identity = False
    
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    if cfg.load_moe:
        model, action_head, proprio_projector, processor = initialize_moe_model(cfg)
    else:
        model, action_head, proprio_projector, processor = initialize_model(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"{get_current_time()} ---------------> Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            total_episodes,
            total_successes,
            log_file,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"{get_current_time()} ---------------> Final results:", log_file)
    log_message(f"{get_current_time()} ---------------> Total episodes: {total_episodes}", log_file)
    log_message(f"{get_current_time()} ---------------> Total successes: {total_successes}", log_file)
    log_message(f"{get_current_time()} ---------------> Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)\n\n{'-'*120}\n", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


def task_identity(cfg: GenerateConfig) -> float:
    """Identify the task through the observations and test-time router."""
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    resize_size = get_image_resize_size(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()

    # Start evaluation
    task = task_suite.get_task(0)
    initial_states, _ = load_initial_states(cfg, task_suite, 0, log_file=None)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    initial_state = initial_states[0]
    env.reset()
    obs = env.set_init_state(initial_state)

    router_logits = {}
    for task in cfg.task2idx.keys():
        cfg.expert_name = task
        cfg.expert_idx = cfg.task2idx[cfg.expert_name]
        model, action_head, proprio_projector, processor = initialize_moe_model(cfg)
        # Initialize action queue
        if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
            print(f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match the NUM_ACTIONS_CHUNK "
                "{NUM_ACTIONS_CHUNK} constant defined in prismatic.vla.constants! For best performance (in terms of "
                "both speed and success rate), we recommend executing the full action chunk.")

        # Do nothing for the first few timesteps to let objects stabilize
        for _ in range(cfg.num_steps_wait):
            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))

        # Run episode
        observation, _ = prepare_observation(obs, resize_size)

        router_logit = get_action(
            cfg,
            model,
            observation,
            task_description,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            use_minivlm=cfg.use_minivlm
        )
        router_logits.update(router_logit)
    
    # find the max routing_weights and return mask and expert
    keys = list(router_logits.keys())
    router_logits = torch.stack(list(router_logits.values()), dim=1)
    routing_weights = F.softmax(router_logits, dim=1) # (1, 4)
    print(routing_weights, keys)
    idx = routing_weights.argmax(dim=1).item()
    expert_name = keys[idx]
    expert_idx = cfg.task2idx[expert_name]

    return expert_name, expert_idx



if __name__ == "__main__":
    eval_libero()
