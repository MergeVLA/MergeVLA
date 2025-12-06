import os
import json
import copy
import torch
from torch import nn
from tqdm import tqdm
import lightning as L
from enum import Enum
from pathlib import Path
from typing import List
import sys
sys.path.append('/path/to/MergeVLA')
sys.path.append('/path/to/LIBERO')
sys.path.append('/path/to/MergeVLA/fusion_bench')
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.robot_utils import (
    get_model,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_moe_action_head,
    get_processor,
    get_proprio_projector,
)
from transformers import AutoConfig, AutoModelForVision2Seq
from prismatic.models import load
from prismatic.models.moe_model import svd
from fusion_bench.method import (
    TaskArithmeticAlgorithm, ISO_CTS_Merge, WeightedAverageAlgorithm, 
    TiesMergingAlgorithm, TallMaskTaskArithmeticAlgorithm, TM_ISO_CTS_Merge,
    TallMaskTiesMergingAlgorithm, TaskSingularVectorMerging,
    WUDIMerging, TallMaskTaskSingularVectorMerging, TallMaskWUDIMerging,
    EMRMerging, KnotsMerging, TallMaskKnotsMerging
)
from fusion_bench.modelpool import BaseModelPool

LLM_DIM = 896
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"

    @classmethod
    def has(cls, value) -> bool:
        assert isinstance(value, (str, cls)), f"Value must be a string or {cls.__name__} Enum."
        value = value.value if isinstance(value, cls) else value # Get the string value if Enum
        return value in cls._value2member_map_

def invert_keys(all_keys, exclude_keys):
    return [k for k in all_keys if k not in exclude_keys]

def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    model = get_model(cfg)

    proprio_projector = get_proprio_projector(
        cfg,
        model.llm_dim,
        proprio_dim=8,  # 8-dimensional proprio for LIBERO
    )

    action_head = get_action_head(cfg, model.llm_dim)

    return model, action_head, proprio_projector

def initialize_moe_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    model = get_model(cfg)

    proprio_projector = get_proprio_projector(
        cfg,
        model.llm_dim,
        proprio_dim=8,  # 8-dimensional proprio for LIBERO
    )

    action_head = get_moe_action_head(cfg, model.llm_dim)

    return model, action_head, proprio_projector

def initialize_proprio_projector(llm_dim, proprio_dim=8, is_zero_init=True):
    from prismatic.models.projectors import ProprioProjector
    proprio_projector = ProprioProjector(
        llm_dim=llm_dim,
        proprio_dim=proprio_dim,
    ).to("cuda")
    proprio_projector = proprio_projector.to(torch.bfloat16).to("cuda")
    proprio_projector.eval()

    if is_zero_init:
        print("Zero initializing proprio projector.")
        with torch.no_grad():
            for name, param in proprio_projector.named_parameters():
                param.zero_()
    return proprio_projector

def initialize_action_head(llm_dim, is_zero_init=True):
    from prismatic.vla.constants import (ACTION_DIM)
    from prismatic.models.action_heads import L1RegressionActionHead

    action_head = L1RegressionActionHead(
        input_dim=llm_dim, 
        hidden_dim=llm_dim, 
        action_dim=ACTION_DIM,
    )
    
    action_head = action_head.to(torch.bfloat16).to("cuda")
    action_head.eval()

    if is_zero_init:
        print("Zero initializing action head.")
        with torch.no_grad():
            for name, param in action_head.named_parameters():
                param.zero_()
    return action_head

tasks = {
    "spatial": TaskSuite.LIBERO_SPATIAL,
    "object": TaskSuite.LIBERO_OBJECT,
    "goal": TaskSuite.LIBERO_GOAL,
    "10": TaskSuite.LIBERO_10
}
ckptsprosigmoid = {
    "spatial": "/path/to/spatial_ckpt",
    "object": "/path/to/object_ckpt",
    "goal": "/path/to/goal_ckpt",
    "10": "/path/to/long10_ckpt"
}

def load_vlm(is_zero_init=True):
    vlm_path = "path/to/prism-qwen25-extra-dinosiglip-224px-0_5b"
    hf_token = 'your_hf_token'
    vlm = load(vlm_path, hf_token=hf_token, load_for_training=True)

    config = AutoConfig.from_pretrained("pretrained_models/configs", trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    replace_map = [
        ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
        ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
        ("llm_backbone.llm", "language_model"),
        ("projector.projector.0", "projector.fc1"),
        ("projector.projector.2", "projector.fc2"),
        ("projector.projector.4", "projector.fc3"),
        ("gamma", "scale_factor"),
    ]

    def rename_state_dict_keys(state_dict, replace_map):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            for old, new in replace_map:
                if old in new_k:
                    new_k = new_k.replace(old, new)
            new_state_dict[new_k] = v
        return new_state_dict

    old_state_dict = vlm.state_dict()
    RAW_STATE_DICT = rename_state_dict_keys(old_state_dict, replace_map)

    missing_keys, unexpected_keys = vla.load_state_dict(RAW_STATE_DICT, strict=False)
    del old_state_dict

    # turn action_queries into zeros
    if is_zero_init:
        print("Zero initializing action queries.")
        with torch.no_grad():
            vla.action_queries.weight.zero_()
    vla.__class__.__module__ = "prismatic.extern.hf.modeling_prismatic"
    return vla

def save_vlm(checkpoint_dir):
        '''
        >>> checkpoint_dir = Path(os.path.join('pretrained_models', 'Pretrained-VLM'))
        >>> save_vlm(checkpoint_dir)
        '''
        pretrained_vlm = load_vlm(is_zero_init=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        pretrained_vlm.save_pretrained(checkpoint_dir)
        pretrained_vlm.state_dict()
        print(f"Saved merged checkpoint to {checkpoint_dir}")

def load_vlm_from_vla():
    vlm_path = "/path/to/Pretrained-VLM"
    cfg = GenerateConfig(pretrained_checkpoint=vlm_path)
    model = get_model(cfg)

    return model

def get_algo(algo_name: str, exclude_keys=None, eval_task=None, return_all_masks: bool = False, WA_weights=[0.5, 0.5]):
    device="cuda"
    if algo_name == "TA":
        algorithm = TaskArithmeticAlgorithm(
            scaling_factor=1.0,
            exclude_keys=exclude_keys
        )
        device = "cpu"
    elif algo_name == "TATallMask":
        algorithm = TallMaskTaskArithmeticAlgorithm(
            tall_mask_lambda=0.6,
            eval_model_name=eval_task,
            exclude_keys=exclude_keys,
            return_all_masks=return_all_masks
        )
        device = "cpu"
    elif algo_name == "ties":
        if exclude_keys is None: exclude_keys = []
        algorithm = TiesMergingAlgorithm(
            scaling_factor=1.0,
            threshold=1, # 1 or 100 means no reset
            remove_keys=exclude_keys,
            merge_func="sum",
        )
        device = "cpu"
    elif algo_name == "ties_TallMask":
        if exclude_keys is None: exclude_keys = []
        algorithm = TallMaskTiesMergingAlgorithm(
            scaling_factor=1.0,
            threshold=100, # 1 or 100 means no reset
            remove_keys=exclude_keys,
            merge_func="sum",
            tall_mask_lambda=0.6,
            eval_model_name=eval_task,
            return_all_masks=return_all_masks
        )
        device = "cpu"
    elif algo_name == "tsv":
        algorithm = TaskSingularVectorMerging(
            alpha=1.0,
            exclude_keys=exclude_keys
        )
        device = "cpu"
    elif algo_name == "tsvTallMask":
        algorithm = TallMaskTaskSingularVectorMerging(
            alpha=1.0,
            exclude_keys=exclude_keys,
            tall_mask_lambda=0.6,
            eval_model_name=eval_task,
            return_all_masks=return_all_masks
        )
        device = "cpu"
    elif algo_name == "wudi":
        algorithm = WUDIMerging(
            iter_num=300,
            exclude_keys=exclude_keys
        )
        algorithm.fabric = L.Fabric(accelerator="cuda", devices=1)
    elif algo_name == "wudi_TallMask":
        exclude_keys.extend(["language_model.model.embed_tokens.weight", "language_model.lm_head.weight"])
        algorithm = TallMaskWUDIMerging(
            iter_num=300,
            exclude_keys=exclude_keys,
            tall_mask_lambda=0.6,
            eval_model_name=eval_task,
            return_all_masks=return_all_masks
        )
        algorithm.fabric = L.Fabric(accelerator="cuda", devices=1)
    elif algo_name == "knots":
        algorithm = KnotsMerging(
            exclude_keys=exclude_keys,
        )
    elif algo_name == "knots_TallMask":
        algorithm = TallMaskKnotsMerging(
            tall_mask_lambda=0.6,
            exclude_keys=exclude_keys,
            eval_model_name=eval_task,
            return_all_masks=return_all_masks
        )
    elif algo_name == "EMR":
        algorithm = EMRMerging(
            eval_model_name=eval_task,
            exclude_keys=exclude_keys,
            return_all_masks=return_all_masks
        )
        device = "cpu"
    elif algo_name == "weighted_average":
        algorithm = WeightedAverageAlgorithm(normalize=False, weights=WA_weights, exclude_keys=exclude_keys)
        device = "cpu"
    else:
        raise ValueError(f"Unknown algorithm name: {algo_name}")
    print(f"num_excluded_keys: {len(exclude_keys) if exclude_keys else 0}, algo_name: {algo_name}, eval_task: {eval_task if eval_task else 'None'}, return_all_masks: {return_all_masks}, WA_weights: {WA_weights}")
    return algorithm, device

def merge_as_single_model(merged_tasks, algo_name, k_gate, action_head_layer_num, save_dir: str = "outputs", note: str = None):
    if isinstance(algo_name, str): # algo_name is like "TA_tall_mask" or ["TA_tall_mask", "weighted_average"]
        algo_name = [algo_name] * 2 
    else:
        assert isinstance(algo_name, list), "algo_name should be a string or a list of two strings."
        if len(algo_name) == 1:
            algo_name = algo_name * 2
        assert len(algo_name) == 2, "algo_name should be a list of two strings."
    assert isinstance(merged_tasks, list), "merged_tasks should be a list of task names."

    vlm_dict, head_dict, proprio_dict, stat = {}, {}, {}, []
    for task in merged_tasks:
        assert TaskSuite.has(tasks[task]), f"Unknown task: {task}"
        cfg = GenerateConfig(pretrained_checkpoint=ckptsprosigmoid[task], task_suite_name=tasks[task])
        vlm, action_head, proprio_projector = initialize_model(cfg)
        vlm_dict[TaskSuite(tasks[task])] = vlm.to("cpu")
        head_dict[TaskSuite(tasks[task])] = action_head.to("cpu")
        proprio_dict[TaskSuite(tasks[task])] = proprio_projector.to("cpu")
        stat.append(vlm.norm_stats)
    stat = {k: v for d in stat for k, v in d.items()}
    processor = get_processor(cfg)
    llm_dim = vlm.llm_dim

    identical_keys = get_identity_keys("model_merging", "libero", list(vlm_dict.values()))

    #####      VLM      #####
    AQ_keys = ["action_queries.weight"] # only contain AQ
    VLM_exclude_keys = identical_keys + AQ_keys
    exclude_AQ_keys = invert_keys(vlm_dict[TaskSuite(tasks['spatial'])].state_dict().keys(), AQ_keys) # only exclude AQ
    model_dict = copy.deepcopy(vlm_dict)
    merged_vlm_TV_model, tall_masks = merge_module(model_dict, "vlm_noAQ", algo_name=algo_name[0], exclude_keys=VLM_exclude_keys)
    if tall_masks:
        for key, tall_mask in tall_masks.items(): # convert to bool to save storage
            for k,v in tall_mask.items():
                if k == 'rescaler':
                    continue
                tall_masks[key][k] = v.to(torch.bool)

    # #####      VLM AQ      #####
    model_dict = copy.deepcopy(vlm_dict)
    merged_AQ_model, _ = merge_module(model_dict, "AQ", algo_name=algo_name[1], exclude_keys=exclude_AQ_keys)

    # # #####      Load checkpoint of AQ      #####
    sd = merged_vlm_TV_model.state_dict()
    sd['action_queries.weight'] = merged_AQ_model.state_dict()['action_queries.weight'].clone()
    merged_vlm_TV_model.load_state_dict(sd)

    #####      Projector      #####
    model_dict = copy.deepcopy(proprio_dict)
    merged_proprio_projector, _ = merge_module(model_dict, "proprio_projector", algo_name=algo_name[1], llm_dim=llm_dim)

    #####      Action Head      #####
    exclude_keys =  [k for k in head_dict[TaskSuite(tasks['spatial'])].state_dict() if "fc2" in k or "layer_norm2" in k or "23." in k]
    model_dict = copy.deepcopy(head_dict)
    merged_action_head, _ = merge_module(model_dict, "action_head", algo_name=algo_name[1], exclude_keys=exclude_keys, llm_dim=llm_dim)
    merged_moe_action_head = load_moe_model(merged_action_head, head_dict, k_gate, action_head_layer_num)
    merged_action_head = merged_moe_action_head

    #####      Save Model      #####
    len_model_dict = len(model_dict) if "_pretrained_" not in model_dict else len(model_dict)-1
    savename = f"merged_{len_model_dict}_libero_{algo_name[0]}_VLM_{algo_name[1]}_AQ_PP_AH_MoE_ALL"
    if note: savename += f"_{note}"
    savepath = Path(os.path.join(save_dir, savename))
    save(merged_vlm_TV_model, merged_proprio_projector, merged_action_head, processor, stat, merged_tasks, savepath, tall_masks=tall_masks)
    print("done")

def merge_module(model_dict, module_name, algo_name="TA", exclude_keys=None, eval_task=None, llm_dim=None, is_zero_init=False):
    if eval_task is not None: assert eval_task in model_dict.keys() is not None, f"eval_task {eval_task} not found in model_dict."

    print(f"merge module: {module_name} from models: {list(model_dict.keys())} using algorithm: {algo_name}")
    if algo_name == "weighted_average":
        len_model_dict = len(model_dict) - 1 if '_pretrained_' in model_dict else len(model_dict)
        WA_weights = [1/len_model_dict] * len_model_dict
    else:
        WA_weights = None
    algorithm, device = get_algo(algo_name, exclude_keys=exclude_keys, eval_task=eval_task, WA_weights=WA_weights, return_all_masks=True)
    for _, model in model_dict.items():
        model.to(device)
    if "average" not in algo_name: # add pretrained model
        print(f"Loading pretrained VLM for {module_name}.")
        if module_name == "vlm_noAQ" or module_name == "AQ":
            if '_pretrained_' not in model_dict:
                model_dict["_pretrained_"] = load_vlm_from_vla().to(device)
            else:
                print(f"Pretrained VLM already in model_dict of {module_name}.")
        else:
            assert llm_dim is not None, "llm_dim must be provided for proprio_projector and action_head."
            if module_name == "proprio_projector":
                model_dict["_pretrained_"] = initialize_proprio_projector(llm_dim, is_zero_init=is_zero_init).to(device)
            elif module_name == "action_head" or module_name == "AH_output_layer":
                model_dict["_pretrained_"] = initialize_action_head(llm_dim, is_zero_init=is_zero_init).to(device)
            else:
                raise ValueError(f"Unknown module name: {module_name}")
    model_pool = BaseModelPool(model_dict)
    merged_model = algorithm.run(model_pool)

    tall_masks = None
    if ("TallMask" in algo_name or 'EMR' in algo_name) and algorithm.return_all_masks:
        merged_model, tall_masks = merged_model # return_all_masks = True

    if eval_task:
        if "TallMask" in algo_name or 'EMR' in algo_name:
            merged_model = merged_model[eval_task]
             
        sd = merged_model.state_dict()
        if exclude_keys and module_name != "AQ" and module_name != "AH_output_layer":
            for k in exclude_keys:
                print(f"replace key: {k} from model {eval_task}")
                sd[k] = model_dict[eval_task].state_dict()[k].clone()
        merged_model.load_state_dict(sd)
    return merged_model, tall_masks

def save(vla, proprio_projector, action_head, processor, stat, merged_tasks, checkpoint_dir, tall_masks=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_dataset_statistics(stat, checkpoint_dir)
    torch.save({TaskSuite(tasks[task]): idx for idx, task in enumerate(merged_tasks)}, checkpoint_dir / "task2idx.pt"); print("Saved task2idx dict.")
    processor.save_pretrained(checkpoint_dir); print("Saved processor.")
    vla.save_pretrained(checkpoint_dir); print("Saved VLA.")
    torch.save(proprio_projector.state_dict(), checkpoint_dir / "proprio_projector--checkpoint.pt"); print("Saved proprio projector.")
    torch.save(action_head.state_dict(), checkpoint_dir / "action_head--checkpoint.pt"); print("Saved action head.")
    if tall_masks: torch.save(tall_masks, checkpoint_dir / "tall_masks.pt"); print("Saved tall masks.")
    print(f"Saved merged checkpoint to {checkpoint_dir}")

def save_dataset_statistics(dataset_statistics, run_dir):
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        json.dump(dataset_statistics, f_json, indent=2)

def load_moe_model(shared_weight_model, expert_models_dict, k_gate, action_head_layer_num):
    from prismatic.models.action_heads import L1RegressionMoEActionHead
    from prismatic.vla.constants import (ACTION_DIM)
    
    action_head = L1RegressionMoEActionHead(
        input_dim=LLM_DIM,
        hidden_dim=LLM_DIM,
        action_dim=ACTION_DIM,
        num_experts=len(expert_models_dict),
        k_gate=k_gate,
        action_head_layer_num=action_head_layer_num,
    ).to("cpu", dtype=torch.bfloat16)
    with torch.no_grad():
        for name, param in action_head.named_parameters():
            param.zero_()
    
    shared_keys = [x for x in list(shared_weight_model.state_dict().keys()) if x in list(action_head.state_dict().keys())]
    moe_keys = [k for k in action_head.state_dict().keys() if k not in shared_keys]

    share_layer_num = 24-action_head_layer_num

    sd = action_head.state_dict()
    for k in tqdm(shared_keys):
        sd[k] = shared_weight_model.state_dict()[k].clone()
    del shared_weight_model

    expert_sds = {}
    for k, v in expert_models_dict.items():
        expert_sds[k] = v.state_dict()

    for k in tqdm(moe_keys):
        if "gating_factor" in k:
            idx = int(k.split('.')[2])
            share_key = f"model.mlp_resnet_blocks.{share_layer_num+idx}.gating_factor"
            gating_factor = [expert_sd[share_key] for expert_sd in expert_sds.values()]
            gating_factor = torch.cat(gating_factor)
            sd[k] = gating_factor.clone()
            tqdm.write(f"{share_key} -> {k}")
        elif "gate_layer_idx" in k:
            sd[k] = torch.tensor(share_layer_num-1, dtype=torch.long)
            tqdm.write(f"{k} -> {share_layer_num-1}")
        elif "experts.1." in k or "experts.2." in k or "experts.3." in k:
            continue
        elif "layer_norm2" in k or "model.fc2" in k:
            prefix, suffix = k.split(".")[:2], k.split(".")[-1]
            share_key = ".".join(prefix + [suffix])
            for idx, (expert_name, expert_sd) in enumerate(expert_sds.items()):
                moe_key = ".".join(prefix) + f".experts.{idx}." + suffix
                assert moe_key in sd, f"{moe_key} not in sd"
                sd[moe_key] = expert_sd[share_key].clone()
                tqdm.write(f"{share_key} [{expert_name}] -> {moe_key}")
        elif "film_gen" in k or "_self" in k:
            print(f"Skip key {k} in moe action head.")
        elif "mlp_resnet_moe_blocks" in k:
            name_list = k.split(".")
            idx = int(name_list[2])
            if "ffn" in k:
                share_key = ".".join([name_list[0], f"mlp_resnet_blocks.{share_layer_num+idx}", name_list[3], name_list[4], name_list[-1]])
            else:
                share_key = ".".join([name_list[0], f"mlp_resnet_blocks.{share_layer_num+idx}", name_list[-4], name_list[-1]])
            for expert_idx, (expert_name, expert_sd) in enumerate(expert_sds.items()):
                if "ffn" in k:
                    moe_key = ".".join(name_list[:5]) + f".experts.{expert_idx}." + name_list[-1]
                else:
                    moe_key = ".".join(name_list[:4]) + f".experts.{expert_idx}." + name_list[-1]
                assert share_key in expert_sd, f"{share_key} not in expert's sd"
                assert moe_key in sd, f"{moe_key} not in sd"
                sd[moe_key] = expert_sd[share_key].clone()
                tqdm.write(f"{share_key} [{expert_name}] -> {moe_key}")
        else:
            print(f"💡 Temporarily skip key {k}")

    sd = create_gate(expert_sds, sd, k=k_gate, action_head_layer_num=action_head_layer_num)
    
    action_head.load_state_dict(sd)
    print("MoE action head loaded.")
    return action_head

def create_gate(expert_sds, moe_model_sd, k=8, action_head_layer_num=1):
    gate_module_name = [
        f"model.mlp_resnet_blocks.{24-action_head_layer_num-1}.v_adapter.weight",
        f"model.mlp_resnet_blocks.{24-action_head_layer_num-1}.v_task.weight",
    ]
    gate_name_template = "model.gate.gate.{}.routers.{}"

    device = moe_model_sd[gate_name_template.format(0, 0)].device
    original_dtype = moe_model_sd[gate_name_template.format(0, 0)].dtype

    for idx, module_name in enumerate(gate_module_name):
        for j, (expert, expert_sd) in enumerate(expert_sds.items()):
            gate_name = gate_name_template.format(idx, j)
            w = expert_sd[module_name].clone().to(
                    device, dtype=torch.float32, non_blocking=True
                ) 
            _, _, v = svd(w)
            v = v[:, :k]
            moe_model_sd[gate_name] = v.T.clone().to(original_dtype).to(device)
            print(f"{module_name} [{expert}] -> {gate_name}")
    return moe_model_sd

def get_identity_keys(path, benchmark, expert: List=None):
    filepath = os.path.join(path, f'{benchmark}_identical_keys.pt')
    if os.path.exists(filepath):
        print(f"Load identical keys at {filepath}.")
        return torch.load(filepath)
    else:
        assert expert is not None, "expert must be provided if the identical keys does not exist."
        expert_sds = [e.state_dict() for e in expert]
        sd_spatial, sd_object, sd_goal, sd_10 = expert_sds
        sd_base = load_vlm_from_vla().to("cpu").state_dict()
        common_keys = set(sd_base.keys()) & set(sd_spatial.keys()) & set(sd_object.keys()) & set(sd_goal.keys()) & set(sd_10.keys())
        identical_keys = []
        for k in common_keys:
            t1, t2, t3, t4, t5 = sd_spatial[k], sd_object[k], sd_goal[k], sd_10[k], sd_base[k]
            if (
                t1.shape == t2.shape == t3.shape == t4.shape == t5.shape
                and t1.dtype == t2.dtype == t3.dtype == t4.dtype == t5.dtype
                and torch.equal(t1, t2)
                and torch.equal(t1, t3)
                and torch.equal(t1, t4)
                and torch.equal(t1, t5)
            ):
                identical_keys.append(k)
        identical_keys = [k for k in sd_spatial if k in identical_keys]

        torch.save(identical_keys, filepath)
        print(f"Saved identical keys of {benchmark} at {filepath}.")
        return identical_keys
    
def get_identity_keys_AH(expert: List=None):
    expert_sds = [e.state_dict() for e in expert]
    sd_spatial, sd_object, sd_goal, sd_10 = expert_sds
    common_keys = set(sd_spatial.keys()) & set(sd_object.keys()) & set(sd_goal.keys()) & set(sd_10.keys())
    identical_keys = []
    for k in common_keys:
        t1, t2, t3, t4 = sd_spatial[k], sd_object[k], sd_goal[k], sd_10[k]
        if (
            t1.shape == t2.shape == t3.shape == t4.shape
            and t1.dtype == t2.dtype == t3.dtype == t4.dtype
            and torch.equal(t1, t2)
            and torch.equal(t1, t3)
            and torch.equal(t1, t4)
        ):
            identical_keys.append(k)
    identical_keys = [k for k in sd_spatial if k in identical_keys and "self" not in k and "film" not in k]
    return identical_keys

if __name__ == "__main__":
    merged_tasks = ["spatial", "object", "goal", "10"]
    algo_name = ["TATallMask", "weighted_average"]
    action_head_layer_num = 1
    k_gate = 8
    merge_as_single_model(merged_tasks=merged_tasks, algo_name=algo_name, k_gate=k_gate, action_head_layer_num=action_head_layer_num, 
                          note=f'{len(merged_tasks)}tasks_AHnum_{action_head_layer_num}_k_{k_gate}', save_dir="outputs")