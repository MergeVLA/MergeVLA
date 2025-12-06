"""
Modified from 
EMR-MERGING: Tuning-Free High-Performance Model Merging
https://arxiv.org/pdf/2405.17461
https://github.com/harveyhuang18/EMR_Merging
"""

import logging
from collections import OrderedDict
from copy import deepcopy
from typing import List, Optional
from tqdm import tqdm

import torch

from fusion_bench import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin, auto_register_config
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_sub,
    state_dict_sum,
)

log = logging.getLogger(__name__)

def emr_merge(task_vectors, sum_param):
    vector_unified = {}
    scales = torch.zeros(len(task_vectors))
    masks = {}
    masks = {model: {} for model in task_vectors.keys()}
    for key in tqdm(sum_param):
        # masks[key] = []
        flag = (sum_param[key] > 0) * 2 - 1 # indicate the sign, True -> 1, False -> -1
        param_max = torch.zeros_like(task_vectors[list(task_vectors.keys())[0]][key])
        for idx, (name, tv) in enumerate(task_vectors.items()):
            param = tv[key]
            mask = (param * flag) > 0 # whether the sign is the same as sum_param
            masks[name][key] = mask
            
            param_abs = torch.abs(mask * param) # only consider the part with the same sign and absolute value
            param_max = torch.where(param_abs > param_max, param_abs, param_max)
            scales[idx] += torch.mean(torch.abs(param))
        vector_unified[key] =  param_max * flag
    
    # compute rescalers
    new_scales = torch.zeros(len(task_vectors))
    for i, name in enumerate(task_vectors.keys()):
        for key in vector_unified:
            p = vector_unified[key] * masks[name][key]
            new_scales[i] += torch.mean(torch.abs(p))
    rescalers = scales / new_scales
    
    for i, name in enumerate(masks.keys()): 
        masks[name]['rescaler'] = rescalers[i]

    return vector_unified, masks

@auto_register_config
class EMRMerging(
    SimpleProfilerMixin,
    BaseAlgorithm,
):
    def __init__(
        self,
        # tall_mask_lambda: float,
        eval_model_name: Optional[str] = None,
        exclude_keys: List[str] = None,
        return_all_masks: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(models=modelpool)

        log.info("Compressing models using tall mask task arithmetic.")
        task_vector = None
        with self.profile("load model"):
            pretrained_model = modelpool.load_model("_pretrained_")

        task_vectors = {}
        models = {}
        for model_name in modelpool.model_names:
            with self.profile("load model"):
                model = modelpool.load_model(model_name)
            task_vector = state_dict_sub(
                model.state_dict(keep_vars=True),
                pretrained_model.state_dict(keep_vars=True),
                exclude_keys=self.exclude_keys,
                strict=True if not self.exclude_keys else False,
                show_pbar=True
            )
            # new_task_vector = OrderedDict() # rearrange the order of keys
            new_task_vector = {k: task_vector[k] for k in list(pretrained_model.state_dict().keys()) if k in list(task_vector.keys())}
            task_vectors[model_name] = new_task_vector

        multi_task_vector = state_dict_sum(list(task_vectors.values()))
        multi_task_vector, emr_masks = emr_merge(task_vectors, multi_task_vector)

        with self.profile("compress and retrieve"):
            if self.eval_model_name is None:
                log.info("return merged task vectors and tall masks for all models.")
                final_model = deepcopy(pretrained_model)
                with torch.no_grad():
                    for param in final_model.parameters():
                        param.zero_()
                
                final_model.load_state_dict(multi_task_vector, strict=True if not self.exclude_keys else False) # return merged task vector

                self.return_all_masks = True
                self.print_profile_summary()
                return final_model, emr_masks
            else:
                for model_name in modelpool.model_names:
                    if model_name != self.eval_model_name:
                        continue
                    
                    task_vector_recon = {}
                    for key in multi_task_vector:
                        task_vector_recon[key] =  multi_task_vector[key] * emr_masks[model_name][key] * emr_masks[model_name]['rescaler']

                    retrieved_state_dict = state_dict_add(
                        pretrained_model.state_dict(keep_vars=True), task_vector_recon,
                        show_pbar=True,
                        strict=False,
                        exclude_keys=self.exclude_keys
                    )
                    retrieved_model = deepcopy(pretrained_model)
                    with torch.no_grad():
                        for param in retrieved_model.parameters():
                            param.zero_()
                    retrieved_model.load_state_dict(retrieved_state_dict, strict=True if not self.exclude_keys else False)
                    models[model_name] = retrieved_model

                self.print_profile_summary()
                if self.return_all_masks:
                    return models, emr_masks
                else:
                    return models