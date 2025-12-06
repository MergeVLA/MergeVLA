from typing import List, Dict

import torch
from copy import deepcopy
from collections import OrderedDict

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_mul,
    state_dict_sub,
    state_dict_hadamard_product
)
from fusion_bench.method.isotropic_merging.iso_utils import check_parameterNamesMatch, iso_c, iso_cts

from .utils import generate_task_masks

# class TallMaskIsotropicMergingInCommonSubspace(BaseAlgorithm, LightningFabricMixin):
#     """
#     Isotropic Merging in Common Subspace (Iso-C)
#     """

#     def __init__(
#         self,
#         scaling_factor: float,
#         exclude_keys: List[str] = None,
#     ):
#         self.scaling_factor = scaling_factor
#         self.exclude_keys = exclude_keys
#         super().__init__()

#     def run(self, modelpool: BaseModelPool):
#         # load the pretrained model and the task vectors of all the finetuned models
#         with torch.no_grad():
#             pretrained_model = modelpool.load_pretrained_model()
#             task_vectors = []
#             for model_name in modelpool.model_names:
#                 finetuned_model = modelpool.load_model(model_name)
#                 task_vectors.append(
#                     state_dict_sub(
#                         finetuned_model.state_dict(), pretrained_model.state_dict()
#                     )
#                 )
#                 del finetuned_model  # free memory
#             check_parameterNamesMatch(task_vectors)

#         # compute the merged task vector
#         merged_tv = iso_c(
#             task_vectors,
#             accelerator=self.fabric.device,
#             exclude_keys=self.exclude_keys,
#         )

#         # merged_parameters = pretrained_parameters + scaling_factor * merged_task_vector
#         pretrained_model.load_state_dict(
#             state_dict_add(
#                 pretrained_model.state_dict(),
#                 state_dict_mul(merged_tv, self.scaling_factor),
#             )
#         )

#         return pretrained_model


class TallMaskIsotropicMergingInCommonAndTaskSubspace(BaseAlgorithm, LightningFabricMixin):
    """
    Isotropic Merging in Common and Task-Specific Subspaces (Iso-CTS)
    """

    def __init__(
        self,
        scaling_factor: float,
        tall_mask_lambda: float,
        common_space_fraction: float,
        eval_model_name: str | None = None,
        exclude_keys: List[str] = None,
        return_all_masks: bool = False,
    ):
        self.scaling_factor = scaling_factor
        self.common_space_fraction = common_space_fraction
        self.tall_mask_lambda = tall_mask_lambda
        self.eval_model_name = eval_model_name
        self.return_all_masks = return_all_masks
        self.exclude_keys = exclude_keys
        super().__init__()

    def run(self, modelpool: BaseModelPool):
        # load the pretrained model and the task vectors of all the finetuned models
        with torch.no_grad():
            pretrained_model = modelpool.load_pretrained_model()
            
            task_vectors_list = []
            task_vectors = {}
            models = {}
            for model_name in modelpool.model_names:
                finetuned_model = modelpool.load_model(model_name)
                task_vector = state_dict_sub(
                    finetuned_model.state_dict(), pretrained_model.state_dict(), 
                    show_pbar=True, exclude_keys=self.exclude_keys,
                    strict=True if not self.exclude_keys else False,
                )
                task_vector = OrderedDict(
                    (k, task_vector[k]) for k in finetuned_model.state_dict().keys() if k in task_vector
                )
                task_vectors[model_name] = {k: v.cpu() for k, v in task_vector.items()} # move to cpu to save gpu memory
                task_vectors_list.append(task_vector)
                del finetuned_model  # free memory
            check_parameterNamesMatch(task_vectors_list)

        # compute the merged task vector
        merged_tv = iso_cts(
            task_vectors_list,
            common_space_fraction=self.common_space_fraction,
            accelerator=self.fabric.device,
            exclude_keys=self.exclude_keys,
        )

        # generate tall masks for each model
        tall_masks = {model: {} for model in modelpool.model_names}
        for model_name in modelpool.model_names:
            merged_tv = {k: v.cpu() for k, v in merged_tv.items()} # move to cpu to save gpu memory
            tall_mask = generate_task_masks(
                merged_tv,
                task_vectors[model_name],
                tall_mask_lambda=self.tall_mask_lambda,
            )
            tall_masks[model_name] = tall_mask

        # merged_parameters = pretrained_parameters + scaling_factor * merged_task_vector * tall_mask
        for model_name in modelpool.model_names:
            if self.eval_model_name is not None and model_name != self.eval_model_name:
                continue
            for key, tensor in tall_masks[model_name].items():
                tall_masks[model_name][key] = tensor.to(merged_tv[key].dtype)
            retrieved_task_vector = state_dict_hadamard_product(
                tall_masks[model_name], merged_tv
            )
            # retrieved_task_vector = merged_tv # NOTE: test without tall mask

            retrieved_state_dict = state_dict_add(
                pretrained_model.to("cpu").state_dict(), state_dict_mul(retrieved_task_vector, self.scaling_factor),
                show_pbar=True,
                strict=True if not self.exclude_keys else False,
                exclude_keys=self.exclude_keys
            )
            retrieved_model = deepcopy(pretrained_model)
            retrieved_model.load_state_dict(retrieved_state_dict, strict=True if not self.exclude_keys else False)
            models[model_name] = retrieved_model

        if self.return_all_masks:
            return models, tall_masks
        else:
            return models

# TM_ISO_C_Merge = TallMaskIsotropicMergingInCommonSubspace  # alias
TM_ISO_CTS_Merge = TallMaskIsotropicMergingInCommonAndTaskSubspace  # alias
