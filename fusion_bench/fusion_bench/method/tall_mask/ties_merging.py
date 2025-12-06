import logging
from copy import deepcopy
from typing import Any, Dict, List, Literal, Mapping, Union  # noqa: F401

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel

from fusion_bench import LazyStateDict
from fusion_bench.compat.modelpool import to_modelpool
from fusion_bench.method import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin, auto_register_config
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.type import StateDictType
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_mul,
    state_dict_sub,
    state_dict_hadamard_product,
    _validate_state_dict_same_keys
)

from fusion_bench.method.ties_merging.ties_merging_utils import state_dict_to_vector, ties_merging, vector_to_state_dict

from .utils import generate_task_masks

log = logging.getLogger(__name__)


@auto_register_config
class TallMaskTiesMergingAlgorithm(
    SimpleProfilerMixin,
    BaseAlgorithm,
):
    def __init__(
        self,
        scaling_factor: float,
        threshold: float,
        remove_keys: List[str],
        merge_func: Literal["sum", "mean", "max"],
        tall_mask_lambda: float,
        eval_model_name: str | None = None,
        return_all_masks: bool = False,
        **kwargs: Any,
    ):
        """
        TiesMergingAlgorithm is a class for fusing multiple models using the TIES merging technique.

        Initialize the TiesMergingAlgorithm with the given parameters.

        Args:
            scaling_factor (float): The scaling factor to apply to the merged task vector.
            threshold (float): The threshold for resetting values in the task vector.
            remove_keys (List[str]): List of keys to remove from the state dictionary.
            merge_func (Literal["sum", "mean", "max"]): The merge function to use for disjoint merging.
            **kwargs: Additional keyword arguments for the base class.
        """
        super().__init__(**kwargs)

    @torch.no_grad()
    def run(
        self, modelpool: BaseModelPool | Dict[str, nn.Module], **kwargs: Any
    ) -> nn.Module:
        """
        Run the TIES merging algorithm to fuse models in the model pool.

        Args:
            modelpool (BaseModelPool | Dict[str, nn.Module]): The model pool containing the models to fuse.

        Returns:
            nn.Module: The fused model.
        """
        log.info("Fusing models using ties merging.")
        modelpool = to_modelpool(modelpool)
        remove_keys = self.config.get("remove_keys", [])
        merge_func = self.config.get("merge_func", "sum")
        scaling_factor = self.scaling_factor
        threshold = self.threshold
        tall_mask_lambda = self.tall_mask_lambda
        eval_model_name = self.eval_model_name
        return_all_masks = self.return_all_masks

        with self.profile("loading models"):
            # Load the pretrained model
            pretrained_model = modelpool.load_model("_pretrained_")

            # Load the state dicts of the models
            ft_checks: List[StateDictType] = [
                modelpool.load_model(model_name).state_dict(keep_vars=True)
                for model_name in modelpool.model_names
            ]
            ptm_check: StateDictType = pretrained_model.state_dict(keep_vars=True)

        task_vectors = {}
        models = {}
        with self.profile("merging models"):
            # Compute the task vectors
            flat_ft: Tensor = torch.vstack(
                [state_dict_to_vector(check, remove_keys) for check in ft_checks]
            )
            flat_ptm: Tensor = state_dict_to_vector(ptm_check, remove_keys)
            tv_flat_checks = flat_ft - flat_ptm
            for model_name, tv_flat_check in zip(modelpool.model_names, tv_flat_checks):
                task_vector = vector_to_state_dict(
                    tv_flat_check, ptm_check, remove_keys=remove_keys
                )
                task_vectors[model_name] = task_vector
            # Perform TIES Merging
            merged_tv = ties_merging(
                tv_flat_checks,
                reset_thresh=threshold,
                merge_func=merge_func,
            )
            merged_tv = vector_to_state_dict(
                merged_tv, ptm_check, remove_keys=remove_keys
            ) # to state_dict
            for k, v in merged_tv.items():
                if isinstance(v, torch.nn.Parameter):
                    merged_tv[k] = v.detach().clone()

        with self.profile("generating task masks"):
            # generate tall masks for each model
            tall_masks = {model: {} for model in modelpool.model_names}
            for model_name in modelpool.model_names:
                for key, tensor in task_vectors[model_name].items():
                    task_vectors[model_name][key] = tensor.to(merged_tv[key].dtype)
                tall_mask = generate_task_masks(
                    merged_tv,
                    task_vectors[model_name],
                    tall_mask_lambda=tall_mask_lambda,
                )
                tall_masks[model_name] = tall_mask

        with self.profile("compress and retrieve"):
            if self.eval_model_name is None:
                log.info("return merged task vectors and tall masks for all models.")
                final_model = deepcopy(pretrained_model)
                with torch.no_grad():
                    for param in final_model.parameters():
                        param.zero_()
                final_model.load_state_dict(merged_tv, strict=True if not remove_keys else False)

                self.return_all_masks = True
                self.print_profile_summary()
                return final_model, tall_masks
            else:
                # merged_parameters = pretrained_parameters + scaling_factor * merged_task_vector * tall_mask
                for model_name in modelpool.model_names:
                    if eval_model_name is not None and model_name != self.eval_model_name:
                        continue
                    try: # match dtype
                        _validate_state_dict_same_keys([tall_masks[model_name], merged_tv])
                    except ValueError:
                        for key, tensor in tall_masks[model_name].items():
                            tall_masks[model_name][key] = tensor.to(merged_tv[key].dtype)
                    retrieved_task_vector = state_dict_hadamard_product(
                        tall_masks[model_name], merged_tv
                    )
                    # retrieved_task_vector = merged_tv # NOTE: test without tall mask

                    retrieved_state_dict = state_dict_add(
                        pretrained_model.to(next(iter(retrieved_task_vector.values())).dtype).state_dict(), state_dict_mul(retrieved_task_vector, scaling_factor),
                        show_pbar=True,
                        strict=True if not remove_keys else False,
                        exclude_keys=remove_keys
                    )
                    retrieved_model = deepcopy(pretrained_model)
                    retrieved_model.load_state_dict(retrieved_state_dict, strict=True if not remove_keys else False)
                    models[model_name] = retrieved_model
                self.print_profile_summary()
                if return_all_masks:
                    return models, tall_masks
                else:
                    return models
    
