"""
Modified from https://github.com/Zhou-Hangyu/randes/tree/main/benchmark/fusion_bench
"""

import logging
from collections import OrderedDict
from copy import deepcopy
from typing import List, Optional

import torch

from fusion_bench import BaseAlgorithm
from fusion_bench.mixins import SimpleProfilerMixin, auto_register_config
from fusion_bench.modelpool import BaseModelPool
from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_binary_mask,
    state_dict_diff_abs,
    state_dict_hadamard_product,
    state_dict_mul,
    state_dict_sub,
    state_dict_sum,
)

from .utils import generate_task_masks

log = logging.getLogger(__name__)

@auto_register_config
class TallMaskTaskArithmeticAlgorithm(
    SimpleProfilerMixin,
    BaseAlgorithm,
):
    def __init__(
        self,
        tall_mask_lambda: float,
        eval_model_name: Optional[str] = None,
        exclude_keys: List[str] = None,
        return_all_masks: bool = False,
        verbose: int = 0,
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
            for layer_name, layer in model.state_dict(keep_vars=True).items():
                if self.verbose >= 1:
                    log.info(f"{layer_name} | {layer.shape}")
            task_vector = state_dict_sub(
                model.state_dict(keep_vars=True),
                pretrained_model.state_dict(keep_vars=True),
                exclude_keys=self.exclude_keys,
                strict=True if not self.exclude_keys else False,
                show_pbar=True
            )
            task_vectors[model_name] = task_vector

        multi_task_vector = state_dict_sum(list(task_vectors.values()))

        tall_masks = {model: {} for model in modelpool.model_names}
        with self.profile("generating task masks"):
            for model_name in modelpool.model_names:
                tall_mask = generate_task_masks(
                    multi_task_vector,
                    task_vectors[model_name],
                    # pretrained_model.state_dict(keep_vars=True),
                    tall_mask_lambda=self.tall_mask_lambda,
                )
                tall_masks[model_name] = tall_mask

        with self.profile("compress and retrieve"):
            if self.eval_model_name is None:
                log.info("return merged task vectors and tall masks for all models.")
                final_model = deepcopy(pretrained_model)
                with torch.no_grad():
                    for param in final_model.parameters():
                        param.zero_()
                final_model.load_state_dict(multi_task_vector, strict=True if not self.exclude_keys else False)

                self.return_all_masks = True
                self.print_profile_summary()
                return final_model, tall_masks
            else:
                for model_name in modelpool.model_names:
                    if model_name != self.eval_model_name:
                        continue
                    
                    for key, tensor in tall_masks[model_name].items():
                        tall_masks[model_name][key] = tensor.to(multi_task_vector[key].dtype)
                    retrieved_task_vector = state_dict_hadamard_product(
                        tall_masks[model_name], multi_task_vector
                    )
                    retrieved_state_dict = state_dict_add(
                        pretrained_model.state_dict(keep_vars=True), retrieved_task_vector,
                        show_pbar=True,
                        strict=False,
                        exclude_keys=self.exclude_keys
                    )
                    retrieved_model = deepcopy(pretrained_model)
                    retrieved_model.load_state_dict(retrieved_state_dict, strict=True if not self.exclude_keys else False)
                    models[model_name] = retrieved_model

                self.print_profile_summary()
                if self.return_all_masks:
                    return models, tall_masks
                else:
                    return models