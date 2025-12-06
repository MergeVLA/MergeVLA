import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Mapping, Union  # noqa: F401
from tqdm import tqdm

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
    state_dict_sub,
    state_dict_sum,
    state_dict_hadamard_product
)

from fusion_bench.method.knots.knots_merging import subspace_alignment
from .utils import generate_task_masks

log = logging.getLogger(__name__)

@auto_register_config
class TallMaskKnotsMerging(
    SimpleProfilerMixin,
    BaseAlgorithm,
):
    def __init__(
        self,
        tall_mask_lambda: float,
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
            new_task_vector = {k: task_vector[k] for k in list(pretrained_model.state_dict().keys()) if k in list(task_vector.keys())}
            task_vectors[model_name] = new_task_vector
        
        multi_task_vector = {}
        for key in tqdm(new_task_vector.keys()):
            task_vectors_list = []
            for tv in task_vectors.values():
                task_vectors_list.append(tv[key])
            if task_vectors_list[0].dim() == 2:
                U, S, Vhs = subspace_alignment(task_vectors_list)
                V_merged = torch.sum(torch.stack(Vhs), dim=0) # TA, sum the Vhs directly
                multi_task_vector[key] = ((U * S) @ V_merged).to('cpu')
            else:
                multi_task_vector[key] = torch.stack(task_vectors_list, dim=0).mean(dim=0).to('cpu')

        tall_masks = {model: {} for model in modelpool.model_names}
        with self.profile("generating task masks"):
            for key, tensor in multi_task_vector.items():
                multi_task_vector[key] = tensor.to('cpu')
            
            for model_name in modelpool.model_names:
                for key, tensor in task_vectors[model_name].items():
                    task_vectors[model_name][key] = tensor.to(multi_task_vector[key].device)
                tall_mask = generate_task_masks(
                    multi_task_vector,
                    task_vectors[model_name],
                    tall_mask_lambda=self.tall_mask_lambda,
                )
                tall_masks[model_name] = tall_mask

        with self.profile("compress and retrieve"):
            if self.eval_model_name is None:
                log.info("return merged task vectors and tall masks for all models.")
                final_model = deepcopy(pretrained_model).to('cpu')
                with torch.no_grad():
                    for param in final_model.parameters():
                        param.zero_()
                final_model.load_state_dict(multi_task_vector, strict=True if not self.exclude_keys else False)
                final_model.to('cuda')
                self.return_all_masks = True
                self.print_profile_summary()
                return final_model, tall_masks
            else:
                models = {}
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