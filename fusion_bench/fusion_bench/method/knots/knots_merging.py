import logging
from copy import deepcopy
from typing import Any, Dict, List, Literal, Mapping, Union  # noqa: F401
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
)

log = logging.getLogger(__name__)

def subspace_alignment(
    delta_weights: list[torch.Tensor],
    svd_dtype: torch.dtype | None = torch.float32,
    eps: float = 1e-4,
):
    """
    Reference: Model merging with SVD to tie the Knots. http://arxiv.org/abs/2410.19735
    """
    if svd_dtype is None:
        svd_dtype = delta_weights[0].dtype
    original_dtype = delta_weights[0].dtype
    output_dim, input_dim = delta_weights[0].size()
    concat_task_vector = torch.cat(delta_weights, dim=1)
    U, S, Vh = torch.linalg.svd(concat_task_vector.to(svd_dtype), full_matrices=False)
    # Keep only supported basis components
    U = U[:, S > eps].to(original_dtype)
    Vh = Vh[S > eps].to(original_dtype)
    S = S[S > eps].to(original_dtype)
    Vhs = torch.split(Vh, input_dim, dim=1)
    return U, S, Vhs


@auto_register_config
class KnotsMerging(
    SimpleProfilerMixin,
    BaseAlgorithm,
):
    def __init__(
        self,
        # tall_mask_lambda: float,
        # eval_model_name: Optional[str] = None,
        exclude_keys: List[str] = None,
        # return_all_masks: bool = False,
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
                multi_task_vector[key] = (U * S) @ V_merged
            else:
                multi_task_vector[key] = torch.stack(task_vectors_list, dim=0).mean(dim=0)

        with self.profile("compress and retrieve"):
            retrieved_state_dict = state_dict_add(
                pretrained_model.state_dict(keep_vars=True), multi_task_vector,
                show_pbar=True,
                strict=False,
                exclude_keys=self.exclude_keys
            )
            merged_model = deepcopy(pretrained_model)
            with torch.no_grad():
                for param in merged_model.parameters():
                    param.zero_()
            merged_model.load_state_dict(retrieved_state_dict, strict=True if not self.exclude_keys else False)

            self.print_profile_summary()
            return merged_model