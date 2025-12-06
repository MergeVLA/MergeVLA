"""
Whoever Started the Interference Should End It:  Guiding Data-Free Model Merging via Task Vectors
Arxiv: http://arxiv.org/abs/2503.08099
"""

from typing import List

import torch
from tqdm import tqdm
from copy import deepcopy

from fusion_bench import BaseAlgorithm, BaseModelPool, auto_register_config
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.utils import timeit_context
from fusion_bench.utils.state_dict_arithmetic import state_dict_add, state_dict_sub, state_dict_hadamard_product
from .utils import generate_task_masks

def wudi_merging(
    task_vectors: List[torch.Tensor],
    accelerator="cuda",
    iter_num: int = 300,
    exclude_keys: List[str] = None,
):
    exclude_keys = [] if exclude_keys is None else exclude_keys

    with timeit_context("WUDI Merging"):
        new_vector = {}
        for key in tqdm(task_vectors[0], desc="WUDI Merging", leave=False):
            tqdm.write(f"key: {key}")
            original_device = task_vectors[0][key].device
            tvs = torch.stack(
                [
                    task_vector[key].to(device=accelerator, non_blocking=True)
                    for task_vector in task_vectors
                ]
            )
            num_tvs = len(tvs)

            if key in exclude_keys:
                continue

            new_vector[key] = torch.nn.Parameter(torch.sum(tvs, dim=0))
            if len(task_vectors[0][key].shape) == 2:
                optimizer = torch.optim.Adam([new_vector[key]], lr=1e-5, weight_decay=0)
                l2_norms = torch.square(
                    torch.norm(tvs.reshape(tvs.shape[0], -1), p=2, dim=-1)
                )
                for i in tqdm(
                    range(iter_num),
                ):
                    disturbing_vectors = new_vector[key].unsqueeze(0) - tvs
                    product = torch.matmul(disturbing_vectors, tvs.transpose(1, 2))
                    loss = torch.sum(
                        torch.square(product) / l2_norms.unsqueeze(-1).unsqueeze(-1)
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            else:
                new_vector[key] = new_vector[key] / num_tvs
            new_vector[key] = new_vector[key].to(
                device=original_device, non_blocking=True
            )
    return new_vector


@auto_register_config
class TallMaskWUDIMerging(
    LightningFabricMixin,
    BaseAlgorithm,
):
    """
    Whoever Started the Interference Should End It:  Guiding Data-Free Model Merging via Task Vectors
    """

    def __init__(
        self,
        iter_num: int,
        tall_mask_lambda: float,
        eval_model_name: str | None = None,
        exclude_keys: List[str] = None,
        return_all_masks: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def run(self, modelpool: BaseModelPool):
        # load the pretrained model and the task vectors of all the finetuned models
        with torch.no_grad():
            pretrained_model = modelpool.load_pretrained_model()
            task_vectors = []
            for model_name in modelpool.model_names:
                finetuned_model = modelpool.load_model(model_name)
                task_vectors.append(
                    state_dict_sub(
                        finetuned_model.state_dict(), pretrained_model.state_dict()
                    )
                )
                del finetuned_model  # free memory
            task_vectors_dict = {}

            for i, model_name in enumerate(modelpool.model_names):
                task_vectors_dict[model_name] = {k:v for k,v in task_vectors[i].items() if k not in self.exclude_keys}

        merged_tv = wudi_merging(
            task_vectors,
            accelerator=self.fabric.device,
            iter_num=self.iter_num,
            exclude_keys=self.exclude_keys,
        )

        for key, tensor in merged_tv.items():
            merged_tv[key] = tensor.to('cpu')
        for model_name, task_vector in task_vectors_dict.items():
            for key, tensor in task_vector.items():
                task_vector[key] = tensor.to('cpu')


        tall_masks = {model: {} for model in modelpool.model_names}
        for model_name in modelpool.model_names:
            tall_mask = generate_task_masks(
                merged_tv,
                task_vectors_dict[model_name],
                tall_mask_lambda=self.tall_mask_lambda,
            )
            tall_masks[model_name] = tall_mask

        if self.eval_model_name is None:
            final_model = deepcopy(pretrained_model)
            with torch.no_grad():
                for param in final_model.parameters():
                    param.zero_()
            final_model.load_state_dict(merged_tv, strict=True if not self.exclude_keys else False)

            self.return_all_masks = True
            return final_model, tall_masks
        
        else:
            models = {}
            for model_name in modelpool.model_names:
                if model_name != self.eval_model_name:
                    continue
                
                for key, tensor in tall_masks[model_name].items():
                    tall_masks[model_name][key] = tensor.to(merged_tv[key].dtype).to('cpu')
                retrieved_task_vector = state_dict_hadamard_product(
                    tall_masks[model_name], merged_tv
                )
                retrieved_state_dict = state_dict_add(
                    pretrained_model.to('cpu').state_dict(keep_vars=True), retrieved_task_vector,
                    show_pbar=True,
                    strict=False,
                    exclude_keys=self.exclude_keys
                )
                retrieved_model = deepcopy(pretrained_model)
                retrieved_model.load_state_dict(retrieved_state_dict, strict=True if not self.exclude_keys else False)
                models[model_name] = retrieved_model

            if self.return_all_masks:
                return models, tall_masks
            else:
                return models

        # pretrained_model.load_state_dict(
        #     state_dict_add(pretrained_model.state_dict(), merged_tv)
        # )

        # return pretrained_model
