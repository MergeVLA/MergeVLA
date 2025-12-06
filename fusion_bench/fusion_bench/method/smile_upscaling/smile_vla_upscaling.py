import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union  # noqa: F401

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor, nn
from tqdm.auto import tqdm

from fusion_bench.method import BaseAlgorithm
from fusion_bench.method.simple_average import simple_average
from fusion_bench.mixins.simple_profiler import SimpleProfilerMixin
from fusion_bench.modelpool import BaseModelPool
# from fusion_bench.models.smile_moe.linear_from_module import (
#     ExpertNotTrainedError,
#     SmileCompressedLinear,
#     SmileGate,
#     SmileMoELinear,
# )
from fusion_bench.models.smile_moe.utils import _is_all_zeros, svd
from fusion_bench.models.utils import get_attr, set_attr
from fusion_bench.utils.devices import get_device
from fusion_bench.utils.parameters import print_parameters

from .smile_upscaling import SmileUpscalingAlgorithm

log = logging.getLogger(__name__)


class SmileGate(nn.Module):
    __constants__ = ["in_features", "num_experts", "k"]
    in_features: int
    num_experts: int
    k: int
    weight: nn.Parameter

    def __init__(
        self,
        input_features: int,
        w_list: List[Tensor],
        k: int,
        upscaling_accelerator=None,
    ):
        super().__init__()
        self.input_features = input_features
        self.num_experts = len(w_list)

        weights = []
        for i, w in enumerate(w_list):
            _, s, v = svd(w, accelerator=upscaling_accelerator)
            # u = u[:, :k]
            s = s[:k]
            v = v[:, :k]

            weights.append(v.T)
        self.k = s.size(0)  # k is the actual k after truncation

        weights = (
            torch.stack(weights, dim=0)
            .reshape(self.num_experts * self.k, -1)
            .contiguous()
        )
        self.weights = nn.Parameter(
            weights
        )  # weights should be a tensor of shape (num_experts * k, n)

    def forward(self, x: Tensor):
        batch_size = x.size(0)
        if self.num_experts == 1:
            return torch.ones(batch_size, 1, device=x.device, dtype=x.dtype)

        routing_weights = F.linear(x, self.weights).view(
            batch_size, -1, self.num_experts, self.k
        ) # (1, 8, 32) -> (1, 8, num_experts, k)
        routing_weights = routing_weights.norm(p=2, dim=3) # (1, 8, 4)
        return routing_weights


class SmileMoEGate(nn.Module):
    # __constants__ = ["in_features", "num_experts", "k"]
    # in_features: int
    # num_experts: int
    # k: int
    # weight: nn.Parameter

    def __init__(
        self,
        modules: nn.Module,
        # k: int,
        upscaling_accelerator=None,
    ):
        super().__init__()
        self.num_experts = None
        gates = []
        for module in modules:
            if self.num_experts is None: self.num_experts = module.num_experts
            device = get_device(module)
            original_dtype = module.weight.dtype
            w_list = [m.weight.to(device, dtype=torch.float32, non_blocking=True) for m in module.experts]
            gates.append(SmileGate(module.experts[0].in_features, w_list, 8).to(device, dtype=original_dtype, non_blocking=True))
            self.gate = nn.ModuleList(gates)

    def forward(self, x: Tensor):        
        routing_weights = []
        for gate in self.gate:
            routing_weight = gate(x)
            routing_weights.append(routing_weight) # (1, 8, 4)
        avg_routing_weights = torch.stack(routing_weights, dim=0).mean(dim=(0, 2))
        return avg_routing_weights
    

class SmileCompressedLinear(nn.Module):
    """
    This module is used to compress a linear layer using SVD decomposition.
    """

    __constants__ = ["in_features", "out_features", "k"]
    in_features: int
    out_features: int
    k: int

    u: nn.Parameter
    svh: nn.Parameter
    bias: Optional[nn.Parameter]

    def __init__(
        self,
        model: nn.Linear,
        k: int,
        svd_cache: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
    ):
        super().__init__()
        self.in_features = model.in_features
        self.out_features = model.out_features
        self.k = k

        if svd_cache is None:
            u, s, v = svd(model.weight)
        else:
            u, s, v = svd_cache
        # if k > 0:
        #     u = u[:, :k]
        #     s = s[:k]
        #     v = v[:, :k]

        self.u = nn.Parameter(u)
        self.svh = nn.Parameter((s * v).T)

        if model.bias is not None:
            self.bias = nn.Parameter(model.bias.data, requires_grad=True)
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        Forward pass of the SmileCompressedLinear module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        x = F.linear(x, self.svh)
        x = F.linear(x, self.u, self.bias)
        return x


class SmileMoELinear(nn.Module):
    # __constants__ = [
    #     "in_features",
    #     "out_features",
    #     "num_experts",
    #     "top_k",
    #     "gate_k",
    #     "k",
    # ]
    # in_features: int
    # out_features: int
    # num_experts: int
    # top_k: int
    # gate_k: int
    # k: int

    @torch.no_grad()
    def __init__(
        self,
        # pretrained_model: nn.Linear,
        expert_models: List[nn.Linear],
        # gate_k: int,
        # k: int,
        top_k: int = 1,
        # full_matrices=True,
        # upscaling_accelerator=None,
        # routing_use_diff=True,
    ):
        super().__init__()
        self.num_experts = len(expert_models)
        self.top_k = top_k
        self.in_features = expert_models[0].in_features
        self.out_features = expert_models[0].out_features

        # construct experts
        experts = expert_models
        self.experts = nn.ModuleList(experts)

    def forward(self, hidden_states: Tensor, router_logits: Tensor):
        input_shape = hidden_states.size()
        hidden_states = hidden_states.view(-1, self.in_features)

        # router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1) # 输入这一步中的routing_weights

        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1 
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = torch.zeros(
            (hidden_states.size(0), self.out_features),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, self.in_features)
            if current_state.numel() == 0:
                continue
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype)
            )
        final_hidden_states = final_hidden_states.reshape(
            *input_shape[:-1], self.out_features
        )
        return final_hidden_states

    @property
    def weight(self):
        """
        Mimic linear layer. Bacause in some cases, user might indicate the device (or dtype of parameters) of the linear layer using `linear_layer.weight.device`
        """
        return self.experts[0].weight

    @property
    def bias(self):
        return self.experts[0].bias

    def __repr__(self):
        return (
            f"SmileMoELinear("
            f"in_features={self.experts[0].in_features}, "
            f"out_features={self.experts[0].out_features}, "
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}, "
            # f"gate_k={self.gate_k}, "
            # f"k={self.k}"
            f")"
        )


class SmileMoENorm(nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        # pretrained_model: nn.Linear,
        expert_models: List[nn.Linear],
        # gate_k: int,
        # k: int,
        top_k: int = 1,
        # full_matrices=True,
        # upscaling_accelerator=None,
        # routing_use_diff=True,
    ):
        super().__init__()
        self.num_experts = len(expert_models)
        self.top_k = top_k
        self.normalized_shape = expert_models[0].normalized_shape

        # construct experts
        experts = expert_models
        self.experts = nn.ModuleList(experts)

    def forward(self, hidden_states: Tensor, router_logits: Tensor):
        input_shape = hidden_states.size()
        # hidden_states = hidden_states.view(-1, self.in_features)

        # router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1) # 输入这一步中的routing_weights

        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1 
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # final_hidden_states = torch.zeros(
        #     (hidden_states.size(0), self.out_features),
        #     dtype=hidden_states.dtype,
        #     device=hidden_states.device,
        # )
        final_hidden_states = torch.zeros_like(hidden_states)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        # expert_mask = torch.nn.functional.one_hot(
        #     selected_experts, num_classes=self.num_experts
        # ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        # for expert_idx in range(self.num_experts):
        for expert_idx, expert_layer in enumerate(self.experts):
            mask = selected_experts == expert_idx
            if not mask.any():
                continue
            current_states = hidden_states[mask]
            final_hidden_states = expert_layer(current_states)
            # expert_layer = self.experts[expert_idx]
            # idx, top_x = torch.where(expert_mask[expert_idx])

            # # Index the correct hidden states and compute the expert hidden state for
            # # the current expert. We need to make sure to multiply the output hidden
            # # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            # current_state = hidden_states[None, top_x].reshape(-1, self.in_features)
            # if current_state.numel() == 0:
            #     continue
            # current_hidden_states = (
            #     expert_layer(current_state) * routing_weights[top_x, idx, None]
            # )

            # # However `index_add_` only support torch tensors for indexing so we'll use
            # # the `top_x` tensor here.
            # final_hidden_states.index_add_(
            #     0, top_x, current_hidden_states.to(hidden_states.dtype)
            # )
        # final_hidden_states = final_hidden_states.reshape(
        #     *input_shape[:-1], self.out_features
        # )
        return final_hidden_states

    @property
    def weight(self):
        """
        Mimic linear layer. Bacause in some cases, user might indicate the device (or dtype of parameters) of the linear layer using `linear_layer.weight.device`
        """
        return self.experts[0].weight

    @property
    def bias(self):
        return self.experts[0].bias

    def __repr__(self):
        return (
            f"SmileMoENorm("
            f"{self.experts[0].normalized_shape}, "
            f"eps={self.experts[0].eps }, "
            f"elementwise_affine={self.experts[0].elementwise_affine}, "
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}, "
            f")"
        )


class SmileVLAUpscalingAlgorithm(SmileUpscalingAlgorithm):
    _linear_layer_cls = (nn.Linear,)
    _parameter_layer_cls = (nn.Parameter,)
    _layer_norm_cls = (nn.LayerNorm,)
    _moe_linear_layer_cls = (SmileMoELinear,)
    _moe_norm_layer_cls = (SmileMoENorm,)


    final_head = [
        "model.mlp_resnet_blocks.23.gating_factor",
        "model.mlp_resnet_blocks.23.ffn.0",
        "model.mlp_resnet_blocks.23.ffn.1",
        "model.mlp_resnet_blocks.23.q_proj_adapter",
        "model.mlp_resnet_blocks.23.k_adapter",
        "model.mlp_resnet_blocks.23.v_adapter",
        "model.mlp_resnet_blocks.23.q_proj_task",
        "model.mlp_resnet_blocks.23.k_task",
        "model.mlp_resnet_blocks.23.v_task",
        "model.mlp_resnet_blocks.23.o_proj",
        "model.layer_norm2",
        "model.fc2",
    ]

    def __init__(
        self, *args,
        exclude_keys: List[str] = [],
        lm_head: str = None,
        **kwargs: Any,
    ):
        self.exclude_keys = exclude_keys
        self.lm_head = lm_head
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool) -> nn.Module:
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)

        # if self.config.model_path is not None and os.path.exists(
        #     self.config.model_path
        # ):
        #     log.info(f"Loading model from {self.config.model_path}")
        #     model = torch.load(self.config.model_path)
        #     print_parameters(model)
        #     return model

        with self.profile("loading expert model"):
            expert_models = [
                m
                for m in tqdm(modelpool.models(), total=len(modelpool.model_names))
            ]

            if self.config.device == "cuda" and torch.cuda.is_available():
                # pretrained_model = pretrained_model.cuda()
                expert_models = [m.cuda() for m in expert_models]

        with self.profile("merge model"):
            self.moe_model = deepcopy(expert_models[1]) # init MoE Model
            model = self.merge(expert_models)

        self.print_profile_summary()
        # if self.config.model_path is not None:
        #     os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
        #     log.info(f"Saving model to {self.config.model_path}")
        #     torch.save(model, self.config.model_path)
        print_parameters(model)
        return model

    def merge(
        self,
        # pretrained_model: nn.Module,
        expert_models: List[nn.Module],
        # in_place: bool = True,
    ) -> nn.Module:
        """
        Merges the pretrained model with the fine-tuned models to create an upscaled model.

        Args:
            pretrained_model (nn.Module): The pretrained model.
            finetuned_models (List[nn.Module]): A list of fine-tuned models.
            in_place (bool): If True, modifies the pretrained model in place. Otherwise, creates a copy.

        Returns:
            nn.Module: The merged model.
        """
        # if in_place:
        #     model = pretrained_model
        # else:
        #     model = deepcopy(pretrained_model)

        self._upscale_submodules(expert_models)
        modules = self._define_gate_modules()
        gate = SmileMoEGate(modules)

        self._change_module_name()
        set_attr(self.moe_model.model, ["gate"], gate)
        return self.moe_model

    def _upscale_norm_layer(
        self,   
        # pretrained_model,
        expert_models,
        # name_list: List[str],
        name: str,
    ):
        # config = self.config

        experts = []
        # for name in tqdm(tuple(name_list),
        #     # tqdm_desc,
        #     leave=False,
        #     dynamic_ncols=True,
        # ):
        name_list = name.split(".")
        module = get_attr(self.moe_model, name_list)
        original_device = get_device(module)
        original_dtype = module.weight.dtype
        module = module.to(self.device, dtype=torch.float32, non_blocking=True)
        experts = [
            get_attr(m, name_list).to(self.device, dtype=torch.float32, non_blocking=True)
            for m in expert_models
        ]
        moe_linear = SmileMoENorm(
            # module,
            experts,
            # gate_k=config.gate_k,
            # k=config.k,
            # top_k=config.top_k,
            # routing_use_diff=self.routing_use_diff,
            # full_matrices=self.full_matrices,
            # upscaling_accelerator=self.upscaling_accelerator,
        )
        # x = torch.randn(1, 896)
        # output = moe_linear(x)
        # print(output.shape)
        moe_linear = moe_linear.to(original_device, dtype=original_dtype, non_blocking=True)
        # return moe_linear
        set_attr(self.moe_model, name_list, moe_linear)
        # remove the original module from fine-tuned models to save memory
        for m in expert_models:
            set_attr(m, name_list, None)

    def _upscale_parameter(
        self,   
        # pretrained_model,
        expert_models,
        # name_list: List[str],
        name: str,
    ):
        # config = self.config

        # for name in tqdm(tuple(name_list),
        #     # tqdm_desc,
        #     leave=False,
        #     dynamic_ncols=True,
        # ):
        name_list = name.split(".")
        module = get_attr(self.moe_model, name_list)
        module = module.to(self.device, dtype=torch.float32, non_blocking=True)
        gating_factors = [get_attr(m, name_list) for m in expert_models]
        gating_factors = nn.Parameter(torch.cat(gating_factors, dim=0))
        # mask = torch.tensor([0, 1, 0, 0], dtype=torch.bool)
        # gating_factor = gating_factors[mask]
        set_attr(self.moe_model, name_list, gating_factors)
        # remove the original module from fine-tuned models to save memory
        for m in expert_models:
            set_attr(m, name_list, None)

    def _upscale_linear_layer(
        self,   
        # pretrained_model,
        expert_models,
        # name_list: List[str],
        name: str,
    ):
        # config = self.config

        experts = []
        # for name in tqdm(tuple(name_list),
        #     # tqdm_desc,
        #     leave=False,
        #     dynamic_ncols=True,
        # ):
        name_list = name.split(".")
        module = get_attr(self.moe_model, name_list)
        original_device = get_device(module)
        original_dtype = module.weight.dtype
        module = module.to(self.device, dtype=torch.float32, non_blocking=True)
        experts = [
            get_attr(m, name_list).to(self.device, dtype=torch.float32, non_blocking=True)
            for m in expert_models
        ]
        moe_linear = SmileMoELinear(
            # module,
            experts,
            # gate_k=config.gate_k,
            # k=config.k,
            # top_k=config.top_k,
            # routing_use_diff=self.routing_use_diff,
            # full_matrices=self.full_matrices,
            # upscaling_accelerator=self.upscaling_accelerator,
        )
        # x = torch.randn(1, 896)
        # output = moe_linear(x)
        # print(output.shape)
        moe_linear = moe_linear.to(original_device, dtype=original_dtype, non_blocking=True)

        set_attr(self.moe_model, name_list, moe_linear)
        # remove the original module from fine-tuned models to save memory
        for m in expert_models:
            set_attr(m, name_list, None)

    def _upscale_submodules(
        self,
        # pretrained_model: nn.Module,
        expert_models: List[nn.Module],
        # tqdm_desc: str = "Upscaling Linear Modules",
    ):
        """
        Upscales the submodules of the pretrained model by merging them with the corresponding submodules from the fine-tuned models.

        Args:
            pretrained_model (nn.Module): The pretrained model.
            finetuned_models (List[nn.Module]): A list of fine-tuned models.
            tqdm_desc (str): Description for the tqdm progress bar.
        """
        # config = self.config
        # self._upscale_linear_layer(
        #     # pretrained_model=pretrained_model,
        #     expert_models=expert_models,
        #     # name_list=self.final_head,
        # )
        for module_name in tqdm(self.final_head,
            # tqdm_desc,
            leave=False,
            dynamic_ncols=True,
        ):
            print( f"Upscaling module: {module_name} ")
            module_attrs = module_name.split(".")
            module = get_attr(self.moe_model, module_attrs)
            if isinstance(module, self._linear_layer_cls):
                # self._upscale_linear_layer(
                #     pretrained_model=pretrained_model,
                #     finetuned_models=finetuned_models,
                #     name=module_name,
                # )
                if module_name == "model.fc2":
                    # print(f"To Do for {module_name}")
                    self._upscale_linear_layer(
                        expert_models=expert_models,
                        name=module_name,
                    )
                else:
                    self._upscale_linear_layer(
                        expert_models=expert_models,
                        name=module_name,
                    )
            elif isinstance(module, self._parameter_layer_cls):
                # print(f"To Do for {module_name}")
                self._upscale_parameter(
                    expert_models=expert_models,
                    name=module_name,
                )
            elif isinstance(module, self._layer_norm_cls):
                # print(f"To Do for {module_name}")
                self._upscale_norm_layer(
                    expert_models=expert_models,
                    name=module_name,
                )
            else:
                raise ValueError(f"Unknown module: {module_name} of type {type(module)} ")

    def _define_gate_modules(self):
        modules = []
        for module_name in self.final_head:
            if module_name == "model.fc2":
                continue
            module_attrs = module_name.split(".")
            module = get_attr(self.moe_model, module_attrs) 
            if isinstance(module, SmileMoELinear):
                modules.append(module)
        return modules
    
    def _change_module_name(self):
        # pass
        class L1RegressionMoEActionHead(self.moe_model.__class__):
            pass
        class MLPMoEResNet(self.moe_model.model.__class__):
            pass
        class MLPResNetBlock_Pro_MoE(self.moe_model.model.mlp_resnet_blocks[23].__class__):
            pass
        self.moe_model.__class__ = L1RegressionMoEActionHead
        self.moe_model.model.__class__ = MLPMoEResNet
        # self.moe_model.__class__.__name__ = "L1RegressionMoEActionHead"
        # self.moe_model.model.__class__.__name__ = "MLPMoEResNet"
        moe_block = self.moe_model.model.mlp_resnet_blocks[23]
        moe_block.__class__ = MLPResNetBlock_Pro_MoE
        del self.moe_model.model.mlp_resnet_blocks[23]
        self.moe_model.model.add_module("mlp_resnet_moe_blocks", moe_block)
        
