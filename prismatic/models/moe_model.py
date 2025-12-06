import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from torch import Tensor, nn

def set_attr(obj, names: List[str], val):
    """
    Sets an attribute of an object recursively.

    Args:
        obj (object): Object to set attribute of.
        names (list): List of attribute names to set recursively.
        val (object): Value to set the attribute to.
    """
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def get_attr(obj, names: List[str]):
    """
    Gets an attribute of an object recursively.

    Args:
        obj (object): Object to get attribute of.
        names (list): List of attribute names to get recursively.

    Returns:
        object: The attribute of the object.
    """
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return get_attr(getattr(obj, names[0]), names[1:])
    
def get_device(obj: Any) -> torch.device:
    """
    Get the device of a given object.

    Args:
        obj: The object whose device is to be determined.

    Returns:
        torch.device: The device of the given object.

    Raises:
        ValueError: If the object type is not supported.
    """
    if isinstance(obj, torch.Tensor):
        return obj.device
    elif isinstance(obj, torch.nn.Module):
        if hasattr(obj, "device"):
            return obj.device
        else:
            return next(iter(obj.parameters())).device
    elif isinstance(obj, torch.device):
        return obj
    else:
        raise ValueError(f"Unsupported object type: {type(obj)}")
    
def _svd(w: Tensor, full_matrices: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Perform Singular Value Decomposition (SVD) on a tensor.

    Args:
        w (Tensor): The input tensor.
        full_matrices (bool): Whether to compute the full-sized U and V matrices.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: The U, S, and V matrices from SVD.
    """
    u, s, vh = torch.linalg.svd(
        w, full_matrices=full_matrices, driver="gesvd" if w.is_cuda else None
    )
    v = vh.T
    return u, s, v

def svd(
    w: Tensor,
    full_matrices: bool = True,
    accelerator: Optional[Union[torch.device, str]] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Perform SVD on a tensor, optionally using a specified accelerator.

    Args:
        w (Tensor): The input tensor.
        full_matrices (bool): Whether to compute the full-sized U and V matrices.
        accelerator (Optional[Union[torch.device, str]]): The device to perform the computation on.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: The U, S, and V matrices from SVD.
    """
    if accelerator is None:
        return _svd(w, full_matrices=full_matrices)
    original_device = w.device
    w = w.to(accelerator)
    u, s, v = _svd(w)
    return u.to(original_device), s.to(original_device), v.to(original_device)

class SmileGate(nn.Module):
    def __init__(
        self,
        input_features: int,
        num_experts: int,
        k: int,
    ):
        super().__init__()
        self.input_features = input_features
        self.num_experts = num_experts
        self.k = k

        self.routers = nn.ParameterList([
            nn.Parameter(torch.zeros(k, input_features))
            for _ in range(num_experts)
        ])

    def forward(self, x: Tensor, expert_idx):
        routing_weights = F.linear(x, self.routers[expert_idx]) # (bs, x_dim, k)
        routing_weights = routing_weights.norm(p=2, dim=2) # (bs, x_dim)
        return routing_weights
    
    def __repr__(self):
        router_shapes = [tuple(r.shape) for r in self.routers]
        return (
            f"SmileGate("
            f"routers={router_shapes}, "
            f"input_features={self.input_features}, "
            f"num_experts={self.num_experts}, "
            f"k={self.k}"
            f")"
        )

class SmileMoEGate(nn.Module):
    def __init__(
        self,
        modules: nn.Module,
        num_experts: int,
        k: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        gates = []
        for module in modules:
            device = get_device(module)
            original_dtype = module.weight.dtype
            gates.append(SmileGate(module.in_features, num_experts, k).to(device, dtype=original_dtype, non_blocking=True))
            self.gate = nn.ModuleList(gates)
    
    def forward(self, h_a, h_t, expert_idx: int):        
        routing_weights = []
        routing_weights.append(self.gate[0](h_a, expert_idx).mean(dim=1))
        routing_weights.append(self.gate[1](h_t, expert_idx).mean(dim=1))
        avg_routing_weights = torch.stack(routing_weights, dim=0).mean(dim=(0))
        return avg_routing_weights
    
class SmileMoENorm(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        
        # construct experts
        experts = [nn.LayerNorm(input_dim) for _ in range(num_experts)]
        self.experts = nn.ModuleList(experts)
        self.normalized_shape = self.experts[0].normalized_shape

    def forward(self, hidden_states: Tensor, expert_idx: int):
        expert_layer = self.experts[expert_idx]
        output = expert_layer(hidden_states)
        return output

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
            f"eps={self.experts[0].eps}, "
            f"elementwise_affine={self.experts[0].elementwise_affine}, "
            f"num_experts={self.num_experts}"
            f")"
        )

class SmileMoELinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int,
    ):
        super().__init__()
        self.num_experts = num_experts

        self.in_features = in_features
        self.out_features = out_features

        # construct experts
        experts = [nn.Linear(in_features, out_features) for _ in range(num_experts)]
        self.experts = nn.ModuleList(experts)

    def forward(self, hidden_states: Tensor, expert_idx: int):
        expert_layer = self.experts[expert_idx]
        output = expert_layer(hidden_states)
        return output

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
            f"num_experts={self.num_experts}"
            f")"
        )