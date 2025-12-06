"""
action_heads.py

Implementations of action heads.
"""

import math
import torch
import torch.nn as nn
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from prismatic.models.moe_model import SmileMoENorm, SmileMoELinear, SmileMoEGate, get_attr

def learnable_random_perturbations(seq_len, dim, device, dtype):
    random_perturbations = nn.Parameter(torch.zeros(seq_len, dim, device=device, dtype=dtype))
    nn.init.normal_(random_perturbations, mean=0.0, std=0.02)
    return random_perturbations



class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""
    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_task_tokens=512,
    ):
        super().__init__()
        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.model = MLPResNet(
            num_blocks=24, 
            input_dim=input_dim*ACTION_DIM, 
            hidden_dim=hidden_dim, 
            output_dim=action_dim,
        )

    def predict_action(
            self, 
            actions_hidden_states, 
            proprio=None, 
            proprio_projector=None,
            phase="Inference"
            ):
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device

        proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)  # (bsz, proprio_dim)
        proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
        proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)

        task_hidden_states = actions_hidden_states[:, :, :self.num_task_tokens, :]
        actions_hidden_states = actions_hidden_states[:, :, self.num_task_tokens:, :]

        cond_actions_hidden_states = torch.zeros(
            (batch_size, self.action_dim * NUM_ACTIONS_CHUNK, self.hidden_dim),
            device=device, dtype=actions_hidden_states.dtype
        ).detach()  

        rearranged_actions_hidden_states = cond_actions_hidden_states.reshape(
            batch_size, NUM_ACTIONS_CHUNK, -1
        )  # (batch, chunk_len, action_dim * hidden_dim)

        if phase == "Training":
            batch_size, seq_len, dim = rearranged_actions_hidden_states.shape
            random_perturbations = learnable_random_perturbations(seq_len, dim, device=rearranged_actions_hidden_states.device, dtype=rearranged_actions_hidden_states.dtype) 
            rearranged_actions_hidden_states = (rearranged_actions_hidden_states + random_perturbations) # (1, seq_len, dim)

        action = self.model(
            rearranged_actions_hidden_states, # [1, 8, 7*896=6272]
            h_a=actions_hidden_states, # [1, 25, 64, 896]
            p=proprio_features, # [1, 1, 896]
            h_t=task_hidden_states # [1, len(token), 512, 896]
        )

        return action
    


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(
            self, 
            num_blocks, 
            input_dim, 
            hidden_dim, 
            output_dim,
        ):
        
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()

        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock_Pro(dim=hidden_dim))
                
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x, h_a=None, h_t=None, p= None):
 
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim) (1, 8, 6272)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim) (1, 8, 896)
        x = self.relu(x)  # shape: (batch_size, hidden_dim) (1, 8, 896)
        for i, block in enumerate(self.mlp_resnet_blocks):
            x = block(x, h_t = h_t[:,i+1,:], h_a = h_a[:,i+1,:], p=p)  # shape: (batch_size, hidden_dim) (1, 8, 896)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim) (1, 8, 896)
        x = self.fc2(x)  # shape: (batch_size, output_dim) (1, 8, 7)
        return x   



def apply_rope(q, k, cos, sin):
    """
    RoPE:
    q, k: (B, H, T, D)   # D must be an even number
    cos/sin: (T, D)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)


    def rotate_half(x):
        # Swap even and odd dimensions and flip the signs
        x1 = x[..., ::2]   # Even subdimension
        x2 = x[..., 1::2]  # odd subdimension

        return torch.stack((-x2, x1), dim=-1).reshape_as(x)


    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot



class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        """
        dim = head_dim
        """
        super().__init__()
        assert dim % 2 == 0, "RoPE head_dim must be an even number"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)            # (T, dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)



class MLPResNetBlock(nn.Module):
    """
    One residual MLP block with cross-attention conditioning.

    This block applies multi-head attention over:
      - token features (self-attention),
      - task-related hidden states (h_t),
      - action/proprioception-related hidden states (h_a, p).
    The outputs are combined via a gating mechanism, projected back to the
    hidden dimension, and passed through a small feedforward sub-network with
    residual connection.

    Args:
        dim (int): Dimensionality of the hidden features. Must be divisible by num_heads.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
        h_t (torch.Tensor, optional): Task-related hidden states of shape
                                      (batch_size, K, hidden_dim).
        h_a (torch.Tensor, optional): Action-related hidden states of shape
                                      (batch_size, 1, hidden_dim).
        p (torch.Tensor, optional): Additional conditioning features
                                    (e.g., proprioception), shape (batch_size, 1, hidden_dim).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Main feedforward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        self.num_heads = 8
        self.head_dim = dim // self.num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.gating_factor = nn.Parameter(torch.zeros(1))



    def forward(self, x, h_t=None, h_a=None, p=None):
        """
        x: (batch_size, seq_len, hidden_dim)
        h, t, p: (batch_size, 1, hidden_dim) or None
        """

        g = self.gating_factor
        ratio_g = nn.Tanh()(g)

        conditions = []
        if h_a is not None:
            conditions.append(h_a)
        if p is not None:
            conditions.append(p)

        h = torch.cat(conditions, dim=1)  # (batch_size, cond_len, hidden_dim)

        B = x.size(0)
        T = x.size(1)
        C = x.size(2)
        K_t = h.size(1)
        K = h_t.size(1)

        task_k = h
        task_v = h

        adapter_k = h_t
        adapter_v = h_t

        q_1 = self.q_proj(x) # (B, T, C)
        k_tokens = self.k_proj(x)             # (B, T, C)
        v_tokens = self.v_proj(x)             # (B, T, C)
        k_task = self.k_proj(task_k)    # (B, K, C)
        v_task = self.v_proj(task_v)    # (B, K, C)

        k_adapter = self.k_proj(adapter_k)    # (B, K, C)
        v_adapter = self.v_proj(adapter_v)    # (B, K, C)

        # (B, seq_len, C) -> (B, num_heads, seq_len, head_dim)
        q_1 = q_1.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        k_tokens = k_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_tokens = v_tokens.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_task = k_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)
        v_task = v_task.view(B, K_t, self.num_heads, self.head_dim).transpose(1, 2)

        k_adapter = k_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)
        v_adapter = v_adapter.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores_tokens = torch.matmul(q_1, k_tokens.transpose(-2, -1)) # (B, H, T, T)
        attn_scores_task = torch.matmul(q_1, k_task.transpose(-2, -1)) * 1 # (B, H, T, K)
        attn_scores_adapter = torch.matmul(q_1, k_adapter.transpose(-2, -1)) * ratio_g # (B, H, T, K)

        attn_scores = torch.cat([attn_scores_tokens, attn_scores_task, attn_scores_adapter], dim=-1) # (B, H, T, T+K)
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1) # (B, H, T, T+K)

        v_combined = torch.cat([v_tokens, v_task, v_adapter], dim=2) # (B, H, T+K, head_dim)
        output = torch.matmul(attn_weights, v_combined) # (B, H, T, head_dim)

        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        x = self.ffn(output + x) 

        return x



class MLPResNetBlock_Pro(nn.Module):
    """One MLP ResNet block with separate projections for self, adapter, task + RoPE."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        # Adapter cross-attention: K, V
        self.q_proj_adapter = nn.Linear(dim, dim)
        self.k_adapter = nn.Linear(dim, dim)
        self.v_adapter = nn.Linear(dim, dim)

        # Task cross-attention: K, V
        self.q_proj_task = nn.Linear(dim, dim)
        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)

        self.o_proj = nn.Linear(dim, dim)

        # gating
        self.gating_factor = nn.Parameter(torch.zeros(1))

        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)

    def forward(self, x, h_a=None, h_t=None, p=None):
        """
        h_a: adapter tokens
        h_t: task tokens
        p:   possible conditioning vector
        """
        g = self.gating_factor
        ratio_g = torch.sigmoid(g)

        conditions = []
        if h_a is not None:
            conditions.append(h_a)
        if p is not None:
            conditions.append(p)

        if len(conditions) > 0:
            h_adapter = torch.cat(conditions, dim=1)  # (batch_size, cond_len, hidden_dim)
        else:
            # if no adapter/p provided, create zero-length tensor to avoid errors later
            # but we will guard usages below
            h_adapter = torch.zeros(x.size(0), 0, x.size(2), device=x.device, dtype=x.dtype)

        h_task = h_t if h_t is not None else torch.zeros(x.size(0), 0, x.size(2), device=x.device, dtype=x.dtype)

        B, T, C = x.shape
        K_a = h_adapter.size(1)
        K_t = h_task.size(1) 

        # adapter tokens
        q_adapter = self.q_proj_adapter(x)
        k_adapter = self.k_adapter(h_adapter)
        v_adapter = self.v_adapter(h_adapter)

        # task tokens
        q_task = self.q_proj_task(x)
        k_task = self.k_task(h_task)
        v_task = self.v_task(h_task)


        # reshape -> multi-head
        def reshape_heads(t, B, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # q_self = reshape_heads(q_self, B, T)
        q_adapter = reshape_heads(q_adapter, B, T)
        q_task = reshape_heads(q_task, B, T)

        k_adapter, v_adapter = reshape_heads(k_adapter, B, K_a), reshape_heads(v_adapter, B, K_a)
        k_task, v_task = reshape_heads(k_task, B, K_t), reshape_heads(v_task, B, K_t)

        # RoPE
        cos_main, sin_main = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        dummy_k = torch.zeros((B, self.num_heads, T, self.head_dim), device=q_adapter.device, dtype=q_adapter.dtype)
        q_adapter, _ = apply_rope(q_adapter, dummy_k, cos_main, sin_main)
        q_task, _ = apply_rope(q_task, dummy_k, cos_main, sin_main)

        cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
        _, k_adapter = apply_rope(k_adapter, k_adapter, cos_a, sin_a)
        cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
        _, k_task = apply_rope(k_task, k_task, cos_t, sin_t)

        # attention scores (each is (B, heads, T, Lk))
        attn_scores_adapter = torch.matmul(q_adapter, k_adapter.transpose(-2, -1))  # (B, heads, T, K_a)
        attn_scores_task = torch.matmul(q_task, k_task.transpose(-2, -1)) * ratio_g  # (B, heads, T, K_t)

        # concat along key-length dimension
        attn_scores = torch.cat([attn_scores_adapter, attn_scores_task], dim=-1) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # combine V: concat along key-length dim
        v_combined = torch.cat([v_adapter, v_task], dim=2) 

        output = torch.matmul(attn_weights, v_combined)  # (B, heads, T, head_dim)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output)

        # residual + FFN
        x = self.ffn(output + x)
        return x
    


class L1RegressionMoEActionHead(nn.Module):

    def __init__(
        self,
        input_dim=4096,
        hidden_dim=4096,
        action_dim=7,
        num_task_tokens=512,
        num_experts=4,
        k_gate=8,
        action_head_layer_num=1,
        expert_idx=None,
        use_router=False
    ):
        super().__init__()
        self.num_task_tokens = num_task_tokens
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.expert_idx = expert_idx
        self.model = MLPMoEResNet(
            num_blocks=24-action_head_layer_num, 
            input_dim=input_dim*ACTION_DIM, 
            hidden_dim=hidden_dim, 
            output_dim=action_dim,
            num_experts=num_experts,
            k_gate=k_gate,
            action_head_layer_num=action_head_layer_num,
            use_router=use_router,
        )

    def predict_action(
            self, 
            actions_hidden_states, 
            proprio=None, 
            proprio_projector=None,
            phase="Inference"
            ):
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device

        proprio = proprio.reshape(batch_size, -1).to(torch.bfloat16)  # (bsz, proprio_dim)
        proprio_features = proprio_projector(proprio)  # (bsz, llm_dim)
        proprio_features = proprio_features.unsqueeze(dim=1)  # (bsz, 1, llm_dim)

        task_hidden_states = actions_hidden_states[:, :, :self.num_task_tokens, :]
        actions_hidden_states = actions_hidden_states[:, :, self.num_task_tokens:, :]

        cond_actions_hidden_states = torch.zeros(
            (batch_size, self.action_dim * NUM_ACTIONS_CHUNK, self.hidden_dim),
            device=device, dtype=actions_hidden_states.dtype
        ).detach()  

        rearranged_actions_hidden_states = cond_actions_hidden_states.reshape(
            batch_size, NUM_ACTIONS_CHUNK, -1
        )  # (batch, chunk_len, action_dim * hidden_dim)

        if phase == "Training":
            batch_size, seq_len, dim = rearranged_actions_hidden_states.shape
            random_perturbations = learnable_random_perturbations(seq_len, dim, device=rearranged_actions_hidden_states.device, dtype=rearranged_actions_hidden_states.dtype) 
            rearranged_actions_hidden_states = (rearranged_actions_hidden_states + random_perturbations) # (1, seq_len, dim)

        action = self.model(
            rearranged_actions_hidden_states,
            h_a=actions_hidden_states,
            p=proprio_features,
            h_t=task_hidden_states,
            expert_idx=self.expert_idx
        )

        return action
    


class MLPMoEResNet(nn.Module):

    def __init__(
            self, 
            num_blocks, 
            input_dim, 
            hidden_dim, 
            output_dim,
            num_experts=4,
            k_gate=8,
            action_head_layer_num=1,
            use_router=False,
        ):
        
        super().__init__()
        self.use_router = use_router

        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        self.mlp_resnet_moe_blocks = nn.ModuleList()
        self.action_head_layer_num = action_head_layer_num

        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock_Pro(dim=hidden_dim))
                
        for _ in range(action_head_layer_num):
            self.mlp_resnet_moe_blocks.append(MLPResNetBlock_Pro_MoE(dim=hidden_dim, num_experts=num_experts))

        self.layer_norm2 = SmileMoENorm(hidden_dim, num_experts)
        self.fc2 = SmileMoELinear(hidden_dim, output_dim, num_experts)

        modules = self._define_gate_modules()
        self.gate = SmileMoEGate(modules, num_experts, k=k_gate)
        self.router_layer_idx = nn.Parameter(torch.tensor(-1, dtype=torch.long), requires_grad=False)

    def _define_gate_modules(self):
        module_names = [f'mlp_resnet_blocks.{24-self.action_head_layer_num-1}.v_adapter',
                        f'mlp_resnet_blocks.{24-self.action_head_layer_num-1}.v_task']
        modules = []

        for module_name in module_names:
            module_attrs = module_name.split(".")
            module = get_attr(self, module_attrs) 
            modules.append(module)
        return modules
    
    def forward(self, x, expert_idx, h_a=None, h_t=None, p=None):

        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for i, block in enumerate(self.mlp_resnet_blocks):
            x = block(x, h_t = h_t[:,i+1,:], h_a = h_a[:,i+1,:], p=p)  # shape: (batch_size, hidden_dim)
        
        # # calculate gate
        if self.use_router:
            assert self.router_layer_idx.item() == i, f"Expected gate to be applied at layer {self.router_layer_idx.item()}, but got layer {i}, the checkpoint does not match the current model configuration."
            router_logits = self.gate(h_t = h_t[:,i+2,:], h_a = h_a[:,i+2,:], expert_idx=expert_idx)
            return router_logits

        for j, block in enumerate(self.mlp_resnet_moe_blocks):
            x = block(x, expert_idx, h_t = h_t[:,i+j+2,:], h_a = h_a[:,i+j+2,:], p=p)  # shape: (batch_size, hidden_dim)

        x = self.layer_norm2(x, expert_idx)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x, expert_idx)  # shape: (batch_size, output_dim)
        return x



class MLPResNetBlock_Pro_MoE(nn.Module):
    """One MLP ResNet block with separate projections for self, adapter, task + RoPE."""

    def __init__(self, dim, num_experts=4, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.ffn = nn.Sequential(
            SmileMoENorm(dim, num_experts),
            SmileMoELinear(dim, dim, num_experts),
            nn.ReLU(),
        )

        # Adapter cross-attention: K, V
        self.q_proj_adapter = SmileMoELinear(dim, dim, num_experts)
        self.k_adapter = SmileMoELinear(dim, dim, num_experts)
        self.v_adapter = SmileMoELinear(dim, dim, num_experts)

        # Task cross-attention: K, V
        self.q_proj_task = SmileMoELinear(dim, dim, num_experts)
        self.k_task = SmileMoELinear(dim, dim, num_experts)
        self.v_task = SmileMoELinear(dim, dim, num_experts)

        self.o_proj = SmileMoELinear(dim, dim, num_experts)

        # gating
        self.gating_factor = nn.Parameter(torch.zeros(num_experts))

        # RoPE
        self.rope = RotaryPositionEmbedding(self.head_dim)

    def forward(self, x, expert_idx, h_a=None, h_t=None, p=None):
        """
        h_a: adapter tokens
        h_t: task tokens
        p:   possible conditioning vector
        """
        g = self.gating_factor[expert_idx]
        
        ratio_g = torch.sigmoid(g)

        conditions = []
        if h_a is not None:
            conditions.append(h_a)
        if p is not None:
            conditions.append(p)

        if len(conditions) > 0:
            h_adapter = torch.cat(conditions, dim=1)  # (batch_size, cond_len, hidden_dim)
        else:
            h_adapter = torch.zeros(x.size(0), 0, x.size(2), device=x.device, dtype=x.dtype)

        h_task = h_t if h_t is not None else torch.zeros(x.size(0), 0, x.size(2), device=x.device, dtype=x.dtype)

        B, T, C = x.shape
        K_a = h_adapter.size(1)
        K_t = h_task.size(1) 

        # adapter tokens
        q_adapter = self.q_proj_adapter(x, expert_idx)
        k_adapter = self.k_adapter(h_adapter, expert_idx)
        v_adapter = self.v_adapter(h_adapter, expert_idx)

        # task tokens
        q_task = self.q_proj_task(x, expert_idx)
        k_task = self.k_task(h_task, expert_idx)
        v_task = self.v_task(h_task, expert_idx)


        # reshape -> multi-head
        def reshape_heads(t, B, L):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # q_self = reshape_heads(q_self, B, T)
        q_adapter = reshape_heads(q_adapter, B, T)
        q_task = reshape_heads(q_task, B, T)

        # k_tokens, v_tokens = reshape_heads(k_tokens, B, T), reshape_heads(v_tokens, B, T)
        k_adapter, v_adapter = reshape_heads(k_adapter, B, K_a), reshape_heads(v_adapter, B, K_a)
        k_task, v_task = reshape_heads(k_task, B, K_t), reshape_heads(v_task, B, K_t)

        # RoPE
        cos_main, sin_main = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        dummy_k = torch.zeros((B, self.num_heads, T, self.head_dim), device=q_adapter.device, dtype=q_adapter.dtype)
        q_adapter, _ = apply_rope(q_adapter, dummy_k, cos_main, sin_main)
        q_task, _ = apply_rope(q_task, dummy_k, cos_main, sin_main)

        cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
        _, k_adapter = apply_rope(k_adapter, k_adapter, cos_a, sin_a)
        cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
        _, k_task = apply_rope(k_task, k_task, cos_t, sin_t)

        # attention scores (each is (B, heads, T, Lk))
        attn_scores_adapter = torch.matmul(q_adapter, k_adapter.transpose(-2, -1))  # (B, heads, T, K_a)
        attn_scores_task = torch.matmul(q_task, k_task.transpose(-2, -1)) * ratio_g  # (B, heads, T, K_t) ######### important #* ratio_g #########

        # concat along key-length dimension
        attn_scores = torch.cat([attn_scores_adapter, attn_scores_task], dim=-1) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # combine V: concat along key-length dim
        v_combined = torch.cat([v_adapter, v_task], dim=2) 

        output = torch.matmul(attn_weights, v_combined)  # (B, heads, T, head_dim)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(output, expert_idx)

        # residual + FFN
        x = output + x
        for block in self.ffn:
            if isinstance(block, (SmileMoENorm, SmileMoELinear)):
                x = block(x, expert_idx)
            else:
                x = block(x)

        return x
