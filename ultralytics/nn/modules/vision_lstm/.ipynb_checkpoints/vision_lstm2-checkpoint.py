# This file is licensed under Apache-2.0
# Copyright (c) NXAI GmbH and its affiliates 2024
# Benedikt Alkin, Maximilian Beck, Korbinian Pöppel
import math
import warnings
from enum import Enum

import einops
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import StochasticDepth
import torch.utils.checkpoint as cp
from typing import Optional, Tuple

from .vision_lstm_util import (
    interpolate_sincos, to_ntuple, VitPatchEmbed, VitPosEmbed2d,
    DropPath, SequenceConv2d, SequenceConv3d
)

# ---------------------------------------------------------------------
# Utilities & helpers (canonical, single definitions)
# ---------------------------------------------------------------------

class SequenceTraversal(Enum):
    ROWWISE_FROM_TOP_LEFT = "rowwise_from_top_left"
    ROWWISE_FROM_BOT_RIGHT = "rowwise_from_bot_right"


def bias_linspace_init_(param: torch.Tensor, start: float = 3.4, end: float = 6.0) -> torch.Tensor:
    """Linearly spaced bias init across dimensions."""
    assert param.dim() == 1, f"param must be 1-dimensional (typically a bias), got {param.dim()}"
    n_dims = param.shape[0]
    init_vals = torch.linspace(start, end, n_dims, device=param.device, dtype=param.dtype)
    with torch.no_grad():
        param.copy_(init_vals)
    return param


def small_init_(param: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Transformers without Tears (Nguyen & Salazar, 2019) style init.
    """
    std = math.sqrt(2 / (5 * dim))
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def wang_init_(param: torch.Tensor, dim: int, num_blocks: int):
    """
    Heuristic scaled-down init for residual projections.
    """
    std = 2 / num_blocks / math.sqrt(dim)
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def round_up_to_next_multiple_of(x: int, multiple_of: int) -> int:
    """Rounds up x to the next multiple of multiple_of."""
    return int(((x + multiple_of - 1) // multiple_of) * multiple_of)


# ---------------------------------------------------------------------
# Reference parallel stabilized mLSTM (kept for completeness)
# ---------------------------------------------------------------------
def parallel_stabilized_simple(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        igate_preact: torch.Tensor,
        fgate_preact: torch.Tensor,
        lower_triangular_matrix: torch.Tensor = None,
        stabilize_rowwise: bool = True,
        eps: float = 1e-6,
) -> torch.Tensor:
    """
    Parallel mLSTM cell with stabilized exponentials.

    Args:
        queries: (B, NH, S, DH)
        keys: (B, NH, S, DH)
        values: (B, NH, S, DH)
        igate_preact: (B, NH, S, 1)
        fgate_preact: (B, NH, S, 1)
        lower_triangular_matrix: (S, S) bool
    """
    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device

    # forget gate matrix
    log_fgates = torch.nn.functional.logsigmoid(fgate_preact)  # (B, NH, S, 1)
    if lower_triangular_matrix is None or S < (lower_triangular_matrix.size(-1) if lower_triangular_matrix is not None else 0):
        ltr = torch.tril(torch.ones((S, S), dtype=torch.bool, device=_device))
    else:
        ltr = lower_triangular_matrix
    assert ltr.dtype == torch.bool, f"lower_triangular_matrix must be bool, got {ltr.dtype}"

    log_fgates_cumsum = torch.cat(
        [torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device),
         torch.cumsum(log_fgates, dim=-2)], dim=-2
    )  # (B, NH, S+1, 1)

    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(1, 1, 1, S + 1)  # (B, NH, S+1, S+1)
    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(-2, -1)
    log_fg_matrix = torch.where(ltr, _log_fg_matrix[:, :, 1:, 1:], -float("inf"))  # (B, NH, S, S)

    log_D_matrix = log_fg_matrix + igate_preact.transpose(-2, -1)  # (B, NH, S, S)
    if stabilize_rowwise:
        max_log_D, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)  # (B, NH, S, 1)
    else:
        max_log_D = torch.max(log_D_matrix.view(B, NH, -1), dim=-1, keepdim=True)[0].unsqueeze(-1)
    log_D_matrix_stabilized = log_D_matrix - max_log_D
    D_matrix = torch.exp(log_D_matrix_stabilized)

    keys_scaled = keys / math.sqrt(DH)

    qk_matrix = queries @ keys_scaled.transpose(-2, -1)  # (B, NH, S, S)
    C_matrix = qk_matrix * D_matrix
    normalizer = torch.maximum(C_matrix.sum(dim=-1, keepdim=True).abs(), torch.exp(-max_log_D))  # (B, NH, S, 1)
    C_matrix_normalized = C_matrix / (normalizer + eps)

    h_tilde_state = C_matrix_normalized @ values  # (B, NH, S, DH)
    return h_tilde_state


# ---------------------------------------------------------------------
# FeedForward
# ---------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(
        self,
        embedding_dim,
        ffn_proj_factor=2.6667,
        ffn_round_up_to_multiple_of=64,
        use_bias=False,
        weight_mode="fused",
        num_blocks=15,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ffn_proj_factor = ffn_proj_factor
        self.ffn_round_up_to_multiple_of = ffn_round_up_to_multiple_of
        self.use_bias = use_bias
        self.weight_mode = weight_mode
        self.num_blocks = num_blocks

        self.up_proj_dim = round_up_to_next_multiple_of(
            embedding_dim * ffn_proj_factor, ffn_round_up_to_multiple_of,
        )

        if self.weight_mode == "single":
            self.proj_up_gate = nn.Linear(embedding_dim, self.up_proj_dim, bias=use_bias)
            self.proj_up = nn.Linear(embedding_dim, self.up_proj_dim, bias=use_bias)
        elif self.weight_mode == "fused":
            self.proj_up_gate_z = nn.Linear(embedding_dim, 2 * self.up_proj_dim, bias=use_bias)
        else:
            raise ValueError(f"Unknown weight_mode: {self.weight_mode}")

        self.proj_down = nn.Linear(self.up_proj_dim, embedding_dim, bias=use_bias)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_mode == "single":
            x = self.act_fn(self.proj_up_gate(x)) * self.proj_up(x)
        else:  # fused
            x = self.proj_up_gate_z(x)
            gate, z = torch.tensor_split(x, (self.up_proj_dim,), dim=-1)
            x = self.act_fn(gate) * z

        y = self.proj_down(x)
        return y

    def reset_parameters(self):
        if self.weight_mode == "fused":
            small_init_(self.proj_up_gate_z.weight, dim=self.embedding_dim)
            if self.proj_up_gate_z.bias is not None:
                nn.init.zeros_(self.proj_up_gate_z.bias)
        else:
            small_init_(self.proj_up_gate.weight, dim=self.embedding_dim)
            small_init_(self.proj_up.weight, dim=self.embedding_dim)
            if self.proj_up_gate.bias is not None:
                nn.init.zeros_(self.proj_up_gate.bias)
            if self.proj_up.bias is not None:
                nn.init.zeros_(self.proj_up.bias)

        wang_init_(self.proj_down.weight, dim=self.embedding_dim, num_blocks=self.num_blocks or 1)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)


# ---------------------------------------------------------------------
# LayerNorm variants
# ---------------------------------------------------------------------
class LayerNorm(nn.Module):
    """LayerNorm with optional no-bias and residual-weight behavior."""
    def __init__(
            self,
            ndim: int = -1,
            weight: bool = True,
            bias: bool = False,
            eps: float = 1e-5,
            residual_weight: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
        self.residual_weight = residual_weight
        self.ndim = ndim
        self.reset_parameters()

    @property
    def weight_proxy(self) -> torch.Tensor:
        if self.weight is None:
            return None
        if self.residual_weight:
            return 1.0 + self.weight
        else:
            return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, normalized_shape=(self.ndim,), weight=self.weight_proxy, bias=self.bias, eps=self.eps,
        )

    def reset_parameters(self):
        if self.weight_proxy is not None:
            if self.residual_weight:
                nn.init.zeros_(self.weight)
            else:
                nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MultiHeadLayerNorm(LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, "Input must be 4D tensor (B, NH, S, DH)"
        B, NH, S, DH = x.shape
        gn_in_1 = x.transpose(1, 2)           # (B, S, NH, DH)
        gn_in_2 = gn_in_1.reshape(B * S, NH * DH)
        out = F.group_norm(gn_in_2, num_groups=NH, weight=self.weight_proxy, bias=self.bias, eps=self.eps)
        out = out.view(B, S, NH, DH).transpose(1, 2)
        return out
    
# The new, fully compatible MultiHeadRMSNorm
class MultiHeadRMSNorm(LayerNorm):
    """
    Identical interface to MultiHeadLayerNorm, but uses RMSNorm logic.
    Inherits __init__, properties, and reset_parameters from LayerNorm.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, "Input must be 4D tensor (B, NH, S, DH)"
        # Get head count and dimension from the input tensor's shape
        _, NH, _, DH = x.shape

        # 1. Perform RMS normalization across the last dimension (DH)
        rrms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        norm_x = x * rrms

        # 2. Apply the gain, reshaping it to match the input dimensions
        if self.weight_proxy is None:
            return norm_x

        # self.weight_proxy has shape (ndim,), which is (NH * DH,)
        # Reshape to (1, NH, 1, DH) for broadcasting against (B, NH, S, DH)
        gain_reshaped = self.weight_proxy.view(1, NH, 1, DH)

        return norm_x * gain_reshaped


class LinearHeadwiseExpand(nn.Module):
    """
    This is a structured projection layer that projects the input to a higher dimension.
    It only allows integer up-projection factors, i.e. the output dimension is a multiple of the input dimension.
    """

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads

        dim_per_head = dim // num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, dim_per_head, dim_per_head))
        if bias:
            self.bias = nn.Parameter(torch.empty(dim))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, mean=0.0, std=math.sqrt(2 / 5 / self.weight.shape[-1]))
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "... (nh d) -> ... nh d", nh=self.num_heads)
        x = einops.einsum(
            x,
            self.weight,
            "... nh d, nh out_d d -> ... nh out_d",
        )
        x = einops.rearrange(x, "... nh out_d -> ... (nh out_d)")
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return (
            f"dim={self.dim}, "
            f"num_heads={self.num_heads}, "
            f"bias={self.bias is not None}, "
        )


class CausalConv1d(nn.Module):
    """Causal depthwise Conv1d over (B,T,F)."""
    def __init__(self, dim, kernel_size=4, bias=True):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=self.pad, groups=dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "b l d -> b d l")
        x = self.conv(x)
        x = x[:, :, :-self.pad]
        x = einops.rearrange(x, "b d l -> b l d")
        return x


# ---------------------------------------------------------------------
# MatrixLSTMCell (production path with fused if-gate + backend)
# ---------------------------------------------------------------------
from mlstm_kernels.torch.backend_module import mLSTMBackendConfig, mLSTMBackend

class MatrixLSTMCell(nn.Module):
    def __init__(self, dim, num_heads, norm_bias=True, eps=1e-6, chunk_size=16,
                 use_autocast=True, autocast_dtype=torch.bfloat16, gate_soft_cap=15.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_autocast = use_autocast
        self.autocast_dtype = autocast_dtype
        self.gate_soft_cap = gate_soft_cap
        self.chunk_size = int(chunk_size)

        # Fused ifgate projection: produces [i, f] for each head
        self.ifgate = nn.Linear(3 * dim, 2 * num_heads)

        # Normalization across heads (group-norm based LN)
        self.outnorm = MultiHeadRMSNorm(ndim=dim, weight=True, bias=norm_bias, eps=1e-6)

        # Backend configs (CPU/GPU, train/infer)
        self.cpu_backend_config_infer = mLSTMBackendConfig(
            chunkwise_kernel="chunkwise--native_autograd",
            sequence_kernel="native_sequence__native",
            step_kernel="native",
            chunk_size=self.chunk_size,
            autocast_kernel_dtype="bfloat16",
            return_last_states=False,
            mode="inference",
            eps=5e-5
        )
        self.cpu_backend_infer = mLSTMBackend(config=self.cpu_backend_config_infer)

        self.gpu_backend_config_infer = mLSTMBackendConfig(
            chunkwise_kernel="chunkwise--triton_xl_chunk_siging",
            sequence_kernel="native_sequence__triton",
            step_kernel="triton",
            chunk_size=self.chunk_size,
            autocast_kernel_dtype="bfloat16",
            return_last_states=False,
            mode="inference",
            eps=5e-5
        )
        self.gpu_backend_infer = mLSTMBackend(config=self.gpu_backend_config_infer)

        self.cpu_backend_config = mLSTMBackendConfig(
            chunkwise_kernel="chunkwise--native_autograd",
            sequence_kernel="native_sequence__native",
            step_kernel="native",
            chunk_size=self.chunk_size,
            autocast_kernel_dtype="bfloat16",
            return_last_states=False,
            mode="train_with_padding",
            eps=5e-5
        )
        self.cpu_backend = mLSTMBackend(config=self.cpu_backend_config)

        self.gpu_backend_config = mLSTMBackendConfig(
            chunkwise_kernel="chunkwise--triton_xl_chunk_siging",
            sequence_kernel="native_sequence__triton",
            step_kernel="triton",
            chunk_size=self.chunk_size,
            autocast_kernel_dtype="bfloat16",
            return_last_states=False,
            mode="train_with_padding",
            eps=5e-5
        )
        self.gpu_backend = mLSTMBackend(config=self.gpu_backend_config)

        self.reset_parameters()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q,k,v: (B, S, H)
        if not (q.device == k.device == v.device):
            raise ValueError("All input tensors (q, k, v) must be on the same device.")
        device = q.device
        Q_dtype = q.dtype
        backend = (self.gpu_backend if device.type == 'cuda' else self.cpu_backend) if self.training \
            else (self.gpu_backend_infer if device.type == 'cuda' else self.cpu_backend_infer)

        # gates
        if_gate_input = torch.cat([q, k, v], dim=-1)  # (B, S, 3H)
        if_preact = self.ifgate(if_gate_input)        # (B, S, 2*NH)
        # soft-cap for stability
        if_preact = self.gate_soft_cap * torch.tanh(if_preact / self.gate_soft_cap)

        i_preact, f_preact = torch.chunk(if_preact, 2, dim=-1)  # each (B, S, NH)
        B, S, H = q.shape
        # reshape heads
        q = q.view(B, S, self.num_heads, -1).transpose(1, 2)  # (B, NH, S, DH)
        k = k.view(B, S, self.num_heads, -1).transpose(1, 2)
        v = v.view(B, S, self.num_heads, -1).transpose(1, 2)

        i = i_preact.transpose(-1, -2)  # (B, NH, S)
        f = f_preact.transpose(-1, -2)  # (B, NH, S)

        # autocast routing (match backend dtype)
        if device.type == 'cuda' and self.use_autocast:
            q = q.to(self.autocast_dtype); k = k.to(self.autocast_dtype)
            v = v.to(self.autocast_dtype); i = i.to(self.autocast_dtype); f = f.to(self.autocast_dtype)

        h_state = backend(q=q, k=k, v=v, i=i, f=f)
        h_state = h_state.to(Q_dtype).contiguous()  # (B, NH, S, DH)

        # post-norm and merge heads
        h_state_norm = self.outnorm(h_state)               # (B, NH, S, DH)
        h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, -1)  # (B, S, H)
        return h_state_norm

    def reset_parameters(self):
        self.outnorm.reset_parameters()
        # Initialize fused ifgate: weights zero, biases set to i≈-10, f∈[3,6]
        torch.nn.init.zeros_(self.ifgate.weight)
        with torch.no_grad():
            i_bias = torch.full((self.num_heads,), -10.0, dtype=self.ifgate.bias.dtype, device=self.ifgate.bias.device)
            f_bias = torch.linspace(3.0, 6.0, steps=self.num_heads, dtype=self.ifgate.bias.dtype, device=self.ifgate.bias.device)
            self.ifgate.bias.data = torch.cat([i_bias, f_bias], dim=0)

# class ViLLayer(nn.Module):
#     def __init__(self,
#                  dim,
#                  direction,
#                  expansion=2,
#                  qkv_block_size=16,
#                  proj_bias=False,
#                  norm_bias=False,
#                  conv_bias=False,
#                  conv_kernel_size=3,
#                  conv_kind="2d",
#                  num_blocks=15,
#                  gate_soft_cap=15.0,
#                  ffn_proj_factor=2.6667,
#                  mlp_ratio = 2.6667,
#                  ffn_round_up_to_multiple_of=64,
#                  weight_mode="fused",
#                  chunk_size=64,
#                  sd_depth_scale=0.0):
#         super().__init__()
#         self.dim = dim
#         self.direction = direction
#         self.sd_depth_scale = max(0.0, min(1.0, float(sd_depth_scale)))
#         self._base_sd = 1.0

#         inner_dim = expansion * dim
#         num_heads = inner_dim // qkv_block_size
#         self.inner_dim = inner_dim
#         self.num_heads = num_heads

#         # MODIFIED: Project to 3*inner_dim to create a path for the gate `z`.
#         self.proj_up = nn.Linear(dim, 3 * inner_dim, bias=proj_bias)

#         assert conv_kernel_size % 2 == 1, "conv_kernel_size must be odd for 2d conv"
#         self.conv = SequenceConv2d(
#             inner_dim, inner_dim, kernel_size=conv_kernel_size,
#             padding=conv_kernel_size // 2, groups=inner_dim,
#             bias=conv_bias
#         )

#         self.qk_proj = nn.Linear(inner_dim, 2 * inner_dim, bias=proj_bias)
#         self.v_proj  = nn.Linear(inner_dim, inner_dim, bias=proj_bias)

#         from .vision_lstm_util import DropPath
#         from .vision_lstm2 import MatrixLSTMCell, FeedForward
#         self.mlstm_cell = MatrixLSTMCell(
#             dim=inner_dim, num_heads=num_heads, norm_bias=norm_bias,
#             eps=1e-6, chunk_size=chunk_size, gate_soft_cap=gate_soft_cap
#         )

#         self.learnable_skip = nn.Parameter(torch.ones(inner_dim))
#         self.proj_down = nn.Linear(inner_dim, dim, bias=proj_bias)
#         self.norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=norm_bias)
#         self.ffn_norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=norm_bias)
#         self.ffn = FeedForward(
#             embedding_dim=dim, ffn_proj_factor=ffn_proj_factor,
#             ffn_round_up_to_multiple_of=ffn_round_up_to_multiple_of,
#             use_bias=proj_bias, weight_mode=weight_mode, num_blocks=num_blocks or 1,
#         )
#         self.sd_attn = DropPath(drop_prob=sd_depth_scale)
#         self.sd_ffn = DropPath(drop_prob=sd_depth_scale)

#         self.set_stochastic_depth(1.0)

#     def _attn_residual(self, x_in: torch.Tensor, H: int, W: int) -> torch.Tensor:
#         x = self.norm(x_in)
#         if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             x = x.flip(dims=[1])

#         x_inner = self.proj_up(x)
#         # MODIFIED: Split into three chunks for qk, v, and the gate z.
#         x_qk, x_v, z = torch.chunk(x_inner, 3, dim=-1)

#         x_qk_grid = einops.rearrange(x_qk, "b (h w) d -> b h w d", h=H, w=W)
#         x_qk_conv_grid = self.conv(x_qk_grid)
#         x_qk_conv_act = F.silu(einops.rearrange(x_qk_conv_grid, "b h w d -> b (h w) d", h=H, w=W))

#         qk = self.qk_proj(x_qk_conv_act)
#         q, k = torch.chunk(qk, 2, dim=-1)
#         v = self.v_proj(x_v)
#         h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)
#         h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_qk_conv_act)
        
#         # MODIFIED: Apply the gate `z` to the mLSTM output before the final projection.
#         gated_output = h_tilde_state_skip * F.silu(z)
#         out = self.proj_down(gated_output)

#         if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             out = out.flip(dims=[1])
#         return out

#     def _ffn_residual(self, x_in: torch.Tensor) -> torch.Tensor:
#         return self.ffn(self.ffn_norm(x_in))

#     def set_stochastic_depth(self, base_p: float):
#         base_p = float(max(0.0, min(1.0, base_p)))
#         self._base_sd = base_p
#         scaled = base_p * self.sd_depth_scale
#         self.sd_attn.drop_prob = scaled
#         self.sd_ffn.drop_prob = scaled

#     def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
#         x = self.sd_attn(x, residual_path=lambda t: self._attn_residual(t, H, W))
#         x = self.sd_ffn(x, residual_path=self._ffn_residual)
#         return x

# class ViLLayer(nn.Module):
#     def __init__(self,
#                  dim,
#                  direction,
#                  expansion=2,
#                  qkv_block_size=32,
#                  proj_bias=False,
#                  norm_bias=False,
#                  conv_bias=False,
#                  conv_kernel_size=3,
#                  conv_kind="2d",
#                  mlp_ratio = 4.0,
#                  num_blocks=1,
#                  gate_soft_cap=15.0,
#                  ffn_proj_factor=2.0,
#                  ffn_round_up_to_multiple_of=96,
#                  weight_mode="fused",
#                  chunk_size=64,
#                  sd_depth_scale=0.0,
#                  use_qk_mixer=True,
#                  qk_mixer_groups=8):
#         super().__init__()

#         self.dim = dim
#         self.direction = direction
#         self.sd_depth_scale = max(0.0, min(1.0, float(sd_depth_scale)))
#         self._base_sd = 1.0

#         inner_dim = expansion * dim
#         num_heads = inner_dim // qkv_block_size
#         self.inner_dim = inner_dim
#         self.num_heads = num_heads

#         # --- Single fused projection for q,k,v,z ---
#         self.qkvz = nn.Linear(dim, 4 * inner_dim, bias=proj_bias)

#         # --- Depthwise conv on [q,k] slice (locality) ---
#         assert conv_kernel_size % 2 == 1, "conv_kernel_size must be odd"
#         self.conv_qk = nn.Conv2d(
#             2 * inner_dim, 2 * inner_dim,
#             kernel_size=conv_kernel_size,
#             padding=conv_kernel_size // 2,
#             groups=2 * inner_dim,
#             bias=conv_bias
#         )

#         # --- Optional light mixer on q,k (grouped 1x1 conv) ---
#         self.use_qk_mixer = use_qk_mixer
#         if use_qk_mixer:
#             self.qk_mixer = nn.Conv1d(
#                 2 * inner_dim, 2 * inner_dim,
#                 kernel_size=1,
#                 groups=qk_mixer_groups,
#                 bias=False
#             )
            
#         self.mlstm_cell = MatrixLSTMCell(
#             dim=inner_dim, num_heads=num_heads, norm_bias=norm_bias,
#             eps=1e-6, chunk_size=chunk_size, gate_soft_cap=gate_soft_cap
#         )

#         self.learnable_skip = nn.Parameter(torch.ones(inner_dim))
#         self.proj_down = nn.Linear(inner_dim, dim, bias=proj_bias)

#         self.norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=norm_bias)
#         self.ffn_norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=norm_bias)

#         self.ffn = FeedForward(
#             embedding_dim=dim,
#             ffn_proj_factor=ffn_proj_factor,
#             ffn_round_up_to_multiple_of=ffn_round_up_to_multiple_of,
#             use_bias=proj_bias,
#             weight_mode=weight_mode,
#             num_blocks=num_blocks or 1,
#         )

#         self.sd_attn = DropPath(drop_prob=sd_depth_scale)
#         self.sd_ffn = DropPath(drop_prob=sd_depth_scale)
#         self.set_stochastic_depth(1.0)

#         self.act = nn.SiLU(inplace=False)

#     def _attn_residual(self, x_in: torch.Tensor, H: int, W: int) -> torch.Tensor:
#         x = self.norm(x_in)
#         if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             x = x.flip(dims=[1])

#         # fused projection → q,k,v,z
#         qkvz = self.qkvz(x)   # [B, HW, 4*inner]
#         q, k, v, z = torch.chunk(qkvz, 4, dim=-1)

#         # conv on [q,k]
#         qk = torch.cat([q, k], dim=-1)        # [B, HW, 2*inner]
#         B, N, D2 = qk.shape
#         qk = qk.view(B, H, W, D2).permute(0, 3, 1, 2).contiguous()
#         qk = self.conv_qk(qk)                 # depthwise conv
#         qk = qk.permute(0, 2, 3, 1).contiguous().view(B, N, D2)

#         if self.use_qk_mixer:
#             qk = self.qk_mixer(qk.transpose(1,2)).transpose(1,2)

#         q, k = torch.chunk(self.act(qk), 2, dim=-1)

#         # mLSTM with v
#         h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)
#         h_tilde_state_skip = h_tilde_state + (self.learnable_skip * q)
#         gated_output = h_tilde_state_skip * F.silu(z, inplace=False)
#         out = self.proj_down(gated_output)

#         if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             out = out.flip(dims=[1])
#         return out

#     def _ffn_residual(self, x_in: torch.Tensor) -> torch.Tensor:
#         return self.ffn(self.ffn_norm(x_in))

#     def set_stochastic_depth(self, base_p: float):
#         base_p = float(max(0.0, min(1.0, base_p)))
#         self._base_sd = base_p
#         scaled = base_p * self.sd_depth_scale
#         self.sd_attn.drop_prob = scaled
#         self.sd_ffn.drop_prob = scaled

#     def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
#         x = self.sd_attn(x, residual_path=lambda t: self._attn_residual(t, H, W))
#         x = self.sd_ffn(x, residual_path=self._ffn_residual)
#         return x



# # ---- tokenwise RMS norm for Q and K (affine scale) ----
class QKNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim)) if affine else None
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, N, D]
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        if self.g is not None:
            x = x * self.g
        return x

# class ViLLayer(nn.Module):
#     def __init__(self,
#                  dim,
#                  direction,
#                  expansion=2,
#                  qkv_block_size=32,
#                  proj_bias=False,
#                  norm_bias=False,
#                  conv_bias=False,
#                  conv_kernel_size=3,
#                  conv_kind="2d",
#                  mlp_ratio=4.0,                 # kept for config parity
#                  num_blocks=1,
#                  gate_soft_cap=15.0,
#                  ffn_proj_factor=2.0,
#                  ffn_round_up_to_multiple_of=96,
#                  weight_mode="fused",
#                  chunk_size=64,
#                  sd_depth_scale=0.0,
#                  # stabilizers
#                  use_qk_norm=False,
#                  use_layerscale=True,
#                  layerscale_init=1e-5,
#                  # optional conditioning vector -> 2D bias on [q,k]
#                  cond_dim: int | None = None):
#         super().__init__()

#         self.dim = dim
#         self.direction = direction
#         self.sd_depth_scale = float(max(0.0, min(1.0, sd_depth_scale)))

#         inner_dim = expansion * dim
#         head_dim  = qkv_block_size
#         num_heads = inner_dim // head_dim
#         self.inner_dim = inner_dim
#         self.num_heads = num_heads

#         # ---- fused input projection → [q, k, v, z] ----
#         self.qkvz = nn.Linear(dim, 4 * inner_dim, bias=proj_bias)

#         # ---- DW-conv on concatenated [q,k] in NHWC (spatial inductive bias) ----
#         assert conv_kernel_size % 2 == 1, "conv_kernel_size must be odd"
#         self.conv_qk = SequenceConv2d(
#             2 * inner_dim, 2 * inner_dim,
#             kernel_size=conv_kernel_size,
#             padding=conv_kernel_size // 2,
#             groups=2 * inner_dim,
#             bias=conv_bias
#         )

#         # ---- attention core (mLSTM) ----
#         # from .vision_lstm2 import MatrixLSTMCell, FeedForward
#         self.mlstm_cell = MatrixLSTMCell(
#             dim=inner_dim, num_heads=num_heads, norm_bias=norm_bias,
#             eps=1e-6, chunk_size=chunk_size, gate_soft_cap=gate_soft_cap
#         )
#         self.learnable_skip = nn.Parameter(torch.ones(inner_dim))
#         self.proj_down = nn.Linear(inner_dim, dim, bias=proj_bias)

#         # ---- pre-norms (parallel branches) ----
#         self.attn_norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=norm_bias)
#         self.ffn_norm  = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=norm_bias)

#         # ---- FFN (SwiGLU-style) ----
#         self.ffn = FeedForward(
#             embedding_dim=dim,
#             ffn_proj_factor=ffn_proj_factor,
#             ffn_round_up_to_multiple_of=ffn_round_up_to_multiple_of,
#             use_bias=proj_bias,
#             weight_mode=weight_mode,
#             num_blocks=num_blocks or 1,
#         )

#         # ---- QK-Norm stabilizer ----
#         self.use_qk_norm = use_qk_norm
#         if use_qk_norm:
#             self.q_norm = QKNorm(inner_dim)
#             self.k_norm = QKNorm(inner_dim)

#         # ---- LayerScale per branch ----
#         self.use_layerscale = use_layerscale
#         if use_layerscale:
#             self.gamma_attn = nn.Parameter(torch.full((dim,), layerscale_init))
#             self.gamma_ffn  = nn.Parameter(torch.full((dim,), layerscale_init))
#         else:
#             self.register_buffer("gamma_attn", torch.ones(dim), persistent=False)
#             self.register_buffer("gamma_ffn",  torch.ones(dim), persistent=False)

#         # ---- DropPath per branch ----
#         # from .vision_lstm_util import DropPath
#         self.sd_attn = DropPath(drop_prob=sd_depth_scale)
#         self.sd_ffn  = DropPath(drop_prob=sd_depth_scale)
#         self.set_stochastic_depth(1.0)

#         self.act = nn.SiLU(inplace=False)

#         # ---- safe start: near-identity for attention branch ----
#         nn.init.zeros_(self.proj_down.weight)
#         if self.proj_down.bias is not None:
#             nn.init.zeros_(self.proj_down.bias)

#     # Attention residual (pre-norm, layerscale, no in-place ops on chunk views)
#     def _attn_residual(self, x_in, H, W, cond=None):
#         x = self.attn_norm(x_in)
#         if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             x = x.flip(dims=[1])

#         q, k, v, z = torch.chunk(self.qkvz(x), 4, dim=-1)

#         # NHWC grid for DW-conv on [q,k]
#         B, N, D = q.shape
#         qk = torch.cat([q, k], dim=-1)                 # [B, N, 2*inner]
#         qk_grid = qk.view(B, H, W, 2*self.inner_dim)   # NHWC

#         if getattr(self, "cond_dim", None) and cond is not None:

#         qk_grid = self.conv_qk(qk_grid)                # DW spatial bias
#         qk = qk_grid.view(B, N, 2*self.inner_dim)
#         q, k = torch.chunk(F.silu(qk, inplace=False), 2, dim=-1)

#         # head-wise norm + temperature (or your tokenwise QK-Norm)
#         if hasattr(self, "qk_headnorm"):
#             q, k = self.qk_headnorm(q, k)
#         elif getattr(self, "use_qk_norm", False):
#             q = self.q_norm(q); k = self.k_norm(k)

#         h = self.mlstm_cell(q=q, k=k, v=v)
#         h = h + (self.learnable_skip * q)
#         gated = h * F.silu(z, inplace=False)
#         out = self.proj_down(gated)

#         if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             out = out.flip(dims=[1])

#         return out * self.gamma_attn  # LayerScale

#     def _ffn_residual(self, x_in):
#         y = self.ffn(self.ffn_norm(x_in))
#         return y * self.gamma_ffn
    
#     def forward(self, x, H, W, cond=None):
#         x = self.sd_attn(x, residual_path=lambda t: self._attn_residual(t, H, W, cond))
#         x = self.sd_ffn(x,  residual_path=self._ffn_residual)
#         return x
    
#     # ---------------- SD scheduling ----------------
#     def set_stochastic_depth(self, base_p: float):
#         base_p = float(max(0.0, min(1.0, base_p)))
#         scaled = base_p * self.sd_depth_scale
#         self.sd_attn.drop_prob = scaled
#         self.sd_ffn.drop_prob  = scaled
#         self.sd_ffn.drop_prob  = scaled



# class ViLLayer(nn.Module):
#     def __init__(self,
#                  dim,
#                  direction,
#                  expansion=2,
#                  qkv_block_size=32,          # head_dim
#                  proj_bias=True,
#                  norm_bias=True,
#                  conv_bias=True,
#                  conv_kernel_size=3,
#                  conv_kind="2d",
#                  mlp_ratio=4.0,              # kept for cfg parity (unused here)
#                  num_blocks=1,
#                  gate_soft_cap=15.0,
#                  ffn_proj_factor=2.0,
#                  ffn_round_up_to_multiple_of=128,
#                  weight_mode="fused",
#                  chunk_size=64,
#                  sd_depth_scale=0.0,
#                  use_qk_norm=False,          # optional stabilizer (tokenwise)
#                  use_layerscale=True,
#                  layerscale_init=1e-5):
#         super().__init__()

#         assert conv_kind == "2d", "This layer currently supports conv_kind='2d' only."

#         self.dim = dim
#         self.direction = direction
#         self.sd_depth_scale = float(max(0.0, min(1.0, sd_depth_scale)))

#         inner_dim = expansion * dim
#         head_dim  = qkv_block_size
#         assert inner_dim % head_dim == 0, "inner_dim must be divisible by qkv_block_size"
#         num_heads = inner_dim // head_dim
#         self.inner_dim = inner_dim
#         self.num_heads = num_heads
#         self.head_dim  = head_dim

#         # ---- fused input projection → [q, k, v, z] ----
#         self.qkvz = nn.Linear(dim, 4 * inner_dim, bias=proj_bias)

#         # ---- DW-conv on concatenated [q,k] (NHWC in/out) ----
#         assert conv_kernel_size % 2 == 1, "conv_kernel_size must be odd"
#         self.conv_qk = SequenceConv2d(
#             2 * inner_dim, 2 * inner_dim,
#             kernel_size=conv_kernel_size,
#             padding=conv_kernel_size // 2,
#             groups=2 * inner_dim,
#             bias=conv_bias
#         )

#         # ---- attention core (mLSTM) ----
#         self.mlstm_cell = MatrixLSTMCell(
#             dim=inner_dim, num_heads=num_heads, norm_bias=norm_bias,
#             eps=1e-6, chunk_size=chunk_size, gate_soft_cap=gate_soft_cap
#         )
#         self.learnable_skip = nn.Parameter(torch.ones(inner_dim))
#         self.proj_down = nn.Linear(inner_dim, dim, bias=proj_bias)

#         # ---- pre-norms ----
#         self.attn_norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=norm_bias)
#         self.ffn_norm  = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=norm_bias)

#         # ---- FFN (SwiGLU-style) ----
#         self.ffn = FeedForward(
#             embedding_dim=dim,
#             ffn_proj_factor=ffn_proj_factor,
#             ffn_round_up_to_multiple_of=ffn_round_up_to_multiple_of,
#             use_bias=proj_bias,
#             weight_mode=weight_mode,
#             num_blocks=num_blocks or 1,
#         )

#         # ---- optional QK-Norm (tokenwise RMS) ----
#         self.use_qk_norm = use_qk_norm
#         if use_qk_norm:
#             self.q_norm = QKNorm(inner_dim)
#             self.k_norm = QKNorm(inner_dim)

#         # ---- LayerScale per branch ----
#         self.use_layerscale = use_layerscale
#         if use_layerscale:
#             self.gamma_attn = nn.Parameter(torch.full((dim,), layerscale_init))
#             self.gamma_ffn  = nn.Parameter(torch.full((dim,), layerscale_init))
#         else:
#             self.register_buffer("gamma_attn", torch.ones(dim), persistent=False)
#             self.register_buffer("gamma_ffn",  torch.ones(dim), persistent=False)

#         # ---- DropPath per branch ----
#         self.sd_attn = DropPath(drop_prob=sd_depth_scale)
#         self.sd_ffn  = DropPath(drop_prob=sd_depth_scale)
#         self.set_stochastic_depth(1.0)

#         # ---- safe start: near-identity for attention branch ----
#         nn.init.zeros_(self.proj_down.weight)
#         if self.proj_down.bias is not None:
#             nn.init.zeros_(self.proj_down.bias)

#         self.act = nn.SiLU(inplace=False)

#     # ---- attention residual (seq -> grid for conv, then back; no in-place on chunk views) ----
#     def _attn_residual(self, x_in: torch.Tensor, H: int, W: int) -> torch.Tensor:
#         x = self.attn_norm(x_in)
#         if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             x = x.flip(dims=[1])

#         # fused projection → q,k,v,z : [B, HW, inner] x3 + [B, HW, inner]
#         q, k, v, z = torch.chunk(self.qkvz(x), 4, dim=-1)

#         # depthwise 2D conv prior on [q,k] (SequenceConv2d expects NHWC)
#         B, N, _ = q.shape
#         assert N == H * W, "sequence length must equal H*W"
#         qk = torch.cat([q, k], dim=-1)                       # [B, HW, 2*inner]
#         qk_grid = qk.view(B, H, W, 2 * self.inner_dim)       # NHWC
#         qk_grid = self.conv_qk(qk_grid)                      # NHWC -> NHWC
#         qk = qk_grid.view(B, N, 2 * self.inner_dim)
#         q, k = torch.chunk(F.silu(qk, inplace=False), 2, dim=-1)

#         # key scaling (stabilizes similarity temperature)
#         k = k * (1.0 / math.sqrt(self.head_dim))

#         # optional tokenwise QK-Norm
#         if self.use_qk_norm:
#             q = self.q_norm(q)
#             k = self.k_norm(k)

#         # mLSTM attention core + gated output
#         h = self.mlstm_cell(q=q, k=k, v=v)
#         h = h + (self.learnable_skip * q)
#         gated = h * F.silu(z, inplace=False)
#         out = self.proj_down(gated)

#         if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             out = out.flip(dims=[1])

#         # LayerScale for the branch
#         return out * self.gamma_attn

#     def _ffn_residual(self, x_in: torch.Tensor) -> torch.Tensor:
#         y = self.ffn(self.ffn_norm(x_in))
#         return y * self.gamma_ffn

#     def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
#         x = self.sd_attn(x, residual_path=lambda t: self._attn_residual(t, H, W))
#         x = self.sd_ffn(x,  residual_path=self._ffn_residual)
#         return x

#     def set_stochastic_depth(self, base_p: float):
#         base_p = float(max(0.0, min(1.0, base_p)))
#         scaled = base_p * self.sd_depth_scale
#         self.sd_attn.drop_prob = scaled
#         self.sd_ffn.drop_prob  = scaled


# class ViLLayer(nn.Module):
#     def __init__(self,
#                  dim,
#                  direction,
#                  expansion=2,
#                  qkv_block_size=32,  # head_dim
#                  proj_bias=True,
#                  norm_bias=True,
#                  conv_bias=True,
#                  conv_kernel_size=3,
#                  conv_kind="2d",
#                  mlp_ratio=4.0,  # kept for cfg parity (unused here)
#                  num_blocks=1,
#                  gate_soft_cap=15.0,
#                  ffn_proj_factor=2.0,
#                  ffn_round_up_to_multiple_of=128,
#                  weight_mode="fused",
#                  chunk_size=64,
#                  sd_depth_scale=0.0,
#                  use_qk_norm=False,  # optional stabilizer (tokenwise)
#                  use_layerscale=True,
#                  layerscale_init=1e-5):
#         super().__init__()
#         assert conv_kind == "2d", "This layer currently supports conv_kind='2d' only."
#         self.dim = dim
#         self.direction = direction
#         self.sd_depth_scale = float(max(0.0, min(1.0, sd_depth_scale)))
#         inner_dim = expansion * dim
#         head_dim = qkv_block_size
#         assert inner_dim % head_dim == 0, "inner_dim must be divisible by qkv_block_size"
#         num_heads = inner_dim // head_dim
#         self.inner_dim = inner_dim
#         self.num_heads = num_heads
#         self.head_dim = head_dim
#         # ---- Projection up to 2*inner_dim (original pattern) ----
#         self.proj_up = nn.Linear(dim, 2 * inner_dim, bias=proj_bias)
#         # ---- DW-conv on x_mlstm (original pattern, NHWC in/out) ----
#         assert conv_kernel_size % 2 == 1, "conv_kernel_size must be odd"
#         self.conv = SequenceConv2d(
#             inner_dim, inner_dim,
#             kernel_size=conv_kernel_size,
#             padding=conv_kernel_size // 2,
#             groups=inner_dim,
#             bias=conv_bias
#         )
#         # ---- attention core (mLSTM) ----
#         self.mlstm_cell = MatrixLSTMCell(
#             dim=inner_dim, num_heads=num_heads, norm_bias=norm_bias,
#             eps=1e-6, chunk_size=chunk_size, gate_soft_cap=gate_soft_cap
#         )
#         self.learnable_skip = nn.Parameter(torch.ones(inner_dim))
#         self.proj_down = nn.Linear(inner_dim, dim, bias=proj_bias)
#         # ---- pre-norms ----
#         self.attn_norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=norm_bias)
#         self.ffn_norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=norm_bias)
#         # ---- FFN (SwiGLU-style) ----
#         self.ffn = FeedForward(
#             embedding_dim=dim,
#             ffn_proj_factor=ffn_proj_factor,
#             ffn_round_up_to_multiple_of=ffn_round_up_to_multiple_of,
#             use_bias=proj_bias,
#             weight_mode=weight_mode,
#             num_blocks=num_blocks or 1,
#         )
#         # ---- optional QK-Norm (tokenwise RMS) ----
#         self.use_qk_norm = use_qk_norm
#         if use_qk_norm:
#             self.q_norm = QKNorm(inner_dim)
#             self.k_norm = QKNorm(inner_dim)
#         # ---- LayerScale per branch ----
#         self.use_layerscale = use_layerscale
#         if use_layerscale:
#             self.gamma_attn = nn.Parameter(torch.full((dim,), layerscale_init))
#             self.gamma_ffn = nn.Parameter(torch.full((dim,), layerscale_init))
#         else:
#             self.register_buffer("gamma_attn", torch.ones(dim), persistent=False)
#             self.register_buffer("gamma_ffn", torch.ones(dim), persistent=False)
#         # ---- DropPath per branch ----
#         self.sd_attn = DropPath(drop_prob=sd_depth_scale)
#         self.sd_ffn = DropPath(drop_prob=sd_depth_scale)
#         self.set_stochastic_depth(1.0)
#         # ---- safe start: near-identity for attention branch ----
#         nn.init.zeros_(self.proj_down.weight)
#         if self.proj_down.bias is not None:
#             nn.init.zeros_(self.proj_down.bias)
#         self.act = nn.SiLU(inplace=False)

#     def _attn_residual(self, x_in: torch.Tensor, H: int, W: int) -> torch.Tensor:
#         x = self.attn_norm(x_in)
#         if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             x = x.flip(dims=[1])
#         # Projection up and chunking (original pattern) → x_mlstm, z : [B, HW, inner]
#         x_inner = self.proj_up(x)
#         x_mlstm, z = torch.chunk(x_inner, 2, dim=-1)
#         # Depthwise 2D conv on x_mlstm (original pattern, SequenceConv2d expects NHWC)
#         B, N, _ = x_mlstm.shape
#         assert N == H * W, "sequence length must equal H*W"
#         x_mlstm_grid = einops.rearrange(x_mlstm, "b (h w) d -> b h w d", h=H, w=W)
#         x_mlstm_grid = self.conv(x_mlstm_grid)  # NHWC -> NHWC
#         x_mlstm = einops.rearrange(x_mlstm_grid, "b h w d -> b (h w) d")
#         x_mlstm_act = F.silu(x_mlstm, inplace=False)
#         # Derive q, k from x_mlstm_act; v from x_mlstm (original pattern, no separate projections)
#         q = x_mlstm_act
#         k = x_mlstm_act
#         v = x_mlstm
#         # Optional tokenwise QK-Norm
#         if self.use_qk_norm:
#             q = self.q_norm(q)
#             k = self.k_norm(k)
#         # mLSTM attention core + gated output (original gating pattern)
#         h = self.mlstm_cell(q=q, k=k, v=v)
#         h = h + (self.learnable_skip * x_mlstm_act)
#         gated = h * F.silu(z, inplace=False)
#         out = self.proj_down(gated)
#         if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             out = out.flip(dims=[1])
#         # LayerScale for the branch
#         return out * self.gamma_attn

#     def _ffn_residual(self, x_in: torch.Tensor) -> torch.Tensor:
#         y = self.ffn(self.ffn_norm(x_in))
#         return y * self.gamma_ffn

#     def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
#         x = self.sd_attn(x, residual_path=lambda t: self._attn_residual(t, H, W))
#         x = self.sd_ffn(x, residual_path=self._ffn_residual)
#         return x

#     def set_stochastic_depth(self, base_p: float):
#         base_p = float(max(0.0, min(1.0, base_p)))
#         scaled = base_p * self.sd_depth_scale
#         self.sd_attn.drop_prob = scaled
#         self.sd_ffn.drop_prob = scaled


class ViLLayer(nn.Module):
    def __init__(self,
                 dim,
                 direction,
                 expansion=2,
                 qkv_block_size=16,
                 proj_bias=False,
                 norm_bias=False,
                 conv_bias=False,
                 conv_kernel_size=3,
                 conv_kind="2d",
                 num_blocks=15,
                 gate_soft_cap=15.0,
                 ffn_proj_factor=2.6667,
                 mlp_ratio=2.6667,
                 ffn_round_up_to_multiple_of=64,
                 weight_mode="fused",
                 chunk_size=64,
                 sd_depth_scale=0.0):
        super().__init__()
        self.dim = dim
        self.direction = direction
        self.sd_depth_scale = max(0.0, min(1.0, float(sd_depth_scale)))
        self._base_sd = 1.0
        inner_dim = expansion * dim
        num_heads = inner_dim // qkv_block_size
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        # Project to 2*inner_dim (matches original proportions)
        self.proj_up = nn.Linear(dim, 2 * inner_dim, bias=proj_bias)
        assert conv_kernel_size % 2 == 1, "conv_kernel_size must be odd for 2d conv"
        self.conv_kind = '2d'
        self.conv = nn.Conv2d(
            inner_dim, inner_dim, kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2, groups=inner_dim,
            bias=conv_bias
        )
        # Separate projections for Q, K, V (matches original's independent subspaces)
        self.q_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.k_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.v_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        from .vision_lstm_util import DropPath
        from .vision_lstm2 import MatrixLSTMCell, FeedForward
        self.mlstm_cell = MatrixLSTMCell(
            dim=inner_dim, num_heads=num_heads, norm_bias=norm_bias,
            eps=1e-6, chunk_size=chunk_size, gate_soft_cap=gate_soft_cap
        )
        self.learnable_skip = nn.Parameter(torch.ones(inner_dim))
        self.proj_down = nn.Linear(inner_dim, dim, bias=proj_bias)
        self.norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=norm_bias)
        self.ffn_norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=norm_bias)
        self.ffn = FeedForward(
            embedding_dim=dim, ffn_proj_factor=ffn_proj_factor,
            ffn_round_up_to_multiple_of=ffn_round_up_to_multiple_of,
            use_bias=proj_bias, weight_mode=weight_mode, num_blocks=num_blocks or 1,
        )
        self.sd_attn = DropPath(drop_prob=sd_depth_scale)
        self.sd_ffn = DropPath(drop_prob=sd_depth_scale)
        self.set_stochastic_depth(1.0)

    def _attn_residual(self, x_in: torch.Tensor) -> torch.Tensor:
        B, H, W, D = x_in.shape
        x = self.norm(x_in)
        if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1, 2])
        x_inner = self.proj_up(x)
        x_qk, x_v = torch.chunk(x_inner, 2, dim=-1)
        if self.conv_kind == "2d":
            x_qk_perm = x_qk.permute(0, 3, 1, 2)
            x_qk_conv = self.conv(x_qk_perm).permute(0, 2, 3, 1)
        else:  # causal1d
            x_qk_flat = self._flatten(x_qk)
            x_qk_conv = self._unflatten(self.conv(x_qk_flat), H, W)
        x_qk_conv_act = torch.nn.functional.silu(x_qk_conv)
        qk = self.qk_proj(x_qk_conv_act)
        q, k = torch.chunk(qk, 2, dim=-1)
        v = self.v_proj(x_v)
        h_tilde_state = self.mlstm_cell(q=self._flatten(q), k=self._flatten(k), v=self._flatten(v))
        h_tilde_state = self._unflatten(h_tilde_state, H, W)
        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_qk_conv_act)
        out = self.proj_down(h_tilde_state_skip)
        if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            out = out.flip(dims=[1, 2])
        return out


    def _ffn_residual(self, x_in: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.ffn_norm(x_in))

    def set_stochastic_depth(self, base_p: float):
        base_p = float(max(0.0, min(1.0, base_p)))
        self._base_sd = base_p
        scaled = base_p * self.sd_depth_scale
        self.sd_attn.drop_prob = scaled
        self.sd_ffn.drop_prob = scaled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sd_attn(x, residual_path=lambda t: self._attn_residual(t))
        x = self.sd_ffn(x, residual_path=self._ffn_residual)
        return x

class ViLBlock(nn.Module):
    def __init__(self, dim, direction, drop_path=0.0, **kwargs):
        super().__init__()
        # Pop seqlens, as it's no longer used for initialization
        kwargs.pop("seqlens", None)
        self.layer = ViLLayer(dim=dim, direction=direction, sd_depth_scale=drop_path, **kwargs)

    def set_stochastic_depth(self, base_p: float):
        self.layer.set_stochastic_depth(base_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass H, W through to the underlying ViLLayer
        return self.layer(x)


class ViLBlockPair(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.0, ckpt_thresh: int = 200 * 200, **kwargs):
        super().__init__()
        self.ckpt_thresh = int(ckpt_thresh)
        # Pop seqlens, as it's no longer used for initialization
        kwargs.pop("seqlens", None)

        self.rowwise_from_top_left = ViLBlock(
            dim=dim, direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT, drop_path=drop_path, **kwargs
        )
        self.rowwise_from_bot_right = ViLBlock(
            dim=dim, direction=SequenceTraversal.ROWWISE_FROM_BOT_RIGHT, drop_path=drop_path, **kwargs
        )

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rowwise_from_top_left(x)
        x = self.rowwise_from_bot_right(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, D = x.shape
        S = H * W
        need_ckpt = self.training and x.requires_grad and S >= self.ckpt_thresh
        if need_ckpt:
            # Checkpoint now correctly passes H and W as args
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False, preserve_rng_state=True)
        return self._forward_impl(x)

    def set_stochastic_depth(self, base_p: float):
        self.rowwise_from_top_left.set_stochastic_depth(base_p)
        self.rowwise_from_bot_right.set_stochastic_depth(base_p)



# ---------------------------------------------------------------------
# VisionLSTM2 (standalone backbone/classifier; not used by your YAML)
# ---------------------------------------------------------------------
class VisionLSTM2(nn.Module):
    def __init__(
            self,
            dim=192,
            input_shape=(3, 224, 224),
            patch_size=16,
            depth=12,
            output_shape=(1000,),
            mode="classifier",
            pooling="bilateral_flatten",
            drop_path_rate=0.0,
            drop_path_decay=False,
            stride=None,
            legacy_norm=False,
            conv_kind="2d",
            conv_kernel_size=3,
            proj_bias=True,
            norm_bias=True,
            init_weights="original",
    ):
        if depth == 24 and dim < 1024:
            warnings.warn(
                "A single VisionLSTM2 block consists of two subblocks (one for each traversal direction). "
                "ViL-T, ViL-S and ViL-B therefore use depth=12 instead of depth=24."
            )
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        ndim = len(self.input_shape) - 1
        self.patch_size = to_ntuple(patch_size, n=ndim)
        self.dim = dim
        self.depth = depth
        self.stride = stride
        self.mode = mode
        self.pooling = pooling
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        self.conv_kind = conv_kind
        self.conv_kernel_size = conv_kernel_size
        self.proj_bias = proj_bias
        self.norm_bias = norm_bias
        self.init_weights = init_weights

        # initialize patch_embed
        self.patch_embed = VitPatchEmbed(
            dim=dim,
            stride=stride,
            num_channels=self.input_shape[0],
            resolution=self.input_shape[1:],
            patch_size=self.patch_size,
        )

        # pos embed
        self.pos_embed = VitPosEmbed2d(seqlens=self.patch_embed.seqlens, dim=dim)

        # stochastic depth per block (internal, unrelated to ViLLayer SD)
        if drop_path_decay and drop_path_rate > 0.:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate] * depth

        # depth blocks (each is a ViLBlockPair)
        self.blocks = nn.ModuleList(
            [
                ViLBlockPair(
                    dim=dim,
                    drop_path=dpr[i],
                    conv_kind=conv_kind,
                    seqlens=self.patch_embed.seqlens,
                    proj_bias=proj_bias,
                    norm_bias=norm_bias,
                    num_blocks=depth * 2,
                    init_weights=init_weights,
                )
                for i in range(depth)
            ],
        )
        head_dim = dim * 2 if (pooling == "bilateral_flatten" and mode == "classifier") else dim
        self.norm = LayerNorm(dim, bias=norm_bias, eps=1e-6)
        self.legacy_norm = nn.LayerNorm(head_dim) if legacy_norm else nn.Identity()

        # head
        if mode == "features":
            if self.output_shape is not None:
                warnings.warn(f"mode=features -> output_shape is ignored ({self.output_shape})")
            self.head = None
            if self.pooling is None:
                self.output_shape = (self.patch_embed.num_patches, dim)
            elif self.pooling == "to_image":
                self.output_shape = (dim, *self.patch_embed.seqlens)
            else:
                warnings.warn(f"invalid pooling -> pooling is ignored ({self.pooling})")
                self.pooling = None
        elif mode == "classifier":
            assert self.output_shape is not None and len(self.output_shape) == 1, \
                "define number of classes via output_shape=(num_classes,)"
            self.head = nn.Linear(head_dim, self.output_shape[0])
            nn.init.trunc_normal_(self.head.weight, std=2e-5)
            nn.init.zeros_(self.head.bias)
        else:
            raise NotImplementedError

    def load_state_dict(self, state_dict, strict=True):
        old_pos_embed = state_dict["pos_embed.embed"]
        if old_pos_embed.shape != self.pos_embed.embed.shape:
            state_dict["pos_embed.embed"] = interpolate_sincos(embed=old_pos_embed, seqlens=self.pos_embed.seqlens)
        if self.mode == "features":
            state_dict.pop("head.weight", None)
            state_dict.pop("head.bias", None)
            cur_sd = self.state_dict()
            state_dict["legacy_norm.weight"] = cur_sd["legacy_norm.weight"]
            state_dict["legacy_norm.bias"] = cur_sd["legacy_norm.bias"]
        return super().load_state_dict(state_dict=state_dict, strict=strict)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed.embed"}

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        x = einops.rearrange(x, "b ... d -> b (...) d")
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        if self.pooling is None:
            x = self.legacy_norm(x)
        elif self.pooling == "to_image":
            x = self.legacy_norm(x)
            seqlen_h, seqlen_w = self.patch_embed.seqlens
            x = einops.rearrange(x, "b (h w) d -> b d h w", h=seqlen_h, w=seqlen_w)
        elif self.pooling == "bilateral_avg":
            x = (x[:, 0] + x[:, -1]) / 2
            x = self.legacy_norm(x)
        elif self.pooling == "bilateral_flatten":
            x = torch.concat([x[:, 0], x[:, -1]], dim=1)
            x = self.legacy_norm(x)
        else:
            raise NotImplementedError(f"pooling '{self.pooling}' is not implemented")

        if self.head is not None:
            x = self.head(x)
        return x


# ---------------------------------------------------------------------
# Fusion MLPs (kept; some models may import these)
# ---------------------------------------------------------------------
class FusionMLPBase(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or 4 * dim
    def forward(self, x):
        raise NotImplementedError


class MLPBaseline(FusionMLPBase):
    def __init__(self, dim, hidden_dim=None):
        super().__init__(dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, dim)
        )
    def forward(self, x): return self.net(x)


class GEGLU(FusionMLPBase):
    def __init__(self, dim, hidden_dim=None):
        super().__init__(dim, hidden_dim)
        self.fc = nn.Linear(dim, self.hidden_dim * 2)
        self.proj = nn.Linear(self.hidden_dim, dim)
    def forward(self, x):
        x1, x2 = self.fc(x).chunk(2, dim=-1)
        return self.proj(F.gelu(x1) * x2)


class SwiGLU(FusionMLPBase):
    def __init__(self, dim, hidden_dim=None):
        super().__init__(dim, hidden_dim)
        self.fc = nn.Linear(dim, self.hidden_dim * 2)
        self.proj = nn.Linear(self.hidden_dim, dim)
    def forward(self, x):
        x1, x2 = self.fc(x).chunk(2, dim=-1)
        return self.proj(F.silu(x1) * x2)


class RGBlock(FusionMLPBase):
    def __init__(self, dim, hidden_dim=None):
        super().__init__(dim, hidden_dim)
        local_dim = self.hidden_dim * 2 // 3
        self.fc1 = nn.Conv2d(dim, local_dim * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(local_dim, local_dim, kernel_size=3, padding=1, groups=local_dim)
        self.fc2 = nn.Conv2d(local_dim, dim, kernel_size=1)
    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = F.gelu(self.dwconv(x) + x) * v
        return self.fc2(x)


class ConvMLP(FusionMLPBase):
    def __init__(self, dim, hidden_dim=None):
        super().__init__(dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, self.hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, groups=self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, dim, kernel_size=1)
        )
    def forward(self, x): return self.mlp(x)


class LoRAMLP(FusionMLPBase):
    def __init__(self, dim, hidden_dim=None, rank=16):
        super().__init__(dim, hidden_dim)
        self.rank = min(rank, self.hidden_dim)
        self.down = nn.Linear(dim, self.rank)
        self.up = nn.Linear(self.rank, dim)
    def forward(self, x): return self.up(F.relu(self.down(x)))


class MLPMixer(FusionMLPBase):
    def __init__(self, dim, seq_len, hidden_dim=None):
        super().__init__(dim, hidden_dim)
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(seq_len, seq_len),
        )
        self.channel_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, dim)
        )
    def forward(self, x):
        x = x.transpose(1, 2)  # B, C, S
        x = self.token_mlp(x)
        x = x.transpose(1, 2)
        return self.channel_mlp(x)


class CrossAttentionMLP(FusionMLPBase):
    def __init__(self, dim, hidden_dim=None):
        super().__init__(dim, hidden_dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, dim)
    def forward(self, x1, x2):
        q = self.q(x1); k = self.k(x2); v = self.v(x2)
        attn = F.softmax(q @ k.transpose(-2, -1) / (self.dim ** 0.5), dim=-1)
        return self.out(attn @ v)


class FiLMMLP(FusionMLPBase):
    def __init__(self, dim, hidden_dim=None):
        super().__init__(dim, hidden_dim)
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, dim)
        )
    def forward(self, x, modulator):
        gamma = self.gamma(modulator)
        beta = self.beta(modulator)
        return self.ffn(x) * gamma + beta


MLP_REGISTRY = {
    "baseline": lambda dim, **kwargs: MLPBaseline(dim, **kwargs),
    "geglu":    lambda dim, **kwargs: GEGLU(dim, **kwargs),
    "swiglu":   lambda dim, **kwargs: SwiGLU(dim, **kwargs),
    "rgblock":  lambda dim, **kwargs: RGBlock(dim, **kwargs),
    "convmlp":  lambda dim, **kwargs: ConvMLP(dim, **kwargs),
    "lora":     lambda dim, **kwargs: LoRAMLP(dim, **kwargs),
    "mixer":    lambda dim, seq_len=64, **kw: MLPMixer(dim, seq_len=seq_len, **kw),
    "crossattn": lambda dim, **kwargs: CrossAttentionMLP(dim, **kwargs),
    "film":     lambda dim, **kwargs: FiLMMLP(dim, **kwargs),
}


# ---------------------------------------------------------------------
# Optional FusionViLLayer (not used by your YAML; fixed & safe)
# ---------------------------------------------------------------------
class FusionViLLayer(nn.Module):
    """
    Optional fusion; not used by your YAML. Kept here for completeness with safe defaults.
    """
    def __init__(
        self,
        dim,
        direction="rowwise_from_top_left",
        mlp_type="baseline",
        mlp_hidden_dim=None,
        use_skip=True,
        use_mlp=True,
        conv_kind="2d",
        conv_kernel_size=3,
        proj_bias=True,
        norm_bias=True,
        seqlens=None,
        num_blocks=1,
        init_weights="original",
        seq_len=None,
        proj_type="linear",  # 'linear', 'conv', or 'sequenceconv'
    ):
        super().__init__()
        self.use_skip = use_skip
        self.use_mlp = use_mlp
        self.seq_len = seq_len
        self.proj_type = proj_type
        self.dim = dim

        # Project
        if proj_type == "linear":
            self.input_proj = nn.Linear(dim * 2, dim)
        elif proj_type == "conv":
            self.input_proj = nn.Sequential(
                nn.Conv2d(dim * 2, dim, kernel_size=1, bias=proj_bias),
                nn.BatchNorm2d(dim),
                nn.SiLU()
            )
        elif proj_type == "sequenceconv":
            self.input_proj = SequenceConv2d(
                in_channels=dim * 2,
                out_channels=dim,
                kernel_size=1,
                padding=0,
                bias=proj_bias,
                seqlens=seqlens
            )
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}")

        self.norm = LayerNorm(ndim=dim, weight=True, bias=norm_bias)

        self.vilayer = ViLLayer(
            dim=dim,
            direction=direction,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            seqlens=seqlens,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            num_blocks=num_blocks,
            init_weights=init_weights,
        )

        self.residual_proj = nn.Identity() if not use_skip else nn.Linear(dim, dim)

        if use_mlp:
            self.norm2 = LayerNorm(ndim=dim)
            self.post_mlp = MLP_REGISTRY[mlp_type](
                dim, hidden_dim=mlp_hidden_dim or dim * 4, seq_len=seq_len
            )
        else:
            self.post_mlp = None

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        # For skip/residual projection from x1, precompute sequence view
        x1_seq = einops.rearrange(x1, "b c h w -> b (h w) c")

        if self.proj_type == "conv":
            x = torch.cat([x1, x2], dim=1)             # [B, 2C, H, W]
            x = self.input_proj(x)                      # [B, C, H, W]
            x_seq = einops.rearrange(x, "b c h w -> b (h w) c")
        else:
            x2_seq = einops.rearrange(x2, "b c h w -> b (h w) c")
            x_cat  = torch.cat([x1_seq, x2_seq], dim=-1)  # [B, S, 2C]
            x_seq  = self.input_proj(x_cat)               # [B, S, C]

        fused = self.norm(x_seq)
        fused_out = self.vilayer(fused)

        if self.use_skip:
            fused_out = fused_out + self.residual_proj(x1_seq)

        if self.use_mlp:
            fused_out = fused_out + self.post_mlp(self.norm2(fused_out))

        return einops.rearrange(fused_out, "b (h w) c -> b c h w", h=H, w=W)
