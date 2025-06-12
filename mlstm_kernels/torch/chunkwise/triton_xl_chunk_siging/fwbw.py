#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
from collections.abc import Callable

import torch
from torch.amp import custom_bwd, custom_fwd

from ...utils import contiguous, int_or_none, tensor_or_none
from .bw import mlstm_siging_chunkwise_bw
from .fw import mlstm_siging_chunkwise_fw


## PyTorch Autograd Function - Boilerplate
def _mlstm_siging_chunkwise_fwbw_generator(
    autocast_kernel_dtype: torch.dtype = torch.bfloat16, normalize: bool = True
) -> Callable:
    class _mlstm_chunkwise_fwbw(torch.autograd.Function):
        @staticmethod
        @custom_fwd(device_type="cuda", cast_inputs=autocast_kernel_dtype)
        @contiguous
        def forward(
            ctx,
            matQ: torch.Tensor,  # (B, NH, S, DHQK)
            matK: torch.Tensor,  # (B, NH, S, DHQK)
            matV: torch.Tensor,  # (B, NH, S, DHV)
            vecI: torch.Tensor,  # (B, NH, S)
            vecF: torch.Tensor,  # (B, NH, S)
            matC_initial: torch.Tensor = None,  # (B, NH, DHQK, DHV)
            vecN_initial: torch.Tensor = None,  # (B, NH, DHQK)
            qk_scale: float = None,
            return_last_states: bool = False,
            eps: float = 0.0,
            chunk_size: int = 128,
            chunk_size_inter: int | None = None,
            chunk_size_intra: int | None = None,
            siz_b_L_parallel: int | None = None,
            siz_b_L_loop: int | None = None,
            siz_b_DH_parallel: int | None = None,
            siz_b_DH_loop: int | None = None,
            num_warps_intra: int | None = None,
            num_warps_inter: int | None = None,
            num_stages_intra: int | None = None,
            num_stages_inter: int | None = None,
            recompute_states_in_bw: bool = True,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            B, NH, S, DHQK = matQ.shape
            if qk_scale is None:
                qk_scale = DHQK**-0.5

            matH_out, vecN_out, last_states, all_states = mlstm_siging_chunkwise_fw(
                matQ=matQ,
                matK=matK,
                matV=matV,
                vecI=vecI,
                vecF=vecF,
                matC_initial=matC_initial,
                vecN_initial=vecN_initial,
                qk_scale=qk_scale,
                return_last_states=return_last_states,
                return_all_states=(not recompute_states_in_bw),
                normalize=normalize,
                chunk_size=chunk_size,
                chunk_size_inter=chunk_size_inter,
                chunk_size_intra=chunk_size_intra,
                siz_b_L_parallel=siz_b_L_parallel,
                siz_b_L_loop=siz_b_L_loop,
                siz_b_DH_parallel=siz_b_DH_parallel,
                siz_b_DH_loop=siz_b_DH_loop,
                num_warps_intra=num_warps_intra,
                num_warps_inter=num_warps_inter,
                num_stages_intra=num_stages_intra,
                num_stages_inter=num_stages_inter,
                output_dtype=matQ.dtype,
                eps=eps,
            )

            if return_last_states:
                (matC_last, vecN_last) = last_states
            else:
                (matC_last, vecN_last) = (None, None)

            if all_states is not None:
                matC_all, vecN_all = all_states
            else:
                matC_all, vecN_all = (None, None)

            ctx.save_for_backward(
                matQ,
                matK,
                matV,
                vecI,
                vecF,
                matC_initial,
                vecN_initial,
                matC_all,
                vecN_all,
                vecN_out,
                torch.tensor(qk_scale),
                torch.tensor(chunk_size),
                tensor_or_none(chunk_size_inter),
                tensor_or_none(chunk_size_intra),
                tensor_or_none(siz_b_L_parallel),
                tensor_or_none(siz_b_L_loop),
                tensor_or_none(siz_b_DH_parallel),
                tensor_or_none(siz_b_DH_loop),
                tensor_or_none(num_warps_intra),
                tensor_or_none(num_warps_inter),
                tensor_or_none(num_stages_intra),
                tensor_or_none(num_stages_inter),
                torch.tensor(eps),
            )
            return matH_out, matC_last, vecN_last

        @staticmethod
        @custom_bwd(device_type="cuda")
        @contiguous
        def backward(ctx, matDeltaH_out, matDeltaC_last, vecDeltaN_last):
            (
                matQ,
                matK,
                matV,
                vecI,
                vecF,
                matC_initial,
                vecN_initial,
                matC_all,
                vecN_all,
                vecN_out,
                qk_scale,
                chunk_size,
                chunk_size_inter,
                chunk_size_intra,
                siz_b_L_parallel,
                siz_b_L_loop,
                siz_b_DH_parallel,
                siz_b_DH_loop,
                num_warps_intra,
                num_warps_inter,
                num_stages_intra,
                num_stages_inter,
                eps,
            ) = ctx.saved_tensors

            (
                matDeltaQ,
                matDeltaK,
                matDeltaV,
                vecDeltaI,
                vecDeltaF,
                matDeltaC_initial,
                vecDeltaN_initial,
            ) = mlstm_siging_chunkwise_bw(
                matQ=matQ,
                matK=matK,
                matV=matV,
                vecI=vecI,
                vecF=vecF,
                matC_initial=matC_initial,
                vecN_initial=vecN_initial,
                matCstate_all=matC_all,
                vecNstate_all=vecN_all,
                vecN_out=vecN_out,
                matDeltaH_out=matDeltaH_out,
                matDeltaC_last=matDeltaC_last,
                qk_scale=float(qk_scale),
                normalize=normalize,
                chunk_size=int(chunk_size),
                chunk_size_inter=int_or_none(chunk_size_inter),
                chunk_size_intra=int_or_none(chunk_size_intra),
                siz_b_L_parallel=int_or_none(siz_b_L_parallel),
                siz_b_L_loop=int_or_none(siz_b_L_loop),
                siz_b_DH_parallel=int_or_none(siz_b_DH_parallel),
                siz_b_DH_loop=int_or_none(siz_b_DH_loop),
                num_warps_intra=int_or_none(num_warps_intra),
                num_warps_inter=int_or_none(num_warps_inter),
                num_stages_intra=int_or_none(num_stages_intra),
                num_stages_inter=int_or_none(num_stages_inter),
                eps=float(eps),
            )

            return (
                matDeltaQ,
                matDeltaK,
                matDeltaV,
                vecDeltaI,
                vecDeltaF,
                matDeltaC_initial,
                vecDeltaN_initial,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    return _mlstm_chunkwise_fwbw


def mlstm_siging_chunkwise__xl_chunk(
    q: torch.Tensor,  # (B, NH, S, DHQK)
    k: torch.Tensor,  # (B, NH, S, DHQK)
    v: torch.Tensor,  # (B, NH, S, DHHV)
    i: torch.Tensor,  # (B, NH, S)
    f: torch.Tensor,  # (B, NH, S)
    c_initial: torch.Tensor = None,  # (B, NH, DHQK, DHHV)
    n_initial: torch.Tensor = None,  # (B, NH, DHQK)
    return_last_states: bool = False,
    eps: float = 1e-6,
    normalize: bool = True,
    chunk_size: int = 128,
    chunk_size_inter: int | None = None,
    chunk_size_intra: int | None = None,
    siz_b_L_parallel: int | None = None,
    siz_b_L_loop: int | None = None,
    siz_b_DH_parallel: int | None = None,
    siz_b_DH_loop: int | None = None,
    num_warps_intra: int | None = None,
    num_warps_inter: int | None = None,
    num_stages_intra: int | None = None,
    num_stages_inter: int | None = None,
    recompute_states_in_bw: bool = True,
    autocast_kernel_dtype: torch.dtype = torch.float32,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    if autocast_kernel_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"Unsupported kernel dtype {autocast_kernel_dtype}.")
    _mlstm_siging_chunkwise_fwbw = _mlstm_siging_chunkwise_fwbw_generator(
        autocast_kernel_dtype=autocast_kernel_dtype, normalize=normalize
    )
    matH_out, matC_last, vecN_last = _mlstm_siging_chunkwise_fwbw.apply(
        q,
        k,
        v,
        i,
        f,
        c_initial,
        n_initial,
        None,  # qk_scale always the default value
        return_last_states,
        eps,
        chunk_size,
        chunk_size_inter,
        chunk_size_intra,
        siz_b_L_parallel,
        siz_b_L_loop,
        siz_b_DH_parallel,
        siz_b_DH_loop,
        num_warps_intra,
        num_warps_inter,
        num_stages_intra,
        num_stages_inter,
        recompute_states_in_bw,
    )
    if return_last_states:
        return matH_out, (matC_last, vecN_last)
    else:
        return matH_out
