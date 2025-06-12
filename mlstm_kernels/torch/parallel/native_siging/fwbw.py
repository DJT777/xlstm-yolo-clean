#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""
PyTorch

mLSTM sigmoid input gate forward and backward pass. Parallel formulation.
"""

from collections.abc import Callable

import torch
from torch.amp import custom_bwd, custom_fwd

from ...utils import contiguous
from .bw import mlstm_siging_parallel_bw
from .fw import mlstm_siging_parallel_fw


def _mlstm_siging_parallel_fwbw_generator(
    autocast_kernel_dtype=torch.float32, stable_fgate=True, normalize=True
) -> Callable:
    class _mlstm_parallel_fwbw(torch.autograd.Function):
        @staticmethod
        @custom_fwd(device_type="cuda", cast_inputs=autocast_kernel_dtype)
        @contiguous
        def forward(
            ctx,
            matQ: torch.Tensor,
            matK: torch.Tensor,
            matV: torch.Tensor,
            vecI: torch.Tensor,
            vecF: torch.Tensor,
            eps: float = 1e-6,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            matH, vecN = mlstm_siging_parallel_fw(
                matQ=matQ,
                matK=matK,
                matV=matV,
                vecI=vecI,
                vecF=vecF,
                eps=eps,
                stable_fgate=stable_fgate,
                normalize=normalize,
            )
            ctx.save_for_backward(matQ, matK, matV, vecI, vecF, vecN, torch.tensor(eps))
            return matH, vecN

        @staticmethod
        @custom_bwd(device_type="cuda")
        @contiguous
        def backward(
            ctx,
            matDeltaHtilde: torch.Tensor,
            vecDeltaN_unused: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            (matQ, matK, matV, vecI, vecF, vecN, eps) = ctx.saved_tensors
            matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF = (
                mlstm_siging_parallel_bw(
                    matDeltaHtilde=matDeltaHtilde,
                    matQ=matQ,
                    matK=matK,
                    matV=matV,
                    vecI=vecI,
                    vecF=vecF,
                    vecN=vecN,
                    eps=float(eps),
                    stable_fgate=stable_fgate,
                    normalize=normalize,
                )
            )
            return matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF, None

    return _mlstm_parallel_fwbw


def _get_parallel_fwbw_kernel(
    autocast_kernel_dtype: torch.dtype, stable_fgate: bool, normalize: bool
) -> Callable:
    if autocast_kernel_dtype == torch.float32:
        return _mlstm_siging_parallel_fwbw_generator(
            autocast_kernel_dtype=torch.float32,
            stable_fgate=stable_fgate,
            normalize=normalize,
        )
    elif autocast_kernel_dtype == torch.float16:
        return _mlstm_siging_parallel_fwbw_generator(
            autocast_kernel_dtype=torch.float16,
            stable_fgate=stable_fgate,
            normalize=normalize,
        )
    elif autocast_kernel_dtype == torch.bfloat16:
        return _mlstm_siging_parallel_fwbw_generator(
            autocast_kernel_dtype=torch.bfloat16,
            stable_fgate=stable_fgate,
            normalize=normalize,
        )
    else:
        raise ValueError(f"Unsupported autocast_kernel_dtype: {autocast_kernel_dtype}")


def mlstm_siging_parallel__native_autograd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    c_initial: torch.Tensor = None,
    n_initial: torch.Tensor = None,
    m_initial: torch.Tensor = None,
    return_last_states: bool = False,
    eps: float = 1e-6,
    stable_fgate: bool = True,
    normalize: bool = True,
    **kwargs,
) -> torch.Tensor:
    assert c_initial is None, "c_initial is not supported"
    assert n_initial is None, "n_initial is not supported"
    assert m_initial is None, "m_initial is not supported"
    assert not return_last_states, "return_last_states is not supported"

    matH, _ = mlstm_siging_parallel_fw(
        matQ=q,
        matK=k,
        matV=v,
        vecI=i,
        vecF=f,
        eps=eps,
        stable_fgate=stable_fgate,
        normalize=normalize,
    )
    return matH


def mlstm_siging_parallel__native_custbw(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    c_initial: torch.Tensor = None,
    n_initial: torch.Tensor = None,
    m_initial: torch.Tensor = None,
    return_last_states: bool = False,
    eps: float = 1e-6,
    stable_fgate: bool = True,
    normalize: bool = True,
    autocast_kernel_dtype: torch.dtype = torch.float32,
    **kwargs,
) -> torch.Tensor:
    assert c_initial is None, "c_initial is not supported"
    assert n_initial is None, "n_initial is not supported"
    assert m_initial is None, "m_initial is not supported"
    assert return_last_states is False, "return_last_states is not supported"

    _mlstm_parallel_fwbw = _get_parallel_fwbw_kernel(
        autocast_kernel_dtype=autocast_kernel_dtype,
        stable_fgate=stable_fgate,
        normalize=normalize,
    )

    matH, _ = _mlstm_parallel_fwbw.apply(q, k, v, i, f, eps)
    return matH
