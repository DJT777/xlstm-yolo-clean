#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

"""This module contains an interface to the different mLSTM cell functions."""

import torch

from ._mlstm_cells import mlstm_exp_stable_fgate, mlstm_sig_stable_fgate
from ._norm_layers import apply_normalize


def mlstm_cell_func(
    mlstm_func_specifier: str,
    norm_specifier: str,
    q: torch.Tensor,  # (B, NH, S, DHQK)
    k: torch.Tensor,  # (B, NH, S, DHQK)
    v: torch.Tensor,  # (B, NH, S, DHHV)
    i: torch.Tensor,  # (B, NH, S)
    f: torch.Tensor,  # (B, NH, S)
    norm_eps: float = 1e-6,
    backend_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:  # (B, NH, S, DHHV)
    """A general interface for the transfer behavior analysis.

    The mlstm_func_specifier has the following format:
        mode__namepart1--namepart2

    mode: either 'mk' for mlstem_kernels or 'tb' transfer behavior
    namepart1 and namepart2: the name of the function

    for 'mk' mode:
        namepart1: either chunkwise or parallel
        namepart2: the name of the kernel

    for 'tb' mode:
        namepart1: the name of the function
        namepart2: the normalizer mode

    Args:
        mlstm_func_specifier: the specifier for the mLSTM function.
        norm_specifier: the specifier for the normalization function

    Returns:
        hidden states (after multihead norm), unnormalized hidden states (before multihead norm)
    """

    B, NH, S, DHQK = q.shape
    DHHV = v.shape[-1]

    # we want all kernels to work seamlessly with the same interface
    # xl chunk kernels are happy with 128 and larger
    assert S >= 128, f"S must be at least 128, got {S}"

    mode_and_nameparts = mlstm_func_specifier.split("__")
    mode = mode_and_nameparts[0]
    nameparts = mode_and_nameparts[1]

    if mode == "mk":
        mlstm_func = apply_mlstm_kernels_func
    elif mode == "tb":
        mlstm_func = apply_mlstm_transfer_behavior_func
    else:
        raise ValueError(f"Unsupported mode {mode}. Supported modes are 'mk' and 'tb'.")

    h_unnormalized = mlstm_func(
        mlstm_func_specifier=nameparts, q=q, k=k, v=v, i=i, f=f, eps=backend_eps
    )
    h_normalized = apply_normalize(norm_specifier, x=h_unnormalized, eps=norm_eps)

    return h_normalized, h_unnormalized


def apply_mlstm_kernels_func(
    mlstm_func_specifier: str,
    q: torch.Tensor,  # (B, NH, S, DHQK)
    k: torch.Tensor,  # (B, NH, S, DHQK)
    v: torch.Tensor,  # (B, NH, S, DHHV)
    i: torch.Tensor,  # (B, NH, S)
    f: torch.Tensor,  # (B, NH, S)
    eps: float = 1e-6,
) -> torch.Tensor:  # (B, NH, S, DHHV)
    from ....torch import get_mlstm_kernel

    mlstm_kernel = get_mlstm_kernel(mlstm_func_specifier)
    ret = mlstm_kernel(q=q, k=k, v=v, i=i, f=f, eps=eps)

    if isinstance(ret, tuple):
        ret = ret[0]
    return ret


def apply_mlstm_transfer_behavior_func(
    mlstm_func_specifier: str,
    q: torch.Tensor,  # (B, NH, S, DHQK)
    k: torch.Tensor,  # (B, NH, S, DHQK)
    v: torch.Tensor,  # (B, NH, S, DHHV)
    i: torch.Tensor,  # (B, NH, S)
    f: torch.Tensor,  # (B, NH, S)
    eps: float = 1e-6,
) -> torch.Tensor:  # (B, NH, S, DHHV)
    mlstm_and_normalization = mlstm_func_specifier.split("--")
    mlstm_specifier = mlstm_and_normalization[0]
    normalization_mode = mlstm_and_normalization[1]

    if mlstm_specifier == "mlstmexp":
        mlstm_func = mlstm_exp_stable_fgate
    elif mlstm_specifier == "mlstmsig":
        mlstm_func = mlstm_sig_stable_fgate
    else:
        raise ValueError(f"Unsupported mLSTM function specifier {mlstm_specifier}.")

    ret = mlstm_func(
        matQ=q,
        matK=k,
        matV=v,
        vecI=i,
        vecF=f,
        eps=eps,
        normalization_mode=normalization_mode,
    )
    return ret[0]
