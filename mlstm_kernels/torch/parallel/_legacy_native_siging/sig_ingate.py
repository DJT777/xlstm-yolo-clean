#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import torch

from .ops import (
    build_forget_gate_matrix,
    qkdecmask_normalize_parallel,
)


def mlstm_siging_parallel(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    igate_preact: torch.Tensor,
    fgate_preact: torch.Tensor,
    lower_triangular_matrix: torch.Tensor = None,
    qk_decmask_normalize: bool = True,
    normalization_mode: str = "max_abs_sum_C_1",
    normalize_sqrt_d: bool = False,  # in order to match with new implementation
    normalizer_offset: float = 0.0,
    eps: float = 1e-6,
    **kwargs,
):
    """This is the core linear hopfield retrieval operation in parallel form.
    It has sigmoid input gates instead of exponential ones.

    Args:
        queries (torch.Tensor): (B, NH, S, DH)
        keys (torch.Tensor): (B, NH, S, DH)
        values (torch.Tensor): (B, NH, S, DH)
        igate_preact (torch.Tensor): (B, NH, S, 1)
        fgate_preact (torch.Tensor): (B, NH, S, 1)
        lower_triangular_matrix (torch.Tensor, optional): (S,S). Defaults to None.
        qk_decmask_normalize (bool, optional): Wether to normalize the combination matrix C. Defaults to True.
        normalization_mode (str, optional): Normalization mode for the combination matrix C. Defaults to "sum_C".
        normalize_sqrt_d (bool, optional): Wether to normalize the combination matrix C by the sqrt of the qk head dimension.
                                           Originally, this was not present. In the new implementation we add this in order to
                                           match with the exponential input gate version. Defaults to False.
        normalizer_offset (float, optional): Offset for the normalizer. This number is added to the denominator.
                                             Defaults to 0.0.
        eps (float, optional): Used for building the forgetgate matrix. Defaults to 1e-6.

    Returns:
        torch.Tensor: (B, NH, S, DH), retrieved values
    """
    B, NH, S, DHQK = queries.shape

    # forget gate matrix
    fgates = torch.nn.functional.logsigmoid(fgate_preact)  # (B, NH, S, 1)
    fg_matrix = build_forget_gate_matrix(
        per_timestep_fg_gate_vals=fgates,
        per_time_step_decay_vals_in_logspace=True,
        return_log_matrix=True,
        lower_triangular_matrix=lower_triangular_matrix,
        eps=0.0,
    )  # (B, NH, S, S)

    # input gates
    igates = torch.nn.functional.logsigmoid(igate_preact)  # (B, NH, S, 1)
    # gate decay matrix D
    log_D_matrix = fg_matrix + igates.transpose(-2, -1)  # (B, NH, S, S)
    D_matrix = torch.exp(log_D_matrix)  # (B, NH, S, S)
    # combination matrix C
    qk_matrix = queries @ keys.transpose(-2, -1)  # (B, NH, S, S)
    C_matrix = qk_matrix * D_matrix  # (B, NH, S, S)
    if normalize_sqrt_d:
        C_matrix = C_matrix * (DHQK**-0.5)
    if qk_decmask_normalize:
        # (B, NH, S, S)
        C_matrix_normalized = qkdecmask_normalize_parallel(
            C_matrix=C_matrix,
            normalization_mode=normalization_mode,
            normalizer_offset=normalizer_offset,
            eps=eps,
        )
    else:
        C_matrix_normalized = C_matrix

    # retrieved values
    retrieved_values = C_matrix_normalized @ values  # (B, NH, S, DH)
    return retrieved_values
