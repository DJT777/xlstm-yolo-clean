#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import torch


def compute_normalizer(
    matrix: torch.Tensor, normalization_mode: str, max_val: torch.Tensor = None
) -> torch.Tensor:
    """Compute the normalizers for the (BxNHxSxS) matrix. The normalizer is a (BxNHSx1) vector.
    'maxval' is used for the stabilized version of 'max_abs_sum_C_1' normalization mode.

    Args:
        matrix (torch.Tensor): (BxNHxSxS) matrix (sequence length x sequence length)
        normalization_mode (str): The normalization mode. One of ['sum_C', 'sum_abs_C', 'abs_sum_C', 'max_abs_sum_C_1']
        max_val (torch.Tensor): The maximum value for the stabilized version of 'max_abs_sum_C_1' normalization mode.

    Raises:
        ValueError: If normalization_mode is not one of ['sum_C', 'sum_abs_C', 'abs_sum_C', 'max_abs_sum_C_1']

    Returns:
        torch.Tensor: The normalizer (BxNHxSx1) vector.
    """
    if normalization_mode == "sum_C":
        return matrix.sum(dim=-1, keepdim=True)
    elif normalization_mode == "sum_abs_C":
        matrix_abs = torch.abs(matrix)
        return matrix_abs.sum(dim=-1, keepdim=True)
    elif normalization_mode == "abs_sum_C":
        return matrix.sum(dim=-1, keepdim=True).abs()
    elif normalization_mode == "max_abs_sum_C_1":
        mval = (
            max_val
            if max_val is not None
            else torch.tensor(1.0, dtype=matrix.dtype, device=matrix.device)
        )
        return torch.maximum(matrix.sum(dim=-1, keepdim=True).abs(), mval)
    else:
        raise ValueError(
            f"normalization_mode must be one of ['sum_C', 'sum_abs_C', 'abs_sum_C', 'max_abs_sum_C_1'], got {normalization_mode}"
        )


def qkdecmask_normalize_parallel(
    C_matrix: torch.Tensor,
    normalization_mode: str,
    normalizer_offset: float = 0.0,
    eps: float = 1e-6,
    max_val: torch.Tensor = None,
) -> torch.Tensor:
    # careful: do not do inplace operation of variables that are used in the computation graph

    normalizer = (
        compute_normalizer(
            matrix=C_matrix, normalization_mode=normalization_mode, max_val=max_val
        )
        + eps
        + normalizer_offset
    )  # (B, NH, S, 1)
    C_matrix_normalized = (
        C_matrix / normalizer
    )  # (B, NH, S, S) = (B, NH, S, S) / (B, NH, S, 1)
    return C_matrix_normalized


def qs_normalizer_recurrent(
    qz_dotproduct: torch.Tensor,
    normalization_mode: str,
    eps: float = 1e-6,
    max_val: torch.Tensor = None,
) -> torch.Tensor:
    if normalization_mode in ["sum_C"]:
        r_denom = qz_dotproduct + eps
    elif normalization_mode in ["abs_sum_C"]:
        r_denom = qz_dotproduct.abs() + eps
    elif normalization_mode in ["max_abs_sum_C_1"]:
        mval = (
            max_val
            if max_val is not None
            else torch.tensor(
                1.0, dtype=qz_dotproduct.dtype, device=qz_dotproduct.device
            )
        )
        r_denom = torch.maximum(qz_dotproduct.abs(), mval) + eps
    else:
        raise ValueError(
            f"normalization_mode must be one of ['sum_C', 'abs_sum_C'], got {normalization_mode}"
        )
    return r_denom


def build_forget_gate_matrix(
    per_timestep_fg_gate_vals: torch.Tensor,
    forget_gate_on_input_too: bool = False,
    lower_triangular_matrix: torch.Tensor = None,
    return_log_matrix: bool = False,
    per_time_step_decay_vals_in_logspace: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """ "Build a forget gate matrix from a tensor of per-timestep forget gate values.
    Used in the parallel implementation of Linear Attention or Retention like layers.
    Args:
        per_timestep_fg_gate_vals: A tensor of shape (batch_size, num_heads, context_length, 1)
            containing the forget gate values for each timestep.
        forget_gate_on_input_too: Whether to apply the forget gate to the current input too.
        lower_triangular_matrix: A tensor of shape (1, 1, context_length, context_length) or
            (context_length, context_length) or (batch_size, num_heads, context_length, context_length) containing the
            lower triangular matrix filled with ones. If None or context_length of `per_timestep_fg_gate_vals`<context_length,
            a lower triangular matrix will be created.
        return_log_matrix: Whether to return the log of the decay matrix.
        per_time_step_decay_vals_in_logspace: Whether the per_timestep_fg_gate_vals are already in log space.
        eps: A small value to avoid log(0) in the computation of the decay matrix.

    Returns:
        A tensor of shape (batch_size, num_heads, context_length, context_length) containing the decay values.

    Example:
        >>> per_timestep_fg_gate_vals = tensor([[[[2.],
                                                  [3.],
                                                  [4.],
                                                  [5.]]]])
        >>> per_timestep_fg_gate_vals.shape
        torch.Size([1, 1, 4, 1])
        >>> fg_matrix = build_forget_gate_matrix(per_timestep_fg_gate_vals, forget_gate_on_input_too=True)
        >>> fg_matrix, fg_matrix.shape
        (tensor([[[[  2.0000,   0.0000,   0.0000,   0.0000],
                   [  6.0000,   3.0000,   0.0000,   0.0000],
                   [ 24.0001,  12.0000,   4.0000,   0.0000],
                   [120.0005,  60.0002,  20.0000,   5.0000]]]]),
         torch.Size([1, 1, 4, 4]))
         >>> fg_matrix = build_forget_gate_matrix(per_timestep_fg_gate_vals, forget_gate_on_input_too=False)
         (tensor([[[[ 1.0000,  0.0000,  0.0000,  0.0000],
                    [ 3.0000,  1.0000,  0.0000,  0.0000],
                    [12.0000,  4.0000,  1.0000,  0.0000],
                    [60.0002, 20.0000,  5.0000,  1.0000]]]]),
         torch.Size([1, 1, 4, 4]))
    """
    if not per_time_step_decay_vals_in_logspace:
        assert (
            per_timestep_fg_gate_vals >= 0.0
        ).all(), f"per_timestep_fg_gate_vals must be larger than or equal to 0.0, got {per_timestep_fg_gate_vals}"
    batch_size, num_heads, context_length, _ = per_timestep_fg_gate_vals.shape
    if lower_triangular_matrix is None or context_length < lower_triangular_matrix.size(
        -1
    ):
        ltr = torch.tril(
            torch.ones(
                (context_length, context_length),
                dtype=torch.bool,
                device=per_timestep_fg_gate_vals.device,
            )
        )
    else:
        ltr = lower_triangular_matrix
    assert (
        ltr.dtype == torch.bool
    ), f"lower_triangular_matrix must be of dtype bool, got {ltr.dtype}"
    # * Slow version: loop over the context length and construct columns of the decay matrix
    # dec_ma_temp = per_timestep_fg_gate_vals.transpose(-2, -1).repeat(1, 1, context_length, 1)
    # dec_ma_temp = dec_ma_temp.log() + eps  # transform to log space
    # decay_matrix = torch.zeros(
    #     (batch_size, num_heads, context_length, context_length),
    #     dtype=per_timestep_fg_gate_vals.dtype,
    #     device=per_timestep_fg_gate_vals.device,
    # )
    # for i in range(context_length):
    #     t0 = dec_ma_temp[:, :, :, i:].cumsum(-1)
    #     # the elements we need to put into the decay matrix are on the diagonal of t0
    #     t1 = t0.diagonal(-i, dim1=-2, dim2=-1)
    #     decay_matrix[:, :, i:, i] = t1
    # decay_matrix = decay_matrix.exp()
    # decay_matrix = torch.where(ltr > 0.0, decay_matrix, ltr)
    device = per_timestep_fg_gate_vals.device
    dtype = per_timestep_fg_gate_vals.dtype
    if per_time_step_decay_vals_in_logspace:
        log_timestep_decay_vals = per_timestep_fg_gate_vals
    else:
        log_timestep_decay_vals = (
            per_timestep_fg_gate_vals + eps
        ).log()  # per_timestep_fg_gate_vals.log() + eps
    ts_cumsum = torch.cat(
        [
            torch.zeros((batch_size, num_heads, 1, 1), dtype=dtype, device=device),
            torch.cumsum(log_timestep_decay_vals, dim=-2),
        ],
        dim=-2,
    )
    # * Fast version
    rep_ts_cumsum = ts_cumsum.repeat(1, 1, 1, context_length + 1)
    ret = rep_ts_cumsum - rep_ts_cumsum.transpose(-2, -1)
    #! put exp() outside of where() to avoid overflow on upper triangular (NaNs)
    if forget_gate_on_input_too:
        decay_matrix = torch.where(ltr, ret[:, :, 1:, :-1], -float("inf"))
    else:
        decay_matrix = torch.where(ltr, ret[:, :, 1:, 1:], -float("inf"))

    if return_log_matrix:
        return decay_matrix
    else:
        return decay_matrix.exp()
