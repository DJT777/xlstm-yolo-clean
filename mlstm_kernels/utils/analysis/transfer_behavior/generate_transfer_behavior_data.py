#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from collections.abc import Callable

import numpy as np
import torch


def generate_qkv(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim_qk: int,
    head_dim_hv: int,
    q_std: float,
    k_std: float,
    v_std: float,
) -> Callable:
    q = q_std * torch.randn(
        (batch_size, num_heads, seq_len, head_dim_qk), requires_grad=False
    )
    k = k_std * torch.randn(
        (batch_size, num_heads, seq_len, head_dim_qk), requires_grad=False
    )
    v = v_std * torch.randn(
        (batch_size, num_heads, seq_len, head_dim_hv), requires_grad=False
    )

    return q, k, v


def compute_gain_metric(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    metric_specifier: str = "abs_max_mean-v",
) -> float:
    """
    The metric specifier has the following format:
    reduce_fn-denominator

    numerator is always h
    if no denominator is specified, the metric is computed on h
    """

    # reduce functions for metrics
    def max_mean(x: torch.Tensor) -> float:
        return x.max(-1)[0].mean().item()

    def abs_max_mean(x: torch.Tensor) -> float:
        return x.abs().max(-1)[0].mean().item()

    def std_mean(x: torch.Tensor) -> float:
        return x.std(-1).mean().item()

    metric_denominator = metric_specifier.split("-")
    assert (
        len(metric_denominator) <= 2
    ), "Invalid metric specifier format. Expected format is 'reduce_fn-denominator'."

    metric_name = metric_denominator[0]

    if metric_name == "max_mean":
        metric_fn = max_mean
    elif metric_name == "abs_max_mean":
        metric_fn = abs_max_mean
    elif metric_name == "std_mean":
        metric_fn = std_mean
    else:
        raise ValueError(
            f"Unsupported metric {metric_name}. Supported metrics are 'max_mean', 'abs_max_mean', 'std_mean'."
        )

    if len(metric_denominator) == 1:
        metric_val = metric_fn(h)
    else:
        denominator = metric_denominator[1]

        if denominator == "v":
            denom_vec = v
        elif denominator == "q":
            denom_vec = q
        elif denominator == "k":
            denom_vec = k
        else:
            raise ValueError(
                f"Unsupported denominator {denominator}. Supported denominators are 'v', 'q', 'k'."
            )

        metric_val = metric_fn(h) / metric_fn(denom_vec)

    return metric_val


def make_gate_offset_sweep(
    mlstm_func: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    igate_preact_offsets: list[float],
    fgate_preact_offsets: list[float],
    igate_preact_init_fn: Callable,  # (B, NH, S)
    fgate_preact_init_fn: Callable,  # (B, NH, S)
    metric_specifier: str = "abs_max_mean-v",
    dtype=torch.float32,
    device=torch.device("cuda"),
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """
    This function runs the forward pass of a given mLSTM function with a range of input and forget gate pre-activations
    and fixed input and forget gates.

    It creates the data for a 2D heatmap of the metric of interest as a function of the input and forget gate pre-activations.
    Computes the metric for the normalized and unnormalized mLSTM outputs.
    """
    B, NH, S, _ = q.shape

    q, k, v = map(
        lambda x: x.to(dtype=dtype, device=device).requires_grad_(False), (q, k, v)
    )

    data = []
    metric_before_norm_data_array = np.zeros(
        (len(igate_preact_offsets), len(fgate_preact_offsets))
    )
    metric_after_norm_data_array = np.zeros(
        (len(igate_preact_offsets), len(fgate_preact_offsets))
    )
    for i, vecI_offset in enumerate(igate_preact_offsets):
        for j, vecF_offset in enumerate(fgate_preact_offsets):
            i_preact = vecI_offset + igate_preact_init_fn((B, NH, S))
            f_preact = vecF_offset + fgate_preact_init_fn((B, NH, S))
            i_preact, f_preact = map(
                lambda x: x.to(dtype=dtype, device=device).requires_grad_(False),
                (i_preact, f_preact),
            )

            h_normalized_unnormalized = mlstm_func(
                q=q, k=k, v=v, i=i_preact, f=f_preact
            )

            metric_after_norm, metric_before_norm = tuple(
                map(
                    lambda h: compute_gain_metric(
                        metric_specifier=metric_specifier, q=q, k=k, v=v, h=h
                    ),
                    h_normalized_unnormalized,
                )
            )

            data_val = {
                "vecI_offset": vecI_offset,
                "vecF_offset": vecF_offset,
                "metric_before_norm": metric_before_norm,
                "metric_after_norm": metric_after_norm,
            }
            data.append(data_val)
            metric_before_norm_data_array[i, j] = metric_before_norm
            metric_after_norm_data_array[i, j] = metric_after_norm

    return metric_after_norm_data_array, metric_before_norm_data_array, data
