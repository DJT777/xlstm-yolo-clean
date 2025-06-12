#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

"""This module contains the normalization functions.

We do not add scale parameters here as they are initialized to 1.0.
"""

import torch


def rms_normalize(
    x: torch.Tensor, eps: float = 1e-6, force_float32_reductions: bool = True
) -> torch.Tensor:
    # x: (B, ..., S,..., D)
    # apply rms norm over the last dimension, i.e. D dimension
    in_dtype = x.dtype
    if force_float32_reductions:
        x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x.to(in_dtype)


def layer_normalize(
    x: torch.Tensor, eps: float = 1e-6, force_float32_reductions: bool = True
) -> torch.Tensor:
    # x: (B, ..., S,..., D)
    # apply layer norm over the last dimension, i.e. D dimension
    in_dtype = x.dtype
    if force_float32_reductions:
        x = x.float()
    x_centered = x - x.mean(dim=-1, keepdim=True)
    y = x_centered * torch.rsqrt(x.var(dim=-1, keepdim=True, unbiased=False) + eps)
    return y.to(in_dtype)


def no_normalize(x: torch.Tensor, **kwargs) -> torch.Tensor:
    return x


def apply_normalize(
    norm_specifier: str,
    x: torch.Tensor,
    eps: float = 1e-6,
    force_float32_reductions: bool = True,
) -> torch.Tensor:
    if norm_specifier == "rms":
        return rms_normalize(
            x=x, eps=eps, force_float32_reductions=force_float32_reductions
        )
    elif norm_specifier == "layer":
        return layer_normalize(
            x=x, eps=eps, force_float32_reductions=force_float32_reductions
        )
    elif norm_specifier == "none":
        return no_normalize(x=x)
    else:
        raise ValueError(f"Unsupported norm specifier {norm_specifier}.")
