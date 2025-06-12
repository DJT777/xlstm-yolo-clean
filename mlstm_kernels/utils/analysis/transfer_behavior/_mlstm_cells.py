#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

"""This module contains experimental mLSTM cells designed solely for the
transfer behavior analysis.
"""

import torch


def _compute_vecN(matCtilde: torch.Tensor, normalization_mode: str) -> torch.Tensor:
    _dtype, _device = matCtilde.dtype, matCtilde.device
    if "max_sum_abs_1" in normalization_mode:
        split = normalization_mode.split("-")
        if len(split) == 2:
            denom_const = float(split[1])
        else:
            denom_const = 1.0
        vecN = torch.maximum(
            matCtilde.sum(dim=-1, keepdim=True).abs(),
            torch.tensor([denom_const], device=_device, dtype=_dtype),
        )  # (B, NH, S, 1)
    elif normalization_mode == "sum_abs":
        vecN = matCtilde.sum(dim=-1, keepdim=True).abs()

    elif normalization_mode == "sum":
        vecN = matCtilde.sum(dim=-1, keepdim=True)
    elif "denom_one" in normalization_mode:
        split = normalization_mode.split("-")
        if len(split) == 2:
            denom_const = float(split[1])
        else:
            denom_const = 1.0
        vecN = torch.tensor([float(denom_const)], device=_device, dtype=_dtype)
    else:
        raise ValueError(f"mstate_mode {normalization_mode} not recognized")

    return vecN


def mlstm_exp_stable_fgate(
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    eps: float = 1e-6,
    normalization_mode: str = "paper",
) -> torch.Tensor:
    import math

    B, NH, S, DHQK = matQ.shape
    assert matK.shape == (B, NH, S, DHQK)
    assert vecI.shape == (B, NH, S)
    assert vecF.shape == (B, NH, S)

    _dtype, _device = matQ.dtype, matQ.device

    vecLogSigF = torch.nn.functional.logsigmoid(vecF)  # (B, NH, S)

    matLogSigF_tril = vecLogSigF[:, :, :, None].repeat(1, 1, 1, S).tril(-1)
    matLogSigF_cum = matLogSigF_tril.cumsum(-2)

    ltr = torch.tril(
        torch.ones(
            (S, S),
            dtype=torch.bool,
            device=_device,
        )
    )

    matLogSigF_mask = torch.where(ltr, matLogSigF_cum, -float("inf"))

    matLogD = matLogSigF_mask + vecI[:, :, None, :]

    vecM, _ = torch.max(matLogD, dim=-1, keepdim=True)  # (B, NH, S, 1)
    matLogD_stabilized = matLogD - vecM

    matD = torch.exp(matLogD_stabilized)  # (B, NH, S, S)

    matS = (matQ @ matK.transpose(-2, -1)) / math.sqrt(DHQK)  # (B, NH, S, S)

    matCtilde = matS * matD  # (B, NH, S, S)

    if normalization_mode == "paper":
        vecN = torch.maximum(
            matCtilde.sum(dim=-1, keepdim=True).abs(), torch.exp(-vecM)
        )  # (B, NH, S, 1)
    else:
        vecN = _compute_vecN(matCtilde, normalization_mode)

    # (B, NH, S, S)
    matC = matCtilde / (vecN + eps)

    matH = matC @ matV  # (B, NH, S, DH)

    return (
        matH,
        vecM.squeeze(-1),
        vecN.squeeze(-1),
        matLogD,
        matLogD_stabilized,
        matD,
        matCtilde,
        matC,
        vecLogSigF,
    )


def mlstm_sig_stable_fgate(
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    eps: float = 1e-6,
    normalization_mode: str = "paper",  # "sum_only", "abs_sum", "max_abs_sum_1", "denom_one--1.0"
):
    import math

    B, NH, S, DHQK = matQ.shape
    assert matK.shape == (B, NH, S, DHQK)
    assert vecI.shape == (B, NH, S)
    assert vecF.shape == (B, NH, S)

    _dtype, _device = matQ.dtype, matQ.device

    vecLogSigF = torch.nn.functional.logsigmoid(vecF)  # (B, NH, S)
    matLogSigF_tril = vecLogSigF[:, :, :, None].repeat(1, 1, 1, S).tril(-1)
    matLogSigF_cum = matLogSigF_tril.cumsum(-2)

    ltr = torch.tril(
        torch.ones(
            (S, S),
            dtype=torch.bool,
            device=_device,
        )
    )

    matLogSigF_mask = torch.where(ltr, matLogSigF_cum, -float("inf"))

    # input gates
    vecLogIGate = torch.nn.functional.logsigmoid(vecI)

    matLogD = matLogSigF_mask + vecLogIGate[:, :, None, :]

    matD = torch.exp(matLogD)  # (B, NH, S, S)

    matS = (matQ @ matK.transpose(-2, -1)) / math.sqrt(DHQK)  # (B, NH, S, S)

    matCtilde = matS * matD  # (B, NH, S, S)

    if normalization_mode == "paper":
        # (B, NH, S, S)
        matC = matCtilde
        vecN = torch.zeros_like(vecI)
    else:
        vecN = _compute_vecN(matCtilde, normalization_mode)
        # (B, NH, S, S)
        matC = matCtilde / (vecN + eps)

    matH = matC @ matV  # (B, NH, S, DH)

    return (
        matH,
        torch.zeros_like(vecI),
        vecN.squeeze(-1),
        matLogD,
        torch.zeros_like(matLogD),
        matD,
        matCtilde,
        matC,
        vecLogSigF,
    )
