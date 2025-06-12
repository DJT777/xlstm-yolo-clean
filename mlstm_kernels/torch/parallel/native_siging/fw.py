#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""
PyTorch

mLSTM sigmoid input gate forward pass. Parallel formulation.
"""

import torch
import torch.nn.functional as F


def mlstm_siging_parallel_fw(
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    eps: float = 1e-6,
    stable_fgate: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    B, NH, S, DHQK = matQ.shape
    assert matK.shape == (B, NH, S, DHQK)
    assert vecI.shape == (B, NH, S)
    assert vecF.shape == (B, NH, S)

    _dtype, _device = matQ.dtype, matQ.device

    vecLogSigF = F.logsigmoid(vecF)  # (B, NH, S)

    if stable_fgate:
        matLogSigF_tril = vecLogSigF[:, :, :, None].repeat(1, 1, 1, S).tril(-1)
        matLogSigF = matLogSigF_tril.cumsum(-2)
    else:
        vecLogSigF_cumsum = vecLogSigF.cumsum(-1)
        matLogSigF = vecLogSigF_cumsum[:, :, :, None] - vecLogSigF_cumsum[:, :, None, :]

    ltr = torch.tril(
        torch.ones(
            (S, S),
            dtype=torch.bool,
            device=_device,
        )
    )

    matLogSigF_mask = torch.where(ltr, matLogSigF, -float("inf"))

    vecLogSigI = F.logsigmoid(vecI)

    matLogD = matLogSigF_mask + vecLogSigI[:, :, None, :]

    matD = torch.exp(matLogD)  # (B, NH, S, S)

    matS = (matQ @ matK.transpose(-2, -1)) * (DHQK**-0.5)  # (B, NH, S, S)

    matCtilde = matS * matD  # (B, NH, S, S)
    if normalize:
        vecN = torch.maximum(
            matCtilde.sum(dim=-1, keepdim=True).abs(),
            torch.tensor([1.0], dtype=_dtype, device=_device),
        )  # (B, NH, S, 1)
        # (B, NH, S, S)
        matC = matCtilde / (vecN + eps)
        vecN = vecN.squeeze(-1)
    else:
        matC = matCtilde
        vecN = None

    matH = matC @ matV  # (B, NH, S, DH)

    return (matH, vecN)
