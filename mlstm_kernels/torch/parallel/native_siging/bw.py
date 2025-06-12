#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
import torch
from torch.nn.functional import logsigmoid


def mlstm_siging_parallel_bw(
    matDeltaHtilde: torch.Tensor,
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    vecN: torch.Tensor,
    eps: float = 1e-6,
    stable_fgate: bool = True,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, NH, S, DHQK = matQ.shape
    assert matK.shape == (B, NH, S, DHQK)
    assert vecI.shape == (B, NH, S)
    assert vecF.shape == (B, NH, S)

    _dtype, _device = matQ.dtype, matQ.device

    vecLogSigF = logsigmoid(vecF)  # (B, NH, S)

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

    vecLogSigI = logsigmoid(vecI)

    matLogD = matLogSigF_mask + vecLogSigI[:, :, None, :]

    matD = torch.exp(matLogD)  # (B, NH, S, S)

    # intermediate delta-errors
    if normalize:
        matDeltaC = (
            matDeltaHtilde @ matV.transpose(-2, -1) / (vecN[:, :, :, None] + eps)
        )
    else:
        matDeltaC = matDeltaHtilde @ matV.transpose(-2, -1)

    matS = (matQ @ matK.transpose(-2, -1)) * (DHQK**-0.5)

    matDeltaDtilde = matDeltaC * matD * matS

    vecDeltaIbar = torch.sum(matDeltaDtilde, dim=-2)

    # output delta-errors / gradients
    matP = matDeltaC * matD

    matDeltaQ = (matP @ matK) * (DHQK**-0.5)
    matDeltaK = (matP.transpose(-2, -1) @ matQ) * (DHQK**-0.5)

    matCtilde = matS * matD

    if normalize:
        matDeltaV = matCtilde.transpose(-2, -1) @ (
            matDeltaHtilde / (vecN[:, :, :, None] + eps)
        )
    else:
        matDeltaV = matCtilde.transpose(-2, -1) @ matDeltaHtilde

    # compute the vecDeltaFbar values with dfbar = rev_cumsum((q*dq - k*dk).sum(-1))
    vecDeltaFbar_acc = (matQ * matDeltaQ - matK * matDeltaK).sum(-1)
    vecDeltaFbar = vecDeltaFbar_acc.flip(-1).cumsum(-1).flip(-1)
    vecDeltaF = vecDeltaFbar * torch.sigmoid(-vecF)

    vecDeltaI = vecDeltaIbar * torch.sigmoid(-vecI)

    return (
        matDeltaQ,
        matDeltaK,
        matDeltaV,
        vecDeltaI,
        vecDeltaF,
    )
