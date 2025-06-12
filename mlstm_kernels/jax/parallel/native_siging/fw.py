#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""
Jax.

mLSTM sigmoid input gate forward pass. Parallel formulation.
"""

import jax
import jax.numpy as jnp


def mlstm_siging_parallel_fw(
    matQ: jax.Array,
    matK: jax.Array,
    matV: jax.Array,
    vecI: jax.Array,
    vecF: jax.Array,
    eps: float = 1e-6,
    stable_fgate: bool = True,
    normalize: bool = True,
) -> jax.Array:
    B, NH, S, DHQK = matQ.shape
    assert matK.shape == (B, NH, S, DHQK)
    assert vecI.shape == (B, NH, S)
    assert vecF.shape == (B, NH, S)

    vecLogSigF = jax.nn.log_sigmoid(vecF)  # (B, NH, S)

    if stable_fgate:
        matLogSigF_tril = jnp.tril(vecLogSigF[:, :, :, None].repeat(S, axis=-1), k=-1)
        matLogSigF = jnp.cumsum(matLogSigF_tril, axis=-2)
    else:
        vecLogSigF_cumsum = jnp.cumsum(vecLogSigF, axis=-1)
        matLogSigF = vecLogSigF_cumsum[:, :, :, None] - vecLogSigF_cumsum[:, :, None, :]

    ltr = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))

    matLogSigF_mask = jnp.where(ltr, matLogSigF, -float("inf"))

    vecLogSigI = jax.nn.log_sigmoid(vecI)

    matLogD = matLogSigF_mask + vecLogSigI[:, :, None, :]

    matD = jnp.exp(matLogD)  # (B, NH, S, S)

    matS = (matQ @ matK.swapaxes(-2, -1)) * (DHQK**-0.5)  # (B, NH, S, S)

    matCtilde = matS * matD  # (B, NH, S, S)
    if normalize:
        vecN = jnp.maximum(
            jnp.abs(jnp.sum(matCtilde, axis=-1, keepdims=True)),
            jnp.array([1.0]),
        )  # (B, NH, S, 1)
        # (B, NH, S, S)
        matC = matCtilde / (vecN + eps)
        vecN = vecN.squeeze(-1)
    else:
        matC = matCtilde
        vecN = None

    matH = matC @ matV  # (B, NH, S, DH)

    return (matH, vecN)
