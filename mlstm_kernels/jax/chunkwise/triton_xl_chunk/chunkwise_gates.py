#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

"""In this file we compute the chunkwise or cumulative gates (i.e. vecA and vecB)
for the forward and backward pass of the mLSTM.
We use the stable formulations, i.e. we avoid subtraction of forget gates.
"""

import jax
import jax.numpy as jnp


def compute_chunkwise_log_gates_vecB_vecA(
    vecI: jax.Array,  # (B, NH, S)
    vecF: jax.Array,  # (B, NH, S)
    chunk_size: int,
):
    B, NH, S = vecI.shape
    assert S % chunk_size == 0, f"S={S} is not divisible by chunk_size={chunk_size}"
    NC = S // chunk_size
    L = chunk_size

    # compute vecB
    vecF_logsig: jax.Array = jax.nn.log_sigmoid(vecF.astype(jnp.float32))
    vecF_logsig_chunked = vecF_logsig.reshape(B, NH, NC, L)
    vecB = vecF_logsig_chunked.cumsum(axis=-1)

    # compute vecA
    vecI_chunked = vecI.reshape(B, NH, NC, L)
    # unstable vecA computation:
    # vecA = (vecB[..., -1, None] - vecB) + vecI  # (B, NH, NC, L)
    # stable vecA computation:
    vecF_cumsum = jnp.flip(
        jnp.flip(vecF_logsig_chunked[..., 1:], axis=-1).cumsum(-1), axis=-1
    )
    vecA = (
        jnp.concat(
            [
                vecF_cumsum,
                jnp.zeros((B, NH, NC, 1), dtype=jnp.float32),
            ],
            axis=-1,
        )
        + vecI_chunked
    )  # (B, NH, NC, L)
    return vecB, vecA


def compute_chunkwise_log_gates_vecB(
    vecF: jax.Array,  # (B, NH, S)
    chunk_size: int,
):
    B, NH, S = vecF.shape
    assert S % chunk_size == 0, f"S={S} is not divisible by chunk_size={chunk_size}"
    NC = S // chunk_size
    L = chunk_size

    # compute vecB
    vecF_logsig: jax.Array = jax.nn.log_sigmoid(vecF.astype(jnp.float32))
    vecF_logsig_chunked = vecF_logsig.reshape(B, NH, NC, L)
    vecB = vecF_logsig_chunked.cumsum(axis=-1)

    return vecB


def compute_gate_grads_vecDeltaI_vecDeltaF(
    matQ: jax.Array,
    matK: jax.Array,
    matDeltaQ: jax.Array,
    matDeltaK: jax.Array,
    vecF: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    # postprocessing: compute deltaF and deltaI gradients
    # vecF = rearrange(vecF, "b nh nc l -> b nh (nc l)")
    # compute the vecDeltaFbar values with dfbar = rev_cumsum((q*dq - k*dk).sum(-1))
    matQ = matQ.astype(jnp.float32)
    matK = matK.astype(jnp.float32)
    matDeltaQ = matDeltaQ.astype(jnp.float32)
    matDeltaK = matDeltaK.astype(jnp.float32)
    vecDeltaFbar_acc = ((matQ * matDeltaQ) - (matK * matDeltaK)).sum(-1)
    #
    # vecDeltaFbar = jnp.flip(jnp.cumsum(jnp.flip(vecDeltaFbar_acc, axis=-1).astype(jnp.float32), axis=-1), axis=-1)
    # vecDeltaF = vecDeltaFbar * jax.nn.sigmoid(-vecF)
    # align with limit_chunk kernel:
    vecDeltaFbar = jnp.flip(vecDeltaFbar_acc, axis=-1).astype(jnp.float32)
    vecDeltaFbar = jnp.flip(vecDeltaFbar.cumsum(axis=-1), axis=-1)
    vecDeltaF = vecDeltaFbar * jax.nn.sigmoid(-vecF)
    # compute deltaI
    # both are equivalent:
    # vecDeltaI = (matV * matDeltaV).sum(-1)
    vecDeltaI = (matK * matDeltaK).sum(-1)

    return vecDeltaI, vecDeltaF
