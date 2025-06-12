#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

# Maximilian Beck
"""This file contains the parallel part of the forward pass of the mLSTM chunkwise formulation.
It is adapted such that it allows for arbitrary large chunk sizes AND head dimensions.
"""

import triton
import triton.language as tl


@triton.jit
def mlstm_siging_chunkwise__parallel_fw_Hintra_kernel(
    matQ,  # (B, NH, S, DHQK)
    matK,  # (B, NH, S, DHQK)
    matV,  # (B, NH, S, DHHV)
    # these are all the states at every chunk, (we only use NC states up to the last chunk, i.e. :-1)
    matC_states,  # (B, NH, (NC+1) * DHQK, DHHV)
    vecN_states,  # (B, NH, (NC+1) * DHQK)
    vecI,  # (B, NH, NC, L)
    vecB,  # (B, NH, NC, L)
    matHout,  # (B, NH, S, DHHV)
    vecNout,  # (B, NH, S)
    qk_scale: tl.constexpr,
    str_matQK_B_NH: tl.constexpr,
    str_matQK_S: tl.constexpr,
    str_matQK_DHQK: tl.constexpr,
    str_matHV_B_NH: tl.constexpr,
    str_matHV_S: tl.constexpr,
    str_matHV_DHHV: tl.constexpr,
    str_matCstates_B_NH: tl.constexpr,
    str_matCstates_NCDHQK: tl.constexpr,
    str_matCstates_DHHV: tl.constexpr,
    str_vecNstates_B_NH: tl.constexpr,
    str_vecNstates_NCDHQK: tl.constexpr,
    str_vecBI_B_NH: tl.constexpr,
    str_vecBI_NC: tl.constexpr,
    str_vecBI_L: tl.constexpr,
    str_vecN_B_NH: tl.constexpr,
    str_vecN_S: tl.constexpr,
    B: tl.constexpr,
    NH: tl.constexpr,
    S: tl.constexpr,
    DHQK: tl.constexpr,
    DHHV: tl.constexpr,
    NC: tl.constexpr,
    L: tl.constexpr,
    siz_b_LQ: tl.constexpr,
    siz_b_LKV: tl.constexpr,
    siz_b_DHQK: tl.constexpr,
    siz_b_DHHV: tl.constexpr,
    NORMALIZE: tl.constexpr = True,
    DTYPE: tl.constexpr = tl.float32,
    OUTPUT_DTYPE: tl.constexpr = tl.float32,
    EPS: tl.constexpr = 0.0,
):
    # our grid has 4 dimensions: (num_b_DHHV, num_b_LQ, (NC, B * NH))
    idx_b_DHHV, idx_b_LQ, idx_b_NC_BNH = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    idx_b_NC = idx_b_NC_BNH % NC
    idx_b_BNH = idx_b_NC_BNH // NC

    # gate pointers for the current thread block
    vecB_ptr = vecB + idx_b_BNH * str_vecBI_B_NH + idx_b_NC * str_vecBI_NC
    vecI_ptr = vecI + idx_b_BNH * str_vecBI_B_NH + idx_b_NC * str_vecBI_NC

    # load vecB_LQ (siz_b_LQ,)
    vecB_LQ_ptr = vecB_ptr + idx_b_LQ * siz_b_LQ + tl.arange(0, siz_b_LQ)
    vecB_LQ_val = tl.load(vecB_LQ_ptr).to(tl.float32)

    # compute vecBbar (siz_b_LQ,)
    vecBbar_val = tl.exp(vecB_LQ_val)  # for inter chunk contribution

    # for causal masking
    b_q_offset = idx_b_LQ * siz_b_LQ
    b_q_idxes = b_q_offset + tl.arange(0, siz_b_LQ)

    # ? compute the intra chunk contribution
    # loop over b_LKV blocks
    # initialize accumulators
    # TODO maybe use only one accumulator for matH and vecN, respectively
    # matH accumulators (siz_b_LQ, siz_b_DHHV_threadblock)
    matH_intra_acc = tl.zeros([siz_b_LQ, siz_b_DHHV], dtype=tl.float32)
    matH_inter_acc = tl.zeros([siz_b_LQ, siz_b_DHHV], dtype=tl.float32)

    # vecN accumulators (siz_b_LQ,)
    if NORMALIZE:
        vecN_inter_acc = tl.zeros([siz_b_LQ], dtype=tl.float32)
        vecN_intra_acc = tl.zeros([siz_b_LQ], dtype=tl.float32)
    # only compute the lower triangular part
    idx_b_LKV_end = ((idx_b_LQ + 1) * siz_b_LQ) // siz_b_LKV
    for idx_b_LKV in range(idx_b_LKV_end):
        # compute matG = matQ @ matK^T
        # matG accumulator (siz_b_LQ, siz_b_LKV)
        matG = tl.zeros([siz_b_LQ, siz_b_LKV], dtype=tl.float32)
        ## loop over DHQK blocks
        for idx_b_DHQK in range(tl.cdiv(DHQK, siz_b_DHQK)):
            # load matQ block (siz_b_LQ, siz_b_DHQK)
            matQ_ptr = tl.make_block_ptr(
                base=matQ + idx_b_BNH * str_matQK_B_NH,
                shape=(S, DHQK),
                strides=(str_matQK_S, str_matQK_DHQK),
                offsets=(idx_b_NC * L + idx_b_LQ * siz_b_LQ, idx_b_DHQK * siz_b_DHQK),
                block_shape=(siz_b_LQ, siz_b_DHQK),
                order=(1, 0),
            )
            matQ_val = tl.load(matQ_ptr, boundary_check=(0, 1))
            # load matK transposed block (siz_b_DHQK, siz_b_LKV)
            matK_ptr = tl.make_block_ptr(
                base=matK + idx_b_BNH * str_matQK_B_NH,
                shape=(DHQK, S),
                strides=(str_matQK_DHQK, str_matQK_S),
                offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_NC * L + idx_b_LKV * siz_b_LKV),
                block_shape=(siz_b_DHQK, siz_b_LKV),
                order=(0, 1),
            )
            matK_val = tl.load(matK_ptr, boundary_check=(0, 1)).to(DTYPE)

            # accumulate in matG (siz_b_LQ, siz_b_LKV)
            matG += tl.dot(matQ_val.to(DTYPE), matK_val)

            ### ? compute the  inter chunk contribution
            # compute this only once on the first iteration
            if idx_b_LKV == 0:
                # load matC_km1 (siz_b_DHQK, siz_b_DHHV)
                matC_km1_ptr = tl.make_block_ptr(
                    base=matC_states
                    + idx_b_BNH * str_matCstates_B_NH
                    + idx_b_NC * DHQK * DHHV,
                    shape=(DHQK, DHHV),
                    strides=(str_matCstates_NCDHQK, str_matCstates_DHHV),
                    offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
                    block_shape=(siz_b_DHQK, siz_b_DHHV),
                    order=(1, 0),
                )
                matC_km1_val = tl.load(matC_km1_ptr, boundary_check=(0, 1)).to(DTYPE)

                matQbar_val = matQ_val.to(tl.float32) * vecBbar_val[:, None] * qk_scale

                # acccumulate matH_inter (siz_b_LQ, siz_b_DHHV)
                matH_inter_acc += tl.dot(matQbar_val.to(DTYPE), matC_km1_val)

                if NORMALIZE:
                    # load vecN_km1 (siz_b_DHQK,)
                    vecN_km1_ptr = (
                        vecN_states
                        + idx_b_BNH * str_vecNstates_B_NH
                        + idx_b_NC * DHQK
                        + idx_b_DHQK * siz_b_DHQK
                        + tl.arange(0, siz_b_DHQK)
                    )
                    vecN_km1_val = tl.load(vecN_km1_ptr).to(tl.float32)

                    # accumulate vecN_inter (siz_b_LQ,1) = matQbar (siz_b_LQ, siz_b_DHQK) @ vecN_km1 (siz_b_DHQK,1)
                    vecN_inter_acc += tl.sum(
                        matQbar_val * vecN_km1_val[None, :], axis=1
                    )

        # load vecB_LKV (siz_B_LKV,)
        vecB_LKV_ptr = vecB_ptr + idx_b_LKV * siz_b_LKV + tl.arange(0, siz_b_LKV)
        vecB_LKV = tl.load(vecB_LKV_ptr).to(tl.float32)

        # load vecI_LKV (siz_B_LKV,)
        vecI_LKV_ptr = vecI_ptr + idx_b_LKV * siz_b_LKV + tl.arange(0, siz_b_LKV)
        vecI_LKV = tl.load(vecI_LKV_ptr).to(tl.float32)
        vecIlogsig_LKV = tl.log(tl.sigmoid(vecI_LKV))

        # construct gate matrix matDtilde (siz_b_LQ, siz_b_LKV)
        matDtilde_val = (
            vecB_LQ_val[:, None] - vecB_LKV[None, :] + vecIlogsig_LKV[None, :]
        )

        b_kv_offset = idx_b_LKV * siz_b_LKV
        # causal masking if on the diagonal
        if b_kv_offset >= b_q_offset:
            b_kv_idxes = b_kv_offset + tl.arange(0, siz_b_LKV)
            mask = b_q_idxes[:, None] >= b_kv_idxes[None, :]
            matDtilde_val = tl.where(mask, matDtilde_val, -float("inf"))

        # compute matD (siz_b_LQ, siz_b_LKV)
        matD_val = tl.exp(matDtilde_val)
        # tl.device_print("matD_val", matD_val)
        # compute matS (siz_b_LQ, siz_b_LKV)
        matS = matG * qk_scale * matD_val

        # load matV (siz_b_LKV, siz_b_DHHV)
        matV_ptr = tl.make_block_ptr(
            base=matV + idx_b_BNH * str_matHV_B_NH,
            shape=(S, DHHV),
            strides=(str_matHV_S, str_matHV_DHHV),
            offsets=(idx_b_NC * L + idx_b_LKV * siz_b_LKV, idx_b_DHHV * siz_b_DHHV),
            block_shape=(siz_b_LKV, siz_b_DHHV),
            order=(1, 0),
        )
        matV_val = tl.load(matV_ptr, boundary_check=(0, 1)).to(DTYPE)

        # accumulate matH (siz_b_LQ, siz_b_DHHV)
        matH_cur = tl.dot(matS.to(DTYPE), matV_val)
        matH_intra_acc = matH_intra_acc + matH_cur

        # compute vecN (siz_b_LQ,)
        if NORMALIZE:
            vecN_intra_acc = vecN_intra_acc + tl.sum(matS, axis=1)

    # ? combine the intra and inter chunk contributions
    # compute the matH_comb_num (siz_b_LQ, siz_b_DHHV)
    matH_comb_num_val = matH_inter_acc + matH_intra_acc

    if NORMALIZE:
        # compute the vecN_comb_denom (siz_b_LQ,)
        vecN_comb_denom_val = tl.maximum(
            tl.abs(vecN_inter_acc + vecN_intra_acc),
            1.0,
        )
        # compute the matH_comb_out_val (siz_b_LQ, siz_b_DHHV)
        matH_comb_out_val = matH_comb_num_val / (vecN_comb_denom_val[:, None] + EPS)
    else:
        # Note: we need to set vecN_comb_denom_val to 1.0 to make the compiler happy
        # if vecN_comb_denom_val is not defined, it gives an error
        # the value is not used in this case
        vecN_comb_denom_val = 1.0
        matH_comb_out_val = matH_comb_num_val

    # store matHout (size_b_LQ, siz_b_DHHV)
    matHout_ptr = tl.make_block_ptr(
        base=matHout + idx_b_BNH * str_matHV_B_NH,
        shape=(S, DHHV),
        strides=(str_matHV_S, str_matHV_DHHV),
        offsets=(idx_b_NC * L + idx_b_LQ * siz_b_LQ, idx_b_DHHV * siz_b_DHHV),
        block_shape=(siz_b_LQ, siz_b_DHHV),
        order=(1, 0),
    )
    tl.store(matHout_ptr, matH_comb_out_val.to(DTYPE), boundary_check=(0, 1))

    # the different thread blocks for different value head dimensions
    # compute the same vecN
    if NORMALIZE and (idx_b_DHHV == 0):
        # store vecNout (siz_b_LQ,)
        vecNout_ptr = (
            vecNout
            + idx_b_BNH * str_vecN_B_NH
            + idx_b_NC * L
            + idx_b_LQ * siz_b_LQ
            + tl.arange(0, siz_b_LQ)
        )
        tl.store(vecNout_ptr, vecN_comb_denom_val.to(OUTPUT_DTYPE))
