#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from typing import Literal

import numpy as np
import sympy as sp

try:
    from .flops_mlstm import (
        F_causal,
        L,
        N_batch,
        N_chunk,
        N_head,
        T,
        d_hv,
        d_qk,
        p_qk,
        simpl_comp_flop_cwp_sig_total,
    )
    from .memops_mlstm import bytes_Cmn, bytes_if, bytes_qkv, comp_memop_cwp_sig_total
except ImportError:
    from flops_mlstm import (
        F_causal,
        L,
        N_batch,
        N_chunk,
        N_head,
        T,
        d_hv,
        d_qk,
        p_qk,
        simpl_comp_flop_cwp_sig_total,
    )
    from memops_mlstm import bytes_Cmn, bytes_if, bytes_qkv, comp_memop_cwp_sig_total


# hardware parameters
# Acc_math [FLOP/s] Accelerator computation speed
# Acc_mem [byte/s] Accelerator memory bandwidth
Acc_math, Acc_mem = sp.symbols("Acc_math Acc_mem", real=True, positive=True)
# Accelerator intensity [FLOP/byte] Acc_math / Acc_mem
Acc_intensity = sp.symbols("Acc_intensity", real=True, positive=True)

# V100 Specs
# https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf
Acc_math_v100 = 120 * 1e12  # FLOP/s without sparsity
Acc_mem_v100 = 0.9 * 1e12  # byte/s
Acc_intensity_v100 = Acc_math_v100 / Acc_mem_v100

# A100 Specs
# https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf
# FLOP/s with sparsity: 624 * 1e12
Acc_math_a100 = 312 * 1e12  # FLOP/s without sparsity
Acc_mem_a100 = 1.935 * 1e12  # byte/s
Acc_intensity_a100 = Acc_math_a100 / Acc_mem_a100

# H100 Specs
# FLOP/s with sparsity: 1_979 * 1e12
# FLOP/s without sparsity
Acc_math_h100 = 989 * 1e12  # FLOP/s without sparsity
# Memory bandwidth: 3.35 * 1e12
Acc_mem_h100 = 3.35 * 1e12  # byte/s
Acc_intensity_h100 = Acc_math_h100 / Acc_mem_h100

# B100 Specs
# data sheet: https://resources.nvidia.com/en-us-blackwell-architecture/datasheet
# FLOP/s with sparsity: 4.5 * 1e15
Acc_math_b200 = 2.25 * 1e15  # FLOP/s without sparsity
Acc_mem_b200 = 7.7 * 1e12  # byte/s
Acc_intensity_b200 = Acc_math_b200 / Acc_mem_b200

## over year development functions since volta
# without blackwell
years_no_blackwell = [0, 3, 5]
math_points_no_blackwell = [Acc_math_v100, Acc_math_a100, Acc_math_h100]
mem_points_no_blackwell = [Acc_mem_v100, Acc_mem_a100, Acc_mem_h100]
intensity_points_no_blackwell = [
    Acc_intensity_v100,
    Acc_intensity_a100,
    Acc_intensity_h100,
]

# with blackwell
years_blackwell = [0, 3, 5, 8]
math_points_blackwell = [Acc_math_v100, Acc_math_a100, Acc_math_h100, Acc_math_b200]
mem_points_blackwell = [Acc_mem_v100, Acc_mem_a100, Acc_mem_h100, Acc_mem_b200]
intensity_points_blackwell = [
    Acc_intensity_v100,
    Acc_intensity_a100,
    Acc_intensity_h100,
    Acc_intensity_b200,
]


def compute_projection(years, math_points, mem_points, intensity_points):
    math_coeffs = np.polyfit(years, math_points, 1)
    mem_coeffs = np.polyfit(years, mem_points, 1)
    intensity_coeffs = np.polyfit(years, intensity_points, 1)
    return math_coeffs, mem_coeffs, intensity_coeffs


math_coeffs_no_blackwell, mem_coeffs_no_blackwell, intensity_coeffs_no_blackwell = (
    compute_projection(
        years_no_blackwell,
        math_points_no_blackwell,
        mem_points_no_blackwell,
        intensity_points_no_blackwell,
    )
)
math_coeffs_blackwell, mem_coeffs_blackwell, intensity_coeffs_blackwell = (
    compute_projection(
        years_blackwell,
        math_points_blackwell,
        mem_points_blackwell,
        intensity_points_blackwell,
    )
)


def lin_func(x, coeffs):
    return coeffs[0] * x + coeffs[1]


def acc_mem_projection(x_years, with_blackwell=False):
    if with_blackwell:
        return lin_func(x_years, mem_coeffs_blackwell)
    return lin_func(x_years, mem_coeffs_no_blackwell)


def acc_math_projection(x_years, with_blackwell=False):
    if with_blackwell:
        return lin_func(x_years, math_coeffs_blackwell)
    return lin_func(x_years, math_coeffs_no_blackwell)


def acc_intensity_projection(x_years, with_blackwell=False):
    if with_blackwell:
        return lin_func(x_years, intensity_coeffs_blackwell)
    return lin_func(x_years, intensity_coeffs_no_blackwell)


Acc_math_mem_dict = {
    "v100": (Acc_math_v100, Acc_mem_v100),
    "a100": (Acc_math_a100, Acc_mem_a100),
    "h100": (Acc_math_h100, Acc_mem_h100),
}

Acc_math_mem_dict_blackwell = {
    **Acc_math_mem_dict,
    "b200": (Acc_math_b200, Acc_mem_b200),
}


def get_flops_per_second_for_acc(
    x_arithmetic_intensity, acc: Literal["v100", "a100", "h100", "b200"]
):
    acc_math, acc_mem = Acc_math_mem_dict_blackwell[acc]

    return min(x_arithmetic_intensity * acc_mem, acc_math)


#######################################
### FLOP optimal chunk size ###
#######################################

## mLSTMsig ##

# We begin with the total number of flops for mLSTMsig
# 1) we substitute the qk head dimension d_qk with p_qk * d_hv
flops_total_sig_cwp_subs = simpl_comp_flop_cwp_sig_total.subs(d_qk, p_qk * d_hv)

# 2) we differentiate the total number of flops with respect to L, to find the minima
diff_flops_total_sig_cwp_subs = sp.simplify(sp.diff(flops_total_sig_cwp_subs, L))

# 3) we set the derivative to zero and solve for L and take the positive solution
L_flop_optimal_sig = sp.solve(sp.Eq(diff_flops_total_sig_cwp_subs, 0), L)[1]

fn_L_flop_optimal_sig = sp.lambdify(
    (d_hv, p_qk, F_causal), L_flop_optimal_sig, modules=["numpy"]
)


def get_flop_optimal_chunk_size_mlstmsig(d_hv, p_qk, factor_causal, **kwargs):
    return fn_L_flop_optimal_sig(d_hv, p_qk, factor_causal)


######################################
### Theoretical Runtime ###
######################################

## mLSTMsig ##


def compute_total_theoretical_runtime_A_math_mem(
    flops_per_seq, memops_per_seq, acc_math, acc_mem
):
    return flops_per_seq / acc_math + memops_per_seq / acc_mem


total_runtime_cwp_sig_math_mem = (
    N_batch
    * N_head
    * compute_total_theoretical_runtime_A_math_mem(
        flops_per_seq=simpl_comp_flop_cwp_sig_total.subs(d_qk, p_qk * d_hv),
        memops_per_seq=comp_memop_cwp_sig_total.subs(d_qk, p_qk * d_hv),
        acc_math=Acc_math,
        acc_mem=Acc_mem,
    )
)

fn_total_runtime_cwp_sig = sp.lambdify(
    (
        N_batch,
        N_head,
        T,
        L,
        d_hv,
        p_qk,
        F_causal,
        Acc_math,
        Acc_mem,
        bytes_if,
        bytes_qkv,
        bytes_Cmn,
    ),
    total_runtime_cwp_sig_math_mem,
    modules=["numpy"],
)


def get_theoretical_runtime_mlstmsig_math_mem_in_ms(
    batch_size,
    num_heads,
    seq_len,
    chunk_size,
    d_hv,
    p_qk,
    factor_causal,
    acc_math,
    acc_mem,
    bytes_if,
    bytes_qkv,
    bytes_Cmn,
    **kwargs,
):
    return (
        fn_total_runtime_cwp_sig(
            batch_size,
            num_heads,
            seq_len,
            chunk_size,
            d_hv,
            p_qk,
            factor_causal,
            acc_math,
            acc_mem,
            bytes_if,
            bytes_qkv,
            bytes_Cmn,
        )
        * 1000
    )  # convert to ms


######################################
### Theoretical Runtime optimal chunk size ###
######################################

## mLSTMsig ##


# 0) compute total theoretical runtime equivalent, dependent on Acc_intensity for simpler terms
# example: we know only the ration Acc_intensity matters, and the optimal chunksize is
# independent of N_batch and N_head
def compute_total_theoretical_runtime_equivalent_A_intensity(
    flops_per_seq, memops_per_seq, acc_intensity
):
    return flops_per_seq + acc_intensity * memops_per_seq


total_runtime_equivalent_cwp_sig_intensity = (
    compute_total_theoretical_runtime_equivalent_A_intensity(
        flops_per_seq=simpl_comp_flop_cwp_sig_total.subs(d_qk, p_qk * d_hv),
        memops_per_seq=comp_memop_cwp_sig_total.subs(d_qk, p_qk * d_hv),
        acc_intensity=Acc_intensity,
    )
)

total_runtime_equivalent_cwp_sig_intensity = sp.collect(
    sp.collect(sp.expand(total_runtime_equivalent_cwp_sig_intensity), T), Acc_intensity
)

# 1) differentiate the total runtime with respect to L
diff_total_runtime_eqivalent_cwp_sig_intensity = sp.simplify(
    sp.diff(total_runtime_equivalent_cwp_sig_intensity, L)
)

# 2) set the derivative to zero and solve for L
# 3) take the positive solution
L_optimal_runtime_sig_intensity = sp.solve(
    sp.Eq(diff_total_runtime_eqivalent_cwp_sig_intensity, 0), L
)[1]

L_optimal_runtime_sig_intensity = sp.collect(L_optimal_runtime_sig_intensity, F_causal)

fn_L_optimal_runtime_sig_intensity = sp.lambdify(
    (d_hv, p_qk, F_causal, Acc_intensity, bytes_Cmn),
    L_optimal_runtime_sig_intensity,
    modules=["numpy"],
)


def get_runtime_optimal_chunk_size_mlstmsig_intensity(
    d_hv, p_qk, factor_causal, Acc_intensity, bytes_Cmn, **kwargs
):
    return fn_L_optimal_runtime_sig_intensity(
        d_hv, p_qk, factor_causal, Acc_intensity, bytes_Cmn
    )


######################################
### Arithmetic Intensity ###
######################################

## mLSTMsig ##

Alg_intensity_cwp_sig = sp.simplify(
    simpl_comp_flop_cwp_sig_total.subs(d_qk, p_qk * d_hv)
    / comp_memop_cwp_sig_total.subs(d_qk, p_qk * d_hv)
)

fn_Alg_intensity_cwp_sig = sp.lambdify(
    (L, d_hv, p_qk, F_causal, bytes_if, bytes_qkv, bytes_Cmn),
    Alg_intensity_cwp_sig,
    modules=["numpy"],
)


def get_arithmetic_intensity_mlstmsig(
    chunk_size, d_hv, p_qk, factor_causal, bytes_if, bytes_qkv, bytes_Cmn, **kwargs
):
    return fn_Alg_intensity_cwp_sig(
        chunk_size, d_hv, p_qk, factor_causal, bytes_if, bytes_qkv, bytes_Cmn
    )
