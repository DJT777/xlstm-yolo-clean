#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import sympy as sp

### Define or import symbols ###
# import symbols
# we use the try-catch block to enable import from the same directory
try:
    from .flops_mlstm import L, N_chunk, T, d_hv, d_qk
except ImportError:
    from flops_mlstm import L, N_chunk, T, d_hv, d_qk

# number of bytes
bytes_qkv, bytes_Cmn, bytes_if = sp.symbols(
    "bytes_qkv bytes_Cmn bytes_if", real=True, positive=True
)

### MEMORY OPERATION COUNTS ###

######################################
### Chunkwise-parallel formulation ###
######################################

## mLSTMexp ##
memop_cwp_exp_rec_load = L * (d_qk + d_hv) * bytes_qkv + 2 * L * bytes_if
memop_cwp_exp_rec_store = (d_qk * d_hv + d_qk + 1) * bytes_Cmn

memop_cwp_exp_par_load = (
    L * (2 * d_qk + d_hv) * bytes_qkv
    + 2 * L * bytes_if
    + (d_qk * d_hv + d_qk + 1) * bytes_Cmn
)
memop_cwp_exp_par_store = L * d_hv * bytes_qkv + 2 * L * bytes_Cmn

total_memop_cwp_exp = (
    memop_cwp_exp_rec_load
    + memop_cwp_exp_rec_store
    + memop_cwp_exp_par_load
    + memop_cwp_exp_par_store
)
total_memop_cwp_exp = sp.simplify(
    sp.collect(total_memop_cwp_exp, syms=[bytes_qkv, bytes_Cmn, bytes_if])
)


## mLSTMsig ##
memop_cwp_sig_rec_load = L * (d_qk + d_hv) * bytes_qkv + 2 * L * bytes_if
memop_cwp_sig_rec_store = d_qk * d_hv * bytes_Cmn

memop_cwp_sig_par_load = (
    L * (2 * d_qk + d_hv) * bytes_qkv + 2 * L * bytes_if + (d_qk * d_hv) * bytes_Cmn
)
memop_cwp_sig_par_store = L * d_hv * bytes_qkv

total_memop_cwp_sig = (
    memop_cwp_sig_rec_load
    + memop_cwp_sig_rec_store
    + memop_cwp_sig_par_load
    + memop_cwp_sig_par_store
)
total_memop_cwp_sig = sp.simplify(
    sp.collect(total_memop_cwp_sig, syms=[bytes_qkv, bytes_Cmn, bytes_if])
)

######################################
### Parallel formulation ###
######################################

## mLSTMexp ##
memop_par_exp_load = T * (2 * d_qk + d_hv) * bytes_qkv + 2 * T * bytes_if
memop_par_exp_store = T * d_hv * bytes_qkv + 2 * T * bytes_Cmn

total_memop_par_exp = memop_par_exp_load + memop_par_exp_store
total_memop_par_exp = sp.simplify(
    sp.collect(total_memop_par_exp, syms=[bytes_qkv, bytes_Cmn, bytes_if])
)

## mLSTMsig ##
memop_par_sig_load = T * (2 * d_qk + d_hv) * bytes_qkv + 2 * T * bytes_if
memop_par_sig_store = T * d_hv * bytes_qkv

total_memop_par_sig = memop_par_sig_load + memop_par_sig_store
total_memop_par_sig = sp.simplify(
    sp.collect(total_memop_par_sig, syms=[bytes_qkv, bytes_Cmn, bytes_if])
)

#######################################
### Recurrent formulation ###
#######################################

## mLSTMexp ##
memop_rec_exp_load = (
    (2 * d_qk + d_hv) * bytes_qkv + 2 * bytes_if + (d_qk * d_hv + d_qk + 1) * bytes_Cmn
)
memop_rec_exp_store = d_hv * bytes_qkv + (d_qk * d_hv + d_qk + 1) * bytes_Cmn

total_memop_rec_exp = memop_rec_exp_load + memop_rec_exp_store
total_memop_rec_exp = sp.simplify(
    sp.collect(total_memop_rec_exp, syms=[bytes_qkv, bytes_Cmn, bytes_if])
)

## mLSTMsig ##
memop_rec_sig_load = (
    (2 * d_qk + d_hv) * bytes_qkv + 2 * bytes_if + d_qk * d_hv * bytes_Cmn
)
memop_rec_sig_store = d_hv * bytes_qkv + d_qk * d_hv * bytes_Cmn

total_memop_rec_sig = memop_rec_sig_load + memop_rec_sig_store
total_memop_rec_sig = sp.simplify(
    sp.collect(total_memop_rec_sig, syms=[bytes_qkv, bytes_Cmn, bytes_if])
)

###################################################
### TOTAL Memory Operations for a full sequence ###
###################################################

## mLSTMexp ##
comp_memop_cwp_exp_total = N_chunk * total_memop_cwp_exp
comp_memop_par_exp_total = total_memop_par_exp
comp_memop_rec_exp_total = T * total_memop_rec_exp

## mLSTMsig ##
comp_memop_cwp_sig_total = N_chunk * total_memop_cwp_sig
comp_memop_par_sig_total = total_memop_par_sig
comp_memop_rec_sig_total = T * total_memop_rec_sig
