#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import sympy as sp

### Define the symbols ###

# dimensions
N_batch, N_head, N_chunk = sp.symbols(
    "N_batch N_head N_chunk", integer=True, positive=True
)
T, L, d_qk, d_hv = sp.symbols("T L d_qk d_hv", real=True, positive=True)
# dimension factor
p_qk = sp.symbols("p_qk")

# flop factors
F_exp, F_log, F_sig, F_max, F_mask, F_abs = sp.symbols(
    "F_exp F_log F_sig F_max F_mask F_abs", real=True, positive=True
)
# causal factor
F_causal = sp.symbols("F_causal", real=True, positive=True)

N_chunk = T / L

### FLOP COUNTS ###

######################################
### Chunkwise-parallel formulation ###
######################################

## mLSTMexp ##

# recurrent computation of the inter chunk states
flop_cwp_exp_gates = (
    2 * L + 0.5 * L * (L + 1) + L * (1 + F_exp + F_log + F_sig) + 3 + F_max + F_exp
)
flop_cwp_exp_numerator = 2 * d_qk * d_hv + 2 * L * d_qk * d_hv + L * d_qk
flop_cwp_exp_denominator = 2 * d_qk + 2 * L * d_qk
flop_cwp_exp_reccomp_inter = (
    flop_cwp_exp_gates + flop_cwp_exp_numerator + flop_cwp_exp_denominator
)

# parallel computation of the intra chunk outputs
flop_cwp_exp_cum_fgates = 0.5 * L * (L + 1) + L * (F_log + F_sig)
flop_cwp_exp_gate_matrix = F_causal * (L**2 * (3 + F_exp + F_max) + L * (1 + F_max))
flop_cwp_exp_intra_outputs = F_causal * (2 * L**2 * (d_qk + d_hv) + 3 * L**2)
flop_cwp_exp_parcomp_intra = (
    flop_cwp_exp_cum_fgates + flop_cwp_exp_gate_matrix + flop_cwp_exp_intra_outputs
)

# parallel computation of the inter chunk outputs
flop_cwp_exp_inter_outputs = 2 * L * d_qk * d_hv + 3 * L * d_qk

# combination of inter and intra chunk outputs
flop_cwp_exp_output_comb = 2 * L * d_hv + L * (1 + F_max + F_abs + F_exp)

flop_cwp_exp_expr_dict = {
    "flop_cwp_exp_gates": flop_cwp_exp_gates,
    "flop_cwp_exp_numerator": flop_cwp_exp_numerator,
    "flop_cwp_exp_denominator": flop_cwp_exp_denominator,
    "flop_cwp_exp_reccomp_inter": flop_cwp_exp_reccomp_inter,
    "flop_cwp_exp_cum_fgates": flop_cwp_exp_cum_fgates,
    "flop_cwp_exp_gate_matrix": flop_cwp_exp_gate_matrix,
    "flop_cwp_exp_intra_outputs": flop_cwp_exp_intra_outputs,
    "flop_cwp_exp_parcomp_intra": flop_cwp_exp_parcomp_intra,
    "flop_cwp_exp_inter_outputs": flop_cwp_exp_inter_outputs,
    "flop_cwp_exp_output_comb": flop_cwp_exp_output_comb,
}

# total flops mLSTMexp
flop_cwp_exp_recurrent = flop_cwp_exp_reccomp_inter
flop_cwp_exp_parallel = (
    flop_cwp_exp_parcomp_intra + flop_cwp_exp_inter_outputs + flop_cwp_exp_output_comb
)

flop_cwp_exp_total = (
    flop_cwp_exp_reccomp_inter
    + flop_cwp_exp_parcomp_intra
    + flop_cwp_exp_inter_outputs
    + flop_cwp_exp_output_comb
)

## mLSTMsig ##

# recurrent computation of the inter chunk states
flop_cwp_sig_gates = (
    2 * L + 0.5 * L * (L + 1) + L * F_exp + F_exp + 2 * L * (F_log + F_sig)
)
flop_cwp_sig_numerator = 2 * d_qk * d_hv + 2 * L * d_qk * d_hv + L * d_qk
flop_cwp_sig_reccomp_inter = flop_cwp_exp_gates + flop_cwp_sig_numerator

# parallel computation of the intra chunk outputs
flop_cwp_sig_cum_fgates = 0.5 * L * (L + 1) + 2 * L * (F_log + F_sig)
flop_cwp_sig_gate_matrix = F_causal * (L**2 * (2 + F_exp))
flop_cwp_sig_intra_outputs = F_causal * (2 * L**2 * (d_qk + d_hv) + 3 * L**2)
flop_cwp_sig_parcomp_intra = (
    flop_cwp_sig_cum_fgates + flop_cwp_sig_gate_matrix + flop_cwp_sig_intra_outputs
)

# parallel computation of the inter chunk outputs
flop_cwp_sig_inter_outputs = 2 * L * d_qk * d_hv + L * d_qk

# combination of inter and intra chunk outputs
flop_cwp_sig_output_comb = L * d_hv

flop_cwp_sig_expr_dict = {
    "flop_cwp_sig_gates": flop_cwp_sig_gates,
    "flop_cwp_sig_numerator": flop_cwp_sig_numerator,
    "flop_cwp_sig_reccomp_inter": flop_cwp_sig_reccomp_inter,
    "flop_cwp_sig_cum_fgates": flop_cwp_sig_cum_fgates,
    "flop_cwp_sig_gate_matrix": flop_cwp_sig_gate_matrix,
    "flop_cwp_sig_intra_outputs": flop_cwp_sig_intra_outputs,
    "flop_cwp_sig_parcomp_intra": flop_cwp_sig_parcomp_intra,
    "flop_cwp_sig_inter_outputs": flop_cwp_sig_inter_outputs,
    "flop_cwp_sig_output_comb": flop_cwp_sig_output_comb,
}

flop_cwp_sig_recurrent = flop_cwp_sig_reccomp_inter
flop_cwp_sig_parallel = (
    flop_cwp_sig_parcomp_intra + flop_cwp_sig_inter_outputs + flop_cwp_sig_output_comb
)
flop_cwp_sig_total = (
    flop_cwp_sig_reccomp_inter
    + flop_cwp_sig_parcomp_intra
    + flop_cwp_sig_inter_outputs
    + flop_cwp_sig_output_comb
)

######################################
### Recurrent formulation ###
######################################

## mLSTMexp ##

flop_rec_exp_gates = 4 + 2 * F_exp + F_log + F_sig + F_max
flop_rec_exp_cell_update = 4 * d_qk * d_hv
flop_rec_exp_denom = 6 * d_qk + d_hv + 1 + F_abs + F_max
flop_rec_exp_output = 2 * d_hv * d_qk + d_qk

flop_rec_exp_total = (
    flop_rec_exp_gates
    + flop_rec_exp_cell_update
    + flop_rec_exp_denom
    + flop_rec_exp_output
)

## mLSTMsig ##

flop_rec_sig_gates = 2 * F_sig
flop_rec_sig_cell_update = 4 * d_qk * d_hv
flop_rec_sig_output = 2 * d_hv * d_qk + d_qk

flop_rec_sig_total = flop_rec_sig_gates + flop_rec_sig_cell_update + flop_rec_sig_output


######################################
### Parallel formulation ###
######################################

## mLSTMexp ##
flop_par_exp_cum_fgates = 0.5 * T * (T + 1) + T * (F_log + F_sig)
flop_par_exp_gate_matrix = T**2 * (3 + F_exp + F_max + F_mask)
flop_par_exp_attn_logits = F_causal * (2 * T**2 * d_qk + 2 * T**2)
flop_par_exp_normalization = F_causal * (T**2 * (3 + F_abs) + T * (F_exp + F_max))
flop_par_exp_outputs = F_causal * 2 * T**2 * d_hv

flop_par_exp_total = (
    flop_par_exp_cum_fgates
    + flop_par_exp_gate_matrix
    + flop_par_exp_attn_logits
    + flop_par_exp_normalization
    + flop_par_exp_outputs
)

## mLSTMsig ##

flop_par_sig_cum_fgates = 0.5 * T * (T + 1) + 2 * T * (F_log + F_sig)
flop_par_sig_gate_matrix = T**2 * (3 + F_exp + F_max + F_mask)
flop_par_sig_attn_logits = F_causal * (2 * T**2 * d_qk + 2 * T**2)

flop_par_sig_outputs = F_causal * 2 * T**2 * d_hv

flop_par_sig_total = (
    flop_par_sig_cum_fgates
    + flop_par_sig_gate_matrix
    + flop_par_sig_attn_logits
    + flop_par_sig_outputs
)

#######################################
### TOTAL FLOPs for a full sequence ###
#######################################

# we set all the flop factors to 1
subs_dict = {F_exp: 1, F_log: 1, F_sig: 1, F_max: 1, F_mask: 1, F_abs: 1}

## mLSTMexp ##

comp_flop_rec_exp_total = T * flop_rec_exp_total
comp_flop_cwp_exp_total = N_chunk * flop_cwp_exp_total
comp_flop_par_exp_total = flop_par_exp_total

# simplify and substitute the flop factors
simpl_comp_flop_rec_exp_total = sp.collect(comp_flop_rec_exp_total.subs(subs_dict), T)
simpl_comp_flop_cwp_exp_total = sp.collect(
    sp.cancel(sp.collect(comp_flop_cwp_exp_total.subs(subs_dict), L)), L
)
simpl_comp_flop_par_exp_total = sp.collect(comp_flop_par_exp_total.subs(subs_dict), T)

# lamdify the expressions
fn_flop_rec_exp = sp.lambdify(
    (T, d_qk, d_hv), simpl_comp_flop_rec_exp_total, modules=["numpy"]
)
fn_flop_cwp_exp = sp.lambdify(
    (T, d_qk, d_hv, F_causal, L), simpl_comp_flop_cwp_exp_total, modules=["numpy"]
)
fn_flop_par_exp = sp.lambdify(
    (T, d_qk, d_hv, F_causal), simpl_comp_flop_par_exp_total, modules=["numpy"]
)


def count_flops_mlstmexp_recurrent(seq_len, d_qk, d_hv, **kwargs):
    return fn_flop_rec_exp(seq_len, d_qk, d_hv)


def count_flops_mlstmexp_chunkwise_parallel(
    seq_len, d_qk, d_hv, factor_causal, chunk_size, **kwargs
):
    return fn_flop_cwp_exp(seq_len, d_qk, d_hv, factor_causal, chunk_size)


def count_flops_mlstmexp_parallel(seq_len, d_qk, d_hv, factor_causal, **kwargs):
    return fn_flop_par_exp(seq_len, d_qk, d_hv, factor_causal)


## mLSTMsig ##

comp_flop_rec_sig_total = T * flop_rec_sig_total
comp_flop_cwp_sig_total = N_chunk * flop_cwp_sig_total
comp_flop_par_sig_total = flop_par_sig_total

# simplify and substitute the flop factors
simpl_comp_flop_rec_sig_total = sp.collect(comp_flop_rec_sig_total.subs(subs_dict), T)
simpl_comp_flop_cwp_sig_total = sp.collect(
    sp.cancel(sp.collect(comp_flop_cwp_sig_total.subs(subs_dict), L)), L
)
simpl_comp_flop_par_sig_total = sp.collect(comp_flop_par_sig_total.subs(subs_dict), T)

# lamdify the expressions
fn_flop_rec_sig = sp.lambdify(
    (T, d_qk, d_hv), simpl_comp_flop_rec_sig_total, modules=["numpy"]
)
fn_flop_cwp_sig = sp.lambdify(
    (T, d_qk, d_hv, F_causal, L), simpl_comp_flop_cwp_sig_total, modules=["numpy"]
)
fn_flop_par_sig = sp.lambdify(
    (T, d_qk, d_hv, F_causal), simpl_comp_flop_par_sig_total, modules=["numpy"]
)


def count_flops_mlstmsig_recurrent(seq_len, d_qk, d_hv, **kwargs):
    return fn_flop_rec_sig(seq_len, d_qk, d_hv)


def count_flops_mlstmsig_chunkwise_parallel(
    seq_len, d_qk, d_hv, factor_causal, chunk_size, **kwargs
):
    return fn_flop_cwp_sig(seq_len, d_qk, d_hv, factor_causal, chunk_size)


def count_flops_mlstmsig_parallel(seq_len, d_qk, d_hv, factor_causal, **kwargs):
    return fn_flop_par_sig(seq_len, d_qk, d_hv, factor_causal)
