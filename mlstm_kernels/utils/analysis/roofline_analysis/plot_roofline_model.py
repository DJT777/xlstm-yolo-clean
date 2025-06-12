#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .flops_mlstm import (
    count_flops_mlstmsig_chunkwise_parallel,
)
from .plot_config import get_plot_mpl_context
from .plot_runtime import (
    calculate_flops_and_flops_per_second_and_arithmetic_intensity_for_runtime_df,
    load_runtime_df,
)
from .roofline_analysis_mlstm import (
    Acc_math_a100,
    Acc_math_b200,
    Acc_math_h100,
    Acc_math_mem_dict,
    Acc_math_mem_dict_blackwell,
    Acc_mem_a100,
    Acc_mem_b200,
    Acc_mem_h100,
    acc_math_projection,
    acc_mem_projection,
    get_arithmetic_intensity_mlstmsig,
    get_flop_optimal_chunk_size_mlstmsig,
    get_flops_per_second_for_acc,
    get_runtime_optimal_chunk_size_mlstmsig_intensity,
    get_theoretical_runtime_mlstmsig_math_mem_in_ms,
)


def get_measured_data_df():
    result_filename = "notebooks/plots_mlstm_kernel_benchmark_tfla_paper/mlstm_tfla_paper_consttoken_benchmark_results.p"
    file = Path(".") / "../.." / result_filename
    runtime_df = load_runtime_df(file)

    d_hv = 512
    p_qk = 0.5

    seq_len = 8192

    num_heads = 8
    batch_size = 8

    bytes_if = 2
    bytes_qkv = 2
    bytes_Cmn = 4

    factors_causal = [0.5, 0.66, 1.0]

    res_df = (
        calculate_flops_and_flops_per_second_and_arithmetic_intensity_for_runtime_df(
            runtime_df=runtime_df,
            fn_count_flops=count_flops_mlstmsig_chunkwise_parallel,
            fn_calculate_arithmetic_intensity=get_arithmetic_intensity_mlstmsig,
            p_qk=p_qk,
            d_hv=d_hv,
            num_heads=num_heads,
            factors_causal=factors_causal,
            seq_len=seq_len,
            batch_size=batch_size,
            bytes_if=bytes_if,
            bytes_qkv=bytes_qkv,
            bytes_Cmn=bytes_Cmn,
        )
    )
    return res_df


def plot_accelerator_roofline(
    ax: plt.Axes,
    peak_flops: float,
    peak_memory_bandwidth: float,
    max_intensity: float = None,
    style_dict: dict = {},
) -> plt.Axes:
    """
    Plot the roofline model for an accelerator with given peak FLOPS and peak memory bandwidth.
    """

    intensity_ridge_point = peak_flops / peak_memory_bandwidth

    if max_intensity is None:
        max_intensity = intensity_ridge_point * 2

    # Plot the memory bandwidth roofline
    ax.plot([0, intensity_ridge_point], [0, peak_flops], **style_dict)

    # Plot the compute roofline
    ax.plot(
        [intensity_ridge_point, max_intensity],
        [peak_flops, peak_flops],
        **style_dict,
    )

    ax.text(
        x=max_intensity + 5,
        y=peak_flops,
        s=style_dict["label"],
        color=style_dict["color"],
        ha="left",
        va="center",
    )

    style_dict["linestyle"] = "-."
    ax.vlines(x=intensity_ridge_point, ymin=0, ymax=peak_flops, **style_dict, zorder=0)

    # ax.set_xscale("log", base=2)

    return ax


def plot_roofline_analysis(
    accelerator_math_mem_dict: dict[str, tuple[float, float]],
    accelerator_style_dict: dict[str, dict],
    measured_data_df: pd.DataFrame,
    max_intensity: float = 550,
    factors_causal: list[float] = [0.5, 0.66, 1.0],
    factors_causal_style_dict: dict[float, dict] = {
        0.5: {"linestyle": ":", "alpha": 0.0},
        0.66: {"linestyle": "-", "alpha": 0.0},
        1.0: {"linestyle": "--", "alpha": 0.0},
    },
    cs_markers=["o", "s", "D", "^", "v", "P", "*"],
    measured_data_color: str = "#9a3c73",
    marker_size: float = 150.0,
    scilimits: tuple[float, float] = (12, 12),
    x_lim: tuple[float, float] = (10, 550),
    y_lim: tuple[float, float] = (10e12, 1100e12),
    y_label: str = "Attainable FLOPs/s",
    x_label: str = r"Arithmetic Intensity [FLOP/byte]",
    ax: plt.Axes = None,
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots()

    for accelerator, (
        peak_flops,
        peak_memory_bandwidth,
    ) in accelerator_math_mem_dict.items():
        plot_accelerator_roofline(
            ax=ax,
            peak_flops=peak_flops,
            peak_memory_bandwidth=peak_memory_bandwidth,
            style_dict=accelerator_style_dict[accelerator],
            max_intensity=max_intensity,
        )

    for fc in factors_causal:
        ax.plot(
            measured_data_df[f"arithmetic_intensity_fc{fc}"],
            measured_data_df[f"flops_per_second_fc{fc}"],
            **factors_causal_style_dict[fc],
            color=measured_data_color,
        )

    if 0.5 in factors_causal and 1.0 in factors_causal:
        num_points = 200
        xnew = np.linspace(0, max_intensity, num_points)
        ynews = []
        for fc in [0.5, 1.0]:
            xold = measured_data_df[f"arithmetic_intensity_fc{fc}"]
            yold = measured_data_df[f"flops_per_second_fc{fc}"]
            ynews.append(np.interp(xnew, xold, yold))

        ax.fill_between(
            x=xnew, y1=ynews[0], y2=ynews[1], alpha=0.2, color=measured_data_color
        )

    # plot values
    for i in range(len(measured_data_df["chunk_size"])):
        ai_val = measured_data_df["arithmetic_intensity_fc0.66"][i]
        if ai_val < max_intensity:
            flops_per_sec_val = measured_data_df["flops_per_second_fc0.66"][i]
            cs = measured_data_df["chunk_size"][i]
            ax.scatter(
                x=ai_val,
                y=flops_per_sec_val,
                marker=cs_markers[i],
                zorder=100,
                color=measured_data_color,
                s=marker_size,
                label=f"{cs}",
            )

        ax.vlines(
            x=ai_val,
            ymin=0,
            ymax=get_flops_per_second_for_acc(
                x_arithmetic_intensity=ai_val, acc="h100"
            ),
            color=measured_data_color,
            alpha=0.7,
            linestyle="--",
            zorder=0,
            linewidth=2,
        )

    ax.set_yticks(np.arange(0, 1100, 100) * 1e12)
    ax.set_xticks(np.arange(0, max_intensity, 50))

    ax.ticklabel_format(style="sci", axis="y", scilimits=scilimits)
    ax.set_xlabel(xlabel=x_label)
    ax.set_ylabel(ylabel=y_label)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    # ax.set_yscale("log")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    from matplotlib.lines import Line2D

    lw = accelerator_style_dict["v100"]["linewidth"]
    custom_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-.",
        ),
        Line2D([0], [0], color=measured_data_color, linewidth=8, alpha=0.2),
        Line2D([0], [0], color="white", linewidth=8, alpha=0.2),
    ]

    legend_labels = [
        "Accelerator Roofline",
        "Memory vs. Compute Bound",
        "TFLA mLSTM Kernels",
        "Chunk Sizes (Measured):",
    ]

    for h, l in zip(handles, labels):
        try:
            cs = int(l)
            custom_handles.append(h)
            legend_labels.append(l)
        except Exception:
            pass

    legend_kwargs = {
        "loc": "upper left",
        "ncol": 1,
        "bbox_to_anchor": (1.1, 1.03),
        "frameon": False,
        "facecolor": "white",
        # "alignment": "top",
        "labelspacing": 1.05,
    }

    ax.legend(handles=custom_handles, labels=legend_labels, **legend_kwargs)

    fig = ax.get_figure()

    return fig


def create_roofline_analysis_plot() -> plt.Figure:
    acc_style_dict = {
        "v100": {"color": "grey", "label": "V100", "alpha": 1.0, "linewidth": 2.5},
        "a100": {"color": "tab:blue", "label": "A100", "alpha": 1.0, "linewidth": 2.5},
        "h100": {
            "color": "tab:orange",
            "label": "H100",
            "alpha": 1.0,
            "linewidth": 2.5,
        },
        # "b200": {"color": "grey", "label": "B200", "alpha": 0.7, "linewidth": 2.5},
    }

    measured_data_df = get_measured_data_df()

    with get_plot_mpl_context():
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        fig = plot_roofline_analysis(
            accelerator_math_mem_dict=Acc_math_mem_dict,  # _blackwell,
            accelerator_style_dict=acc_style_dict,
            measured_data_df=measured_data_df,
            ax=ax,
        )

    return fig
