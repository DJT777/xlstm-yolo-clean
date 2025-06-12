#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from collections.abc import Callable
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .plot_config import get_plot_mpl_context
from .roofline_analysis_mlstm import (
    Acc_intensity_a100,
    Acc_intensity_h100,
    Acc_intensity_v100,
    Acc_math_a100,
    Acc_math_b200,
    Acc_math_h100,
    Acc_mem_a100,
    Acc_mem_b200,
    Acc_mem_h100,
    get_flop_optimal_chunk_size_mlstmsig,
    get_runtime_optimal_chunk_size_mlstmsig_intensity,
)


def plot_flop_optimal_chunk_size(
    calculate_optimal_chunk_size: Callable,
    p_qk: float,
    ax: Axes = None,
    num_points: int = 50,
    color: str = "tab:blue",
    show_title: bool = True,
    show_ylabel: bool = True,
    legend_args: dict = None,
    yticks: np.ndarray = None,
) -> Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    head_dims_hv = np.logspace(0, 10, num_points, base=2)
    optimal_chunk_size_fcausal05 = calculate_optimal_chunk_size(
        d_hv=head_dims_hv, p_qk=p_qk, factor_causal=0.5
    )
    optimal_chunk_size_fcausal1 = calculate_optimal_chunk_size(
        d_hv=head_dims_hv, p_qk=p_qk, factor_causal=1
    )
    optimal_chunk_size_fcausal066 = calculate_optimal_chunk_size(
        d_hv=head_dims_hv, p_qk=p_qk, factor_causal=0.66
    )

    ax.fill_between(
        head_dims_hv,
        optimal_chunk_size_fcausal05,
        optimal_chunk_size_fcausal1,
        alpha=0.1,
        color=color,
    )
    ax.plot(
        head_dims_hv,
        optimal_chunk_size_fcausal05,
        linestyle=":",
        color=color,
        label=r"$F_\text{causal}$=0.5",
    )
    ax.plot(
        head_dims_hv,
        optimal_chunk_size_fcausal066,
        label=r"$F_\text{causal}$=0.66",
        color=color,
    )
    ax.plot(
        head_dims_hv,
        optimal_chunk_size_fcausal1,
        linestyle="--",
        color=color,
        label=r"$F_\text{causal}$=1.0",
    )
    ax.set_xlabel(r"Head Dimension $d_{hv}$")
    if show_ylabel:
        ax.set_ylabel(r"Chunk Size $L_\text{opt,FLOP}$")

    if legend_args is not None:
        ax.legend()

    # max_exp = int(np.log2(head_dims_hv[-1]))
    # ax.set_xticks(np.logspace(0, max_exp, base=2, num=max_exp+1))
    # ax.set_xticks([16, 256, 512, 1024, 1536, 2048])
    ax.set_xticks([32, 128, 256, 512, 768, 1024])

    if yticks is not None:
        ax.set_yticks(yticks)

    if show_title:
        ax.set_title(rf"$d_{{qk}}={p_qk}d_{{hv}}$   ($p_{{qk}}$={p_qk})")

    ax.grid(alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ax.set_xscale("log", base=2)
    return ax.get_figure()


def create_mlstm_flop_optimal_chunksize_plot() -> Figure:
    n_cols = 2
    fig_height = 4

    with get_plot_mpl_context():
        fig, ax = plt.subplots(
            1,
            n_cols,
            figsize=(1.5 * n_cols * fig_height, fig_height),
            sharey=True,
            sharex=True,
            gridspec_kw={"wspace": 0.1, "hspace": 0.2},
            # gridspec_kw={"width_ratios": [0.46, 0.54]},
        )
        fig = plot_flop_optimal_chunk_size(
            ax=ax[0],
            calculate_optimal_chunk_size=get_flop_optimal_chunk_size_mlstmsig,
            p_qk=0.5,
            yticks=np.arange(0, 40, 5),
        )
        fig = plot_flop_optimal_chunk_size(
            ax=ax[1],
            calculate_optimal_chunk_size=get_flop_optimal_chunk_size_mlstmsig,
            p_qk=1.0,
            show_ylabel=False,
            yticks=np.arange(0, 40, 5),
        )
        handles, labels = ax[1].get_legend_handles_labels()
        legend_kwargs = {
            "loc": "upper right",
            "ncol": 1,
            "bbox_to_anchor": (0.0, 0.9, 1.06, 0.0),
            "frameon": False,
            "facecolor": "white",
            # "alignment": "top",
            "labelspacing": 1.1,
        }
        fig.legend(handles, labels, **legend_kwargs)

    return fig


def plot_runtime_optimal_chunk_size(
    fns_calculate_optimal_chunk_size: list[Callable],
    p_qk: float,
    ax: Axes = None,
    num_points: int = 50,
    colors: list[str] = ["tab:blue", "tab:orange"],
    labels: list[str] = None,
    show_title: bool = True,
    title_suffix: str = None,
    show_ylabel: bool = True,
    legend_args: dict = None,
    yticks: np.ndarray = np.arange(0, 50, 5),
) -> Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    head_dims_hv = np.logspace(0, 10, num_points, base=2)

    for calculate_optimal_chunk_size, color in zip(
        fns_calculate_optimal_chunk_size, colors
    ):
        optimal_chunk_size_fcausal05 = calculate_optimal_chunk_size(
            d_hv=head_dims_hv, p_qk=p_qk, factor_causal=0.5
        )
        optimal_chunk_size_fcausal1 = calculate_optimal_chunk_size(
            d_hv=head_dims_hv, p_qk=p_qk, factor_causal=1
        )
        optimal_chunk_size_fcausal066 = calculate_optimal_chunk_size(
            d_hv=head_dims_hv, p_qk=p_qk, factor_causal=0.66
        )

        ax.fill_between(
            head_dims_hv,
            optimal_chunk_size_fcausal05,
            optimal_chunk_size_fcausal1,
            alpha=0.1,
            color=color,
        )
        ax.plot(
            head_dims_hv,
            optimal_chunk_size_fcausal05,
            linestyle=":",
            color=color,
            label=r"$F_\text{causal}$=0.5",
        )
        ax.plot(
            head_dims_hv,
            optimal_chunk_size_fcausal1,
            linestyle="--",
            color=color,
            label=r"$F_\text{causal}$=1.0",
        )
        ax.plot(
            head_dims_hv,
            optimal_chunk_size_fcausal066,
            label=r"$F_\text{causal}$=0.66",
            color=color,
        )

    ax.set_xlabel(r"Head Dimension $d_{hv}$")
    if show_ylabel:
        ax.set_ylabel(r"Chunk Size $L_\text{opt,Runtime}$")

    if legend_args is not None:
        # Custom legend handles and labels
        from matplotlib.lines import Line2D

        custom_lines = [
            Line2D([0], [0], color="black", linestyle=":"),
            Line2D([0], [0], color="black", linestyle="-"),
            Line2D([0], [0], color="black", linestyle="--"),
        ]

        legend_labels = [
            r"$F_\text{causal}$=0.5",
            r"$F_\text{causal}$=0.66",
            r"$F_\text{causal}$=1.0",
        ]
        for color in colors:
            custom_lines.append(Line2D([0], [0], color=color, lw=4))

        if labels is not None:
            legend_labels += labels
        ax.legend(
            handles=custom_lines,
            labels=legend_labels,
            **legend_args,
        )

    # max_exp = int(np.log2(head_dims_hv[-1]))
    # ax.set_xticks(np.logspace(0, max_exp, base=2, num=max_exp+1))
    # ax.set_xticks([16, 256, 512, 1024, 1536, 2048])
    ax.set_xticks([32, 128, 256, 512, 768, 1024])

    if yticks is not None:
        ax.set_yticks(yticks)

    if show_title:
        title = rf"$d_{{qk}}={p_qk}d_{{hv}}$ ($p_{{qk}}$={p_qk})"
        if title_suffix is not None:
            title += f" {title_suffix}"
        ax.set_title(title)

    ax.grid(alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ax.set_xscale("log", base=2)
    return ax.get_figure()


def create_mlstm_runtime_optimal_chunk_size_plot() -> Figure:
    acc_intensity_h100 = Acc_math_h100 / Acc_mem_h100
    acc_intensity_b200 = Acc_math_b200 / Acc_mem_b200
    acc_intensity_a100 = Acc_math_a100 / Acc_mem_a100

    bytes_cmn = 4  # 4 bytes per float32
    fn_optimal_chunk_size_h100_fp32 = partial(
        get_runtime_optimal_chunk_size_mlstmsig_intensity,
        Acc_intensity=acc_intensity_h100,
        bytes_Cmn=bytes_cmn,
    )
    fn_optimal_chunk_size_a100_fp32 = partial(
        get_runtime_optimal_chunk_size_mlstmsig_intensity,
        Acc_intensity=acc_intensity_a100,
        bytes_Cmn=bytes_cmn,
    )
    fn_optimal_chunk_size_b200_fp32 = partial(
        get_runtime_optimal_chunk_size_mlstmsig_intensity,
        Acc_intensity=acc_intensity_b200,
        bytes_Cmn=bytes_cmn,
    )
    bytes_cmn = 2  # 2 bytes per bfloat16
    fn_optimal_chunk_size_h100_bf16 = partial(
        get_runtime_optimal_chunk_size_mlstmsig_intensity,
        Acc_intensity=acc_intensity_h100,
        bytes_Cmn=bytes_cmn,
    )
    fn_optimal_chunk_size_a100_bf16 = partial(
        get_runtime_optimal_chunk_size_mlstmsig_intensity,
        Acc_intensity=acc_intensity_a100,
        bytes_Cmn=bytes_cmn,
    )
    fn_optimal_chunk_size_b200_bf16 = partial(
        get_runtime_optimal_chunk_size_mlstmsig_intensity,
        Acc_intensity=acc_intensity_b200,
        bytes_Cmn=bytes_cmn,
    )

    p_qk = 0.5
    n_cols = 2
    fig_height = 4

    measured_optimal_chunk_size = 256
    measured_d_hv = 512
    measured_color = "#9a3c73"

    colors = ["tab:blue", "tab:orange"]
    labels = [
        r"A100 Intensity$\approx$160 FLOP/byte",
        r"H100 Intensity$\approx$295 FLOP/byte",
    ]

    with get_plot_mpl_context():
        fig, ax = plt.subplots(
            1,
            n_cols,
            figsize=(1.5 * n_cols * fig_height, fig_height),
            sharey=True,
            sharex=True,
            gridspec_kw={"wspace": 0.04, "hspace": 0.5},
            # gridspec_kw={"width_ratios": [0.46, 0.54]},
        )
        fig = plot_runtime_optimal_chunk_size(
            [fn_optimal_chunk_size_a100_fp32, fn_optimal_chunk_size_h100_fp32],
            p_qk,
            ax=ax[0],
            yticks=np.arange(0, 950, 100),
            colors=colors,
            labels=labels,
            show_title=True,
            title_suffix="| States: float32",
        )
        fig = plot_runtime_optimal_chunk_size(
            [fn_optimal_chunk_size_a100_bf16, fn_optimal_chunk_size_h100_bf16],
            p_qk,
            ax=ax[1],
            yticks=np.arange(0, 950, 100),
            colors=colors,
            labels=labels,
            show_ylabel=False,
            show_title=True,
            title_suffix="| States: bfloat16",
        )

        measured_handle = ax[0].scatter(
            measured_d_hv,
            measured_optimal_chunk_size,
            color=measured_color,
            s=250,
            zorder=10,
            marker="*",
            alpha=1.0,
        )

        from matplotlib.lines import Line2D

        custom_lines = [
            Line2D([0], [0], color="black", linestyle=":"),
            Line2D([0], [0], color="black", linestyle="-"),
            Line2D([0], [0], color="black", linestyle="--"),
        ]

        legend_labels = [
            r"$F_\text{causal}$=0.5",
            r"$F_\text{causal}$=0.66",
            r"$F_\text{causal}$=1.0",
        ]
        for color in colors:
            custom_lines.append(Line2D([0], [0], color=color, lw=4))

        if labels is not None:
            legend_labels += labels

        custom_lines.append(measured_handle)
        legend_labels.append("TFLA mLSTMsig\nOptimal Chunk Size\n" + r"($d_{hv}$=512)")

        legend_kwargs = {
            "loc": "upper left",
            "ncol": 1,
            "bbox_to_anchor": (0.9, 0.9),
            "frameon": False,
            "facecolor": "white",
            # "alignment": "top",
            "labelspacing": 1.1,
        }
        fig.legend(
            handles=custom_lines,
            labels=legend_labels,
            **legend_kwargs,
        )

    return fig


def plot_runtime_optimal_chunk_size_over_acc_intensity(
    fns_calculate_optimal_chunk_size: list[Callable],
    p_qk: float,
    ax: Axes = None,
    num_points: int = 50,
    colors: list[str] = ["tab:blue"],
    labels: list[str] = None,
    show_title: bool = True,
    title_suffix: str = None,
    show_ylabel: bool = True,
    accelerator_intensities: dict = None,
    legend_args: dict = None,
    yticks: np.ndarray = None,
) -> Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    acc_intensities = np.linspace(
        50,
        350,
        num_points,
    )
    if accelerator_intensities is not None:
        for acc_name, acc_intensity in accelerator_intensities.items():
            ax.axvline(
                x=acc_intensity,
                color="grey",
                linestyle="--",
                alpha=0.75,
            )
            ax.text(
                acc_intensity - 13,
                655,
                acc_name,
                rotation=90,
                verticalalignment="center",
                color="grey",
            )

    for calculate_optimal_chunk_size, color in zip(
        fns_calculate_optimal_chunk_size, colors
    ):
        optimal_chunk_size_fcausal05 = calculate_optimal_chunk_size(
            Acc_intensity=acc_intensities, p_qk=p_qk, factor_causal=0.5
        )
        optimal_chunk_size_fcausal1 = calculate_optimal_chunk_size(
            Acc_intensity=acc_intensities, p_qk=p_qk, factor_causal=1
        )
        optimal_chunk_size_fcausal066 = calculate_optimal_chunk_size(
            Acc_intensity=acc_intensities, p_qk=p_qk, factor_causal=0.66
        )

        ax.fill_between(
            acc_intensities,
            optimal_chunk_size_fcausal05,
            optimal_chunk_size_fcausal1,
            alpha=0.1,
            color=color,
        )
        ax.plot(
            acc_intensities,
            optimal_chunk_size_fcausal05,
            linestyle=":",
            color=color,
            label=r"$F_\text{causal}$=0.5",
        )
        ax.plot(
            acc_intensities,
            optimal_chunk_size_fcausal1,
            linestyle="--",
            color=color,
            label=r"$F_\text{causal}$=1.0",
        )
        ax.plot(
            acc_intensities,
            optimal_chunk_size_fcausal066,
            label=r"$F_\text{causal}$=0.66",
            color=color,
        )

    measured_optimal_chunk_size = 256
    measured_acc_intensity = Acc_intensity_h100
    measured_color = "#9a3c73"

    measured_handle = ax.scatter(
        measured_acc_intensity,
        measured_optimal_chunk_size,
        color=measured_color,
        s=250,
        zorder=10,
        marker="*",
        alpha=1.0,
    )

    ax.set_xlabel(r"Accelerator Intensity [FLOP/byte]")
    if show_ylabel:
        ax.set_ylabel(r"Chunk Size $L_\text{opt,Runtime}$")

    if legend_args is not None:
        # Custom legend handles and labels
        from matplotlib.lines import Line2D

        custom_lines = [
            Line2D([0], [0], color="black", linestyle=":"),
            Line2D([0], [0], color="black", linestyle="-"),
            Line2D([0], [0], color="black", linestyle="--"),
        ]

        legend_labels = [
            r"$F_\text{causal}$=0.5",
            r"$F_\text{causal}$=0.66",
            r"$F_\text{causal}$=1.0",
        ]
        for color in colors:
            custom_lines.append(Line2D([0], [0], color=color, lw=4))

        if labels is not None:
            legend_labels += labels

        custom_lines.append(measured_handle)
        legend_labels.append("TFLA mLSTMsig\nOptimal Chunk Size\n" + r"($d_{hv}$=512)")

        ax.legend(
            handles=custom_lines,
            labels=legend_labels,
            **legend_args,
        )

    # max_exp = int(np.log2(head_dims_hv[-1]))
    # ax.set_xticks(np.logspace(0, max_exp, base=2, num=max_exp+1))
    # ax.set_xticks([16, 256, 512, 1024, 1536, 2048])
    # ax.set_xticks([32, 128, 256, 512, 768, 1024])

    if yticks is not None:
        ax.set_yticks(yticks)

    if show_title:
        title = rf"$d_{{qk}}={p_qk}d_{{hv}}$ ($p_{{qk}}$={p_qk})"
        if title_suffix is not None:
            title += f" {title_suffix}"
        ax.set_title(title)

    ax.grid(alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ax.set_xscale("log", base=2)
    return ax.get_figure()


def create_mlstm_runtime_optimal_chunk_size_over_acc_intensity_plot() -> Figure:
    p_qk = 0.5

    fn_optimal_chunk_size_fp32_dhv512 = partial(
        get_runtime_optimal_chunk_size_mlstmsig_intensity,
        bytes_Cmn=4,  # 4 bytes per float32
        d_hv=512,
    )
    fn_optimal_chunk_size_fp32_dhv128 = partial(
        get_runtime_optimal_chunk_size_mlstmsig_intensity,
        bytes_Cmn=4,  # 4 bytes per float32
        d_hv=128,
    )
    # fn_optimal_chunk_size_bf16 = partial(
    #     get_runtime_optimal_chunk_size_mlstmsig_intensity,
    #     bytes_Cmn=2, # 2 bytes per bfloat16
    # )
    with get_plot_mpl_context():
        legend_kwargs = {
            "loc": "upper left",
            "ncol": 1,
            "bbox_to_anchor": (1.0, 1.0, 1.1, 0.0),
            "frameon": False,
            "facecolor": "white",
            # "alignment": "top",
            "labelspacing": 1.1,
        }
        fig = plot_runtime_optimal_chunk_size_over_acc_intensity(
            fns_calculate_optimal_chunk_size=[
                fn_optimal_chunk_size_fp32_dhv512,
                fn_optimal_chunk_size_fp32_dhv128,
            ],
            p_qk=p_qk,
            colors=["tab:orange", "tab:blue"],
            show_title=True,
            title_suffix="| States: float32",
            labels=[r"$d_{hv}=512$", r"$d_{hv}=128$"],
            legend_args=legend_kwargs,
            accelerator_intensities={
                "V100": Acc_intensity_v100,
                "A100": Acc_intensity_a100,
                "H100": Acc_intensity_h100,
            },
        )
    return fig
