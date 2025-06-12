#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_double_bar_plot(
    data: pd.DataFrame,
    y_col_left: str,
    y_col_right: str,
    x_col: str,
    left_color,
    right_color,
    x_label: str,
    figsize: tuple[float, float],
    bar_width: float = 0.4,
    ax_grid: Literal["left", "right", "none"] = "left",
    left_alpha: float = 1.0,
    right_alpha: float = 1.0,
    left_label: str = None,
    right_label: str = None,
    left_scilimits: tuple[int, int] = None,
    right_scilimits: tuple[int, int] = None,
    left_yerr: str | tuple[str, str] | None = None,
    right_yerr: str | tuple[str, str] | None = None,
    capsize: int = 5,
    left_ylim: tuple[float, float] = None,
    right_ylim: tuple[float, float] = None,
    left_labelpad: int = None,
    right_labelpad: int = None,
    left_yaxis_label: str = None,
    right_yaxis_label: str = None,
    add_legend: bool = True,
    ax: plt.Axes = None,
    bbox_to_anchor: tuple[float, float] = None,
) -> plt.Figure:
    if ax is None:
        fig, ax_left = plt.subplots(figsize=figsize)
    else:
        ax_left = ax
        fig = ax_left.get_figure()

    x_positions = np.arange(len(data))

    if left_yerr is not None:
        if isinstance(left_yerr, tuple):
            left_yerr_data = data[left_yerr[0]], data[left_yerr[1]]
        else:
            left_yerr_data = data[left_yerr]
    else:
        left_yerr_data = None

    if right_yerr is not None:
        if isinstance(right_yerr, tuple):
            right_yerr_data = data[right_yerr[0]], data[right_yerr[1]]
        else:
            right_yerr_data = data[right_yerr]
    else:
        right_yerr_data = None

    bars_left = ax_left.bar(
        x=x_positions - bar_width / 2,
        height=data[y_col_left],
        width=bar_width,
        color=left_color,
        label=left_label,
        alpha=left_alpha,
        yerr=left_yerr_data,
        capsize=capsize,
    )

    left_yaxis_label = left_yaxis_label if left_yaxis_label is not None else left_label
    ax_left.set_ylabel(ylabel=left_yaxis_label, labelpad=left_labelpad)
    # ax_left.tick_params(axis="y")
    ax_left.set_xticks(x_positions)
    ax_left.set_xticklabels(data[x_col])
    ax_left.set_xlabel(x_label)

    ax_right = ax_left.twinx()

    bars_right = ax_right.bar(
        x=x_positions + bar_width / 2,
        height=data[y_col_right],
        width=bar_width,
        color=right_color,
        label=right_label,
        alpha=right_alpha,
        yerr=right_yerr_data,
        capsize=capsize,
    )
    right_yaxis_label = (
        right_yaxis_label if right_yaxis_label is not None else right_label
    )
    ax_right.set_ylabel(ylabel=right_yaxis_label, labelpad=right_labelpad)
    ax_left.spines.top.set_visible(False)
    ax_right.spines.top.set_visible(False)
    if ax_grid == "left":
        ax_left.grid(alpha=0.2, which="both", zorder=0)
    elif ax_grid == "right":
        ax_left.xaxis.grid(alpha=0.2, which="both", zorder=0)
        ax_right.grid(alpha=0.2, which="both", zorder=0)

    if left_scilimits is not None:
        ax_left.ticklabel_format(style="sci", axis="y", scilimits=left_scilimits)
    if right_scilimits is not None:
        ax_right.ticklabel_format(style="sci", axis="y", scilimits=right_scilimits)

    if left_ylim is not None:
        ax_left.set_ylim(left_ylim)
    if right_ylim is not None:
        ax_right.set_ylim(right_ylim)

    if add_legend:
        legend_kwargs = {
            "loc": "upper center",
            "bbox_to_anchor": bbox_to_anchor,  # Adjust the padding here
            "ncol": 5,
            # "bbox_to_anchor": (0.0, 0.87, 1.05, 0.102),
            "frameon": False,
            "facecolor": "white",
        }
        fig.legend([bars_left, bars_right], [left_label, right_label], **legend_kwargs)
    return fig
