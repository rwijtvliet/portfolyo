from __future__ import annotations

from typing import Dict

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from .. import tools
from ..core.pfline.plot import defaultkwargs
from ..core.pfstate import PfState
from ..tools import visualize as vis


def plot_pfstates(dic: Dict[str, PfState], freq: str = "MS") -> plt.Figure:
    """Plot multiple PfState instances.

    Parameters
    ----------
    dic : Dict[str, PfState]
        Dictionary with PfState instances as values, and their names as the keys.

    Returns
    -------
    plt.Figure
        The figure object to which the instances were plotted.
    """

    gridspec = {"width_ratios": [0.3, 1, 1], "height_ratios": [4, 1] * len(dic)}
    figsize = (15, 5 * len(dic))
    fig, axes = plt.subplots(len(dic) * 2, 3, gridspec_kw=gridspec, figsize=figsize)
    axesgroups = axes.flatten().reshape((len(dic), 6))

    # Share x axes.
    sharex = axesgroups[:, (1, 2, 4)].flatten()
    for ax1, ax2 in zip(sharex[1:], sharex[:-1]):
        ax1.sharex(ax2)
    # Share y axes.
    sharey = axesgroups[:, 2]
    for ax1, ax2 in zip(sharey[1:], sharey[:-1]):
        ax1.sharey(ax2)

    # TODO: resample all to have same index (frequency and length).

    for i, ((pfname, pfs), axes) in enumerate(zip(dic.items(), axesgroups)):
        # If freq is MS or longer: use categorical axes. Plot volumes in MWh.
        # If freq is D or shorter: use time axes. Plot volumes in MW.
        is_category = tools.freq.shortest(pfs.index.freq, "MS") == "MS"

        # Portfolio name.
        axes[0].text(
            0,
            1,
            pfname.replace(" ", "\n"),
            fontsize=12,
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment="left",
        )
        axes[0].axis("off")

        # Volumes.
        if is_category:
            s, kwargs = -1 * pfs.offtakevolume.q, defaultkwargs("q", is_category)
        else:
            s, kwargs = -1 * pfs.offtakevolume.w, defaultkwargs("w", is_category)
        vis.plot_timeseries(axes[1], s, **kwargs)

        # Sourced fraction.
        vis.plot_timeseries(
            axes[2], pfs.sourcedfraction, **defaultkwargs("f", is_category)
        )

        # Empty.
        axes[3].axis("off")

        # Procurement Price.
        vis.plot_timeseries(axes[4], pfs.pnl_cost.p, **defaultkwargs("p", is_category))

        # Empty.
        axes[5].axis("off")

        # Tick formatting.
        axes[2].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        axes[1].yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                lambda x, p: "{:,.0f}".format(x).replace(",", " ")
            )
        )

        for a, ax in enumerate(axes):
            if i == 0 and a in [1, 2]:
                ax.xaxis.set_tick_params(labelbottom=False, labeltop=True, pad=25)
            else:
                ax.xaxis.set_tick_params(labelbottom=False, labeltop=False)

        if i == 0:
            axes[1].set_title("Offtake Volume &\nprocurement price", y=1.27)
            axes[2].set_title("Sourced fraction", y=1.27)

    return
    draw_horizontal_lines(fig, axes)  # draw horizontal lines between portfolios


def draw_horizontal_lines(fig, axes):
    """Function to draw horizontal lines between multiple portfolios.
    This function does not return anything, but tries to plot a 2D line after every 2 axes, eg.
    after (0,2), (0,4),... beacuse each portfolio requires 2x4 axes in the fig (where rows=2, columns=4).

    Parameters
    ----------
    fig : plt.subplots()
    axes : plt.subplots()
    """
    # rearange the axes for no overlap
    fig.tight_layout()

    # Get the bounding boxes of the axes including text decorations
    r = fig.canvas.get_renderer()
    bboxes = np.array(
        [
            ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
            for ax in axes.flat
        ],
        matplotlib.transforms.Bbox,
    ).reshape(axes.shape)

    """TO CORRECT: the horizontal line is not exactly in the middle of two graphs.
    It is more inclined towards the second or next graph in the queue.
    Each pftstate has 4x4 grid and this is plotted in the same graph, but as subgraphs.
    """

    # Get the minimum and maximum extent, get the coordinate half-way between those
    ymax = (
        np.array(list(map(lambda b: b.y1, bboxes.flat))).reshape(axes.shape).max(axis=1)
    )
    ymin = (
        np.array(list(map(lambda b: b.y0, bboxes.flat))).reshape(axes.shape).min(axis=1)
    )
    ys = np.c_[ymax[2:-1:2], ymin[1:-2:2]].mean(axis=1)
    ys = [ymax[0], *ys]

    # Draw a horizontal lines at those coordinates
    for y in ys:
        line = plt.Line2D([0, 1], [y, y], transform=fig.transFigure, color="black")
        fig.add_artist(line)
