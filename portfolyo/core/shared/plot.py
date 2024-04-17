"""
Module with mixins, to add 'plot-functionality' to PfLine and PfState classes.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Dict, Tuple

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ... import tools
from ... import visualize as vis
from ..pfline import classes
from ..pfline.enums import Kind

if TYPE_CHECKING:  # needed to avoid circular imports
    from ..pfline import PfLine
    from ..pfstate import PfState


DEFAULTFMT = {
    "w": "{:,.1f}",
    "q": "{:,.0f}",
    "p": "{:,.2f}",
    "r": "{:,.0f}",
    "f": "{:.0%}",
}


def plotfn_and_kwargs(
    col: str, freq: str, name: str
) -> Tuple[vis.PlotTimeseriesToAxFunction, Dict]:
    """Get correct function to plot as well as default kwargs. ``col``: one of 'qwprf',
    ``freq``: frequency; ``name``: name of the child. If name is emptystring, it is the
    parent of a plot which also has children. If it is None, there are no children."""
    # Get plot function.
    if tools.freq.shortest(freq, "MS") == "MS":  # categorical
        if name == "" or name is None:  # parent
            fn = vis.plot_timeseries_as_bar
        else:  # child
            fn = vis.plot_timeseries_as_hline
    else:  # timeaxis
        if col in ["w", "q"]:
            if name == "" or name is None:  # parent
                fn = vis.plot_timeseries_as_area
            else:  # child
                fn = vis.plot_timeseries_as_step
        else:  # col in ['p', 'r']
            if name == "" or name is None:  # parent
                fn = vis.plot_timeseries_as_step
            else:  # child
                fn = vis.plot_timeseries_as_step

    # Get plot default kwargs.
    if name is None:  # no children
        kwargs = {
            "color": getattr(vis.Colors.Wqpr, col, "grey"),
            "alpha": 0.7,
            "labelfmt": DEFAULTFMT.get(col, "{:.2f}"),
        }
    elif name == "":  # parent with children
        kwargs = {
            "color": "grey",
            "alpha": 0.7,
            "labelfmt": DEFAULTFMT.get(col, "{:.2f}"),
        }
    else:  # child with name
        hashed_value = hashlib.sha256(name.encode()).hexdigest()
        hashed_int = int(hashed_value, 16)
        index = hashed_int % len(vis.Colors.General)
        kwargs = {
            "color": list(vis.Colors.General)[index].value,
            "alpha": 0.9,
            "labelfmt": "",  # no labels on children
            "label": name,
            "linewidth": 0.5,
        }

    return fn, kwargs


class PfLinePlot:
    def plot_to_ax(
        self, ax: plt.Axes, children: bool = False, kind: Kind = None, **kwargs
    ) -> None:
        """Plot a specific dimension (i.e., kind) of the PfLine to a specific axis.

        Parameters
        ----------
        ax : plt.Axes
            The axes object to which to plot the timeseries.
        children : bool, optional (default: False)
            If True, plot also the direct children of the PfLine.
        kind : Kind, optional (default: None)
            What dimension of the data to plot. Ignored unless PfLine.kind is COMPLETE.
        **kwargs
            Any additional kwargs are passed to the pd.Series.plot function when drawing
            the parent.

        Returns
        -------
        None
        """
        # Ensure ``kind`` is volume, price, or revenue.
        if self.kind is not Kind.COMPLETE:
            kind = self.kind
        elif kind not in [Kind.VOLUME, Kind.PRICE, Kind.REVENUE]:
            raise ValueError(
                "To plot a complete portfolio line, the dimension to be plotted must be specified. "
                f"Parameter ``kind`` must be one of {{Kind.VOLUME, Kind.PRICE, Kind.REVENUE}}; got {kind}."
            )

        # Create function to select correct series of the pfline.
        def col_and_series(pfl: PfLine) -> Tuple[str, pd.Series]:
            if kind is Kind.PRICE:
                return "p", pfl.p
            elif kind is Kind.REVENUE:
                return "r", pfl.r
            elif tools.freq.longest(pfl.index.freq, "D") == "D":  # timeaxis
                return "w", pfl.w  # kind is Kind.VOLUME
            else:
                return "q", pfl.q  # kind is Kind.VOLUME

        # Plot top-level data first.
        col, s = col_and_series(self)
        fn, d_kwargs = plotfn_and_kwargs(col, self.index.freq, "" if children else None)
        fn(ax, s, **(d_kwargs | kwargs))

        # Plot children if wanted and available.
        if not children or not isinstance(self, classes.NestedPfLine):
            return
        for name, child in self.items():
            col, s = col_and_series(child)
            fn, d_kwargs = plotfn_and_kwargs(col, self.index.freq, name)
            fn(ax, s, **d_kwargs)
        ax.legend()

    def plot(self, children: bool = False) -> plt.Figure:
        """Plot the PfLine.

        Parameters
        ----------
        children : bool, optional (default: False)
            If True, plot also the direct children of the PfLine.

        Returns
        -------
        plt.Figure
            The figure object to which the series was plotted.
        """

        if self.kind is not Kind.COMPLETE:
            # one axes
            fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(10, 3))
            self.plot_to_ax(ax, children)

        else:
            fig, axes = plt.subplots(3, 1, sharex=True, squeeze=True, figsize=(10, 9))
            for ax, kind in zip(axes, [Kind.VOLUME, Kind.PRICE, Kind.REVENUE]):
                self.plot_to_ax(ax, children, kind)

        return fig


# TODO: ----- below here must still be rewritten ---


def defaultkwargs(*args):
    return {}


class PfStatePlot:
    def plot(self: PfState, children: bool = False) -> plt.Figure:
        """Plot the portfolio state.

        Parameters
        ----------
        None

        Returns
        -------
        plt.Figure
            The figure object to which the series was plotted.
        """
        gridspec = {"width_ratios": [1, 1, 1], "height_ratios": [4, 1]}
        fig, axes = plt.subplots(
            2, 3, sharex=True, gridspec_kw=gridspec, figsize=(10, 6)
        )
        axes = axes.flatten()
        axes[1].sharey(axes[0])
        axes[2].sharey(axes[0])
        axes[4].sharey(axes[3])
        axes[5].sharey(axes[3])

        # If freq is MS or longer: use categorical axes. Plot volumes in MWh.
        # If freq is D or shorter: use time axes. Plot volumes in MW.
        is_category = tools.freq.shortest(self.index.freq, "MS") == "MS"
        so, ss, usv = (
            -1 * self.offtakevolume,
            self.sourced,
            self.unsourced,
        )
        pr_kwargs = defaultkwargs("p", is_category)
        # Volumes.
        if is_category:
            kwargs = defaultkwargs("q", is_category)
            value = "q"
        else:
            kwargs = defaultkwargs("w", is_category)
            value = "w"
        so.plot_to_ax(axes[0], value, children=children, **kwargs)
        ss.plot_to_ax(axes[1], value, children=children, **kwargs)
        # Unsourced volume.
        usv.plot_to_ax(axes[2], value, **kwargs)
        # Procurement Price.
        self.pnl_cost.plot_to_ax(
            axes[3],
            "p",
            **pr_kwargs,
        )
        self.sourced.plot_to_ax(
            axes[4],
            "p",
            children=children,
            **pr_kwargs,
        )
        # Unsourced price
        self.unsourced.plot_to_ax(
            axes[5],
            "p",
            **pr_kwargs,
        )
        # Set titles.
        axes[0].set_title("Offtake volume")
        axes[1].set_title("Sourced volume")
        axes[2].set_title("Unsourced volume")
        axes[3].set_title("Procurement price")
        axes[4].set_title("Sourced price")
        axes[5].set_title("Unsourced price")

        # Format tick labels.
        formatter = matplotlib.ticker.FuncFormatter(
            lambda x, p: "{:,.0f}".format(x).replace(",", " ")
        )
        axes[0].yaxis.set_major_formatter(formatter)
        axes[1].yaxis.set_major_formatter(formatter)
        # axes[3].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

        # Set ticks.
        axes[0].xaxis.set_tick_params(labeltop=False, labelbottom=True)
        axes[1].xaxis.set_tick_params(labeltop=False, labelbottom=True)
        axes[2].xaxis.set_tick_params(labeltop=False, labelbottom=True)
        axes[3].xaxis.set_tick_params(labeltop=False, labelbottom=False)
        axes[4].xaxis.set_tick_params(labeltop=False, labelbottom=False)
        axes[5].xaxis.set_tick_params(labeltop=False, labelbottom=False)

        fig.tight_layout()
        return fig


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
