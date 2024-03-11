"""
Module with mixins, to add 'plot-functionality' to PfLine and PfState classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from ... import tools
from ... import visualize as vis

if TYPE_CHECKING:  # needed to avoid circular imports
    from ..pfline import PfLine
    from ..pfstate import PfState


DEFAULTHOW = {"r": "bar", "q": "bar", "p": "hline", "w": "area", "f": "area"}
DEFAULTFMT = {
    "w": "{:,.1f}",
    "q": "{:,.0f}",
    "p": "{:,.2f}",
    "r": "{:,.0f}",
    "f": "{:.0%}",
}


def defaultkwargs(col: str, is_cat: bool):
    """Styling and type of graph, depending on column ``col`` and whether or not the x-axis
    is a category axis (``is_cat``)."""
    kwargs = {}
    kwargs["alpha"] = 0.8
    # Add defaults for each column.
    kwargs["color"] = getattr(vis.Colors.Wqpr, col, "grey")
    kwargs["labelfmt"] = DEFAULTFMT.get(col, "{:.2f}")
    kwargs["how"] = DEFAULTHOW.get(col, "hline")
    kwargs["cat"] = is_cat
    # Override specific cases.
    if not is_cat and col == "p":
        kwargs["how"] = "step"
    if is_cat and col == "f":
        kwargs["how"] = "bar"
    return kwargs


class PfLinePlot:
    def plot_to_ax(
        self: PfLine,
        ax: plt.Axes,
        col: str,
        how: str,
        labelfmt: str = "",
        children: bool = False,
        **kwargs,
    ) -> None:
        """Plot a timeseries of the PfLine to a specific axes.

        Parameters
        ----------
        ax : plt.Axes
            The axes object to which to plot the timeseries.
        col : str
            The column to plot. One of {'w', 'q', 'p', 'r'}.
        how : str
            How to plot the data. One of {'bar', 'area', 'step', 'hline'}.
        labelfmt : str
            Labels are added to each datapoint in the specified format. ('' to add no labels)
        Any additional kwargs are passed to the pd.Series.plot function.
        """

        if col not in self.kind.available:
            raise ValueError(
                f"For this PfLine, parameter ``col`` must be one of {', '.join(self.kind.available)}; got {col}."
            )
        if children:
            # Plot on category axis if freq monthly or longer, else on time axis.
            is_category = tools.freq.shortest(self.index.freq, "MS") == "MS"
            # adjust kwargs for parent if plotting children
            if how == "bar":
                kwargs["color"] = "none"
                kwargs["edgecolor"] = "seagreen"
                kwargs["linewidth"] = 2
            if how == "area":
                how = "step"

            self.plot_children(col, ax, is_category)
            ax.legend()
        vis.plot_timeseries(ax, getattr(self, col), how, labelfmt, **kwargs)

    def plot(self: PfLine, cols: str = None, children: bool = False) -> plt.Figure:
        """Plot one or more timeseries of the PfLine.

        Parameters
        ----------
        cols : str, optional
            The columns to plot. Default: plot volume (in [MW] for daily values and
            shorter, [MWh] for monthly values and longer) and price `p` [Eur/MWh]
            (if available).
        children : bool, optional (default: False)
            If True, plot also the direct children of the PfLine.

        Returns
        -------
        plt.Figure
            The figure object to which the series was plotted.
        """
        # Plot on category axis if freq monthly or longer, else on time axis.
        is_category = tools.freq.shortest(self.index.freq, "MS") == "MS"

        # If columns are specified, plot these. Else: take defaults, based on what's available
        if cols is None:
            cols = ""
            if "q" in self.kind.available:
                cols += "q" if is_category else "w"
            if "p" in self.kind.available:
                cols += "p"
        else:
            cols = [col for col in cols if col in self.kind.available]
            if not cols:
                raise ValueError("No columns to plot.")

        # Create the plots.
        size = (10, len(cols) * 3)
        fig, axes = plt.subplots(
            len(cols), 1, sharex=True, sharey=False, squeeze=False, figsize=size
        )

        for col, ax in zip(cols, axes.flatten()):
            kwargs = defaultkwargs(col, is_category)

            if children:
                # adjust kwargs for parent if plotting children
                if kwargs["how"] == "bar":
                    kwargs["color"] = "none"
                    kwargs["edgecolor"] = "seagreen"
                    kwargs["linewidth"] = 2
                if kwargs["how"] == "area":
                    kwargs["how"] = "step"

                self.plot_children(col, ax, is_category)
                ax.legend()
            kwargs["alpha"] = 0.8
            s = getattr(self, col)
            vis.plot_timeseries(ax, s, **kwargs)

        return fig

    def get_stacked_offsets(self: PfLine, col: str) -> Dict[str, List[List[float]]]:
        # Calculates offset for each child based on the height of the previous child
        # It disdinguishes between postive and negative offsets
        # Saves values of offsets in 2 dimmensional array: for positive and negative offsets
        return_val = {}

        bottom_offset = [[0.0, 0.0] for i in range(0, self.index.size)]
        for name, child in self.items():
            bar_heights = getattr(child, col)
            offsets = [0.0 for i in range(0, self.index.size)]
            for i in range(0, len(bar_heights)):
                obj = bar_heights[i]
                if obj.magnitude > 0:
                    offsets[i] = bottom_offset[i][0]
                    bottom_offset[i][0] += obj.magnitude
                else:
                    offsets[i] = bottom_offset[i][1]
                    bottom_offset[i][1] += obj.magnitude

            return_val[name] = offsets

        return return_val

    def get_children_with_colors(self: PfLine) -> List[Tuple[str, PfLine, vis.Color]]:
        return [
            (name, child, self.hash_and_map_to_color(name))
            for (name, child) in self.items()
        ]

    def hash_and_map_to_color(self: PfLine, name: str) -> vis.Color:
        # Use a hash function to hash the name
        hashed_value = hash(name)
        # Calculate the index in the General colors enum based on the hashed value
        index = hashed_value % len(vis.Colors.General)
        # Return the color associated with the index
        return list(vis.Colors.General)[index].value

    def plot_children(self: PfLine, col: str, ax: plt.Axes, is_category: bool) -> None:
        """Plot children of the PfLine to the same ax as parent.

        Parameters
        ----------
        cols : str, optional
            The columns to plot. Default: plot volume (in [MW] for daily values and
            shorter, [MWh] for monthly values and longer) and price `p` [Eur/MWh]
            (if available).
        ax : plt.Axes
            The axes object to which to plot the timeseries.

        """
        kwargs = defaultkwargs(col, is_category)
        kwargs["labelfmt"] = ""
        is_stacked_type = kwargs["how"] == "bar" or kwargs["how"] == "area"

        if is_stacked_type:
            offsets = self.get_stacked_offsets(col)
        colors = []
        for name, child, color in self.get_children_with_colors():
            kwargs["color"] = color
            kwargs["label"] = name
            colors.append(color)
            if is_stacked_type:
                kwargs["bottom"] = offsets[name]
                kwargs["hatch"] = ".."
                kwargs["alpha"] = 0.7

            vis.plot_timeseries(ax, getattr(child, col), **kwargs)


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
        axes[2].xaxis.set_tick_params(labeltop=False, labelbottom=False)
        axes[3].xaxis.set_tick_params(labeltop=False, labelbottom=False)

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
