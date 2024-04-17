"""
Module with mixins, to add 'plot-functionality' to PfLine and PfState classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

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
        self: PfLine, ax: plt.Axes, col: str, how: str, labelfmt: str, **kwargs
    ) -> None:
        """Plot a timeseries of the PfLine to a specific axes.

        Parameters
        ----------
        ax : plt.Axes
            The axes object to which to plot the timeseries.
        col : str
            The column to plot. One of {'w', 'q', 'p', 'r'}.
        how : str
            How to plot the data. One of {'jagged', 'bar', 'area', 'step', 'hline'}.
        labelfmt : str
            Labels are added to each datapoint in the specified format. ('' to add no labels)
        Any additional kwargs are passed to the pd.Series.plot function.
        """
        if col not in self.kind.available:
            raise ValueError(
                f"For this PfLine, parameter ``col`` must be one of {', '.join(self.kind.available)}; got {col}."
            )
        vis.plot_timeseries(ax, getattr(self, col), how, labelfmt, **kwargs)

    def plot(self: PfLine, cols: str = None) -> plt.Figure:
        """Plot one or more timeseries of the PfLine.

        Parameters
        ----------
        cols : str, optional
            The columns to plot. Default: plot volume (in [MW] for daily values and
            shorter, [MWh] for monthly values and longer) and price `p` [Eur/MWh]
            (if available).

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
            s = getattr(self, col)
            vis.plot_timeseries(ax, s, **kwargs)

        return fig


class PfStatePlot:
    def plot(self: PfState) -> plt.Figure:
        """Plot the portfolio state.

        Parameters
        ----------
        None

        Returns
        -------
        plt.Figure
            The figure object to which the series was plotted.
        """
        gridspec = {"width_ratios": [1, 1], "height_ratios": [4, 1]}
        fig, axes = plt.subplots(
            2, 2, sharex=True, gridspec_kw=gridspec, figsize=(10, 6)
        )
        axes = axes.flatten()
        axes[0].sharey(axes[1])

        # If freq is MS or longer: use categorical axes. Plot volumes in MWh.
        # If freq is D or shorter: use time axes. Plot volumes in MW.
        is_category = tools.freq.shortest(self.index.freq, "MS") == "MS"

        # Volumes.
        if is_category:
            so, ss = -1 * self.offtakevolume.q, self.sourced.q
            kwargs = defaultkwargs("q", is_category)
        else:
            so, ss = -1 * self.offtakevolume.w, self.sourced.w
            kwargs = defaultkwargs("w", is_category)
        vis.plot_timeseries(axes[0], so, **kwargs)
        vis.plot_timeseries(axes[1], ss, **kwargs)

        # Procurement Price.
        vis.plot_timeseries(axes[2], self.pnl_cost.p, **defaultkwargs("p", is_category))

        # Sourced fraction.
        vis.plot_timeseries(
            axes[3], self.sourcedfraction, **defaultkwargs("f", is_category)
        )

        # Empty.

        # Set titles.
        axes[0].set_title("Offtake volume")
        axes[1].set_title("Sourced volume")
        axes[2].set_title("Procurement price")
        axes[3].set_title("Sourced fraction")

        # Format tick labels.
        formatter = matplotlib.ticker.FuncFormatter(
            lambda x, p: "{:,.0f}".format(x).replace(",", " ")
        )
        axes[0].yaxis.set_major_formatter(formatter)
        axes[1].yaxis.set_major_formatter(formatter)
        axes[3].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

        # Set ticks.
        axes[0].xaxis.set_tick_params(labeltop=False, labelbottom=True)
        axes[1].xaxis.set_tick_params(labeltop=False, labelbottom=True)
        axes[2].xaxis.set_tick_params(labeltop=False, labelbottom=False)
        axes[3].xaxis.set_tick_params(labeltop=False, labelbottom=False)

        fig.tight_layout()
        return fig
