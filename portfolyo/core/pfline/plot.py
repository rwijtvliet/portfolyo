"""
Module with mixins, to add 'plot-functionality' to PfLine and PfState classes.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Dict, Tuple

import pandas as pd
from matplotlib import pyplot as plt

from ... import tools
from . import classes
from .enums import Kind

if TYPE_CHECKING:
    from .classes import PfLine


DEFAULTFMT = {
    "w": "{:,.1f}",
    "q": "{:,.0f}",
    "p": "{:,.2f}",
    "r": "{:,.0f}",
    "f": "{:.0%}",
}


def defaultkwargs(name: str, col: str):
    # Get plot default kwargs.
    if name is None:  # no children
        kwargs = {
            "color": getattr(tools.visualize.Colors.Wqpr, col, "grey"),
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
        index = hashed_int % len(tools.visualize.Colors.General)
        kwargs = {
            "color": list(tools.visualize.Colors.General)[index].value,
            "alpha": 0.9,
            "labelfmt": "",  # no labels on children
            "label": name,
            "linewidth": 0.9,
        }

    return kwargs


def plotfn_and_kwargs(
    col: str, freq: str, name: str
) -> Tuple[tools.visualize.PlotTimeseriesToAxFunction, Dict]:
    """Get correct function to plot as well as default kwargs. ``col``: one of 'qwprf',
    ``freq``: frequency; ``name``: name of the child. If name is emptystring, it is the
    parent of a plot which also has children. If it is None, there are no children."""
    # Get plot function.
    if tools.freq.shortest(freq, "MS") == "MS":  # categorical
        if name == "" or name is None:  # parent
            fn = tools.visualize.plot_timeseries_as_bar
        else:  # child
            fn = tools.visualize.plot_timeseries_as_hline
    else:  # timeaxis
        if col in ["w", "q"]:
            if name == "" or name is None:  # parent
                fn = tools.visualize.plot_timeseries_as_area
            else:  # child
                fn = tools.visualize.plot_timeseries_as_step
        else:  # col in ['p', 'r']
            if name == "" or name is None:  # parent
                fn = tools.visualize.plot_timeseries_as_step
            else:  # child
                fn = tools.visualize.plot_timeseries_as_step

    kwargs = defaultkwargs(name, col)

    return fn, kwargs


class PfLinePlot:
    def plot_to_ax(
        self: PfLine,
        ax: plt.Axes,
        children: bool = False,
        kind: Kind = None,
        **kwargs,
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
        def col_and_series(pfl: classes.PfLine) -> Tuple[str, pd.Series]:
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
        # s = s.pint.m
        fn, d_kwargs = plotfn_and_kwargs(col, self.index.freq, "" if children else None)
        fn(ax, s, **(d_kwargs | kwargs))

        # Plot children if wanted and available.
        if not children or not isinstance(self, classes.NestedPfLine):
            return
        for name, child in self.items():
            col, s = col_and_series(child)
            # s = s.pint.m
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
