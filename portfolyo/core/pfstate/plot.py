from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
from matplotlib import pyplot as plt

from ..pfline import Kind

if TYPE_CHECKING:
    from .pfstate import PfState


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
        fig, (volumeaxes, priceaxes) = plt.subplots(
            2, 3, sharex=True, sharey="row", gridspec_kw=gridspec, figsize=(10, 6)
        )

        so, ss, usv = (
            -1 * self.offtakevolume,
            self.sourced,
            self.unsourced,
        )

        so.plot_to_ax(volumeaxes[0], children=children, kind=so.kind, labelfmt="")
        ss.plot_to_ax(volumeaxes[1], children=children, kind=Kind.VOLUME, labelfmt="")
        # Unsourced volume.
        usv.plot_to_ax(volumeaxes[2], kind=Kind.VOLUME, labelfmt="")
        # Procurement Price.
        self.pnl_cost.plot_to_ax(priceaxes[0], kind=Kind.PRICE, labelfmt="")
        self.sourced.plot_to_ax(
            priceaxes[1], children=children, kind=Kind.PRICE, labelfmt=""
        )
        # Unsourced price
        self.unsourced.plot_to_ax(priceaxes[2], kind=Kind.PRICE, labelfmt="")
        # Set titles.
        volumeaxes[0].set_title("Offtake volume")
        volumeaxes[1].set_title("Sourced volume")
        volumeaxes[2].set_title("Unsourced volume")
        priceaxes[0].set_title("Procurement price")
        priceaxes[1].set_title("Sourced price")
        priceaxes[2].set_title("Unsourced price")

        PfStatePlot.set_max_min_limits(volumeaxes)
        PfStatePlot.set_max_min_limits(priceaxes)

        # Format tick labels.
        formatter = matplotlib.ticker.FuncFormatter(
            lambda x, p: "{:,.0f}".format(x).replace(",", " ")
        )
        volumeaxes[0].yaxis.set_major_formatter(formatter)
        priceaxes[0].yaxis.set_major_formatter(formatter)
        # axes[3].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

        # Set ticks.
        for ax in volumeaxes:
            ax.xaxis.set_tick_params(labeltop=False, labelbottom=True)
        for ax in priceaxes:
            ax.xaxis.set_tick_params(labeltop=False, labelbottom=False)

        fig.tight_layout()
        return fig

    def set_max_min_limits(axes: plt.Axes):
        limit = [ax.get_ylim() for ax in axes]
        mins_vol, maxs_vol = zip(*limit)

        themin, themax = min(mins_vol), max(maxs_vol)
        for ax in axes:
            ax.set_ylim(themin * 1.1, themax * 1.1)
