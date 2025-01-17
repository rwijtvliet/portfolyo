"""Functionality to hedge an offtake profile with a price profile."""

from typing import Tuple

import pandas as pd

from . import isboundary as tools_boundary

from . import duration as tools_duration
from . import intersect as tools_intersect
from . import peakconvert as tools_peakconvert
from . import peakfn as tools_peakfn
from . import trim as tools_trim
from . import wavg as tools_wavg


def one_hedge(df: pd.DataFrame, how: str) -> pd.Series:
    """Hedge over all timestamps in dataframe. Dataframe must have columns
    'w', 'p', 'duration'. Returns Series with index 'w', 'p', with hedge values.
    for the entire period."""
    # Prepare weights.
    if how == "val":  # value hedge
        # solve for w_hedge: sum(w * duration * p) == w_hedge * sum(duration * p)
        # so: w_hedge = sum(w * duration * p) / sum(duration * p)
        weights = df["p"] * df["duration"]
    else:  # how == "vol":  # volume hedge
        # solve for w_hedge: sum(w * duration) == w_hedge * sum(duration)
        # so: w_hedge = sum(w * duration) / sum(duration)
        weights = df["duration"]

    p_hedge = tools_wavg.series(df["p"], df["duration"])
    w_hedge = tools_wavg.series(df["w"], weights)
    return pd.Series({"w": w_hedge, "p": p_hedge})


def hedge(
    w: pd.Series,
    p: pd.Series,
    how: str = "val",
    peak_fn: tools_peakfn.PeakFunction = None,
    freq: str = "MS",
) -> Tuple[pd.Series, pd.Series]:
    """
    Make hedge of power timeseries, for given price timeseries.

    Parameters
    ----------
    w : Series
        Power timeseries with spot market frequency.
    p: Series
        Price timeseries with same frequency.
    how : str, optional (Default: 'val')
        Hedge-constraint. 'vol' for volumetric hedge, 'val' for value hedge.
    peak_fn : PeakFunction, optional (default: None)
        Function that returns boolean Series indicating if timestamps in index lie in peak period.
        If None, hedge with base products.
    freq : {'D' (days), 'MS' (months), 'QS' (quarters), 'YS' (years)}, optional (default: 'MS')
        Frequency of hedging products. E.g. 'QS' to hedge with quarter products.

    Returns
    -------
    Tuple[Series, Series]
        Power timeseries and price timeseries with hedge of ``w`` (with same index).
    """
    if w.index.freq is None or p.index.freq is None:
        raise ValueError(
            "Parameters ``w`` and ``p`` must have a DatetimeIndex with a set frequency attribute."
        )
    if w.index.freq != p.index.freq:
        raise ValueError(
            f"Parameters ``w`` and ``p`` must have same frequency; got {w.index.freq} and {p.index.freq}."
        )
    if w.index.freq not in ["15min", "h", "D"]:
        raise ValueError("Can only hedge a timeseries with daily (or shorter) values.")
    # ATTN!: changed due to new freq
    if not any(
        tools_boundary.freq_to_string(freq).startswith(t)
        for t in ["D", "MS", "QS", "YS"]
    ):
        raise ValueError(
            f"Parameter ``freq`` must be one of 'D', 'MS', 'QS', 'YS'; got '{freq}'."
        )
    if peak_fn is not None and not (w.index.freq in ["15min", "h"] and freq != "D"):
        raise ValueError(
            "Split into peak and offpeak only possible when (a) hedging with monthly (or "
            "longer) products, and (b) if timeseries have hourly (or shorter) values."
        )
    if how not in ["vol", "val"]:
        raise ValueError(f"Parameter `how` must be 'val' or 'vol'; got {how}.")

    # Handle possible units.
    win, wunits = (w.pint.magnitude, w.pint.units) if hasattr(w, "pint") else (w, None)
    pin, punits = (p.pint.magnitude, p.pint.units) if hasattr(p, "pint") else (p, None)

    # Only keep full periods of overlapping timestamps.

    win, pin = tools_intersect.frames(win, pin)
    dfin = tools_trim.frame(pd.DataFrame({"w": win, "p": pin}), freq)
    if len(dfin) == 0:
        return (
            dfin["w"],
            dfin["p"],
        )  # No full periods; don't do hedge; return empty series

    # Do actual hedge.
    # . helper values
    dfin["duration"] = tools_duration.index(dfin.index)
    grouping = tools_peakconvert.group_index(dfin.index, peak_fn, freq)
    # . calculation
    vals = dfin.groupby(grouping).apply(lambda subdf: one_hedge(subdf, how))
    vals.index = pd.MultiIndex.from_tuples(vals.index)
    # . broadcast to original timeseries
    dfout = vals.loc[grouping, :].set_axis(dfin.index)
    wout, pout = dfout["w"], dfout["p"]

    # Handle possible units.
    if wunits or punits:
        wout, pout = wout.astype(f"pint[{wunits}]"), pout.astype(f"pint[{punits}]")

    return wout, pout
