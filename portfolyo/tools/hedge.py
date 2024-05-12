"""Functionality to hedge an offtake profile with a price profile."""

from typing import Tuple

import pandas as pd

from . import duration as tools_duration
from . import intersect as tools_intersect
from . import peakconvert as tools_peakconvert
from . import peakfn as tools_peakfn
from . import trim as tools_trim
from . import wavg as tools_wavg


def _hedge(
    df: pd.DataFrame,
    how: str,
    peak_fn: tools_peakfn.PeakFunction = None,
    freq: str = "MS",
) -> pd.Series:
    """
    Hedge a power timeseries, for given price timeseries.

    Parameters
    ----------
    df : pd.DataFrame
        with 'w' [MW] and 'p' [Eur/MWh] columns.
    how : str. One of {'vol', 'val'}
        Hedge-constraint. 'vol' for volumetric hedge, 'val' for value hedge.
    peak_fn : PeakFunction, optional (default: None)
        Function that returns boolean Series indicating if timestamps in index lie in peak period.
        If None, hedge with base products.
    freq : {'D' (days), 'MS' (months, default), 'QS' (quarters), 'AS' (years)}
        Frequency of hedging products. E.g. 'QS' to hedge with quarter products.

    Returns
    -------
    pd.Series
        With float values or quantities.
        If peak_fn is None, Series with index with 2 values (['w', 'p']; power and
        price in entire period).
        If peak_fn is not None, Series with multiindex (['peak', 'offpeak'] x ['w', 'p'];
        power and price, split between peak and offpeak intervals in the period).
    """
    df = df.copy()

    # Use magnitude of duration only, so that, if w and p are float series, their
    # return series are also floats (instead of dimensionless Quantities).
    df["duration"] = tools_duration.frame(df).pint.m

    # Prepare weights.
    if how == "vol":  # volume hedge
        # solve for w_hedge: sum(w * duration) == w_hedge * sum(duration)
        # so: w_hedge = sum(w * duration) / sum(duration)
        df["weights"] = df["duration"]
    elif how == "val":  # value hedge
        # solve for w_hedge: sum(w * duration * p) == w_hedge * sum(duration * p)
        # so: w_hedge = sum(w * duration * p) / sum(duration * p)
        df["weights"] = df["p"] * df["duration"]
    else:
        raise ValueError(f"Parameter `how` must be 'val' or 'vol'; got {how}.")

    # Grouping.
    grouping = tools_peakconvert.group_arrays(df.index, freq, peak_fn)

    # Do hedge.
    def do_hedge(sub_df):
        p_hedge = tools_wavg.series(sub_df["p"], sub_df["duration"])
        w_hedge = tools_wavg.series(sub_df["w"], sub_df["weights"])
        return pd.Series({"w": w_hedge, "p": p_hedge})

    return df.groupby(grouping).transform(do_hedge)


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
        Power timeseries with hourly or quarterhourly frequency.
    p: Series
        Price timeseries with same frequency.
    how : str, optional (Default: 'val')
        Hedge-constraint. 'vol' for volumetric hedge, 'val' for value hedge.
    peak_fn : PeakFunction, optional (default: None)
        Function that returns boolean Series indicating if timestamps in index lie in peak period.
        If None, hedge with base products.
    freq : {'D' (days), 'MS' (months, default), 'QS' (quarters), 'AS' (years)}
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
    if w.index.freq not in ["15T", "H", "D"]:
        raise ValueError("Can only hedge a timeseries with daily (or shorter) values.")
    if freq not in ["D", "MS", "QS", "AS"]:
        raise ValueError(
            f"Parameter ``freq`` must be one of 'D', 'MS', 'QS', 'AS'; got '{freq}'."
        )
    if peak_fn is not None and not (w.index.freq in ["15T", "H"] and freq != "D"):
        raise ValueError(
            "Split into peak and offpeak only possible when (a) hedging with monthly (or "
            "longer) products, and (b) if timeseries have hourly (or shorter) values."
        )

    # Handle possible units.
    win, wunits = (w.pint.magnitude, w.pint.units) if hasattr(w, "pint") else (w, None)
    pin, punits = (p.pint.magnitude, p.pint.units) if hasattr(p, "pint") else (p, None)

    # Only keep full periods of overlapping timestamps.

    win, pin = tools_intersect.frames(win, pin)
    df = tools_trim.frame(pd.DataFrame({"w": win, "p": pin}), freq)
    if len(df) == 0:
        return df["w"], df["p"]  # No full periods; don't do hedge; return empty series

    # Do actual hedge.
    df2 = _hedge(df, how, freq, peak_fn)

    # Handle possible units.
    if wunits or punits:
        df2 = df2.astype({"w": f"pint[{wunits}]", "p": f"pint[{punits}]"})

    return df2["w"], df2["p"]
