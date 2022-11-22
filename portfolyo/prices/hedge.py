"""Functionality to hedge an offtake profile with a price profile."""

from typing import Tuple

import pandas as pd

from .. import tools
from . import convert
from .utils import is_peak_hour


def _hedge(df: pd.DataFrame, how: str, po: bool) -> pd.Series:
    """
    Hedge a power timeseries, for given price timeseries.

    Parameters
    ----------
    df : pd.DataFrame
        with 'w' [MW] and 'p' [Eur/MWh] columns.
    how : str. One of {'vol', 'val'}
        Hedge-constraint. 'vol' for volumetric hedge, 'val' for value hedge.
    po : bool
        Set to True to split hedge into peak and offpeak values. (Only sensible
        for timeseries with freq=='H' or shorter.)

    Returns
    -------
    pd.Series
        With float values or quantities.
        If po==False, Series with index ['w', 'p'] (power and price in entire period).
        If po==True, Series with multiindex ['peak', 'offpeak'] x ['w', 'p'] (power and
        price, split between peak and offpeak intervals in the period.)

    Notes
    -----
    If the index of `df` doesn't have a .duration attribute, all rows are assumed to be
    of equal duration.
    """

    if not po:
        if df.index.freq:
            # Use magnitude only, so that, if w and p are float series, their return
            # series are also floats (instead of dimensionless Quantities).
            df["dur"] = df.index.duration.pint.m
        else:
            df["dur"] = 1

        # Get single power and price values.
        p_hedge = (df.p * df.dur).sum() / df.dur.sum()
        if how == "vol":  # volume hedge
            # solve for w_hedge: sum (w * duration) == w_hedge * sum (duration)
            w_hedge = (df.w * df.dur).sum() / df.dur.sum()
        elif how == "val":  # value hedge
            # solve for w_hedge: sum (w * duration * p) == w_hedge * sum (duration * p)
            w_hedge = (df.w * df.dur * df.p).sum() / (df.dur * df.p).sum()
        else:
            raise ValueError(f"Parameter `how` must be 'val' or 'vol'; got {how}.")
        return pd.Series({"w": w_hedge, "p": p_hedge})
    else:
        apply_f = lambda df: _hedge(df, how, po=False)  # noqa
        s = df.groupby(is_peak_hour).apply(apply_f)
        return s.rename(index={True: "peak", False: "offpeak"}).stack()


def hedge(
    w: pd.Series,
    p: pd.Series,
    how: str = "val",
    freq: str = "MS",
    po: bool = None,
) -> Tuple[pd.Series]:
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
    freq : {'D' (days), 'MS' (months, default), 'QS' (quarters), 'AS' (years)}
        Frequency of hedging products. E.g. 'QS' to hedge with quarter products.
    po : bool, optional
        Type of hedging products. Set to True to split hedge into peak and offpeak.
        (Default: split if volume timeseries has hourly values or shorter and hedging
        products have monthly frequency or longer.)

    Returns
    -------
    Tuple[pd.Series]
        Power timeseries and price timeseries with hedge of `w` (with same index).
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
    if po is None:  # default: split in peak/offpeak if frequency is short enough
        po = w.index.freq in ["15T", "H"] and freq != "D"
    if po and not (w.index.freq in ["15T", "H"] and freq != "D"):
        raise ValueError(
            "Split into peak and offpeak only possible when (a) hedging with monthly (or "
            "longer) products, and (b) if timeseries have hourly (or shorter) values."
        )

    # Handle possible units.
    win, wunits = (w.pint.magnitude, w.pint.units) if hasattr(w, "pint") else (w, None)
    pin, punits = (p.pint.magnitude, p.pint.units) if hasattr(p, "pint") else (p, None)

    # Only keep full periods of overlapping timestamps.
    i = win.index.intersection(pin.index)
    df = tools.trim.frame(pd.DataFrame({"w": win, "p": pin}).loc[i, :], freq)
    if len(df) == 0:
        return df["w"], df["p"]  # No full periods; don't do hedge; return empty series

    # Do actual hedge.
    group_f = convert.group_function(freq, po)
    vals = df.groupby(group_f).apply(lambda df: _hedge(df, how, False))
    vals.index = pd.MultiIndex.from_tuples(vals.index)
    for c in ["w", "p"]:
        df[c] = df[c].groupby(group_f).transform(lambda gr: vals.loc[gr.name, c])

    # Handle possible units.
    if wunits or punits:
        df = df.astype({"w": f"pint[{wunits:P}]", "p": f"pint[{punits:P}]"})

    return df["w"], df["p"]
