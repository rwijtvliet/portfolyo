"""Module to convert between timeseries and base/peak/offpeak-values."""

import warnings
from typing import List

import numpy as np
import pandas as pd

from . import changefreq as tools_changefreq
from . import duration as tools_duration
from . import freq as tools_freq
from . import peakfn as tools_peakfn
from . import trim as tools_trim
from . import wavg as tools_wavg

BPO = ("base", "peak", "offpeak")


def group_arrays(
    i: pd.DatetimeIndex, freq: str, peak_fn: tools_peakfn.PeakFunction = None
) -> List:
    """Function to group all rows that belong to same delivery period."""
    # Grouping due to delivery period.
    groups = [i.year]
    if freq == "MS":
        groups.append(i.month)
    elif freq == "QS":
        groups.append(i.quarter)
    elif freq == "AS":
        pass
    else:
        raise ValueError(
            f"Parameter ``freq`` must be one of 'MS', 'QS', 'AS'; got '{freq}'."
        )

    # Add grouping due to peak.
    if peak_fn is not None:
        groups.append(peak_fn(i))

    return groups


def complete_bpoframe(
    df: pd.DataFrame, peak_fn: tools_peakfn.PeakFunction
) -> pd.DataFrame:
    """Complete a dataframe so it contains arbitrage-free base, peak, and offpeak values.

    Parameters
    ----------
    df : DataFrame
        With at least 2 of following columns: {'base', 'peak', 'offpeak'}. If all 3 are
        present, no verification is done. Additional columns are dropped. DatetimeIndex
        with freq in {'MS', 'QS', 'AS'}.
    peak_fn : PeakFunction
        Function that returns boolean Series indicating if timestamps in index lie in peak period.

    Returns
    -------
    DataFrame
        Same as ``df``, with all 3 columns ('base', 'peak', 'offpeak').

    Notes
    -----
    Can only be used for values that are 'averagable' over a time period, like power [MW]
    and price [Eur/MWh]. Not for e.g. energy [MWh], revenue [Eur], and duration [h].

    In:

                                peak        offpeak
    ts_left
    2020-01-01 00:00:00+01:00   42.530036   30.614701
    2020-02-01 00:00:00+01:00   33.295167   15.931557
                                ...         ...
    2020-11-01 00:00:00+01:00   49.110873   33.226004
    2020-12-01 00:00:00+01:00   57.872246   35.055449
    12 rows × 2 columns

    Out:

                                base        peak        offpeak
    ts_left
    2020-01-01 00:00:00+01:00   35.034906   42.530036   30.614701
    2020-02-01 00:00:00+01:00   21.919009   33.295167   15.931557
                                ...         ...         ...
    2020-11-01 00:00:00+01:00   38.785706   49.110873   33.226004
    2020-12-01 00:00:00+01:00   43.519745   57.872246   35.055449
    12 rows × 3 columns
    """
    # Guard clauses.
    found = [c for c in df.columns if c in BPO]
    if len(found) <= 1:  # too few present
        raise ValueError(
            f"At least 2 of 'base', 'peak', 'offpeak' must be in columns; found {found}."
        )
    elif len(found) == 3:  # all present, nothing to do
        return df[BPO]  # correct order

    # Make copy of relevant columns, and fill.
    df = df[found].copy()
    b = tools_duration.index(df.index)
    p = tools_peakfn.peak_duration(df.index, peak_fn)
    # Solve: peak * duration_peak + offpeak * duration_offpeak = base * duration_base
    # with:  duration_offpeak = duration_base - duration_peak
    if "peak" not in df.columns:
        # df["peak"] = (df["base"] * b - df["offpeak"] * (b - p)) / p
        df["peak"] = (df["base"] - df["offpeak"]) * b / p + df["offpeak"]
    elif "offpeak" not in df.columns:
        df["offpeak"] = (df["base"] * b - df["peak"] * p) / (b - p)
    else:  # 'base' not in fr.columns
        # df["base"] = (df["peak"] * p + df["offpeak"] * (b - p)) / b
        df["base"] = (df["peak"] - df["offpeak"]) * p / b + df["offpeak"]

    return df[BPO]  # correct order


def _tseries2singlebpo(s: pd.Series, peak_fn: tools_peakfn.PeakFunction) -> pd.Series:
    """
    Aggregate timeseries with varying (float) values to a single base, peak and offpeak
    (float) value.

    In:

    ts_left
    2020-01-01 00:00:00+01:00    41.88
    2020-01-01 01:00:00+01:00    38.60
    2020-01-01 02:00:00+01:00    36.55
                                 ...
    2020-12-31 21:00:00+01:00    52.44
    2020-12-31 22:00:00+01:00    51.86
    2020-12-31 23:00:00+01:00    52.26
    Freq: H, Name: p, Length: 8784, dtype: float64

    Out:

    base       31.401369
    peak       51.363667
    offpeak    20.311204
    dtype: float64
    """
    is_peak = peak_fn(s.index)
    duration = tools_duration.index(s.index).pint.m  # floats
    if duration.nunique() == 1:  # All have same duration: use normal mean (faster).
        return pd.Series(
            {
                "base": s.mean(),
                "peak": s[is_peak].mean(),
                "offpeak": s[~is_peak].mean(),
            }
        )
    else:
        return pd.Series(
            {
                "base": tools_wavg.series(s, duration),
                "peak": tools_wavg.series(s[is_peak], duration[is_peak]),
                "offpeak": tools_wavg.series(s[~is_peak], duration[~is_peak]),
            }
        )


def tseries2bpoframe(
    s: pd.Series, peak_fn: tools_peakfn.PeakFunction, freq: str = "MS"
) -> pd.DataFrame:
    """
    Aggregate timeseries with varying values to a dataframe with base, peak and offpeak
    timeseries, grouped by provided time interval.

    Parameters
    ----------
    s : Series
        Timeseries with hourly or quarterhourly frequency.
    peak_fn : PeakFunction
        Function that returns boolean Series indicating if timestamps in index lie in peak period.
    freq : {'MS' (month, default) 'QS' (quarter), 'AS' (year)}
        Target frequency.

    Returns
    -------
    DataFrame
        Dataframe with base, peak and offpeak values (as columns). Index: downsampled
        timestamps at provided frequency.

    Notes
    -----
    Can only be used for values that are 'averagable' over a time period, like power [MW]
    and price [Eur/MWh]. Not for e.g. energy [MWh], revenue [Eur], and duration [h].

    In:

    ts_left
    2020-01-01 00:00:00+01:00    41.88
    2020-01-01 01:00:00+01:00    38.60
    2020-01-01 02:00:00+01:00    36.55
                                 ...
    2020-12-31 21:00:00+01:00    52.44
    2020-12-31 22:00:00+01:00    51.86
    2020-12-31 23:00:00+01:00    52.26
    Freq: H, Name: p, Length: 8784, dtype: float64

    Out:

                                base        peak        offpeak
    ts_left
    2020-01-01 00:00:00+01:00   35.034906   42.530036   30.614701
    2020-02-01 00:00:00+01:00   21.919009   33.295167   15.931557
                                ...         ...         ...
    2020-11-01 00:00:00+01:00   38.785706   49.110873   33.226004
    2020-12-01 00:00:00+01:00   43.519745   57.872246   35.055449
    12 rows × 3 columns
    """
    if freq not in ("MS", "QS", "AS"):
        raise ValueError(
            f"Parameter ``freq`` must be one of 'MS', 'QS', 'AS'; got '{freq}'."
        )

    # Remove partial data.
    s = tools_trim.frame(s, freq)

    # Handle possible units.
    sin, units = (s.pint.magnitude, s.pint.units) if hasattr(s, "pint") else (s, None)

    # Do calculations.
    sout = sin.resample(freq, group_keys=True).apply(
        lambda s: _tseries2singlebpo(s, peak_fn)
    )

    # Handle possible units.
    if units is not None:
        sout = sout.astype(f"pint[{units}]")
    return sout.unstack()  # base, peak, offpeak as columns


def bpoframe2tseries(
    df: pd.DataFrame, peak_fn: tools_peakfn.PeakFunction, freq: str = "H"
) -> pd.Series:
    """
    Convert a dataframe with base, peak and/or offpeak values, to a single (quarter)hourly
    timeseries.

    Parameters
    ----------
    df : DataFrame
        Dataframe with values. Columns must include at least 2 of {'peak', 'offpeak',
        'base'}. Datetimeindex with frequency in {'MS', 'QS', 'AS'}.
    peak_fn : PeakFunction
        Function that returns boolean Series indicating if timestamps in index lie in peak period.
    freq : {'H' (hour, default) '15T' (quarterhour)}
        Target frequency.

    Returns
    -------
    Series
        Timeseries with values as provided in ``df``.

    Notes
    -----
    Can only be used for values that are 'averagable' over a time period, like power [MW]
    and price [Eur/MWh]. Not for e.g. energy [MWh], revenue [Eur], and duration [h].

    In:

                                peak        offpeak
    ts_left
    2020-01-01 00:00:00+01:00   42.530036   30.614701
    2020-02-01 00:00:00+01:00   33.295167   15.931557
                                ...         ...
    2020-11-01 00:00:00+01:00   49.110873   33.226004
    2020-12-01 00:00:00+01:00   57.872246   35.055449
    12 rows × 2 columns

    Out:

    ts_left
    2020-01-01 00:00:00+01:00    30.614701
    2020-01-01 01:00:00+01:00    30.614701
    2020-01-01 02:00:00+01:00    30.614701
                                 ...
    2020-12-31 21:00:00+01:00    35.055449
    2020-12-31 22:00:00+01:00    35.055449
    2020-12-31 23:00:00+01:00    35.055449
    Freq: H, Length: 8784, dtype: float64
    """
    if "peak" not in df or "offpeak" not in df:  # ensure peak and offpeak available
        df = complete_bpoframe(df, peak_fn)

    # Stretch the dataframe to higher frequency (repeat parent value)
    df2 = tools_changefreq.averagable(df[["peak", "offpeak"]], freq)
    df2["ispeak"] = peak_fn(df2.index)

    return df2["offpeak"].where(df2["ispeak"], df2["peak"])


def tseries2tseries(
    s: pd.Series, peak_fn: tools_peakfn.PeakFunction = None, freq: str = "MS"
) -> pd.Series:
    """
    Transform timeseries (with possibly variable values) to one with (at certain
    frequency) uniform peak and offpeak values.

    Parameters
    ----------
    s : Series
        Timeseries with hourly or quarterhourly frequency.
    peak_fn : PeakFunction, optional (default: None)
        Function that returns boolean Series indicating if timestamps in index lie in peak period.
        If None, calculate uniform base values.
    freq : {'MS' (month, default) 'QS' (quarter), 'AS' (year)}
        Target frequency within which peak and offpeak values will be uniform.

    Returns
    -------
    Series
        Timeseries where each peak hour within the target frequency has the same
        value. Idem for offpeak hours. Index: as original series.

    Notes
    -----
    Can only be used for values that are 'averagable' over a time period, like power [MW]
    and price [Eur/MWh]. Not for e.g. energy [MWh], revenue [Eur], and duration [h].

    In:

    ts_left
    2020-01-01 00:00:00+01:00    41.88
    2020-01-01 01:00:00+01:00    38.60
    2020-01-01 02:00:00+01:00    36.55
                                 ...
    2020-12-31 21:00:00+01:00    52.44
    2020-12-31 22:00:00+01:00    51.86
    2020-12-31 23:00:00+01:00    52.26
    Freq: H, Name: p, Length: 8784, dtype: float64

    Out:

    ts_left
    2020-01-01 00:00:00+01:00    30.614701
    2020-01-01 01:00:00+01:00    30.614701
    2020-01-01 02:00:00+01:00    30.614701
                                ...
    2020-12-31 21:00:00+01:00    35.055449
    2020-12-31 22:00:00+01:00    35.055449
    2020-12-31 23:00:00+01:00    35.055449
    Freq: H, Name: p, Length: 8784, dtype: float64
    """
    if s.index.freq not in ("H", "15T"):
        raise ValueError(
            f"Frequency of provided timeseries must be hourly or quarterhourly; got '{s.index.freq}'."
        )

    # Remove partial data.
    s = tools_trim.frame(s, freq)

    # Handle possible units.
    sin, units = (s.pint.magnitude, s.pint.units) if hasattr(s, "pint") else (s, None)

    # Return normal mean, because all rows have same duration.
    grouping = group_arrays(freq, peak_fn)
    sout = sin.groupby(grouping).transform(np.mean)

    # Handle possible units.
    if units is not None:
        sout = sout.astype(f"pint[{units}]")
    return sout


def bpoframe2bpoframe(
    df: pd.DataFrame, peak_fn: tools_peakfn.PeakFunction, freq: str = "AS"
) -> pd.DataFrame:
    """
    Convert a dataframe with base, peak and/or offpeak values to a similar dataframe
    with a different frequency.

    Parameters
    ----------
    df : DataFrame
        Columns must include at least 2 of {'peak', 'offpeak', 'base'}. Datetimeindex
        with frequency in {'MS', 'QS', 'AS'}.
    peak_fn : PeakFunction
        Function that returns boolean Series indicating if timestamps in index lie in peak period.
    freq : {'MS' (month), 'QS' (quarter), 'AS' (year, default)}
        Target frequency.

    Returns
    -------
    DataFrame
        Dataframe with base, peak and offpeak values (as columns). Index: timestamps at
        specified frequency.

    Notes
    -----
    Can only be used for values that are 'averagable' over a time period, like power [MW]
    and price [Eur/MWh]. Not for e.g. energy [MWh], revenue [Eur], and duration [h].

    In:

                                base        peak        offpeak
    ts_left
    2020-01-01 00:00:00+01:00   35.034906   42.530036   30.614701
    2020-02-01 00:00:00+01:00   21.919009   33.295167   15.931557
                                ...         ...         ...
    2020-11-01 00:00:00+01:00   38.785706   49.110873   33.226004
    2020-12-01 00:00:00+01:00   43.519745   57.872246   35.055449
    12 rows × 3 columns

    Out:

                                base        peak        offpeak
    ts_left
    2020-01-01 00:00:00+01:00   30.490036   38.003536   26.312894
    2020-04-01 00:00:00+02:00   25.900919   35.295167   20.681892
    2020-07-01 00:00:00+02:00   32.706785   44.033511   26.371498
    2020-10-01 00:00:00+02:00   39.455197   54.468722   31.063728
    """
    if freq not in ("MS", "QS", "AS"):
        raise ValueError(
            f"Parameter ``freq`` must be one of 'MS', 'QS', 'AS'; got {freq}."
        )
    if tools_freq.up_or_down(df.index.freq, freq) == 1:
        warnings.warn(
            "This conversion includes upsampling, e.g. from yearly to monthly values."
            " The result will be uniform at the frequency of the original frame ``df``."
        )

    return tseries2bpoframe(bpoframe2tseries(df, peak_fn, "H"), peak_fn, freq)
