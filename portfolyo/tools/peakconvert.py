"""Module to convert between timeseries and base/peak/offpeak-values."""

import warnings

import pandas as pd

from . import changefreq as tools_changefreq
from . import duration as tools_duration
from . import freq as tools_freq
from . import peakfn as tools_peakfn
from . import trim as tools_trim
from . import wavg as tools_wavg

BPO = ["base", "peak", "offpeak"]


def group_index(
    i: pd.DatetimeIndex, peak_fn: tools_peakfn.PeakFunction, freq: str
) -> pd.MultiIndex:
    """Multiindex, that is the same for all rows that belong to same delivery period."""
    # Grouping due to delivery period.
    groups = [i.year]
    if freq == "MS":
        groups.append(i.month)
    elif freq == "QS":
        groups.append(i.quarter)
    elif freq == "YS":
        pass
    else:
        raise ValueError(
            f"Parameter ``freq`` must be one of 'MS', 'QS', 'YS'; got '{freq}'."
        )

    # Add grouping due to peak.
    if peak_fn is not None:
        groups.append(peak_fn(i))

    return pd.MultiIndex.from_arrays(groups)


def complete_bpoframe(
    df: pd.DataFrame, peak_fn: tools_peakfn.PeakFunction, is_summable: bool = False
) -> pd.DataFrame:
    """Complete a dataframe so it contains arbitrage-free base, peak, and offpeak values.

    Parameters
    ----------
    df : DataFrame
        With at least 2 of following columns: {'base', 'peak', 'offpeak'}. If all 3 are
        present, no verification is done. Additional columns are dropped. DatetimeIndex
        with freq in {'MS', 'QS', 'YS'}.
    peak_fn : PeakFunction
        Function that returns boolean Series indicating if timestamps in index lie in peak period.
    is_summable : bool, optional (default: False)
        True if data is summable, False if it is averagable.

    Returns
    -------
    DataFrame
        Same as ``df``, with all 3 columns ('base', 'peak', 'offpeak').

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
    found = [c for c in df if c in BPO]
    if len(found) <= 1:  # too few present
        raise ValueError(
            f"At least 2 of 'base', 'peak', 'offpeak' must be in columns; found {found}."
        )
    elif len(found) == 3:  # all present, nothing to do
        return df[BPO]  # correct order

    # Make copy of relevant columns, and fill.
    df = df[found].copy()
    if is_summable:
        # Solve: peak + offpeak = base
        if "peak" not in df:
            df["peak"] = df["base"] - df["offpeak"]
        elif "offpeak" not in df:
            df["offpeak"] = df["base"] - df["peak"]
        else:  # 'base' not in df
            df["base"] = df["peak"] + df["offpeak"]
    else:
        # Solve: peak * duration_peak + offpeak * duration_offpeak = base * duration_base
        # (with: duration_offpeak = duration_base - duration_peak)
        b = tools_duration.index(df.index).pint.m  # as float
        p = tools_peakfn.peak_duration(df.index, peak_fn).pint.m  # as float
        if "peak" not in df:
            # df["peak"] = (df["base"] * b - df["offpeak"] * (b - p)) / p
            df["peak"] = (df["base"] - df["offpeak"]) * b / p + df["offpeak"]
        elif "offpeak" not in df:
            df["offpeak"] = (df["base"] * b - df["peak"] * p) / (b - p)
        else:  # 'base' not in df
            # df["base"] = (df["peak"] * p + df["offpeak"] * (b - p)) / b
            df["base"] = (df["peak"] - df["offpeak"]) * p / b + df["offpeak"]

    return df[BPO]  # correct order


def _tseries2po(
    s: pd.Series, peak_fn: tools_peakfn.PeakFunction, is_summable: bool
) -> pd.Series:
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

    peak       51.363667
    offpeak    20.311204
    dtype: float64
    """
    is_peak = peak_fn(s.index)
    duration = tools_duration.index(s.index).pint.m  # floats
    if is_summable:
        return pd.Series(
            {
                "peak": s[is_peak].sum(),
                "offpeak": s[~is_peak].sum(),
            }
        )
    else:
        return pd.Series(
            {
                "peak": tools_wavg.series(s[is_peak], duration[is_peak]),
                "offpeak": tools_wavg.series(s[~is_peak], duration[~is_peak]),
            }
        )


def tseries2poframe(
    s: pd.Series,
    peak_fn: tools_peakfn.PeakFunction,
    freq: str = "MS",
    is_summable: bool = False,
) -> pd.DataFrame:
    """
    Aggregate timeseries with varying values to a dataframe with peak and offpeak
    timeseries, grouped by specified frequency.

    Parameters
    ----------
    s : Series
        Timeseries with hourly or quarterhourly frequency.
    peak_fn : PeakFunction
        Function that returns boolean Series indicating if timestamps in index lie in peak period.
    freq : str, optional (default: 'MS')
        Target frequency; monthly-or-longer.
    is_summable : bool, optional (default: False)
        True if data is summable, False if it is averagable.

    Returns
    -------
    DataFrame
        Dataframe with base, peak and offpeak values (as columns). Index: downsampled
        timestamps at provided frequency.

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

                                peak        offpeak
    ts_left
    2020-01-01 00:00:00+01:00   42.530036   30.614701
    2020-02-01 00:00:00+01:00   33.295167   15.931557
                                ...         ...
    2020-11-01 00:00:00+01:00   49.110873   33.226004
    2020-12-01 00:00:00+01:00   57.872246   35.055449
    12 rows × 3 columns
    """
    if tools_freq.up_or_down(freq, "MS") < 0:
        raise ValueError(f"Parameter ``freq`` be monthly-or-longer; got '{freq}'.")

    # Remove partial data.
    s = tools_trim.frame(s, freq)

    # Handle possible units.
    sin, units = (s.pint.magnitude, s.pint.units) if hasattr(s, "pint") else (s, None)

    # Do calculations.
    sout = sin.resample(freq, group_keys=True).apply(
        lambda s: _tseries2po(s, peak_fn, is_summable)
    )

    # Handle possible units.
    if units is not None:
        sout = sout.astype(f"pint[{units}]")
    return sout.unstack()  # peak, offpeak as columns


def poframe2tseries(
    df: pd.DataFrame,
    peak_fn: tools_peakfn.PeakFunction,
    freq: str = "h",
    is_summable: bool = False,
) -> pd.Series:
    """
    Convert a dataframe with peak and offpeak values, to a single timeseries.

    Parameters
    ----------
    df : DataFrame
        Dataframe with values. Columns must include at least {'peak', 'offpeak'}.
        Datetimeindex with monthly-or-longer frequency.
    peak_fn : PeakFunction
        Function that returns boolean Series indicating if timestamps in index lie in peak period.
    freq : str
        Target frequency; short enough for ``peak_fn``.
    is_summable : bool, optional (default: False)
        True if data is summable, False if it is averagable.

    Returns
    -------
    Series
        Timeseries with values as provided in ``df``.

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
    # Prep.
    if "peak" not in df or "offpeak" not in df:  # ensure peak and offpeak available
        df = complete_bpoframe(df, peak_fn)
    df = df[["peak", "offpeak"]]

    # Stretch the dataframe to higher frequency
    if is_summable:
        df2 = tools_changefreq.summable(df, freq)
    else:
        df2 = tools_changefreq.averagable(df, freq)
    df2["ispeak"] = peak_fn(df2.index)  # will raise error if frequency not short enough

    return df2["offpeak"].where(df2["ispeak"], df2["peak"])


# def tseries2tseries(
#     s: pd.Series,
#     peak_fn: tools_peakfn.PeakFunction,
#     freq: str = "MS",
#     is_summable: bool = False,
# ) -> pd.Series:
#     """
#     Transform timeseries (with possibly variable values) to one with (at certain
#     frequency) uniform peak and offpeak values.
#
#     Parameters
#     ----------
#     s : Series
#         Timeseries with hourly or quarterhourly frequency.
#     peak_fn : PeakFunction, optional (default: None)
#         Function that returns boolean Series indicating if timestamps in index lie in peak period.
#     freq : {'MS' (month, default) 'QS' (quarter), 'YS' (year)}
#         Target frequency within which peak and offpeak values will be uniform.
#     is_summable : bool, optional (default: False)
#         True if data is summable, False if it is averagable.
#
#     Returns
#     -------
#     Series
#         Timeseries where each peak hour within the target frequency has the same
#         value. Idem for offpeak hours. Index: as original series.
#
#     In:
#
#     ts_left
#     2020-01-01 00:00:00+01:00    41.88
#     2020-01-01 01:00:00+01:00    38.60
#     2020-01-01 02:00:00+01:00    36.55
#                                  ...
#     2020-12-31 21:00:00+01:00    52.44
#     2020-12-31 22:00:00+01:00    51.86
#     2020-12-31 23:00:00+01:00    52.26
#     Freq: H, Name: p, Length: 8784, dtype: float64
#
#     Out:
#
#     ts_left
#     2020-01-01 00:00:00+01:00    30.614701
#     2020-01-01 01:00:00+01:00    30.614701
#     2020-01-01 02:00:00+01:00    30.614701
#                                 ...
#     2020-12-31 21:00:00+01:00    35.055449
#     2020-12-31 22:00:00+01:00    35.055449
#     2020-12-31 23:00:00+01:00    35.055449
#     Freq: H, Name: p, Length: 8784, dtype: float64
#     """
#     # Remove partial data.
#     s = tools_trim.frame(s, freq)
#
#     # Handle possible units.
#     sin, units = (s.pint.magnitude, s.pint.units) if hasattr(s, "pint") else (s, None)
#
#     # Calculate.
#     # grouping will raise error if frequency not short enough
#     grouping = group_arrays(sin.index, freq, peak_fn)
#     fn = tools_changefreq.summable if is_summable else tools_changefreq.averagable
#     sout = sin.groupby(grouping).transform(
#
#     # Handle possible units.
#     if units is not None:
#         sout = sout.astype(f"pint[{units}]")
#     return sout
#


def poframe2poframe(
    df: pd.DataFrame,
    peak_fn: tools_peakfn.PeakFunction,
    freq: str = "YS",
    is_summable: bool = False,
) -> pd.DataFrame:
    """
    Convert a dataframe with peak and offpeak values to a similar dataframe
    with a different frequency.

    Parameters
    ----------
    df : DataFrame
        Columns must include at least {'peak', 'offpeak'}. Datetimeindex with monthly-
        or-longer frequency.
    peak_fn : PeakFunction
        Function that returns boolean Series indicating if timestamps in index lie in peak period.
    freq : str, optional (default: 'YS')
        Target frequency; monthly-or-longer.
    is_summable : bool, optional (default: False)
        True if data is summable, False if it is averagable.

    Returns
    -------
    DataFrame
        Dataframe with base, peak and offpeak values (as columns). Index: timestamps at
        specified frequency.

    In:

                                peak        offpeak
    ts_left
    2020-01-01 00:00:00+01:00   42.530036   30.614701
    2020-02-01 00:00:00+01:00   33.295167   15.931557
                                ...         ...
    2020-11-01 00:00:00+01:00   49.110873   33.226004
    2020-12-01 00:00:00+01:00   57.872246   35.055449
    12 rows × 3 columns

    Out:

                                peak        offpeak
    ts_left
    2020-01-01 00:00:00+01:00   38.003536   26.312894
    2020-04-01 00:00:00+02:00   35.295167   20.681892
    2020-07-01 00:00:00+02:00   44.033511   26.371498
    2020-10-01 00:00:00+02:00   54.468722   31.063728
    """
    if tools_freq.up_or_down(freq, "MS") < 0:
        raise ValueError(f"Parameter ``freq`` be monthly-or-longer; got '{freq}'.")

    if tools_freq.up_or_down(df.index.freq, freq) == 1:
        warnings.warn(
            "This conversion includes upsampling, e.g. from yearly to monthly values."
            " The result will be uniform at the frequency of the original frame ``df``."
        )

    upsampled = poframe2tseries(df, peak_fn, "h", is_summable)
    downsampled = tseries2poframe(upsampled, peak_fn, freq, is_summable)
    return downsampled
