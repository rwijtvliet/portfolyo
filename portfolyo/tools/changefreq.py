"""Functions to change frequency of a pandas dataframe."""

import datetime as dt
from typing import Union

import pandas as pd

from . import freq as tools_freq
from . import right as tools_right
from . import trim as tools_trim


def _emptyseries(s_ref: pd.Series, freq) -> pd.Series:
    i = pd.DatetimeIndex([], freq=freq, tz=s_ref.index.tz)
    return pd.Series([], i, dtype=s_ref.dtype, name=s_ref.name)


def _downsample_avgable(s: pd.Series, freq: str) -> pd.Series:
    """Downsample averagble series."""
    # For averagable series: first make summable.
    summable = s.mul(s.index.duration, axis=0)  # now has a pint dtype
    summable2 = _downsample_summable(summable, freq)
    s2 = summable2.div(summable2.index.duration, axis=0)
    return s2.astype(s.dtype).rename(s.name)


def _downsample_summable(s: pd.Series, freq: str) -> pd.Series:
    """Downsample summable series."""
    # Downsampling is easiest for summable series: sum child values.

    s = tools_trim.frame(s, freq)  # keep only full periods in target freq

    if not len(s):  # Empty series.
        return _emptyseries(s, freq)

    start_of_day = s.index[0].time()
    offset = dt.timedelta(hours=start_of_day.hour, minutes=start_of_day.minute)
    source_vs_daily = tools_freq.up_or_down(s.index.freq, "D")
    target_vs_daily = tools_freq.up_or_down(freq, "D")

    # We cannot simply `.resample()`, e.g. from hourly to monthly, because in that
    # case the start-of-day is lost. We need to do it in two steps.

    # Downsample to days.
    s2 = s
    if source_vs_daily < 0:
        if target_vs_daily < 0:
            return s2.resample(freq).sum()
        s2 = s2.resample("D", offset=offset).sum()
    # Downsample to longer-than-days.
    if target_vs_daily > 0:
        # workaround: (a) first downsample to days...
        s2 = s2.resample(freq).sum()
        s2.index += offset  # ...(b) add the offset manually...
        s2.index.freq = freq  # ...(c) and set the frequency manually as well.
    return s2.rename(s.name)


def _upsample_summable(s: pd.Series, freq: str) -> pd.Series:
    """Upsample summable series."""
    # For summable series: first make averagable.
    avgable = s.div(s.index.duration, axis=0)  # now has a pint dtype
    avgable2 = _upsample_avgable(avgable, freq)
    s2 = avgable2.mul(avgable2.index.duration, axis=0)
    return s2.astype(s.dtype).rename(s.name)


def _upsample_avgable(s: pd.Series, freq: str) -> pd.Series:
    """Upsample averagable series."""
    # Upsampling is easiest for averagable series: duplicate value to all children.

    if not len(s):  # Empty series.
        return _emptyseries(s, freq)

    start_of_day = s.index[0].time()
    offset = dt.timedelta(hours=start_of_day.hour, minutes=start_of_day.minute)

    source_vs_daily = tools_freq.up_or_down(s.index.freq, "D")
    target_vs_daily = tools_freq.up_or_down(freq, "D")

    # Several isuses with pandas resampling:

    # 1. When upsampling from to yearly to monthly values with `.resample()`, the
    # start-of-day is lost. We need to do it in two steps; first upsampling to days and
    # then downsampling to months.

    if source_vs_daily > 0 and target_vs_daily > 0:
        return _downsample_avgable(_upsample_avgable(s, "D"), freq)

    # 2. We cannot simply `.resample()`, because in that case the final value is not
    # duplicated. Solution: add a dummy value, which we eventually remove again.

    # So, first, add additional row...
    # (original code to add additional row, does not work if unit-aware. Maybe with future release of pint_pandas?)
    # s = s.copy()
    # s.loc[s.index.right[-1]] = None
    # (Workaround: turn into dataframe, change frequency, and turn back into series.)
    df = pd.DataFrame(s)
    additional_stamp = tools_right.stamp(s.index[-1], s.index.freq)
    df.loc[additional_stamp, :] = None
    # ... then do upsampling ...
    df2 = df.resample(freq, offset=offset).asfreq().ffill()  # duplicate value
    # ... and then remove final row (and turn back into series).
    return df2.iloc[:-1, 0].rename(s.name)


def _general(is_summable: bool, s: pd.Series, freq: str = "MS") -> pd.Series:
    f"""Change frequency of a Series, depending on the type of data it contains.

    Parameters
    ----------
    is_summable : bool
        True if data is summable, False if it is averagable.
    s : pd.Series
        Series that needs to be resampled.
    freq : {{{', '.join(tools_freq.FREQUENCIES)}}}, optional (default: 'MS')
        Target frequency.

    Returns
    -------
    pd.Series
        Resampled series at target frequency.
    """

    # TODO: Add tests with multiindex columns

    # Eliminate integers.
    if pd.api.types.is_integer_dtype(s.dtype):
        s = s.astype(float)

    # s is now a Series with a 'float' or 'pint' dtype.

    up_or_down = tools_freq.up_or_down(s.index.freq, freq)

    # Nothing more needed; portfolio already in desired frequency.
    if up_or_down == 0:
        return s

    # Must downsample.
    elif up_or_down == -1:
        if is_summable:
            return _downsample_summable(s, freq)
        else:
            return _downsample_avgable(s, freq)

    # Must upsample.
    else:
        if is_summable:
            return _upsample_summable(s, freq)
        else:
            return _upsample_avgable(s, freq)


def index(i: pd.DatetimeIndex, freq: str = "MS") -> pd.DatetimeIndex:
    f"""Resample index.

    Parameters
    ----------
    i : pd.DatetimeIndex
        Index to resample.
    freq : {{{', '.join(tools_freq.FREQUENCIES)}}}
        Target frequency.

    Returns
    -------
    pd.DatetimeIndex
    """
    up_or_down = tools_freq.up_or_down(i.freq, freq)

    # Nothing more needed; index already in desired frequency.
    if up_or_down == 0:
        return i

    # Must downsample.
    elif up_or_down == -1:
        # We must jump through a hoop: can't directly resample an Index.
        return _downsample_summable(pd.Series(0, i), freq).index

    # Must upsample.
    else:  # up_or_down == 1
        return _upsample_avgable(pd.Series(0, i), freq).index


def summable(
    fr: Union[pd.Series, pd.DataFrame], freq: str = "MS"
) -> Union[pd.Series, pd.DataFrame]:
    f"""
    Resample and aggregate a DataFrame or Series with 'time-summable' quantities.

    Parameters
    ----------
    fr : pd.Series or pd.DataFrame
        Pandas Series or DataFrame to be resampled.
    freq : {{{', '.join(tools_freq.FREQUENCIES)}}}, optional (default: 'MS')
        Target frequency.

    Returns
    -------
    DataFrame or Series

    Notes
    -----
    A 'time-summable' quantity is one that can be summed to get to an aggregate
    value, like revenue [Eur] or energy [MWh]. Prices [Eur/MWh] and powers [MW]
    are not time-summable.
    See https://portfolyo.readthedocs.io/en/latest/specialized_topics/resampling.html
    for more information.

    For shorter-than-daily indices, it is assumed that the index starts with a full day.
    I.e., the time-of-day of the first element is assumed to be the start time for the
    day-or-longer delivery periods. (E.g., if the index has hourly values and starts with
    "2020-04-21 06:00:00", it is assumed that a delivery day is from 06:00:00 (incl)
    until 06:00:00 (excl).)
    """
    if isinstance(fr, pd.DataFrame):
        # Turn into series, change frequency, and turn back into dataframe.
        return pd.DataFrame({key: summable(s, freq) for key, s in fr.items()})

    return _general(True, fr, freq)


def averagable(
    fr: Union[pd.Series, pd.DataFrame], freq: str = "MS"
) -> Union[pd.Series, pd.DataFrame]:
    f"""
    Resample and aggregate a DataFrame or Series with 'time-averagable' quantities.

    Parameters
    ----------
    fr : pd.Series or pd.DataFrame
        Pandas Series or DataFrame to be resampled.
    freq : {{{', '.join(tools_freq.FREQUENCIES)}}}, optional (default: 'MS')
        Target frequency.

    Returns
    -------
    DataFrame or Series

    Notes
    -----
    A 'time-averagable' quantity is one that can be averaged to an aggregate value,
    like power [MW]. When downsampling, the values are weighted with their duration.
    See https://portfolyo.readthedocs.io/en/latest/specialized_topics/resampling.html
    for more information.

    For shorter-than-daily indices, it is assumed that the index starts with a full day.
    I.e., the time-of-day of the first element is assumed to be the start time for the
    day-or-longer delivery periods. (E.g., if the index has hourly values and starts with
    "2020-04-21 06:00:00", it is assumed that a delivery day is from 06:00:00 (incl)
    until 06:00:00 (excl).)
    """
    if isinstance(fr, pd.DataFrame):
        # Turn into series, change frequency, and turn back into dataframe.
        return pd.DataFrame({key: averagable(value, freq) for key, value in fr.items()})

    return _general(False, fr, freq)
