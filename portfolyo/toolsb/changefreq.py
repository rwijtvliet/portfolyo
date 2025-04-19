"""Functions to change frequency of a pandas dataframe."""

from typing import overload

import pandas as pd
from pandas.tseries.offsets import BaseOffset

from . import frame as tools_frame
from . import freq as tools_freq
from . import index as tools_index
from . import stamp as tools_stamp
from . import startofday as tools_sod
from . import unit as tools_unit
from .types import Series_or_DataFrame


def _emptyseries(s_ref: pd.Series, freq) -> pd.Series:
    s = s_ref.copy().iloc[:0]
    s.index.freq = freq
    return freq


def _downsample_avgable(s: pd.Series, freq: BaseOffset) -> pd.Series:
    """Downsample averagble series."""
    # Downsampling is easiest for summable series. Therefore, make `s` summable first.
    duration = tools_index.duration(s.index)
    summable = s.mul(duration)
    summable2 = _downsample_summable(summable, freq)
    duration2 = tools_index.duration(summable2.index)
    s2 = summable2.div(duration2)
    return s2.rename(s.name)


def _downsample_summable(s: pd.Series, freq: BaseOffset) -> pd.Series:
    """Downsample summable series."""
    # Downsampling is easiest for summable series: sum child values.

    s = tools_frame.trim(s, freq)  # keep only full periods in target freq

    if not len(s):  # Empty series.
        return _emptyseries(s, freq)

    offset = tools_sod.to_tdelta(s.index[0].time())
    source, target = s.index.freq, freq
    name = s.name

    # HACK: We cannot always simply `.resample()`. When resampling e.g. from hourly to monthly,
    # the start-of-day is lost, and we need to do it in two steps.
    # Cases and approach:
    # A) Source < daily and target < daily: normal downsampling
    # B) Source < daily and target = daily: downsample to daily with offset
    # C) Source < daily and target > daily: downsample to daily with offset, downsample to source with fix
    # D) Source >= daily and target > daily: downsample to source with fix

    # Case A: Both shorter than daily: normal downsampling.
    if tools_freq.is_shorter_than_daily(source) and tools_freq.is_shorter_than_daily(target):
        return s.resample(target).sum()

    # If we are here, `source` or `target` (or both) is daily-or-longer.

    # Case B or C: downsample to daily with offset.
    if tools_freq.is_shorter_than_daily(source):
        s = s.resample("D", offset=offset).sum()  # workaround: (a) first downsample to days...

    # Case C or D: downsample to source and fix.
    if tools_freq.is_longer_than_daily(target):
        s = s.resample(freq).sum()  # (b) ...then downsample further to source freq...
        s.index += offset  # ...(c) add the offset manually...
        s.index.freq = freq  # ...(d) and set the frequency manually as well.

    return s.rename(name)


def _upsample_summable(s: pd.Series, freq: BaseOffset) -> pd.Series:
    """Upsample summable series."""
    # Upsampling is easiest for averagable series. Therefore, make `s` averagable first.
    duration = tools_index.duration(s.index)
    avgable = s.div(duration)
    avgable2 = _upsample_avgable(avgable, freq)
    duration2 = tools_index.duration(avgable2.index)
    s2 = avgable2.mul(duration2)
    return s2.rename(s.name)


def _upsample_avgable(s: pd.Series, freq: BaseOffset) -> pd.Series:
    """Upsample averagable series."""
    # Upsampling is easiest for averagable series: duplicate value to all children.

    if not len(s):  # Empty series.
        return _emptyseries(s, freq)

    offset = tools_sod.to_tdelta(s.index[0].time())
    source, target = s.index.freq, freq

    # Several isuses with pandas resampling:

    # HACK: (1): When upsampling from to yearly to monthly values with `.resample()`, the
    # start-of-day is lost. We need to do it in two steps; first upsampling to days and
    # then downsampling to months.

    if tools_freq.is_longer_than_daily(source) and tools_freq.is_longer_than_daily(target):
        return _downsample_avgable(_upsample_avgable(s, "D"), freq)

    # HACK: (2): We cannot simply `.resample()`, because in that case the final value is not
    # duplicated. Solution: add a dummy value, which we eventually remove again.

    # So, first, add additional row...
    additional_stamp = tools_stamp.to_right(s.index[-1], s.index.freq)
    new_index = s.index.append(pd.DatetimeIndex([additional_stamp], freq=s.index.freq))
    s = s.reindex(new_index)  # adds nan in final row
    # ... then do upsampling ...
    s2 = s.resample(freq, offset=offset).asfreq().ffill()
    # ... and then remove final row (and turn back into series).
    return s2.iloc[:-1].rename(s.name)


@tools_unit.apply_coercion_pintframe("s")
@tools_freq.apply_coercion("freq")
def _general(s: pd.Series, freq: str | BaseOffset, *, summable: bool) -> pd.Series:
    """Change frequency of a Series, depending on the type of data it contains.

    Parameters
    ----------
    s
        Timeseries that needs to be resampled.
    freq
        Target frequency.
    summable
        True if data is summable, False if it is averagable.

    Returns
    -------
        Resampled series at target frequency.
    """

    # TODO: Add tests with multiindex columns

    up_or_down = tools_freq.up_or_down(s.index.freq, freq)

    # No resampling; portfolio already in desired frequency.
    if up_or_down == 0:
        if s.index.freq != freq:  # equivalent but unequal
            s = s.copy()
            s.index.freq = freq
        return s

    # Must downsample.
    elif up_or_down == -1:
        if summable:
            return _downsample_summable(s, freq)
        else:
            return _downsample_avgable(s, freq)

    # Must upsample.
    else:
        if summable:
            return _upsample_summable(s, freq)
        else:
            return _upsample_avgable(s, freq)


@tools_index.apply_coercion("idx")
@tools_freq.apply_coercion("freq")
def index(idx: pd.DatetimeIndex, freq: str | BaseOffset) -> pd.DatetimeIndex:
    """Resample index.

    Parameters
    ----------
    idx
        Index to resample.
    freq
        Target frequency.

    Returns
    -------
    pd.DatetimeIndex
    """
    up_or_down = tools_freq.up_or_down(idx.freq, freq)

    # Nothing more needed; index already in desired frequency.
    if up_or_down == 0:
        if idx.freq != freq:  # equivalent but unequal
            idx = idx.copy()
            idx.freq = freq
        return idx

    # Must downsample.
    elif up_or_down == -1:
        # HACK: we must jump through a hoop: can't directly resample an Index.
        # (Use summable because faster.)
        return _downsample_summable(pd.Series(0, idx), freq).index

    # Must upsample.
    else:  # up_or_down == 1
        # HACK: as above.
        # (Use averagable because faster.)
        return _upsample_avgable(pd.Series(0, idx), freq).index


@overload
def summable(fr: pd.Series, freq: str | BaseOffset) -> pd.Series: ...


@overload
def summable(fr: pd.DataFrame, freq: str | BaseOffset) -> pd.DataFrame: ...


def summable(fr: Series_or_DataFrame, freq: str | BaseOffset) -> Series_or_DataFrame:
    """Resample and aggregate a Series or DataFrame with 'time-summable' timeseries data.

    Parameters
    ----------
    fr
        Timeseries to change frequency of.
    freq
        Target frequency.

    Returns
    -------
        Resampled timeseries at target frequency.

    Notes
    -----
    When downsampling, the values are summed. When upsampling, the values are split according to
    their duration.

    'Time-summable' data is data that must be SUMMED to an aggregate value, like revenue (e.g. [Eur])
    or energy (e.g. [MWh]). Prices (e.g. [Eur/MWh]) and powers (e.g. [MW]) are not time-summable.
    See https://portfolyo.readthedocs.io/en/latest/specialized_topics/resampling.html for more
    information.

    For shorter-than-daily indices, it is assumed that the index starts with a full day.
    I.e., the time-of-day of the first element is assumed to be the start time for the
    day-or-longer delivery periods. (E.g., if the index has hourly values and starts with
    "2020-04-21 06:00:00", it is assumed that a delivery day is from 06:00:00 (incl)
    until 06:00:00 (excl).)

    See also
    --------
    .averagable
    """
    if isinstance(fr, pd.DataFrame):
        return pd.DataFrame({c: summable(s, freq) for c, s in fr.items()})

    return _general(fr, freq, summable=True)


@overload
def averagable(fr: pd.Series, freq: str | BaseOffset) -> pd.Series: ...


@overload
def averagable(fr: pd.DataFrame, freq: str | BaseOffset) -> pd.DataFrame: ...


def averagable(fr: Series_or_DataFrame, freq: str | BaseOffset) -> Series_or_DataFrame:
    """Resample and aggregate a Series or DataFrame with 'time-averagable' timeseries data.

    Parameters
    ----------
    fr
        Timeseries to change frequency of.
    freq
        Target frequency.

    Returns
    -------
        Resampled timeseries.

    Notes
    -----
    When downsampling, the values are weighted with their duration. When upsampling, the values are
    duplicated.

    'Time-averagable' data is data that must be AVERAGED to an aggregate value, like power (e.g.
    [MW]). Revenues (e.g. [Eur]), energies (e.g. [MWh]) and prices (e.g. [Eur/MWh]) are not
    time-averagable. See https://portfolyo.readthedocs.io/en/latest/specialized_topics/resampling.html
    for more information.

    For shorter-than-daily indices, it is assumed that the index starts with a full day.
    I.e., the time-of-day of the first element is assumed to be the start time for the
    day-or-longer delivery periods. (E.g., if the index has hourly values and starts with
    "2020-04-21 06:00:00", it is assumed that a delivery day is from 06:00:00 (incl)
    until 06:00:00 (excl).)

    See also
    --------
    .summable
    """
    if isinstance(fr, pd.DataFrame):
        return pd.DataFrame({c: averagable(s, freq) for c, s in fr.items()})

    return _general(fr, freq, summable=False)
