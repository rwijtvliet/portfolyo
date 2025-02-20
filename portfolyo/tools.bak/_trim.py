"""
Trim objects to only contain 'full' delivery periods.
"""

from typing import overload

import pandas as pd

from . import ceil as tools_ceil
from . import floor as tools_floor
from . import freq as tools_freq
from . import right as tools_right
from . import startofday as tools_startofday


def index(i: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    f"""Trim index to only keep full periods of certain frequency.

    Parameters
    ----------
    i : pd.DatetimeIndex
        The (untrimmed) DatetimeIndex
    freq : {tools_freq.ALLOWED_FREQUENCIES_DOCS}
        Frequency to trim to. E.g. 'MS' to only keep full months.

    Returns
    -------
    pd.DatetimeIndex
        Subset of ``i``, with same frequency.

    Notes
    -----
    For shorter-than-daily indices, it is assumed that the index starts with a full day.
    I.e., the time-of-day of the first element is assumed to be the start time for the
    day-or-longer delivery periods. (E.g., if the index has hourly values and starts with
    "2020-04-21 06:00:00", it is assumed that a delivery day is from 06:00:00 (incl)
    until 06:00:00 (excl).)
    """
    if not i.freq:
        raise ValueError("Index ``i`` does not have a frequency.")
    # Use index to find start_of_day.
    start_of_day = tools_startofday.get(i)
    # check if the freq are compatible
    try:
        tools_freq.up_or_down(i.freq, freq)
    except ValueError:
        # print(f"The passed frequency {i.freq} can't be aggregated to {freq}.")
        return pd.DatetimeIndex([], freq=i.freq, tz=i.tz)
    # Trim on both sides.
    mask_start = i >= tools_ceil.stamp(i[0], freq, 0, start_of_day)
    i_right = tools_right.index(i)
    mask_end = i_right <= tools_floor.stamp(i_right[-1], freq, 0, start_of_day)
    return i[mask_start & mask_end]


@overload
def frame(fr: pd.Series, freq: str) -> pd.Series:
    ...


@overload
def frame(fr: pd.DataFrame, freq: str) -> pd.DataFrame:
    ...


def frame(fr: pd.Series | pd.DataFrame, freq: str) -> pd.Series | pd.DataFrame:
    f"""Trim index of series or dataframe to only keep full periods of certain frequency.

    Parameters
    ----------
    fr : Series or DataFrame
        The (untrimmed) pandas series or dataframe.
    freq : {tools_freq.ALLOWED_FREQUENCIES_DOCS}
        Frequency to trim to. E.g. 'MS' to only keep full months.

    Returns
    -------
    Series or DataFrame
        Subset of ``fr``, with same frequency.

    Notes
    -----
    For shorter-than-daily indices, it is assumed that the index starts with a full day.
    I.e., the time-of-day of the first element is assumed to be the start time for the
    day-or-longer delivery periods. (E.g., if the index has hourly values and starts with
    "2020-04-21 06:00:00", it is assumed that a delivery day is from 06:00:00 (incl)
    until 06:00:00 (excl).)
    """
    i = index(fr.index, freq)
    return fr.loc[i]
