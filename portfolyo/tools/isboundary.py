"""
Check if timestamp is delivery period start (or end).
"""

import datetime as dt
from typing import Union

import numpy as np
import pandas as pd

from . import freq as tools_freq


# Necessary because pd.DatetimeIndex.is_year_start is broken.
# e.g.
# i = pd.date_range('2020-01-01', freq='MS', periods=12)
# i.is_year_start
# array([False, False, False, False, False, False, False, False, False, False, False, True])
def is_year_start(i: Union[pd.Timestamp, pd.DatetimeIndex]) -> Union[bool, np.ndarray]:
    return (i.day == 1) & (i.month == 1)


def is_quarter_start(
    i: Union[pd.Timestamp, pd.DatetimeIndex]
) -> Union[bool, np.ndarray]:
    return (i.day == 1) & ((i.month - 1) % 3 == 0)


def is_month_start(i: Union[pd.Timestamp, pd.DatetimeIndex]) -> Union[bool, np.ndarray]:
    return i.day == 1


def is_X_start(
    i: Union[pd.Timestamp, pd.DatetimeIndex], freq: str
) -> Union[bool, np.ndarray]:
    if freq == "MS":
        return is_month_start(i)
    elif freq == "QS":
        return is_quarter_start(i)
    elif freq == "AS":
        return is_year_start(i)
    else:
        raise ValueError(f"Unexpected frequency ``freq``; got {freq}.")


# TODO: see if we can remove usage of this function.
def stamp(ts: pd.Timestamp, freq: str, start_of_day: dt.time = None) -> bool:
    f"""Check if timestamp is a valid delivery period start (or end).

    Parameters
    ----------
    ts : pd.Timestamp
        Timestamp for which to do the assertion.
    freq : {{{', '.join(tools_freq.FREQUENCIES)}}}
        Frequency for which to check if the timestamp is a valid start (or end) timestamp.
    start_of_day : dt.time, optional (default: midnight)
        Time of day at which daily-or-longer delivery periods start. E.g. if
        dt.time(hour=6), a delivery day is from 06:00:00 (incl) until 06:00:00 (excl).

    Returns
    -------
    bool
    """
    start_of_day = start_of_day or dt.time(0, 0)
    if freq == "15T":
        return ts.minute % 15 == 0
    elif freq == "H":
        return ts.minute == 0
    elif freq == "D":
        return ts.time() == start_of_day
    elif freq == "MS":
        return (ts.time() == start_of_day) & is_month_start(ts)
    elif freq == "QS":
        return (ts.time() == start_of_day) & is_quarter_start(ts)
    elif freq == "AS":
        return (ts.time() == start_of_day) & is_year_start(ts)
    else:
        raise ValueError(f"Unexpected frequency ``freq``; got {freq}.")


def index(i: pd.DatetimeIndex, freq: str) -> pd.Series:
    f"""Check if timestamps in index are a valid delivery period start (or end).

    Parameters
    ----------
    ts : pd.Timestamp
        Timestamp for which to do the assertion.
    freq : {{{', '.join(tools_freq.FREQUENCIES)}}}
        Frequency for which to check if the timestamp is a valid start (or end) timestamp.

    Returns
    -------
    bool-Series
        With ``i`` as its index, and boolean indicating if timestamp is valid start as the values.

    Notes
    -----
    For shorter-than-daily indices, it is assumed that the index starts with a full day.
    I.e., the time-of-day of the first element is assumed to be the start time for the
    day-or-longer delivery periods. (E.g., if the index has hourly values and starts with
    "2020-04-21 06:00:00", it is assumed that a delivery day is from 06:00:00 (incl)
    until 06:00:00 (excl).)
    """
    # When comparing index to shorter (or same) frequency, all timestamps are on boundary.
    if tools_freq.up_or_down(i.freq, freq) >= 0:
        values = True

    # When comparing daily-or-longer index to other daily-or-longer frequency X,
    # we only need check only if the stamps are on first day of X.
    elif tools_freq.up_or_down(i.freq, "D") >= 0:
        values = is_X_start(i, freq)

    # Comparing shorter-than-daily index to other shorter-than-daily frequency X,
    # (i.e., '15T' with 'H')
    elif tools_freq.up_or_down(freq, "H") <= 0:
        if i.freq == "15T" and freq == "H":
            values = i.minute == 0
        else:
            raise ValueError(
                f"Unexpected frequencies for ``i`` and ``freq``; got {i.freq} and {freq}."
            )

    # Comparing shorter-than-daily index to daily-or-longer frequency X,
    # we need to check if the stamps are on the first day of X AND if they have the correct time.
    else:
        # . Check time of day.
        start_of_day = i[0].time()
        values = i.time == start_of_day
        # . Check day of X.
        if tools_freq.up_or_down(freq, "D") > 0:
            values &= is_X_start(i, freq)

    return pd.Series(values, i, name="isboundary")
