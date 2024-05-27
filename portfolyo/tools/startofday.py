"""
Tools to get/set start-of-day.
"""

import datetime as dt

import pandas as pd

from . import freq as tools_freq
from . import right as tools_right


def get(i: pd.DatetimeIndex, returntype: str = "time") -> dt.time | str | dt.timedelta:
    """Get start-of-day of an index.

    Parameters
    ----------
    i : pd.DatetimeIndex
    returntype : {'time', 'str', 'timedelta'}, optional (default: 'time')
        If 'time', return as datetime.time object.
        If 'str', return as HH:MM:SS string.
        If 'timedelta', return as datetime.timedelta object with timedelta to previous
        midnight (24h day).

    Returns
    -------
    dt.time | str | dt.timedelta
    """
    start_of_day = i[0].time()
    if returntype == "time":
        return start_of_day
    elif returntype == "str":
        return f"{start_of_day.hour:02}:{start_of_day.minute:02}:00"
    elif returntype == "timedelta":
        return dt.timedelta(hours=start_of_day.hour, minutes=start_of_day.minute)
    raise ValueError(
        f"Unknown value for parameter 'returntype'. Expected one of {'time', 'str', 'timedelta'}, got {returntype}."
    )


def set(i: pd.DatetimeIndex, start_of_day: dt.time) -> pd.DatetimeIndex:
    """Set the start-of-day of an index. Done by changing the time-part of the index
    elements (if index has daily-or-longer frequency) or by trimming the index (if index
    has hourly-or-shorter frequency).

    Parameters
    ----------
    i : pd.DatetimeIndex
    start_of_day : dt.time

    Returns
    -------
    pd.DatetimeIndex
        With wanted start-of-day.
    """
    if start_of_day.second != 0 or start_of_day.minute % 15 != 0:
        raise ValueError("Start of day must coincide with a full quarterhour.")

    if tools_freq.up_or_down(i.freq, "D") >= 0:
        return _set_to_longfreq(i, start_of_day)
    else:
        return _set_to_shortfreq(i, start_of_day)


def _set_to_longfreq(i: pd.DatetimeIndex, start_of_day: dt.time) -> pd.DatetimeIndex:
    """Set start-of-day of index with daily-or-longer frequency."""
    tss = (ts.replace(hour=start_of_day.hour, minute=start_of_day.minute) for ts in i)
    return pd.DatetimeIndex(tss, freq=i.freq, tz=i.tz)


def _set_to_shortfreq(i: pd.DatetimeIndex, start_of_day: dt.time) -> pd.DatetimeIndex:
    """Set start-of-day of index with hourly-or-shorter frequency. Destructive; values
    at start and/or end of index are removed to get wanted start_of_day."""
    # Remove from start if necessary.
    for _ in range(0, 100):  # max 100 quarterhours in a day (@ end of DST)
        if i[0].time() == start_of_day:
            break
        i = i[1:]
    else:
        raise ValueError("Did not find any timestamp with correct time at index start.")
    # Remove from end if necessary.
    for _ in range(0, 100):  # max 100 quarterhours in a day (@ end of DST)
        if tools_right.stamp(i[-1], i.freq).time() == start_of_day:
            break
        i = i[:-1]
    else:
        raise ValueError("Did not find any timestamp with correct time at index end.")
    return i
