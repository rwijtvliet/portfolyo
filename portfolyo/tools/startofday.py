"""
Tools to get/set start-of-day.
"""


import pandas as pd
from . import freq as tools_freq
import datetime as dt
from typing import Union


def get(
    i: pd.DatetimeIndex, returntype: str = "time"
) -> Union[dt.time, str, dt.timedelta]:
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
    Union[dt.time, str]
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


def set(i: pd.DatetimeIndex, start_of_day: dt.time = None) -> pd.DatetimeIndex:
    """Set the start-of-day of an index by changing the time-part of the index elements.

    Parameters
    ----------
    i : pd.DatetimeIndex
        Index; must have a daily (or longer) frequency.
    start_of_day : dt.time, optional (default: midnight)
        The new time that replaces the time-part of index elements.

    Returns
    -------
    pd.DatetimeIndex
        With adjusted time-part. Date-part is unchanged, so new timestamps are on same
        calendar day as original timestamps.
    """
    if tools_freq.up_or_down(i.freq, "D") < 0:
        raise ValueError(
            "Can only set the start-of-day of an index with daily (or longer) frequency."
        )

    if start_of_day is None:
        start_of_day = dt.time(hour=0, minute=0, second=0)
    elif start_of_day.second != 0 or start_of_day.minute % 15 != 0:
        raise ValueError("Start of day must coincide with a full quarterhour.")

    tss = (ts.replace(hour=start_of_day.hour, minute=start_of_day.minute) for ts in i)
    return pd.DatetimeIndex(tss, freq=i.freq, tz=i.tz)
