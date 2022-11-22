"""
Round timestamp(s) to beginning of delivery period.

Not exposed directly; used by floor and ceil.
"""

import datetime as dt

import pandas as pd

from . import freq as tools_freq
from . import isboundary as tools_isboundary


def stamp_general(
    fn: str, ts: pd.Timestamp, freq: str, future: int = 0, start_of_day: dt.time = None
) -> pd.Timestamp:
    f"""Floor or ceil a timestamp.

    Parameters
    ----------
    fn : {'floor', 'ceil'}
    ts : pd.Timestamp
        Timestamp for which to do the rounding.
    freq : {{{', '.join(tools_freq.FREQUENCIES)}}}
        Frequency for which to round the timestamp.
    future : int, optional (default: 0)
        0 to round to current period. 1 (-1) to round to period after (before) that, etc.
    start_of_day : dt.time, optional (default: midnight)
        Time of day at which daily-or-longer delivery periods start. E.g. if
        dt.time(hour=6), a delivery day is from 06:00:00 (incl) until 06:00:00 (excl).

    Returns
    -------
    pd.Timestamp
    """
    rounded_current = stamp_current(fn, ts, freq, start_of_day)
    return rounded_current + future * tools_freq.to_offset(freq)


def stamp_current(
    fn: str, ts: pd.Timestamp, freq: str, start_of_day: dt.time = None
) -> pd.Timestamp:
    f"""Floor or ceil a timestamp to the delivery period it's contained in.

    Parameters
    ----------
    fn : {'floor', 'ceil'}
    ts : pd.Timestamp
        Timestamp for which to do the rounding.
    freq : {{{', '.join(tools_freq.FREQUENCIES)}}}
        Frequency for which to round the timestamp.
    start_of_day : dt.time, optional (default: midnight)
        Time of day at which daily-or-longer delivery periods start. E.g. if
        dt.time(hour=6), a delivery day is from 06:00:00 (incl) until 06:00:00 (excl).

    Returns
    -------
    pd.Timestamp
    """
    if start_of_day is None:
        start_of_day = dt.time(0, 0)

    if tools_isboundary.stamp(ts, freq, start_of_day):
        return ts

    # If we land here, the timestamp is not on a boundary.

    # Fixed-duration frequency (= (quarter)hour): simply floor/ceil.
    if tools_freq.up_or_down(freq, "D") < 0:
        if fn == "floor":
            return ts.floor(freq, nonexistent="shift_backward")
        else:
            return ts.ceil(freq, nonexistent="shift_forward")

    # If we land here, the frequency is daily-or-longer.

    # . Correct for the time-of-day.
    if ts.time() == start_of_day:
        rounded = ts
    else:
        part_of_prevday = ts.time() < start_of_day
        rounded = ts.replace(hour=start_of_day.hour, minute=start_of_day.minute).floor(
            "15T"
        )
        if part_of_prevday and fn == "floor":
            rounded -= tools_freq.to_offset("D")
        elif not part_of_prevday and fn == "ceil":
            rounded += tools_freq.to_offset("D")
    # . Correct for the day-of-X.
    if freq == "D" or tools_isboundary.is_X_start(rounded, freq):
        return rounded
    elif fn == "floor":
        return rounded + _offset(freq, -1)
    else:
        return rounded + _offset(freq, 1)


# def index_general(
#     fn: str, i: pd.DatetimeIndex, freq: str, future: int = 0
# ) -> pd.DatetimeIndex:
#     f"""Floor or ceil an index.

#     Parameters
#     ----------
#     fn : {'floor', 'ceil'}
#     i : pd.DatetimeIndex
#         Timestamps for which to do the rounding.
#     freq : {{{', '.join(tools_freq.FREQUENCIES)}}}
#         Frequency for which to round the index.
#     future : int, optional (default: 0)
#         0 to round to current period. 1 (-1) to round to period after (before) that, etc.

#     Returns
#     -------
#     pd.DatetimeIndex

#     Notes
#     -----
#     For shorter-than-daily indices, it is assumed that the index starts with a full day.
#     I.e., the time-of-day of the first element is assumed to be the start time for the
#     day-or-longer delivery periods. (E.g., if the index has hourly values and starts with
#     "2020-04-21 06:00:00", it is assumed that a delivery day is from 06:00:00 (incl)
#     until 06:00:00 (excl).)
#     """
#     rounded_current = index_current(fn, i, freq)
#     return rounded_current + future * tools_freq.to_offset(freq)


# def index_current(fn: str, i: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
#     f"""Floor or ceil an index to the delivery period it's contained in.

#     Parameters
#     ----------
#     fn : {'floor', 'ceil'}
#     i : pd.DatetimeIndex
#         Timestamps for which to do the rounding.
#     freq : {{{', '.join(tools_freq.FREQUENCIES)}}}
#         Frequency for which to round the timestamp.

#     Returns
#     -------
#     pd.DatetimeIndex

#     Notes
#     -----
#     For shorter-than-daily indices, it is assumed that the index starts with a full day.
#     I.e., the time-of-day of the first element is assumed to be the start time for the
#     day-or-longer delivery periods. (E.g., if the index has hourly values and starts with
#     "2020-04-21 06:00:00", it is assumed that a delivery day is from 06:00:00 (incl)
#     until 06:00:00 (excl).)
#     """
#     start_of_day = i[0].time()
#     return pd.DatetimeIndex([stamp_current(fn, ts, freq, start_of_day) for ts in i])


def _offset(freq: str, future: int):
    if freq == "15T":
        return pd.Timedelta(minutes=future * 15)
    elif freq == "H":
        return pd.Timedelta(hours=future)
    elif freq == "D":
        return pd.Timedelta(days=future)
    elif freq == "MS":
        return pd.offsets.MonthBegin(future)
    elif freq == "QS":
        return pd.offsets.QuarterBegin(future, startingMonth=1)
    elif freq == "AS":
        return pd.offsets.YearBegin(future)
    else:
        raise ValueError(
            f"Parameter ``freq`` must be one of {', '.join(tools_freq.FREQUENCIES)}; got {freq}."
        )
