"""
Round timestamp(s) to beginning of delivery period.

Not exposed directly; used by floor and ceil.
"""

import datetime as dt
from typing import Union

import pandas as pd
from pytz import AmbiguousTimeError

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
    if tools_isboundary.stamp(ts, freq, start_of_day):
        return ts

    # If we land here, the timestamp is not on a boundary.

    up_or_down = tools_freq.up_or_down(freq, "D")

    # Fixed-duration frequency (= (quarter)hour): simply floor/ceil.
    if up_or_down < 0:
        return ts.floor(freq) if fn == "floor" else ts.ceil(freq)

    # If we land here, the frequency is daily-or-longer.

    # . Correct for the time-of-day.
    if ts.time() == start_of_day:
        rounded = ts
    else:
        part_of_prevday = ts.time() < start_of_day
        rounded = ts.replace(hour=start_of_day.hour, minute=start_of_day.minute)
        if part_of_prevday and fn == "floor":
            rounded -= tools_freq.to_offset("D")
        elif not part_of_prevday and fn == "ceil":
            rounded += tools_freq.to_offset("D")
    # . Correct for the day-of-X.
    if tools_isboundary.is_X_start(rounded, freq):
        return rounded
    elif fn == "floor":
        return rounded + _offset(freq, -1)
    else:
        return rounded + _offset(freq, 1)


def index_general(
    fn: str, i: pd.DatetimeIndex, freq: str, future: int = 0
) -> pd.DatetimeIndex:
    f"""Floor or ceil an index.

    Parameters
    ----------
    fn : {'floor', 'ceil'}
    i : pd.DatetimeIndex
        Timestamps for which to do the rounding.
    freq : {{{', '.join(tools_freq.FREQUENCIES)}}}
        Frequency for which to round the index.
    future : int, optional (default: 0)
        0 to round to current period. 1 (-1) to round to period after (before) that, etc.

    Returns
    -------
    pd.DatetimeIndex

    Notes
    -----
    For shorter-than-daily indices, it is assumed that the index starts with a full day.
    I.e., the time-of-day of the first element is assumed to be the start time for the
    day-or-longer delivery periods. (E.g., if the index has hourly values and starts with
    "2020-04-21 06:00:00", it is assumed that a delivery day is from 06:00:00 (incl)
    until 06:00:00 (excl).)
    """
    rounded_current = index_current(fn, i, freq)
    return rounded_current + future * tools_freq.to_offset(freq)


def index_current(fn: str, i: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    f"""Floor or ceil an index to the delivery period it's contained in.

    Parameters
    ----------
    fn : {'floor', 'ceil'}
    i : pd.DatetimeIndex
        Timestamps for which to do the rounding.
    freq : {{{', '.join(tools_freq.FREQUENCIES)}}}
        Frequency for which to round the timestamp.

    Returns
    -------
    pd.DatetimeIndex

    Notes
    -----
    For shorter-than-daily indices, it is assumed that the index starts with a full day.
    I.e., the time-of-day of the first element is assumed to be the start time for the
    day-or-longer delivery periods. (E.g., if the index has hourly values and starts with
    "2020-04-21 06:00:00", it is assumed that a delivery day is from 06:00:00 (incl)
    until 06:00:00 (excl).)
    """
    start_of_day = i[0].time()
    return pd.DatetimeIndex([stamp_current(fn, ts, freq, start_of_day) for ts in i])

    # # When comparing index to shorter (or same) frequency, all timestamps are on boundary,
    # # and we don't need to round.
    # if tools_freq.up_or_down(i.freq, freq) >= 0:
    #     return i

    # unchanged =  tools_isboundary.index(i, freq)
    # if all(unchanged):
    #     return i

    # # If we land here, not all timestamps are on a boundary.

    # up_or_down = tools_freq.up_or_down(freq, "D")

    # # Fixed-duration frequency (= (quarter)hour): simply floor/ceil.
    # if up_or_down < 0:
    #     return ts.floor(freq) if fn == "floor" else ts.ceil(freq)

    # # If we land here, the frequency is daily-or-longer.

    # # . Correct for the time-of-day.
    # if ts.time() == start_of_day:
    #     rounded = ts
    # else:
    #     part_of_prevday = ts.time() < start_of_day
    #     rounded = ts.replace(hour=start_of_day.hour, minute=start_of_day.minute)
    #     if part_of_prevday and fn == "floor":
    #         rounded -= tools_freq.to_offset("D")
    #     elif not part_of_prevday and fn == "ceil":
    #         rounded += tools_freq.to_offset("D")
    # # . Correct for the day-of-X.
    # if tools_isboundary.is_X_start(rounded, freq):
    #     return rounded
    # elif fn == "floor":
    #     return rounded + _offset(freq, -1)
    # else:
    #     return rounded + _offset(freq, 1)

    # # When comparing daily-or-longer index to other daily-or-longer frequency X,
    # # we only need check only if the stamps are on first day of X.
    # elif tools_freq.up_or_down(i.freq, "D") >= 0:
    #     values = is_X_start(i, freq)

    # # Comparing shorter-than-daily index to other shorter-than-daily frequency X,
    # # (i.e., '15T' with 'H')
    # elif tools_freq.up_or_down(freq, "H") <= 0:
    #     if i.freq == "15T" and freq == "H":
    #         values = i.minute == 0
    #     else:
    #         raise ValueError(
    #             f"Unexpected frequencies for ``i`` and ``freq``; got {i.freq} and {freq}."
    #         )

    # # Comparing shorter-than-daily index to daily-or-longer frequency X,
    # # we need to check if the stamps are on the first day of X AND if they have the correct time.
    # else:
    #     # . Check time of day.
    #     start_of_day = i.time[0]
    #     values = i.time == start_of_day
    #     # . Check day of X.
    #     if tools_freq.up_or_down(freq, "D") > 0:
    #         values &= is_X_start(i, freq)

    # return pd.Series(values, i, name="isboundary")


def general(
    fn: str,
    obj: Union[pd.Timestamp, pd.DatetimeIndex],
    freq: str,
    future: int = 0,
    offset_hours: int = 0,
) -> pd.Timestamp:
    f"""Floor or ceil timestamp or index to start of delivery period.

    Parameters
    ----------
    fn : {'floor', 'ceil'}
    obj : pd.Timestamp or pd.DatetimeIndex
        Timestamp to floor or ceil.
    freq : {{{','.join(tools_freq.FREQUENCIES)}}}
        What to round to, e.g. 'QS' to get start of quarter it's contained in.
    future : int, optional (default: 0)
        0 to floor/ceil to period start. 1 (-1) to get start of period after (before)
        that, etc.
    offset_hours : int, optional (default: 0)
        Offset of delivery period compared to midnight, in hours. E.g. 6 if delivery
        periods start at 06:00:00.
        Used only when flooring/ceiling a below-daily index to a daily-or-longer frequency.

    Returns
    -------
    pd.Timestamp or pd.DatetimeIndex
        At begin of delivery period.
    """
    if fn == "floor":
        rounding = type(obj).floor
    elif fn == "ceil":
        rounding = type(obj).ceil
    else:
        raise ValueError(f"``fn`` should be 'ceil' or 'floor'; got '{fn}'.")

    # Initial rounding.
    up_or_down = tools_freq.up_or_down(freq, "D")

    # For short (< day) frequencies.
    if up_or_down == -1:
        try:
            rounded = rounding(obj, freq)
        except AmbiguousTimeError:
            rounded = rounding(obj, freq, ambiguous="infer")

    # For day frequencies.
    elif up_or_down == 0:
        # rounded = _round_to_days(fn, obj, offset_hours)
        rounded = obj
        if offset_hours:
            rounded = _round_to_days(fn, rounded, offset_hours)
        # +1 and then -1 needed to get correct result; 0 is incorrect at e.g. month start.
        if fn == "floor":
            rounded = rounded + _offset(freq, 1) + _offset(freq, -1)
        else:
            rounded = rounded + _offset(freq, 0)
        if not offset_hours:
            rounded = _round_to_days(fn, rounded, offset_hours)

    # For longer (> day) frequencies.
    else:
        rounded = obj
        if offset_hours:
            rounded = _round_to_days("floor", rounded, offset_hours)
        # +1 and then -1 needed to get correct result; 0 is incorrect at e.g. month start.
        if fn == "floor":
            rounded = rounded + _offset(freq, 1) + _offset(freq, -1)
        else:
            rounded = rounded + _offset(freq, 0)
        if not offset_hours:
            rounded = _round_to_days("floor", rounded, offset_hours)
        # obj.floor("D")

    # Final rounding.
    if future != 0:
        return rounded + _offset(freq, future)
    else:
        return rounded


def _round_to_days(
    fn: str, obj: Union[pd.Timestamp, pd.DatetimeIndex], offset_hours: int = 0
) -> Union[pd.Timestamp, pd.DatetimeIndex]:
    if not offset_hours:
        if fn == "floor":
            return obj.floor("D")
        else:
            return obj.ceil("D")
    else:
        if isinstance(obj, pd.Timestamp):
            obj = _floor_index_to_daily_with_offset([obj], offset_hours)[0]
        else:
            obj = _floor_index_to_daily_with_offset(obj, offset_hours)
        if fn == "floor":
            return obj
        else:
            return obj + _offset("D", 1)


def _floor_index_to_daily_with_offset(
    i: pd.DatetimeIndex, offset_hours: int
) -> pd.DatetimeIndex:
    gr = pd.Grouper(freq="D", offset=pd.Timedelta(hours=offset_hours))
    g = pd.Series(index=i, dtype=float).groupby(gr)
    idx = g.size()
    return idx.repeat(idx).index


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
