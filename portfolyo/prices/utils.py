"""Utilities for calculating / manipulating price data."""

import datetime as dt
import warnings
from typing import Tuple, Union

import pandas as pd

from .. import tools
from ..tools.unit import Q_

german_peakperiod = tools.peakperiod.factory(dt.time(8), dt.time(20), [1, 2, 3, 4, 5])


def is_peak_hour(
    ts_left: Union[pd.Timestamp, pd.DatetimeIndex]
) -> Union[bool, pd.Series]:
    """
    Boolean value indicating if a timestamp is in a peak period or not.

    Parameters
    ----------
    ts_left : Union[pd.Timestamp, pd.DatetimeIndex]
        Timestamp(s) for which to calculate if it falls in a peak period.

    More precisely: if timestamp lies in one of the (left-closed) time intervals that
    define the peak hour periods.

    Returns
    -------
    bool (if ts_left is Timestamp) or Series (if ts_left is DatetimeIndex).
    """
    if isinstance(ts_left, pd.Timestamp):
        warnings.warn(
            "Calling this function with a single timestamp is deprecated and will be removed in a future version",
            FutureWarning,
        )
        return is_peak_hour(pd.DatetimeIndex([ts_left], freq="15T"))[0]

    return german_peakperiod(ts_left)


def duration_base(
    ts_left: Union[pd.Timestamp, pd.DatetimeIndex], freq: str = None
) -> Union[Q_, pd.Series]:
    """
    Total duration of base periods in a timestamp.

    See also
    --------
    .tools.duration
    """
    if isinstance(ts_left, pd.DatetimeIndex):
        hours = (duration_base(ts, freq) for ts in ts_left)  # has unit
        return pd.Series(hours, ts_left, dtype="pint[h]")

    # Assume it's a single timestamp.
    return tools.duration.stamp(ts_left, freq)


def duration_peak(
    ts_left: Union[pd.Timestamp, pd.DatetimeIndex], freq: str = None
) -> Union[Q_, pd.Series]:
    """
    Total duration of peak periods in a timestamp.

    See also
    --------
    .tools.duration
    """
    if freq is None:
        freq = ts_left.freq

    if freq not in tools.freq.FREQUENCIES:
        raise ValueError(
            f"Parameter ``freq`` must be one of {', '.join(tools.freq.FREQUENCIES)}; got {freq}."
        )

    if isinstance(ts_left, pd.DatetimeIndex):
        if freq in ["15T", "H"]:
            return tools.duration.index(ts_left, freq) * is_peak_hour(ts_left)
        elif freq == "D":
            hours = ts_left.map(lambda ts: ts.isoweekday() < 6) * 12  # no unit
            return pd.Series(hours, ts_left, dtype="pint[h]")  # works even during dst
        else:
            # dur = ts_left.map(duration_peak)  # crashes due to behaviour of .map method
            hours = (duration_peak(ts, freq) for ts in ts_left)  # has unit
            return pd.Series(hours, ts_left, dtype="pint[h]")

    # Assume it's a single timestamp.
    if freq in ["15T", "H"]:
        return tools.duration.stamp(ts_left, freq) * is_peak_hour(ts_left)
    elif freq == "D":
        return Q_(0 if ts_left.isoweekday() >= 6 else 12, "h")
    else:
        ts_right = tools.right.stamp(ts_left, freq)
        days = pd.date_range(ts_left, ts_right, freq="D", inclusive="left")
        return Q_(sum(days.map(lambda day: day.isoweekday() < 6) * 12), "h")


def duration_offpeak(
    ts_left: Union[pd.Timestamp, pd.DatetimeIndex], freq: str = None
) -> Union[Q_, pd.Series]:
    """
    Total duration of offpeak periods in a timestamp.

    See also
    --------
    .tools.duration
    """
    return duration_base(ts_left, freq) - duration_peak(ts_left, freq)


def duration_bpo(
    ts_left: Union[pd.Timestamp, pd.DatetimeIndex], freq: str = None
) -> Union[pd.Series, pd.DataFrame]:
    """
    Duration of base, peak and offpeak periods in a timestamp.

    Parameters
    ----------
    ts_left : Union[pd.Timestamp, pd.DatetimeIndex]
        Timestamp(s) for which to calculate the base, peak and offpeak durations.
    freq : {'15T' (quarter-hour), 'H' (hour), 'D' (day), 'MS' (month), 'QS' (quarter),
        'AS' (year)}, optional
        Frequency to use in determining the durations.
        If none specified, use ``.freq`` attribute of ``ts_left``.

    Returns
    -------
    Series (if ts_left is Timestamp) or DataFrame (if ts_left is DatetimeIndex).
    """
    if freq is None and isinstance(ts_left, pd.DatetimeIndex):
        freq = ts_left.freq

    b = duration_base(ts_left, freq)  # quantity or pint-series
    p = duration_peak(ts_left, freq)  # quantity or pint-series

    if isinstance(ts_left, pd.DatetimeIndex):
        return pd.DataFrame({"base": b, "peak": p, "offpeak": b - p}, dtype="pint[h]")

    # Assume it's a single timestamp.
    return pd.Series({"base": b, "peak": p, "offpeak": b - p}, dtype="pint[h]")


def delivery_period(
    ts_trade: pd.Timestamp,
    period_type: str = "m",
    front_count: int = 1,
    start_of_day: dt.time = None,
) -> Tuple[pd.Timestamp]:
    """
    Find start and end of delivery period.

    Parameters
    ----------
    ts_trade : pd.Timestamp
        Trading day. The time part of the timestamp is ignored and assumed to be after
        the start_of_day of the market.
    period_type : {'d' (day), 'm' (month, default), 'q' (quarter), 's' (season), 'a' (year)}
    front_count : int, optional (default: 1)
        1 = next/coming (full) period, 2 = period after that, etc.
    start_of_day : dt.time, optional (default: midnight)
        Start of day for delivery periods with a longer-than-daily frequency.

    Returns
    -------
    (pd.Timestamp, pd.Timestamp)
        Left (inclusive) and right (exclusive) timestamp of delivery period.
    """
    ts_trade = ts_trade.replace(hour=23, minute=59)  # ensure after start_of_day
    if period_type in ["m", "q", "a"]:
        freq = period_type.upper() + "S"
        ts_left = tools.floor.stamp(ts_trade, freq, front_count, start_of_day)
        ts_right = tools.right.stamp(ts_left, freq)
    elif period_type == "d":
        ts_left = tools.floor.stamp(ts_trade, "D", front_count, start_of_day)
        ts_right = tools.right.stamp(ts_left, "D")
    elif period_type == "s":
        front_count_q = front_count * 2 - 1
        ts_left, ts_right = delivery_period(ts_trade, "q", front_count_q, start_of_day)
        ts_right = tools.right.stamp(ts_right, "QS")  # make 6 months long
        if ts_left.month % 2 == 1:  # season must start on even month
            ts_left = tools.right.stamp(ts_left, "QS")
            ts_right = tools.right.stamp(ts_right, "QS")
    else:
        raise ValueError(
            f"Parameter ``period_type`` must be one of 'd', 'm', 'q', 's', 'a'; got '{period_type}'."
        )
    return ts_left, ts_right
