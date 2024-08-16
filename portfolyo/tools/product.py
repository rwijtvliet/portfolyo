"""Utilities for calculating / manipulating price data."""

import datetime as dt
from typing import Tuple

import pandas as pd

from . import floor as tools_floor
from . import peakfn as tools_peakfn
from . import right as tools_right

germanpower_peakfn = tools_peakfn.factory(dt.time(8), dt.time(20), [1, 2, 3, 4, 5])


def is_peak_hour(i: pd.DatetimeIndex) -> pd.Series:
    raise DeprecationWarning(
        "``pf.is_peak_hour`` has been deprecated and will be removed in a future version."
        " Use ``pf.germanpower_peakfn`` instead, or create your own peak function with"
        " ``pf.create_peakfn()``."
    )


def delivery_period(
    ts_trade: pd.Timestamp,
    period_type: str = "m",
    front_count: int = 1,
    start_of_day: dt.time = None,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
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
        Start of day for delivery periods with a daily-or-longer frequency.

    Returns
    -------
    (pd.Timestamp, pd.Timestamp)
        Left (inclusive) and right (exclusive) timestamp of delivery period.
    """
    ts_trade = ts_trade.replace(hour=23, minute=59)  # ensure after start_of_day
    if period_type in ["m", "q", "y"]:
        freq = period_type.upper() + "S"
        ts_left = tools_floor.stamp(ts_trade, freq, front_count, start_of_day)
        ts_right = tools_right.stamp(ts_left, freq)
    elif period_type == "d":
        ts_left = tools_floor.stamp(ts_trade, "D", front_count, start_of_day)
        ts_right = tools_right.stamp(ts_left, "D")
    elif period_type == "s":
        front_count_q = front_count * 2 - 1
        ts_left, ts_right = delivery_period(ts_trade, "q", front_count_q, start_of_day)
        ts_right = tools_right.stamp(ts_right, "QS")  # make 6 months long
        if ts_left.month % 2 == 1:  # season must start on even month
            ts_left = tools_right.stamp(ts_left, "QS")
            ts_right = tools_right.stamp(ts_right, "QS")
    else:
        raise ValueError(
            f"Parameter ``period_type`` must be one of 'd', 'm', 'q', 's', 'y'; got '{period_type}'."
        )
    return ts_left, ts_right
