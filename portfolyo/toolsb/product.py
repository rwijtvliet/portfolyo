"""Utilities for calculating / manipulating price and product data."""

import datetime as dt
from typing import Tuple

import pandas as pd

from . import peakfn as tools_peakfn
from . import stamp as tools_stamp
from . import startofday as tools_sod

germanpower_peakfn = tools_peakfn.factory("08:00", "20:00", [1, 2, 3, 4, 5])


def delivery_period(
    ts_trade: pd.Timestamp,
    delivery_duration: str,
    front_count: int,
    startofday: dt.time | str = tools_sod.MIDNIGHT,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Find start and end of delivery period.

    Parameters
    ----------
    ts_trade
        Trading day. The time part of the timestamp is ignored and assumed to be after
        the start_of_day of the market.
    delivery_duration
        Duration of the delivery period.
    front_count
        0 = period that `ts_trade` falls in. 1 = following (full) period, 2 = period after that, etc.
    startofday, optional
        Start of day for delivery periods with a daily-or-longer frequency.

    Returns
    -------
    (pd.Timestamp, pd.Timestamp)
        Left (inclusive) and right (exclusive) timestamp of delivery period.
    """
    startofday = tools_sod.coerce(startofday)
    ts_trade = ts_trade.replace(hour=23, minute=59)  # ensure after start_of_day
    if delivery_duration in ["m", "q", "y"]:
        freq = delivery_duration.upper() + "S"
        ts_left = tools_stamp.floor(ts_trade, freq, startofday)
        ts_right = tools_stamp.to_right(ts_left, freq)
    elif delivery_duration == "d":
        ts_left = tools_stamp.floor(ts_trade, "D", startofday)
        ts_right = tools_stamp.to_right(ts_left, "D")
    elif delivery_duration == "s":
        front_count_q = front_count * 2 - 1
        ts_left, ts_right = delivery_period(ts_trade, "q", front_count_q, startofday)
        ts_right = tools_stamp.to_right(ts_right, "QS")  # make 6 months long
        if ts_left.month % 2 == 1:  # season must start on even month
            ts_left = tools_stamp.to_right(ts_left, "QS")
            ts_right = tools_stamp.to_right(ts_right, "QS")
    else:
        raise ValueError(
            f"Parameter ``period_type`` must be one of 'd', 'm', 'q', 's', 'y'; got '{delivery_duration}'."
        )
    return ts_left, ts_right
