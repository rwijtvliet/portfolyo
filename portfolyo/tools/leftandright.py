"""
Tools for working for with timestamps.
"""

import datetime as dt
from typing import Tuple

import pandas as pd

from . import floor as tools_floor
from . import isboundary as tools_isboundary


def stamps(
    left: pd.Timestamp = None,
    right: pd.Timestamp = None,
    tz: str = None,
    start_of_day: dt.time = None,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Makes 2 timestamps coherent to one another to describe a valid delivery period.

    Parameters
    ----------
    left : timestamp, optional
    right : timestamp, optional
        If no values is given for either, the entire next year is given.
        If no value for ``left`` is given, the beginning of the year of ts_right is given.
        If no value for ``right`` is given, the end of the year of ``left`` is given.
        If a value is given for each, they are swapped if their order is incorrect.
    tz : str, optional (default: None)
        Timezone for the returned timestamps. Only used if both ``left`` and ``right``
        are missing.
    start_of_day : dt.time, optional (default: midnight)
        Time of day at which daily-or-longer delivery periods start. E.g. if
        dt.time(hour=6), a delivery day is from 06:00:00 (incl) until 06:00:00 (excl).
        Only used if both ``left`` and ``right`` are missing.

    Returns
    -------
    (timestamp, timestamp)

    Notes
    -----
    - Parameters ``tz`` and ``start_of_day`` are only used if ``left`` and ``right`` are
      both not specified.
    - If both ``left`` and ``right`` are specified, an error is raised if their timezones
      are not identical or if their times are not identical.
    """
    # Convert both into timestamps, if possible. None is converted into pd.NaT
    left, right = pd.Timestamp(left), pd.Timestamp(right)

    if right is pd.NaT:
        if left is pd.NaT:
            left = tools_floor.stamp(pd.Timestamp.now(tz=tz), "AS", 1, start_of_day)
        right = tools_floor.stamp(left, "AS", 1, left.time())

    # if we land here, we at least know right.
    if left is pd.NaT:
        start_of_day = right.time()
        if tools_isboundary.stamp(right, "AS", start_of_day):
            back = -1
        else:
            back = 0
        left = tools_floor.stamp(right, "AS", back, start_of_day)

    # if we land here, we know left and right.
    zones = [None if ts.tz is None else ts.tz.zone for ts in [left, right]]
    if len(set(zones)) == 2:  # distinct timezones
        raise ValueError(
            f"The timestamps have distinct timezones: {left.tz} and {right.tz}."
        )
    times = [ts.time() for ts in [left, right]]
    if len(set(times)) == 2:  # distinct times
        raise ValueError(
            f"The timestamps have distinct times: {left.time()} and {right.time()}."
        )

    # if we land here, we know left and right, and both have same timezone.
    if left > right:
        left, right = right, left

    return left, right
