"""
Tools for working for with timestamps.
"""

from typing import Tuple

import pandas as pd

from . import floor as tools_floor
from . import isboundary as tools_isboundary


def ts_leftright(
    left: pd.Timestamp = None,
    right: pd.Timestamp = None,
    tz: str = None,
    offset_hours: int = None,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Makes 2 timestamps coherent to one another.

    Parameters
    ----------
    left : timestamp, optional
    right : timestamp, optional
        If no values is given for either, the entire next year is given.
        If no value for ``left`` is given, the beginning of the year of ts_right is given.
        If no value for ``right`` is given, the end of the year of ``left`` is given.
        If a value is given for each, they are swapped if their order is incorrect.
    tz : str, optional (default: None)
        Timezone for the returned timestamps. Timezones of ``left`` and ``right`` take
        priority, so this values is ignored unless both ``left`` and ``right`` are not
        specified.
    offset_hours : int, optional (default: 0)
        Offset of delivery period compared to midnight, in hours. E.g. 6 if delivery
        periods start at 06:00:00.
        Used to determine the start (and end) of the year if one of the timestamps is
        omitted.

    Returns
    -------
    (timestamp, timestamp)

    Notes
    -----
    If both ``left`` and ``right`` are specified, an error is raised if their timezones
    are not identical.
    """
    # Convert both into timestamps, if possible. None is converted into pd.NaT
    left, right = pd.Timestamp(left), pd.Timestamp(right)

    if right is pd.NaT:
        if left is pd.NaT:
            left = tools_floor.stamp(pd.Timestamp.now(tz=tz), "AS", 1, offset_hours)
        right = tools_floor.stamp(left, "AS", 1, offset_hours)

    # if we land here, we at least know right.
    if left is pd.NaT:
        if tools_isboundary.stamp(right, "AS", offset_hours):
            back = -1
        else:
            back = 0
        left = tools_floor.stamp(right, "AS", back, offset_hours)

    # if we land here, we know left and right.
    zones = [None if ts.tz is None else ts.tz.zone for ts in [left, right]]
    if len(set(zones)) == 2:  # distinct timezones
        raise ValueError(
            f"The timestamps have distinct timezones: {left.tz} and {right.tz}."
        )

    # if we land here, we know left and right, and both have same timezone.
    if left > right:
        left, right = right, left

    return left, right
