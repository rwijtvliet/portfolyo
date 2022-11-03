"""
Ceil timestamp(s) to end of delivery period.
"""

import pandas as pd

from . import floor as tools_floor
from . import freq as tools_freq


def stamp(
    ts: pd.Timestamp, freq: str, future: int = 0, offset_hours: float = 0
) -> pd.Timestamp:
    f"""Ceil timestamp to end of delivery period.

    i.e., find (earliest) period start that is on or after the timestamp.

    Parameters
    ----------
    ts : pd.Timestamp
        Timestamp to ceil.
    freq : {{{','.join(tools_freq.FREQUENCIES)}}}
        What to ceil to, e.g. 'QS' to get end of quarter it's contained in.
    future : int, optional (default: 0)
        0 to ceil to period end. 1 (-1) to get end of period after (before) that, etc.
    offset_hours : float, optional (default: 0)
        Offset of delivery period compared to midnight, in hours. E.g. 6 if delivery
        periods start at 06:00:00.
        Used only when ceiling a below-daily index to a daily-or-longer frequency.

    Returns
    -------
    pd.Timestamp
        At end of delivery period.

    Notes
    -----
    If ``ts`` is exactly at the start of the period, ceiling and flooring both return the
    original timestamp.

    Examples
    --------
    >>> ceil.stamp(pd.Timestamp('2020-04-21 15:42'), 'AS')
    Timestamp('2021-01-01 00:00:00')
    >>> ceil.stamp(pd.Timestamp('2020-04-21 15:42'), 'MS')
    Timestamp('2020-05-01 00:00:00')
    >>> ceil.stamp(pd.Timestamp('2020-04-21 15:42'), '15T')
    Timestamp('2020-04-21 15:45:00')
    >>> ceil.stamp(pd.Timestamp('2020-04-21 15:42', tz='Europe/Berlin'), 'MS')
    Timestamp('2020-05-01 00:00:00+0200', tz='Europe/Berlin')
    >>> ceil.stamp(pd.Timestamp('2020-04-21 15:42'), 'MS', offset_hours=6)
    Timestamp('2020-05-01 06:00:00')
    >>> ceil.stamp(pd.Timestamp('2020-04-21 15:42'), 'MS', 2)
    Timestamp('2020-07-01 00:00:00')
    """
    if ts != tools_floor.stamp(ts, freq):
        additional_future = 1
    else:
        additional_future = 0
    return tools_floor.stamp(ts, freq, future + additional_future, offset_hours)


def index(
    i: pd.DateimeIndex, freq: str, future: int = 0, offset_hours: float = 0
) -> pd.DatetimeIndex:
    f"""Ceil timestamps of index to end of delivery period.

    i.e., find (earliest) period start that is on or before the timestamp.

    Parameters
    ----------
    i : pd.DatetimeIndex
        Timestamps to ceil.
    freq : {{{', '.join(tools_freq.FREQUENCIES)}}}
        What to ceil to, e.g. 'QS' to get end of quarter it's contained in.
    future : int, optional (default: 0)
        0 to ceil to period end. 1 (-1) to get end of period after (before) that, etc.
    offset_hours : float, optional (default: 0)
        Offset of delivery period compared to midnight, in hours. E.g. 6 if delivery
        periods start at 06:00:00.
        Used only when ceiling a below-daily index to a daily-or-longer frequency.

    Returns
    -------
    pd.DatetimeIndex
        With timestamsp at end of delivery period.
    """
    return pd.DatetimeIndex([stamp(ts, freq, future, offset_hours) for ts in i])
