"""
Check if timestamp is delivery period start (or end).
"""
import pandas as pd

from . import floor as tools_floor
from . import freq as tools_freq


def stamp(ts: pd.Timestamp, freq: str, offset_hours: float = 0) -> bool:
    f"""Check if timestamp is a valid delivery period start (or end).

    Parameters
    ----------
    ts : pd.Timestamp
        Timestamp for which to do the assertion.
    freq : {{{', '.join(tools_freq.FREQUENCIES)}}}
        Frequency for which to check if the timestamp is a valid start (or end) timestamp.
    offset_hours : float, optional (default: 0)
        Offset of delivery period compared to midnight, in hours. E.g. 6 if delivery
        periods start at 06:00:00.
        Used only when assessing a daily-or-longer frequency.

    Returns
    -------
    bool

    Examples
    --------
    >>> isboundary.stamp(pd.Timestamp('2020-02-01'), 'MS')
    True
    >>> isboundary.stamp(pd.Timestamp('2020-02-01'), 'AS')
    False
    >>> isboundary.stamp(pd.Timestamp('2020-02-01 06:00'), 'D')
    False
    >>> isboundary.stamp(pd.Timestamp('2020-02-01 06:00'), 'D', 6)
    True
    """
    return ts == tools_floor.stamp(ts, freq, offset_hours=offset_hours)


def index(i: pd.DatetimeIndex, freq: str, offset_hours: float = 0) -> bool:
    f"""Check if all timestamps in index are a valid delivery period start (or end).

    Parameters
    ----------
    ts : pd.Timestamp
        Timestamp for which to do the assertion.
    freq : {{{', '.join(tools_freq.FREQUENCIES)}}}
        Frequency for which to check if the timestamp is a valid start (or end) timestamp.
    offset_hours : float, optional (default: 0)
        Offset of delivery period compared to midnight, in hours. E.g. 6 if delivery
        periods start at 06:00:00.
        Used only when assessing a daily-or-longer frequency.

    Returns
    -------
    bool
    """
    return all(stamp(ts, freq, offset_hours) for ts in i)
