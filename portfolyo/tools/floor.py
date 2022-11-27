"""
Floor timestamp(s) to beginning of delivery period.
"""

import datetime as dt

import pandas as pd

from . import freq as tools_freq
from . import round as tools_round


def stamp(
    ts: pd.Timestamp, freq: str, future: int = 0, start_of_day: dt.time = None
) -> pd.Timestamp:
    f"""Floor timestamp to beginning of delivery period.

    i.e., find (latest) period start that is on or before the timestamp.

    Parameters
    ----------
    ts : pd.Timestamp
        Timestamp to floor.
    freq : {{{', '.join(tools_freq.FREQUENCIES)}}}
        Frequency for which to floor the timestamp.
    future : int, optional (default: 0)
        0 to floor to current period. 1 (-1) to round to period after (before) that, etc.
    start_of_day : dt.time, optional (default: midnight)
        Time of day at which daily-or-longer delivery periods start. E.g. if
        dt.time(hour=6), a delivery day is from 06:00:00 (incl) until 06:00:00 (excl).

    Returns
    -------
    pd.Timestamp

    Notes
    -----
    If ``ts`` is exactly at the start of the period, ceiling and flooring both return the
    original timestamp.

    Examples
    --------
    >>> floor.stamp(pd.Timestamp('2020-04-21 15:42'), 'AS')
    Timestamp('2020-01-01 00:00:00')
    >>> floor.stamp(pd.Timestamp('2020-04-21 15:42'), 'MS')
    Timestamp('2020-04-01 00:00:00')
    >>> floor.stamp(pd.Timestamp('2020-04-21 15:42'), '15T')
    Timestamp('2020-04-21 15:30:00')
    >>> floor.stamp(pd.Timestamp('2020-04-21 15:42', tz='Europe/Berlin'), 'MS')
    Timestamp('2020-04-01 00:00:00+0200', tz='Europe/Berlin')
    >>> floor.stamp(pd.Timestamp('2020-04-21 15:42'), 'MS', start_of_day=dt.time(hour=6))
    Timestamp('2020-04-01 06:00:00')
    >>> floor.stamp(pd.Timestamp('2020-04-21 15:42'), 'MS', 2)
    Timestamp('2020-06-01 00:00:00')
    """
    return tools_round.stamp_general("floor", ts, freq, future, start_of_day)


# def index(i: pd.DatetimeIndex, freq: str, future: int = 0) -> pd.DatetimeIndex:
#     f"""Floor timestamps of index to start of delivery period.

#     i.e., find (latest) period start that is on or before the timestamp.

#     Parameters
#     ----------
#     i : pd.DatetimeIndex
#         Timestamps to floor.
#     freq : {{{', '.join(tools_freq.FREQUENCIES)}}}
#         What to floor to, e.g. 'QS' to get start of quarter it's contained in.
#     future : int, optional (default: 0)
#         0 to floor to period start. 1 (-1) to get start of period after (before) that, etc.

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
#     return tools_round.index_general("floor", i, freq, future)
