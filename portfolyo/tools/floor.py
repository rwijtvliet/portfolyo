"""
Floor timestamp(s) to beginning of delivery period.
"""

from typing import Union

import pandas as pd

from . import freq as tools_freq


def stamp(
    ts: pd.Timestamp, freq: str, future: int = 0, offset_hours: float = 0
) -> pd.Timestamp:
    f"""Floor timestamp to start of delivery period.

    i.e., find (latest) period start that is on or before the timestamp.

    Parameters
    ----------
    ts : pd.Timestamp
        Timestamp to floor.
    freq : {{{','.join(tools_freq.FREQUENCIES)}}}
        What to floor to, e.g. 'QS' to get start of quarter it's contained in.
    future : int, optional (default: 0)
        0 to floor to period start. 1 (-1) to get start of period after (before) that, etc.
    offset_hours : float, optional (default: 0)
        Offset of delivery period compared to midnight, in hours. E.g. 6 if delivery
        periods start at 06:00:00.
        Used only when flooring a below-daily index to a daily-or-longer frequency.

    Returns
    -------
    pd.Timestamp
        At begin of delivery period.

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
    >>> floor.stamp(pd.Timestamp('2020-04-21 15:42'), 'MS', offset_hours=6)
    Timestamp('2020-04-01 06:00:00')
    >>> floor.stamp(pd.Timestamp('2020-04-21 15:42'), 'MS', 2)
    Timestamp('2020-06-01 00:00:00')
    """
    # Rounding to short (< day) frequencies.
    if freq == "15T":
        return ts.floor("15T") + pd.Timedelta(minutes=future * 15)
    elif freq == "H":
        return ts.floor("H") + pd.Timedelta(hours=future)

    # Rounding to longer (>= day) frequencies.
    # . Floor to nearest daily period start
    if not offset_hours:
        ts = ts.floor("D")
    else:
        gr = pd.Grouper(freq="D", offset=pd.Timedelta(hours=offset_hours))
        g = pd.Series(index=[ts], dtype=float).groupby(gr)
        ts = g.size().index[0]
    # Get correct flooring.
    return _offset(ts, freq, future)


def index(
    i: pd.DateimeIndex, freq: str, future: int = 0, offset_hours: float = 0
) -> pd.DatetimeIndex:
    f"""Floor timestamps of index to start of delivery period.

    i.e., find (latest) period start that is on or before the timestamp.

    Parameters
    ----------
    i : pd.DatetimeIndex
        Timestamps to floor.
    freq : {{{', '.join(tools_freq.FREQUENCIES)}}}
        What to floor to, e.g. 'QS' to get start of quarter it's contained in.
    future : int, optional (default: 0)
        0 to floor to period start. 1 (-1) to get start of period after (before) that, etc.
    offset_hours : float, optional (default: 0)
        Offset of delivery period compared to midnight, in hours. E.g. 6 if delivery
        periods start at 06:00:00.
        Used only when flooring a below-daily index to a daily-or-longer frequency.

    Returns
    -------
    pd.DatetimeIndex
        With timestamsp at begin of delivery period.
    """
    # Rounding to short (< day) frequencies.
    if freq == "15T":
        return i.floor("15T") + pd.Timedelta(minutes=future * 15)
    elif freq == "H":
        return i.floor("H") + pd.Timedelta(hours=future)

    # Rounding to longer (>= day) frequencies.
    # . Floor to nearest daily period start
    if not offset_hours:
        i = i.floor("D")
    else:
        gr = pd.Grouper(freq=freq, offset=pd.Timedelta(hours=offset_hours))
        g = pd.Series(index=i, dtype=float).groupby(gr)
        count = g.size()
        i = count.repeat(count).index
    # . Get correct flooring.
    return _offset(i, freq, future)


def _offset(obj: Union[pd.Timestamp, pd.DatetimeIndex], freq: str, future: int):
    if freq == "D":
        return obj + pd.Timedelta(days=future)
    elif freq == "MS":
        return obj + pd.offsets.MonthBegin(1) + pd.offsets.MonthBegin(future - 1)
    elif freq == "QS":
        return (
            obj
            + pd.offsets.QuarterBegin(1, startingMonth=1)
            + pd.offsets.QuarterBegin(future - 1, startingMonth=1)
        )
    elif freq == "AS":
        return obj + pd.offsets.YearBegin(1) + pd.offsets.YearBegin(future - 1)
    else:
        raise ValueError(
            f"Parameter ``freq`` must be one of {', '.join(tools_freq.FREQUENCIES)}; got {freq}."
        )
