"""Module to work with timestamps."""

import datetime as dt
from typing import Literal

import pandas as pd
from pandas.core.dtypes.dtypes import BaseOffset

from . import freq as tools_freq
from . import startofday as tools_startofday
from . import unit as tools_unit
from pint import Quantity


@tools_freq.check()
def to_right(stamp: pd.Timestamp, freq: BaseOffset) -> pd.Timestamp:
    """Right-bound timestamp belonging to left-bound timestamp.

    Parameters
    ----------
    stamp
        Left-bound (i.e., start) timestamp of a delivery period.
    freq
        Frequency of delivery period.

    Returns
    -------
        Corresponding right-bound (i.e., end) timestamp.

    Notes
    -----
    Does not verify that ``stamp`` is indeed a left-bound timestamp for the previded frequency.
    """
    return stamp + tools_freq.to_jump(freq)


@tools_freq.check()
def duration(stamp: pd.Timestamp, freq: pd.DateOffset) -> Quantity:
    """Duration of a delivery period.

    Parameters
    ----------
    stamp
        Left-bound (i.e., start) timestamp of delivery period for which to calculate the duration.
    freq
        Frequency of delivery period.

    Returns
    -------
        Duration.

    Notes
    -----
    Does not verify that ``stamp`` is indeed a left-bound timestamp for the previded frequency.

    Example
    -------
    >>> duration(pd.Timestamp('2020-04-21'), 'D')
    24.0 h
    >>> duration(pd.Timestamp('2020-03-29'), 'D')
    24.0 h
    >>> duration(pd.Timestamp('2020-03-29', tz='Europe/Berlin'), 'D')
    23.0 h
    """
    jump = tools_freq.to_jump(freq)

    if isinstance(jump, pd.Timedelta):
        tdelta = jump  # one timedelta
    else:
        assert freq.is_on_offset(stamp)
        tdelta = (stamp + jump) - stamp  # one timedelta
    hours = tdelta.total_seconds() / 3600
    return tools_unit.Q_(hours, "h")


def replace_time(stamp: pd.Timestamp, startofday: dt.time) -> pd.Timestamp:
    """Replace the time-part of ``stamp`` with ``startofday``."""
    return stamp.replace(
        hour=startofday.hour, minute=startofday.minute, second=startofday.second
    )


@tools_freq.check()
@tools_startofday.check()
def is_boundary(
    stamp: pd.Timestamp, freq: BaseOffset, startofday: dt.time | None = None
) -> bool:
    """Check if timestamp is a valid delivery period start.

    Parameters
    ----------
    stamp
        Timestamp to check.
    freq
        Frequency of delivery period.
    startofday, optional (default: midnight)
        Time of day at which daily-or-longer delivery periods start. E.g. if
        dt.time(hour=6), a delivery day is from 06:00:00 (incl) until 06:00:00 (excl).

    Returns
    -------
        True if stamp is at start of a delivery period described by ``freq`` and
        ``startofday``.
    """
    if tools_freq.is_shorter_than_daily(freq):
        return stamp.floor(freq) == stamp
    else:
        return stamp.time() == startofday and freq.is_on_offset(stamp)


def _round(
    stamp: pd.Timestamp,
    freq: BaseOffset,
    startofday: dt.time,
    fn: Literal["floor", "ceil"],
) -> pd.Timestamp:
    """Floor (or ceil) a timestamp to start (or end) of delivery period it's contained in."""
    # Fixed-duration frequency: simply floor/ceil.
    if tools_freq.is_shorter_than_daily(freq):
        return stamp.floor(freq) if fn == "floor" else stamp.ceil(freq)

    # If we reach here, we have daily-or-longer frequency.

    # Correct for the time-of-day.
    if stamp.time() == startofday:
        rounded = stamp
    else:
        is_part_of_prevday = stamp.time() < startofday
        rounded = replace_time(stamp, startofday)
        if is_part_of_prevday and fn == "floor":
            rounded -= tools_freq.to_jump("D")
        elif not is_part_of_prevday and fn == "ceil":
            rounded += tools_freq.to_jump("D")

    # Correct for the day-of-X.
    return freq.rollback(rounded) if fn == "floor" else freq.rollforward(rounded)


@tools_freq.check()
@tools_startofday.check()
def floor(
    stamp: pd.Timestamp,
    freq: BaseOffset,
    startofday: dt.time = tools_startofday.MIDNIGHT,
) -> pd.Timestamp:
    """Floor timestamp to beginning of delivery period it's contained in.
    I.e., find (latest) delivery period start that is on or before the timestamp.

    Parameters
    ----------
    stamp
        Timestamp to floor.
    freq
        Frequency of delivery period.
    startofday, optional (default: midnight)
        Time of day at which daily-or-longer delivery periods start. E.g. if
        dt.time(hour=6), a delivery day is from 06:00:00 (incl) until 06:00:00 (excl).

    Returns
    -------
        The floored timestamp.

    Notes
    -----
    If ``stamp`` is exactly at the start of the period, ceiling and flooring both return
    the original timestamp.

    Examples
    --------
    >>> floor(pd.Timestamp('2020-04-21 15:42'), 'YS')
    Timestamp('2020-01-01 00:00:00')
    >>> floor(pd.Timestamp('2020-04-21 15:42'), 'YS-FEB')
    Timestamp('2020-02-01 00:00:00')
    >>> floor(pd.Timestamp('2020-04-21 15:42'), 'MS')
    Timestamp('2020-04-01 00:00:00')
    >>> floor(pd.Timestamp('2020-04-21 15:42'), '15min')
    Timestamp('2020-04-21 15:30:00')
    >>> floor(pd.Timestamp('2020-04-21 15:42', tz='Europe/Berlin'), 'MS')
    Timestamp('2020-04-01 00:00:00+0200', tz='Europe/Berlin')
    >>> floor(pd.Timestamp('2020-04-21 15:42'), 'MS', dt.time(hour=6))
    Timestamp('2020-04-01 06:00:00')
    """
    return _round(stamp, freq, startofday, "floor")


@tools_freq.check()
@tools_startofday.check()
def ceil(
    stamp: pd.Timestamp,
    freq: BaseOffset,
    startofday: dt.time = tools_startofday.MIDNIGHT,
) -> pd.Timestamp:
    """Ceil timestamp to end of delivery period it's contained in.
    I.e., find (earliest) delivery period start that is on or after the timestamp.

    Parameters
    ----------
    stamp
        Timestamp to ceil.
    freq
        Frequency of delivery period.
    startofday, optional (default: midnight)
        Time of day at which daily-or-longer delivery periods start. E.g. if
        dt.time(hour=6), a delivery day is from 06:00:00 (incl) until 06:00:00 (excl).

    Returns
    -------
        The ceiled timestamp.

    Notes
    -----
    If ``stamp`` is exactly at the start of the period, ceiling and flooring both return
    the original timestamp.

    Examples
    --------
    >>> ceil(pd.Timestamp('2020-04-21 15:42'), 'YS')
    Timestamp('2021-01-01 00:00:00')
    >>> ceil(pd.Timestamp('2020-04-21 15:42'), 'YS-FEB')
    Timestamp('2021-02-01 00:00:00')
    >>> ceil(pd.Timestamp('2020-04-21 15:42'), 'MS')
    Timestamp('2020-05-01 00:00:00')
    >>> ceil(pd.Timestamp('2020-04-21 15:42'), '15min')
    Timestamp('2020-04-21 15:45:00')
    >>> ceil(pd.Timestamp('2020-04-21 15:42', tz='Europe/Berlin'), 'MS')
    Timestamp('2020-04-01 00:00:00+0200', tz='Europe/Berlin')
    >>> ceil(pd.Timestamp('2020-04-21 15:42'), 'MS', dt.time(hour=6))
    Timestamp('2020-05-01 06:00:00')
    """
    return _round(stamp, freq, startofday, "ceil")
