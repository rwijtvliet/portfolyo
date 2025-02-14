"""Module to work with indices."""

import pandas as pd
from . import freq as tools_freq
from . import startofday as tools_startofday
from . import stamp as tools_stamp
import datetime as dt


def assert_valid(i: pd.DatetimeIndex) -> None:
    f"""Validate if the given index has the necessary properties to be used in portfolio lines.

    Parameters
    ----------
    i
        Index to be checked. Frequency must be valid (one of {tools_freq.DOCS}), and
        for shorter-than-daily frequencies, the index must contain entire days and start at a full
        hour.

    Raises
    ------
    AssertionError
        If the index is not valid.
    """
    # Check on frequency.
    tools_freq.assert_freq_valid(i.freq)

    # Check on start_of_day.
    startofday = get_startofday(i)
    tools_startofday.assert_valid(startofday)

    # Check on number of days.
    if tools_freq.is_shorter_than_daily(i.freq):
        end_time = tools_stamp.to_right(i[-1], i.freq).time()
        if end_time != startofday:
            raise AssertionError(
                "Index must contain an integer number of days, i.e., the end time of the final delivery period "
                f"(here: {end_time}) must equal the start time of the first delivery period (here: {startofday})."
            )


def to_right(i: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Right-bound timestamps, belonging to left-bound timestamps of delivery periods
    in index.

    Parameters
    ----------
    i
        Index for which to calculate the right-bound timestamps.

    Returns
    -------
        Index with corresponding right-bound timestamps.
    """
    # HACK:
    # Tried and rejected:
    # . i + pd.DateOffset(nanoseconds=i.freq.nanos)
    #   This one breaks for non-fixed frequencies, like months and quarters, e.g.
    #   i = pd.date_range('2020', freq='MS', periods=5)
    # . i.shift()
    #   This one breaks near DST transitions, e.g. it drops a value in this example:
    #   i = pd.date_range('2020-03-29', freq='D', periods=5, tz='Europe/Berlin')
    # . i + i.freq
    #   Same example, different error: time is moved to 01:00 for first timestamp.
    i2 = (i + tools_freq.to_jump(i.freq)).rename("right")
    i2.freq = i.freq
    return i2


def get_startofday(i: pd.DatetimeIndex) -> dt.time:
    """Get start-of-day of an index.

    Parameters
    ----------
    i
        The index.

    Returns
    -------
        The time component of the first timestamp.
    """
    # TODO: is this function really necessary?
    return i[0].time()


def replace_startofday(i: pd.DatetimeIndex, startofday: dt.time) -> pd.DatetimeIndex:
    """For indices with a daily-or-longer frequency, replace the time-part of each
    timestamp, so that the returned index has the specified start-of-day.

    Parameters
    ----------
    i
        The index.
    startofday
        The desired start-of-day.

    Returns
    -------
        Index with wanted start-of-day.

    Notes
    -----
    This process changes the timestamps and must only be used as a correction to data
    that was recorded with the incorrect timestamps.
    """
    tools_startofday.assert_valid(startofday)

    if tools_freq.is_shorter_than_daily(i.freq):
        raise ValueError(
            "This function works on indices with a daily-or-longer frequency. To trim an index with a"
            " shorter-than-daily frequency to a given start-of-day, use the `trim_to_startofday` method."
        )

    # Daily or longer: change timepart of each. Minute and second must be zero, so specify directly.
    stamps = (stamp.replace(hour=startofday.hour, minute=0, second=0) for stamp in i)
    return pd.DatetimeIndex(stamps, freq=i.freq, tz=i.tz)


def trim_to_startofday(i: pd.DatetimeIndex, startofday: dt.time) -> pd.DatetimeIndex:
    """For indices with a shorter-than-daily frequency, drop timestamps from the index
    so that the returned index has the specified start-of-day.

    Parameters
    ----------
    i
        The index.
    startofday
        The desired start-of-day.

    Returns
    -------
        Index with wanted start-of-day.

    Notes
    -----
    This process is lossy; timestamps are dropped.
    """
    tools_startofday.assert_valid(startofday)

    if not tools_freq.is_shorter_than_daily(i.freq):
        raise ValueError(
            "This function works on indices with a shorter-than-daily frequency. To replace the time-part"
            " of an index with a daily-or-longer frequency with a given start-of-day, use the `replace_startofday` method."
        )

    # Find start. Check at most one entire day (25h @ end of DST).
    to_check = i - i[0] < dt.timedelta(hours=25)
    for pos0, stamp in enumerate(i[to_check]):
        if stamp.time() == startofday:
            break
    else:
        raise ValueError("Did not find any timestamp with correct time at index start.")

    # Find end. Check at most one entire day (25h @ end of DST).
    to_check = i[-1] - i < dt.timedelta(hours=25)
    for pos1, stamp in enumerate(reversed(i[to_check])):
        if tools_stamp.to_right(stamp, i.freq).time() == startofday:
            break
    else:
        raise ValueError("Did not find any timestamp with correct time at index end.")

    return i[pos0:-pos1]
