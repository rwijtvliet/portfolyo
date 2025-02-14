"""Module to work with indices."""

import pandas as pd
from . import freq2 as tools_freq


def assert_index_valid(i: pd.DatetimeIndex) -> None:
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
    start_of_day = i[0].time()
    if not (start_of_day.minute == start_of_day.second == 0):
        raise AssertionError(
            "Days must start on a full hour (which is not necessarily midnight)."
        )

    # Check on number of days.
    if tools_freq.is_shorter_than_daily(i.freq):
        end_time = (i[-1] + tools_freq.to_jump(i.freq)).time()
        if end_time != start_of_day:
            raise AssertionError(
                "Index must contain an integer number of days, i.e., the end time of the final delivery period "
                f"(here: {end_time}) must equal the start time of the first delivery period (here: {start_of_day})."
            )


def to_right(i: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Right-bound timestamps, belonging to left-bound timestamps
    of delivery periods in index..

    Parameters
    ----------
    i
        Index for which to calculate the right-bound timestamps.

    Returns
    -------
       Corresponding right-bound timestamps.
    """
    # Get right timestamp for each index value, based on the frequency.
    # Tried and rejected, with example where outcome is not correct:
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
