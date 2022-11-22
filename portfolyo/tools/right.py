"""
Calculate the right-bound timestamp of a time period.
"""

import pandas as pd

from . import freq as tools_freq


def stamp(ts: pd.Timestamp, freq: str = None) -> pd.Timestamp:
    f"""Right-bound timestamp belonging to left-bound timestamp.

    Parameters
    ----------
    ts : pd.Timestamp
        Timestamp for which to calculate the right-bound timestamp.
    freq : {{{', '.join(tools_freq.FREQUENCIES)}}}
        Frequency to use in determining the right-bound timestamp.

    Returns
    -------
    pd.Timestamp
        Corresponding right-bound timestamp.
    """
    return ts + tools_freq.to_offset(freq)


def index(i: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Right-bound timestamp belonging to left-bound timestamp.

    Parameters
    ----------
    i : pd.DatetimeIndex
        Index for which to calculate the right-bound timestamps.

    Returns
    -------
    pd.DatetimeIndex
        With the corresponding right-bound timestamps.
    """
    # Get right timestamp for each index value, based on the frequency.
    # . This one breaks for 'MS':
    # i + pd.DateOffset(nanoseconds=i.freq.nanos)
    # . This drops a value at some DST transitions:
    # i.shift(1)
    # (e.g. when i = pd.date_range('2020-03-29', freq='D', periods=5, tz='Europe/Berlin'))
    # . This one gives wrong value at DST transitions:
    # i + i.freq
    i2 = (i + tools_freq.to_offset(i.freq)).rename("right")
    i2.freq = i.freq
    return i2
