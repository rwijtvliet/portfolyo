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


def index(i: pd.DatetimeIndex, freq: str = None) -> pd.Series:
    f"""Right-bound timestamp belonging to left-bound timestamp.

    Parameters
    ----------
    i : pd.DatetimeIndex
        Index for which to calculate the right-bound timestamps.
    freq : {{{', '.join(tools_freq.FREQUENCIES)}}}, optional
        Frequency to use in determining the right-bound timestamp.
        If none specified, use ``.freq`` attribute of ``i``.

    Returns
    -------
    pd.Series
        With ``i`` as its index, and the corresponding right-bound timestamps as the values.
    """
    freq = freq or i.freq
    # i.shift gives error in edge cases, e.g. if
    # i = pd.date_range('2020-03-29', freq='D', periods=5, tz='Europe/Berlin'))
    return pd.Series(i + tools_freq.to_offset(freq), i, name="ts_right")
