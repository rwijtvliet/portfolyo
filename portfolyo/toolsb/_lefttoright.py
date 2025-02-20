"""Functionality to change a left-bound timestamp/index to a right-bound timestamp/index."""

import pandas as pd
from pandas.tseries.offsets import BaseOffset

from . import freq as tools_freq
from .types import Frequencylike


@tools_freq.check()
def jump(freq: BaseOffset) -> pd.Timedelta | pd.DateOffset:
    """Jump object corresponding to a frequency. Can be added to a left-bound delivery
    period timestamp to get the right-bound timestamp of that delivery period (which
    is the left-bound timestamp of the following delivery period).

    Parameters
    ----------
    freq
        Frequency of delivery period.

    Returns
    -------
        Term that can be (repeatedly) added to / subtracted from a left-bound
        timestamp to find the next / previous left-bound timestamps.

    Notes
    -----
    Only gives correct result if added to a valid left-bound stamp for the frequency.

    Examples
    --------
    >>> freq.lefttoright_jump("h")
    Timedelta('0 days 01:00:00')
    >>> freq.lefttoright_jump("MS")
    <DateOffset: months=1>
    """
    # Custom handling for specific simple frequencies
    if isinstance(freq, pd.tseries.offsets.Minute) and freq.n in (1, 5, 15, 30):
        return pd.Timedelta(minutes=freq.n)
    elif isinstance(freq, pd.tseries.offsets.Hour) and freq.n == 1:
        return pd.Timedelta(hours=1)
    elif isinstance(freq, pd.tseries.offsets.Day) and freq.n == 1:
        return pd.DateOffset(days=1)
    elif isinstance(freq, pd.tseries.offsets.MonthBegin) and freq.n == 1:
        return pd.DateOffset(months=1)
    elif isinstance(freq, pd.tseries.offsets.QuarterBegin) and freq.n == 1:
        return pd.DateOffset(months=3)
    elif isinstance(freq, pd.tseries.offsets.YearBegin) and freq.n == 1:
        return pd.DateOffset(years=1)
    else:
        raise ValueError(
            f"Parameter ``freq`` must be one of {tools_freq.ALLOWED_FREQUENCIES_DOCS}; got '{freq}'."
        )


@tools_freq.check()
def stamp(stamp: pd.Timestamp, freq: Frequencylike) -> pd.Timestamp:
    f"""Right-bound timestamp belonging to left-bound timestamp.

    Parameters
    ----------
    stamp
        Left-bound (i.e., start) timestamp of a delivery period.
    freq : {tools_freq.ALLOWED_FREQUENCIES_DOCS}
        Frequency of delivery period.

    Returns
    -------
        Corresponding right-bound timestamp.
    """
    return stamp + jump(freq)


def index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Right-bound timestamps, belonging to left-bound timestamps of delivery periods
    in index.

    Parameters
    ----------
    idx
        Index for which to calculate the right-bound timestamps.

    Returns
    -------
        Index with corresponding right-bound timestamps.
    """
    # HACK:
    # Tried and rejected:
    # . idx + pd.DateOffset(nanoseconds=i.freq.nanos)
    #   This one breaks for non-fixed frequencies, like months and quarters, e.g.
    #   idx = pd.date_range('2020', freq='MS', periods=5)
    # . idx.shift()
    #   This one breaks near DST transitions, e.g. it drops a value in this example:
    #   idx = pd.date_range('2020-03-29', freq='D', periods=5, tz='Europe/Berlin')
    # . idx + i.freq
    #   Same example, different error: time is moved to 01:00 for first timestamp.
    idx2 = idx + jump(idx.freq)
    idx2.freq = idx.freq
    return idx2
