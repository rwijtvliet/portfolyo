"""
Tools for dealing with frequencies.
"""

from typing import Union

import numpy as np
import pandas as pd

# Allowed frequencies.
# Perfect containment; a short-frequency time period always entirely falls within a single high-frequency time period.
# AS -> 4 QS; QS -> 3 MS; MS -> 28-31 D; D -> 23-25 H; H -> 4 15T
FREQUENCIES = ["AS", "QS", "MS", "D", "H", "15T"]


def up_or_down(
    freq_source: str, freq_target: str, common_ts: pd.Timestamp = None
) -> int:
    """
    Compare source frequency with target frequency to see if it needs up- or downsampling.

    Upsampling means that the number of values increases - one value in the source
    corresponds to multiple values in the target.

    Parameters
    ----------
    freq_source, freq_target : frequencies to compare.
    common_ts : timestamp, optional
        Timestamp to use as anchor from which to compare the two.

    Returns
    -------
    1 (-1, 0) if source frequency must be upsampled (downsampled, no change) to obtain
        target frequency.

    Notes
    -----
    Arbitrarily using a time point as anchor to calculate the length of the time period
    from. May have influence on the ratio (duration of a month, quarter, year etc are
    influenced by this), but, for most common frequencies, not on which is larger.

    Examples
    --------
    >>> freq.up_or_down('D', 'MS')
    -1
    >>> freq.up_or_down('MS', 'D')
    1
    >>> freq.up_or_down('MS', 'MS')
    0
    """
    standard_common_ts = pd.Timestamp("2020-01-01 0:00")
    backup_common_ts = pd.Timestamp("2020-02-03 04:05:06")
    if common_ts is None:
        common_ts = standard_common_ts
    ts1 = common_ts + pd.tseries.frequencies.to_offset(freq_source)
    ts2 = common_ts + pd.tseries.frequencies.to_offset(freq_target)
    if ts1 > ts2:
        return 1
    elif ts1 < ts2:
        return -1
    if common_ts == standard_common_ts:
        # If they are the same, try with another timestamp.
        return up_or_down(freq_source, freq_target, backup_common_ts)
    return 0  # only if both give the same answer.


def _longestshortest(shortest: bool, *freqs: str):
    """Determine which frequency denotes the shortest or longest time period."""
    common_ts = pd.Timestamp("2020-01-01")
    ts = [common_ts + pd.tseries.frequencies.to_offset(fr) for fr in freqs]
    i = (np.argmin if shortest else np.argmax)(ts)
    return freqs[i]


def shortest(*freqs: str) -> str:
    """Find shortest of several frequencies.

    Parameters
    ----------
    *freqs : str
        Frequencies to compare.

    Returns
    -------
    The shortest of the provided frequencies.

    Examples
    --------
    >>> freq.shortest('MS', 'H', 'AS', 'D')
    'H'
    """
    return _longestshortest(True, *freqs)


def longest(*freqs: str) -> str:
    """Find longest of several frequencies.

    Parameters
    ----------
    *freqs : str
        Frequencies to compare.

    Returns
    -------
    The longest of the provided frequencies.

    Examples
    --------
    >>> freq.longest('MS', 'H', 'AS', 'D')
    'AS'
    """
    return _longestshortest(False, *freqs)


def to_offset(freq: str) -> Union[pd.Timedelta, pd.DateOffset]:
    """Object that can be added to a left-bound timestamp to find corresponding right-bound timestamp.

    Parameters
    ----------
    freq : str
        Frequency denoting the length of the time period.

    Returns
    -------
    pd.Timedelta | pd.DateOffset

    Examples
    --------
    >>> freq.to_offset("H")
    Timedelta('0 days 01:00:00')
    >>> freq.to_offset("MS")
    <DateOffset: months=1>
    """
    # Get right timestamp for each index value, based on the frequency.
    # . This one breaks for 'MS':
    # (i + pd.DateOffset(nanoseconds=i.freq.nanos))
    # . This drops a value at some DST transitions:
    # (i.shift(1))
    # . This one gives wrong value at DST transitions:
    # i + i.freq

    if freq == "15T":
        return pd.Timedelta(hours=0.25)
    elif freq == "H":
        return pd.Timedelta(hours=1)
    elif freq == "D":
        return pd.DateOffset(days=1)
    elif freq == "MS":
        return pd.DateOffset(months=1)
    elif freq == "QS":
        return pd.DateOffset(months=3)
    elif freq == "AS":
        return pd.DateOffset(years=1)
    else:
        for freq2 in ["MS", "QS"]:  # Edge case: month-/quarterly but starting != Jan.
            if up_or_down(freq2, freq) == 0:
                return to_offset(freq2)
        raise ValueError(
            f"Parameter ``freq`` must be one of {', '.join(FREQUENCIES)}; got '{freq}'."
        )


def from_tdelta(tdelta: pd.Timedelta) -> str:
    f"""Guess the frequency from a time delta.

    Parameters
    ----------
    tdelta : pd.Timedelta
        Time delta between start and end of delivery period.

    Returns
    -------
    str
        One of {', '.join(FREQUENCIES)}.
    """
    if tdelta == pd.Timedelta(minutes=15):
        return "15T"
    elif tdelta == pd.Timedelta(hours=1):
        return "H"
    elif pd.Timedelta(hours=23) <= tdelta <= pd.Timedelta(hours=25):
        return "D"
    elif pd.Timedelta(days=27) <= tdelta <= pd.Timedelta(days=32):
        return "MS"
    elif pd.Timedelta(days=89) <= tdelta <= pd.Timedelta(days=93):
        return "QS"
    elif pd.Timedelta(days=364) <= tdelta <= pd.Timedelta(days=367):
        return "AS"
    else:
        raise ValueError(
            f"The timedelta ({tdelta}) doesn't seem to be fit to any of the allowed "
            f"frequencies ({', '.join(FREQUENCIES)})."
        )
