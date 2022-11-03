"""
Calculate the left-bound timestamp of delivery period, given the right-bound one.
"""

import pandas as pd

from . import freq as tools_freq


def index(i: pd.DatetimeIndex, how: str = "A") -> pd.DatetimeIndex:
    """Turn an index with right-bound timestamps into one with left-bound timestamps.

    Parameters
    ----------
    i : pd.DatetimeIndex
        The index that needs its timestamps changed.
    how : {'A', 'B'}, optional (default: 'A')
        If ``i`` is not localized, and contains a DST-transition, there are two ways
        in which it may be right-bound. E.g. at start of DST: (A) contains 2:00 but not
        3:00 (like left-bound timestamps do); or (B) contains 3:00 but not 2:00.
        Ignored for timezone-aware ``i`` or if ``i.freq`` can be inferred.

    Returns
    -------
    pd.DatetimeIndex
        With left-bound timestamps.

    Notes
    -----
    If frequency is not set, guess from the spacing of the timestamps. Assumes values
    are in order. Does no error checking to see if index actually makes sense. Does not
    assess if 'A' or 'B' makes most sense for tz-naive ``i``.
    """
    # Must be able to handle cases where .freq is not set. A tz-naive index that contains
    # a DST-changeover will have missing or repeated timestamps.

    # If frequency is known, we can use pandas built-in to make leftbound.
    if (freq := i.freq) is not None or (freq := pd.infer_freq(i)) is not None:
        return i - tools_freq.to_offset(freq)

    # Couldn't infer frequency. Try from median timedelta and turn into time offset.
    offst = tools_freq.to_offset(tools_freq.from_tdelta((i[1:] - i[:-1]).median()))
    if i.tz or how == "A":  # if tz-aware, only one way to make leftbound.
        return i - offst
    else:  # how == "B"
        return pd.DatetimeIndex([i[0] - offst, *i[:-1]])
