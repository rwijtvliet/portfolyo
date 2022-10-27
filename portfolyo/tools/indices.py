"""
Module for doing basic operations on DatetimeIndex instances.
"""

import pandas as pd
from . import stamps


def right_to_left(i: pd.DatetimeIndex, how: str = "A") -> pd.DatetimeIndex:
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
    if i.freq is not None:
        return i.shift(-1)

    # Same goes if frequency can be inferred.
    if (freq := pd.infer_freq(i)) is not None:
        i = i.copy()
        i.freq = freq  # can't just supply as parameter to shift method
        return i.shift(-1)

    # Couldn't infer frequency. Try from median timedelta and turn into time offset.
    offst = stamps.offset(stamps.guess_frequency((i[1:] - i[:-1]).median()))
    if i.tz or how == "A":  # if tz-aware, only one way to make leftbound.
        return i - offst
    else:  # how == "B"
        return pd.DatetimeIndex([i[0] - offst, *i[:-1]])


def intersection(*indices: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Intersect several DatetimeIndices.

    Parameters
    ----------
    *indices : pd.DatetimeIndex
        The indices to intersect.

    Returns
    -------
    pd.DatetimeIndex
        The intersection, i.e., datetimeindex with values that exist in each index.

    Notes
    -----
    If indices have distinct timezones or names, the values from the first index are used.
    """
    if len(indices) == 0:
        raise ValueError("Must specify at least one index.")

    distinct_freqs = set([i.freq for i in indices])
    if len(indices) > 1 and len(distinct_freqs) != 1:
        raise ValueError(
            f"Indices must not have equal frequencies; got {distinct_freqs}."
        )

    tznaive = [i.tz is None for i in indices]
    if any(tznaive) and not all(tznaive):
        raise ValueError(
            "All indices must be either timezone-aware or timezone-naive; got "
            f"{sum(tznaive)} naive (out of {len(indices)})."
        )

    freq, name, tz = indices[0].freq, indices[0].name, indices[0].tz

    # Calculation is cumbersome: pandas DatetimeIndex.intersection not working correctly on timezone-aware indices.
    values = set(indices[0])
    for idx in indices[1:]:
        values = values.intersection(set(idx))
    idx = pd.DatetimeIndex(sorted(list(values)), freq=freq, name=name, tz=tz)

    return idx
