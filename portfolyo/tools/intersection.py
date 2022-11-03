import pandas as pd


def index(*indices: pd.DatetimeIndex) -> pd.DatetimeIndex:
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
    for i in indices[1:]:
        values = values.intersection(set(i))

    return pd.DatetimeIndex(sorted(list(values)), freq=freq, name=name, tz=tz)
