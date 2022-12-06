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
    The indices must have equal frequency, timezone, start-of-day. Otherwise, an error
    is raised. If there is no overlap, an empty Index is returned.
    """
    if len(indices) == 0:
        raise ValueError("Must specify at least one index.")

    if len(indices) == 1:
        return indices[0]

    # If we land here, we have at least 2 indices.

    distinct_freqs = set([i.freq for i in indices])
    if len(distinct_freqs) != 1:
        raise ValueError(f"Indices must have equal frequencies; got {distinct_freqs}.")

    distinct_tzs = set([i.tz for i in indices])
    if len(distinct_tzs) != 1:
        raise ValueError(f"Indices must have equal timezones; got {distinct_tzs}.")

    freq, name, tz = indices[0].freq, indices[0].name, indices[0].tz

    empty_idx = [len(i) == 0 for i in indices]
    if any(empty_idx):
        return pd.DatetimeIndex([], freq=freq, tz=tz, name=name)

    # If we land here, we have at least 2 indices, all are not empty, with equal tz and freq.

    distinct_sod = set([i[0].time() for i in indices])
    if len(distinct_sod) != 1:
        raise ValueError(f"Indices must have equal start-of-day; got {distinct_sod}.")

    # Calculation is cumbersome: pandas DatetimeIndex.intersection not working correctly on timezone-aware indices.
    values = set(indices[0])
    for i in indices[1:]:
        values = values.intersection(set(i))

    return pd.DatetimeIndex(sorted(list(values)), freq=freq, name=name, tz=tz)
