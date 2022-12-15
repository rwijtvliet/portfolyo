from typing import List, Union

import pandas as pd


def indices(*idxs: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Intersect several DatetimeIndices.

    Parameters
    ----------
    *idxs : pd.DatetimeIndex
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
    if len(idxs) == 0:
        raise ValueError("Must specify at least one index.")

    if len(idxs) == 1:
        return idxs[0]

    # If we land here, we have at least 2 indices.

    distinct_freqs = set([i.freq for i in idxs])
    if len(distinct_freqs) != 1:
        raise ValueError(f"Indices must have equal frequencies; got {distinct_freqs}.")

    distinct_tzs = set([i.tz for i in idxs])
    if len(distinct_tzs) != 1:
        raise ValueError(f"Indices must have equal timezones; got {distinct_tzs}.")

    freq, name, tz = idxs[0].freq, idxs[0].name, idxs[0].tz

    empty_idx = [len(i) == 0 for i in idxs]
    if any(empty_idx):
        return pd.DatetimeIndex([], freq=freq, tz=tz, name=name)

    # If we land here, we have at least 2 indices, all are not empty, with equal tz and freq.

    distinct_sod = set([i[0].time() for i in idxs])
    if len(distinct_sod) != 1:
        raise ValueError(f"Indices must have equal start-of-day; got {distinct_sod}.")

    # Calculation is cumbersome: pandas DatetimeIndex.intersection not working correctly on timezone-aware indices.
    values = set(idxs[0])
    for i in idxs[1:]:
        values = values.intersection(set(i))

    return pd.DatetimeIndex(sorted(list(values)), freq=freq, name=name, tz=tz)


def frames(
    *frames: Union[pd.Series, pd.DataFrame]
) -> List[Union[pd.Series, pd.DataFrame]]:
    """Intersect several dataframes and/or series.

    Parameters
    ----------
    *frames : pd.Series and/or pd.DataFrame
        The frames to intersect.

    Returns
    -------
    list of series and/or dataframes
        As input, but trimmed to their intersection.

    Notes
    -----
    The indices must have equal frequency, timezone, start-of-day. Otherwise, an error
    is raised. If there is no overlap, empty frames are returned.
    """
    common_index = indices(*[fr.index for fr in frames])
    return [fr.loc[common_index] for fr in frames]
