from typing import List, Union

import pandas as pd


def indices(
    *idxs: pd.DatetimeIndex,
    ignore_freq: bool = False,
    ignore_tz: bool = False,
    ignore_start_of_day: bool = False,
) -> pd.DatetimeIndex:
    """Intersect several DatetimeIndices.

    Parameters
    ----------
    *idxs : pd.DatetimeIndex
        The indices to intersect.

    ignore_freq: bool
        Ignore the frequencies of indices, perform an intersection.
        Return the result with the frequencies of the first index from the list.

    ignore_tz: bool
        Ignore the timezones of indices, perform an intersection.
        Return the result with the timezone of the first index from the list.

    ignore_start_of_day: bool
        Ignore the start of the day of indices, perform an intersection.
        Return the result with start_of_day time of the first index from the list.
        If frequencies are shorter then daily, transforms it into daily freq.


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
    # convert tuple object into a list
    idxs = list(idxs)
    # If we land here, we have at least 2 indices.

    distinct_freqs = set([i.freq for i in idxs])
    if len(distinct_freqs) != 1 and ignore_freq is False:
        raise ValueError(f"Indices must have equal frequencies; got {distinct_freqs}.")

    distinct_tzs = set([i.tz for i in idxs])
    if len(distinct_tzs) != 1 and ignore_tz is False:
        raise ValueError(f"Indices must have equal timezones; got {distinct_tzs}.")

    freq, name, tz = (idxs[0].freq, idxs[0].name, idxs[0].tz)

    empty_idx = [len(i) == 0 for i in idxs]
    if any(empty_idx):
        return pd.DatetimeIndex([], freq=freq, tz=tz, name=name)

    # If we land here, we have at least 2 indices, all are not empty, with equal tz and freq.

    distinct_sod = set([i[0].time() for i in idxs])
    if len(distinct_sod) != 1 and ignore_start_of_day is False:
        raise ValueError(f"Indices must have equal start-of-day; got {distinct_sod}.")

    if ignore_tz is True:
        # set timezone to none for all values
        for i in range(len(idxs)):
            idxs[i] = idxs[i].tz_localize(None)

    if ignore_start_of_day is True:
        # Save a copy of the original hours and minutes
        start_of_day = idxs[0][0].time()
        for i in range(len(idxs)):
            if is_less_one_day(idxs[i].freq):
                idxs[i] = pd.date_range(
                    start=idxs[i].min(), end=idxs[i].max(), freq="D"
                )
                freq = "D"
        # Set the time components to midnight for each timestamp in the list
        idxs = [timestamp.normalize() for timestamp in idxs]

    if ignore_freq is True:
        # Find the smallest frequency
        smallest_freq = min(idxs, key=lambda x: freq_to_timestamp(x.freq)).freq
        # change bigger freq into small one
        for i in range(len(idxs)):
            start = idxs[i].min()
            end = idxs[i].max()
            if is_less_one_day(smallest_freq):
                end = end + pd.Timedelta(hours=23, minutes=59, seconds=59)
            idxs[i] = pd.date_range(start, end, freq=smallest_freq)

    # Calculation is cumbersome: pandas DatetimeIndex.intersection not working correctly on timezone-aware indices (#46702)
    values = set(idxs[0])
    for i in idxs[1:]:
        values = values.intersection(set(i))

    if ignore_start_of_day is True:
        for timestamp in values.copy():
            new_timestamp = timestamp.replace(
                hour=start_of_day.hour, minute=start_of_day.minute
            )
            values.remove(timestamp)
            values.add(new_timestamp)

        # values turn into timestampt
        # add saved start of the day to all of them
        # change back into datatimeindex
    if len(values) > 0:
        values = pd.date_range(start=min(values), end=max(values), freq=freq, tz=tz)

    return pd.DatetimeIndex(sorted(list(values)), freq=freq, name=name, tz=tz)


def is_less_one_day(dateoffset):
    """Returns True if frequency is shorter than daily
    otherwise, returns False."""
    ts = pd.Timestamp("1990-01-01")
    day = pd.tseries.offsets.DateOffset(days=1)
    return (ts + dateoffset) < (ts + day)


def freq_to_timestamp(dateoffset):
    """Transform frequency (f.e. "15T") into timestamp to enable comparison between frequencies"""
    ts = pd.Timestamp("1990-01-01")
    return ts + dateoffset


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
