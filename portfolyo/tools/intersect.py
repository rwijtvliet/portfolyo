from typing import List, Union, Tuple

import pandas as pd

from portfolyo.tools.freq import longest
from datetime import datetime


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

    # Calculation is cumbersome: pandas DatetimeIndex.intersection not working correctly on timezone-aware indices (#46702)
    values = set(idxs[0])
    for i in idxs[1:]:
        values = values.intersection(set(i))

    return pd.DatetimeIndex(sorted(list(values)), freq=freq, name=name, tz=tz)


def indices_flex(
    *idxs: pd.DatetimeIndex,
    ignore_freq: bool = False,
    ignore_tz: bool = False,
    ignore_start_of_day: bool = False,
) -> Tuple[pd.DatetimeIndex]:
    """Intersect several DatetimeIndices, but allow for more flexibility of ignoring
    certain properties.

    Parameters
    ----------
    *idxs : pd.DatetimeIndex
        The indices to intersect.
    ignore_freq: bool, optional (default: False)
        If True, do the intersection even if the frequencies do not match; drop the
        time periods that do not (fully) exist in either of the frames.
    ignore_tz: bool, optional (default: False)
        If True, ignore the timezones; perform the intersection using 'wall time'.
    ignore_start_of_day: bool, optional (default: False)
        If True, perform the intersection even if the frames have a different start-of-day.
        The start-of-day of the original frames is preserved, even if the frequency is shorter
        than daily.

    Returns
    -------
    Tuple[pd.DatetimeIndex]
        The intersection for each datetimeindex (in same order as input idxs).

    See also
    --------
    indices
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

    freq, name, tz = [], [], []
    for i in range(len(idxs)):
        freq.append(idxs[i].freq)
        name.append(idxs[i].name)
        tz.append(idxs[i].tz)

    empty_idx = [len(i) == 0 for i in idxs]
    if any(empty_idx):
        return pd.DatetimeIndex([])

    # If we land here, we have at least 2 indices, all are not empty, with equal tz and freq.

    distinct_sod = set([i[0].time() for i in idxs])
    if len(distinct_sod) != 1 and ignore_start_of_day is False:
        raise ValueError(f"Indices must have equal start-of-day; got {distinct_sod}.")

    # add one interval of the respective freq to each index (this way, a given date-range from A-B that was exclusive
    # of B is now inclusive of B - this helps when we need to convert frequencies or times-of-day without loosing
    # data. At the end, we exclude the end-date of the final result again.)
    idxs = [
        idx.append(
            pd.DatetimeIndex([idx[-1] + pd.tseries.frequencies.to_offset(idx.freq)])
        )
        for idx in idxs
    ]

    if ignore_freq is True:
        # Find the smallest frequency
        biggest_freq = longest(*freq)
        # change bigger freq into small one
        for i in range(len(idxs)):
            start = idxs[i].min()
            end = idxs[i].max()
            idxs[i] = pd.date_range(start, end, freq=biggest_freq, inclusive="both")

    if ignore_tz is True:
        # set timezone to none for all values
        for i in range(len(idxs)):
            idxs[i] = idxs[i].tz_localize(None)

    if ignore_start_of_day is True:
        # Save a copy of the original hours and minutes
        start_of_day = [x[0].time() for x in idxs]
        # Set the time components to midnight for each timestamp in the list
        idxs = [timestamp.normalize() for timestamp in idxs]

    # Calculation is cumbersome: pandas DatetimeIndex.intersection not working correctly on timezone-aware indices (#46702)
    values = set(idxs[0])
    for i in idxs[1:]:
        values = values.intersection(set(i))

    if len(values) == 0:
        return tuple([pd.DatetimeIndex([]) for _i in idxs])

    idxs_out = []
    for i in range(len(idxs)):
        start = min(values)
        end = max(values)
        inclusive = "left"

        if ignore_start_of_day is True:
            start = datetime.combine(pd.to_datetime(start).date(), start_of_day[i])
            end = datetime.combine(pd.to_datetime(end).date(), start_of_day[i])
            # inclusive = "left"

        idxs_out.append(
            pd.date_range(
                start=start,
                end=end,
                freq=freq[i],
                name=name[i],
                tz=tz[i],
                inclusive=inclusive,
            )
        )

    return tuple(idxs_out)


def frames(
    *frames: Union[pd.Series, pd.DataFrame],
    ignore_freq: bool = False,
    ignore_tz: bool = False,
    ignore_start_of_day: bool = False,
) -> List[Union[pd.Series, pd.DataFrame]]:
    """Intersect several dataframes and/or series.

    Parameters
    ----------
    *frames : pd.Series and/or pd.DataFrame
        The frames to intersect.
    ignore_freq: bool, optional (default: False)
        If True, do the intersection even if the frequencies do not match; drop the
        time periods that do not (fully) exist in either of the frames.
    ignore_tz: bool, optional (default: False)
        If True, ignore the timezones; perform the intersection using 'wall time'.
    ignore_start_of_day: bool, optional (default: False)
        If True, perform the intersection even if the frames have a different start-of-day.
        The start-of-day of the original frames is preserved, even if the frequency is shorter
        than daily.

    Returns
    -------
    list of series and/or dataframes
        As input, but trimmed to their intersection.

    Notes
    -----
    The indices must have equal frequency, timezone, start-of-day. Otherwise, an error
    is raised. If there is no overlap, empty frames are returned.
    """
    new_idx = indices_flex(
        *[fr.index for fr in frames],
        ignore_freq=ignore_freq,
        ignore_tz=ignore_tz,
        ignore_start_of_day=ignore_start_of_day,
    )
    return [fr.loc[idx] for idx, fr in zip(new_idx, frames)]
