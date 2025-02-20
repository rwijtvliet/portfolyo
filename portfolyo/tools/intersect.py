from datetime import datetime
from typing import List, Tuple

import pandas as pd

from . import freq as tools_freq
from . import right as tools_right
from . import trim as tools_trim
from .types import Series_or_DataFrame


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

    empty_idx = [len(i) == 0 for i in idxs]
    if any(empty_idx):
        return pd.DatetimeIndex([])

    # If we land here, we have at least 2 indices, all are not empty.

    distinct_sod = set([i[0].time() for i in idxs])
    if len(distinct_sod) != 1 and ignore_start_of_day is False:
        raise ValueError(f"Indices must have equal start-of-day; got {distinct_sod}.")
    for i in range(len(idxs)):
        if len(distinct_sod) != 1 and tools_freq.up_or_down(idxs[i].freq, "D") == -1:
            raise ValueError(
                "Downsample all indices to daily-or-longer, or trim them so they have the same start-of-day, before attempting to calculate the intersection"
            )

    freq, name, tz = [], [], []
    for i in range(len(idxs)):
        freq.append(idxs[i].freq)
        name.append(idxs[i].name)
        tz.append(idxs[i].tz)

    longest_freq = freq[0]
    if ignore_freq is True and len(distinct_freqs) != 1:
        # Find the longest frequency
        longest_freq = tools_freq.longest(*freq)
        # trim datetimeindex
        for i in range(len(idxs)):
            # if idxs[i].freq is not the same as longest freq, we trim idxs[i]
            if idxs[i].freq != longest_freq:
                idxs[i] = tools_trim.index(idxs[i], longest_freq)

    if ignore_tz is True and len(distinct_tzs) != 1:
        # set timezone to none for all values
        for i in range(len(idxs)):
            idxs[i] = idxs[i].tz_localize(None)

    if ignore_start_of_day is True and len(distinct_sod) != 1:
        # Save a copy of the original hours and minutes
        start_of_day = [x[0].time() for x in idxs]
        # Set the time components to midnight for each index in the list
        idxs = [index.normalize() for index in idxs]

    # Calculation is cumbersome: pandas DatetimeIndex.intersection not working correctly on timezone-aware indices (#46702)
    values = set(idxs[0])
    # intersection is not working on datetimeindex with different freq->we need to use mask
    for i in idxs[1:]:
        values = values.intersection(set(i))
    values = sorted(values)

    if len(values) == 0:
        return tuple([pd.DatetimeIndex([]) for _ in idxs])

    idxs_out = []
    for i in range(len(idxs)):
        start = min(values)
        # end = stamp(start, longest_freq._prefix)
        end = max(values)
        end = tools_right.stamp(end, longest_freq)

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
                inclusive="left",
            )
        )

    return tuple(idxs_out)


def frames(
    *frames: Series_or_DataFrame,
    ignore_freq: bool = False,
    ignore_tz: bool = False,
    ignore_start_of_day: bool = False,
) -> List[Series_or_DataFrame]:
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
    As input, but trimmed to their intersection.

    Notes
    -----
    The indices must have equal frequency, timezone, start-of-day. Otherwise, an error
    is raised. If there is no overlap, empty frames are returned.
    """
    new_idxs = indices_flex(
        *[fr.index for fr in frames],
        ignore_freq=ignore_freq,
        ignore_tz=ignore_tz,
        ignore_start_of_day=ignore_start_of_day,
    )
    return [fr.loc[i] for i, fr in zip(new_idxs, frames)]
