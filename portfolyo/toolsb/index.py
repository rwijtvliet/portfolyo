"""Module to work with indices."""

import datetime as dt
from typing import Iterable

import pandas as pd

from portfolyo.toolsb.types import Frequencylike

from . import _decorator as tools_decorator
from . import freq as tools_freq
from . import stamp as tools_stamp
from . import startofday as tools_startofday
from .types import PintSeries


# Conversion and validation.
# --------------------------


def validate(idx: pd.DatetimeIndex) -> None:
    """Validate if argument has necessary properties to be used in portfolio lines."""
    # Check on frequency.
    freq = idx.freq
    tools_freq.validate(freq)  # conversion not necessary

    # Check on start_of_day.
    startofday = idx[0].time()
    tools_startofday.validate(startofday)  # conversion not necessary

    # Check on integer number of days.
    if tools_freq.is_shorter_than_daily(freq):
        end_time = tools_stamp.to_right(idx[-1], freq).time()
        if end_time != startofday:
            raise ValueError(
                "Index must contain an integer number of days, i.e., the end time of the final delivery period "
                f"(here: {end_time}) must equal the start time of the first delivery period (here: {startofday})."
            )


check = tools_decorator.create_checkdecorator(validation=validate, default_param="idx")


# --------------------------


@check()
def to_right(idx: pd.DatetimeIndex) -> pd.Series:
    """Right-bound timestamps, belonging to left-bound timestamps of delivery periods
    in index.

    Parameters
    ----------
    idx
        Index for which to calculate the right-bound timestamps.

    Returns
    -------
        Series, with ``idx`` as index and corresponding right-bound timestamps as values.
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
    return pd.Series(idx + tools_freq.to_jump(idx.freq), idx)


@check()
def duration(idx: pd.DatetimeIndex) -> PintSeries:
    """Duration of the delivery periods in a datetime index.

    Parameters
    ----------
    idx
        Index for which to calculate the durations.

    Returns
    -------
        Series, with ``idx`` as index and durations as values.
    """
    jump = tools_freq.to_jump(idx.freq)
    if isinstance(jump, pd.Timedelta):
        tdelta = jump  # one timedelta
    else:
        tdelta = (idx + jump) - idx  # timedeltaindex
    hours = tdelta.total_seconds() / 3600  # one value or index
    return pd.Series(hours, idx, dtype="pint[h]")


# TODO: move to `preprocess.py`?
@check()
@tools_startofday.check()
def replace_startofday(idx: pd.DatetimeIndex, startofday: dt.time) -> pd.DatetimeIndex:
    """For indices with a daily-or-longer frequency, replace the time-part of each
    timestamp, so that the returned index has the specified start-of-day.

    Parameters
    ----------
    idx
        The index (with daily-or-longer frequency).
    startofday
        The desired start-of-day.

    Returns
    -------
        Index with wanted start-of-day.

    Notes
    -----
    This process changes the timestamps and must only be used as a correction to data
    that was recorded with the incorrect timestamps.
    """

    if tools_freq.is_shorter_than_daily(idx.freq):
        raise ValueError(
            "This function works on indices with a daily-or-longer frequency. To trim an index with a"
            " shorter-than-daily frequency to a given start-of-day, use the `trim_to_startofday` method."
        )

    # Daily or longer: change timepart of each.
    stamps = (tools_stamp.replace_time(stamp, startofday) for stamp in idx)
    return pd.DatetimeIndex(stamps, freq=idx.freq, tz=idx.tz)


# TODO: move to `preprocess.py`?
@check(validate=False)
@tools_startofday.check()
def trim_to_startofday(idx: pd.DatetimeIndex, startofday: dt.time) -> pd.DatetimeIndex:
    """For indices with a shorter-than-daily frequency, drop timestamps from the index
    so that the returned index has the specified start-of-day.

    Parameters
    ----------
    idx
        The index (with shorter-than-daily frequency).
    startofday
        The desired start-of-day.

    Returns
    -------
        Index with wanted start-of-day.

    Notes
    -----
    This process is lossy; timestamps are dropped.
    """
    if not tools_freq.is_shorter_than_daily(idx.freq):
        raise ValueError(
            "This function works on indices with a shorter-than-daily frequency. To replace the time-part"
            " of an index with a daily-or-longer frequency with a given start-of-day, use the `replace_startofday` method."
        )

    # Find start. Check at most one entire day (25h @ end of DST).
    to_check = idx - idx[0] < dt.timedelta(hours=25)
    for pos0, stamp in enumerate(idx[to_check]):
        if stamp.time() == startofday:
            break
    else:
        raise ValueError("Did not find any timestamp with correct time at index start.")

    # Find end. Check at most one entire day (25h @ end of DST).
    to_check = idx[-1] - idx < dt.timedelta(hours=25)
    for pos1, stamp in enumerate(reversed(idx[to_check])):
        if tools_stamp.lefttoright(stamp, idx.freq).time() == startofday:
            break
    else:
        raise ValueError("Did not find any timestamp with correct time at index end.")

    return idx[pos0:-pos1]


@check()
@tools_freq.check()
def trim(idx: pd.DatetimeIndex, freq: Frequencylike) -> pd.DatetimeIndex:
    """Trim index to only keep full periods of certain frequency.

    Parameters
    ----------
    idx
        (Untrimmed) index.
    freq
        Delivery period frequency to trim to. E.g. 'MS' to only keep full months.

    Returns
    -------
        Subset of ``idx``, with same frequency. If frequencies are incompatible (i.e., if
        one is not a subset of the other), an error is raised.

    Notes
    -----
    Only if ``idx`` has shorter frequency than ``freq`` might actual trimming occur.
    """
    if tools_freq.up_or_down(idx.freq, freq) >= 0:
        return idx  # no trimming needed when upsampling

    # The frequencies are compatible and ``idx`` has shorter than ``freq``.
    startofday = idx[0].time()
    mask_start = idx >= tools_stamp.ceil(idx[0], freq, startofday)
    i_right = to_right(idx)
    mask_end = i_right <= tools_stamp.floor(i_right[-1], freq, startofday)
    return idx[mask_start & mask_end]


def intersect(idxs: Iterable[pd.DatetimeIndex]) -> pd.DatetimeIndex:
    """Intersect several datetime indices.

    Parameters
    ----------
    idxs
        Indices to intersect.

    Returns
    -------
        Intersection, i.e., datetimeindex with values that exist in each index.

    Notes
    -----
    Indices must have equivalent frequencies, equal timezones and equal start-of-day.
    Otherwise, an error is raised. If there is no overlap, an empty datetimeindex is
    returned.
    """
    idxs = list(idxs)  # Iterable does not (necessarily) have __len__. List does.

    if len(idxs) == 0:
        raise ValueError("Must specify at least one index.")

    if len(idxs) == 1:
        return idxs[0]

    # If we land here, we have at least 2 indices.

    # Assert frequencies equivalent.
    freqs = tools_freq.sorted(set(idx.freq for idx in idxs))  # ensures compatible
    if tools_freq.up_or_down(freqs[0], freqs[-1]) != 0:
        raise ValueError(
            f"Indices must have equal (or equivalent) frequencies; got {freqs[0]} and {freqs[-1]}."
        )

    # Assert timezones equal.
    unequal_tzs = set([idx.tz for idx in idxs])
    if len(unequal_tzs) != 1:
        raise ValueError(f"Indices must have equal timezones; got {unequal_tzs}.")

    # If we land here, we have at least 2 indices with equivalent freq and equal tz. But, one or more might be empty.

    if any([len(idx) == 0 for idx in idxs]):
        return idxs[:0]  # empty index

    # Assert start-of-day equal.
    unequal_sod = set([idx[0].time() for idx in idxs])
    if len(unequal_sod) != 1:
        raise ValueError(f"Indices must have equal start-of-day; got {unequal_sod}.")

    # If we land here, we have at least 2 indices, all not empty, with equivalent freq, equal tz, and equal start-of-day.

    # Do actual intersection.
    # TODO: remove this comment after verification that indeed fixed
    # Calculation is cumbersome: pandas DatetimeIndex.intersection not working correctly on timezone-aware indices (#46702)
    # values = set(idxs[0])
    # for i in idxs[1:]:
    #     values = values.intersection(set(i))
    # return pd.DatetimeIndex(sorted(list(values)), freq=freq, name=name, tz=tz)
    intersected_idx = idxs[0]
    for idx in idxs[1:]:
        intersected_idx = intersected_idx.intersection(idx)
    return intersected_idx


def intersect_flex(
    idxs: Iterable[pd.DatetimeIndex],
    *,
    ignore_freq: bool = False,
    ignore_tz: bool = False,
    ignore_startofday: bool = False,
) -> tuple[pd.DatetimeIndex, ...]:
    """Intersect several datetime indices, but allow for more flexibility of ignoring
    certain properties.

    Parameters
    ----------
    idxs
        Indices to intersect.
    ignore_freq, optional (default: False)
        If True, do intersection even if frequencies are not equivalent; drop time
        periods that do not (fully) exist in either of the indices. The frequencies
        of original indices are preserved. If frequencies are incompatible, an error is
        raised.
    ignore_tz, optional (default: False)
        If True, ignore timezones; perform intersection using 'wall time'. The timezones
        of original indices are preserved.
    ignore_startofday, optional (default: False)
        If True, do intersection even if indices have a different start-of-day. The
        start-of-day of original indices are preserved (even if frequency is shorter
        than daily).

    Returns
    -------
        Intersection for each datetimeindex (in same order as input idxs).

    See also
    --------
    .intersect()
    """
    idxs = list(idxs)  # Iterable does not (necessarily) have __len__. List does.

    if len(idxs) == 0:
        raise ValueError("Must specify at least one index.")

    if len(idxs) == 1:
        return (idxs[0],)

    # If we land here, we have at least 2 indices.

    # Assert frequencies equivalent.
    # (Even if we want to ignore the frequency, the frequencies should be compatible,
    # which is (also) tested in the sorted method.)
    short, *_, long = tools_freq.sorted(set(idx.freq for idx in idxs))
    all_equivalent_freq = tools_freq.up_or_down(short, long) == 0  # compare extremes
    if not ignore_freq and not all_equivalent_freq:
        raise ValueError(
            f"Indices must have equal (or equivalent) frequencies; got {short} and {long}."
        )

    # Assert timezones equal.
    unique_tzs = set([idx.tz for idx in idxs])
    all_equal_tz = len(unique_tzs) == 1
    if not ignore_tz and not all_equal_tz:
        raise ValueError(f"Indices must have equal timezones; got {unique_tzs}.")

    # If we land here, we have at least 2 indices with equivalent (or ignored) freq and equal (or ignored) tz. But, one or more might be empty.

    if any([len(idx) == 0 for idx in idxs]):
        return tuple((idx[:0] for idx in idxs))

    # Assert start-of-day equal.
    unique_sods = set([idx[0].time() for idx in idxs])
    all_equal_sod = len(unique_sods) == 1
    if not ignore_startofday and not all_equal_sod:
        raise ValueError(f"Indices must have equal start-of-day; got {unique_sods}.")

    # If we land here, we have at least 2 indices, all not empty, with equivalent (or ignored) freq, equal (or ignored) tz, and equal (or ignored) start-of-day.

    # Prepare adjusted indices, for intersection.
    l_and_r = [(idx, pd.DatetimeIndex(to_right(idx).values)) for idx in idxs]
    if not all_equal_sod:  # convert to wall time to ignore timezones
        l_and_r = [(le.tz_localize(None), ri.tz_localize(None)) for (le, ri) in l_and_r]
    if not all_equal_sod:  # remove time-part to ignore start-of-day
        l_and_r = [(le.normalize(), ri.normalize()) for (le, ri) in l_and_r]

    # Do actual intersection.
    # Find stretch of time present in all indices.
    common_l, common_r = l_and_r[0]
    for le, ri in l_and_r[1:]:
        common_l, common_r = common_l.intersection(le), common_r.intersection(ri)
    # If empty: return empty indices.
    if len(common_l) == 0 or len(common_r) == 0:
        return tuple([idx[:0] for idx in idxs])
    # If not empty: return overlapping part in all indices.
    first_left_stamp, last_right_stamp = min(common_l), max(common_r)
    intersected_idxs = []
    for idx, (le, ri) in zip(idxs, l_and_r):
        keep = (le >= first_left_stamp) & (ri <= last_right_stamp)
        intersected_idxs.append(idx[keep])
    return tuple(intersected_idxs)
