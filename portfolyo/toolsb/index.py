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


def convert(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert argument to correct/expected type."""
    if idx.freq is None:
        idx = pd.DatetimeIndex(idx, freq=idx.inferred_freq)
    return idx


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


def convert_and_validate(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = convert(idx)
    validate(idx)
    return idx


coerce = tools_decorator.create_coercedecorator(
    conversion=convert, validation=validate, default_param="idx"
)


# --------------------------


@coerce()
def to_right(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Right-bound timestamps, belonging to left-bound timestamps of delivery periods
    in index.

    Parameters
    ----------
    idx
        Index for which to calculate the right-bound timestamps.

    Returns
    -------
        Right-bound timestamps.
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
    return pd.DatetimeIndex(idx + tools_freq.to_jump(idx.freq), idx.freq)


@coerce()
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
@coerce()
@tools_startofday.coerce()
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
@coerce(validate=False)
@tools_startofday.coerce()
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


@coerce()
@tools_freq.coerce()
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
    idx_right = to_right(idx)
    mask_end = idx_right <= tools_stamp.floor(idx_right[-1], freq, startofday)
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
    unique_tzs = set(idx.tz for idx in idxs)
    if len(unique_tzs) != 1:
        raise ValueError(f"Indices must have equal timezones; got {unique_tzs}.")

    # If we land here, we have at least 2 indices with equivalent freq and equal tz. But, one or more might be empty.

    if any(idx.empty for idx in idxs):
        return idxs[:0]  # empty index

    # Assert start-of-day equal.
    unique_sods = set(idx[0].time() for idx in idxs)
    if len(unique_sods) != 1:
        raise ValueError(f"Indices must have equal start-of-day; got {unique_sods}.")

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

    # Finally: set frequency if frequencies equivalant (but not equal).
    if len(freqs) != 1:
        intersected_idx.freq = idxs[0].freq
    return intersected_idx


def _convert_to_dailymidnight(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # Helper function for intersect_flex.
    if tools_freq.is_shorter_than_daily(idx.freq):
        # HACK: Can't .asfreq on index, so take round-trip via Series.
        return idx.to_frame().asfreq("D").index.normalize()  # will be with freq == 'D'
    else:
        # HACK: .normalize() loses frequency, so explicitly set again.
        return pd.DatetimeIndex(idx.normalize(), freq=idx.freq)


def _trim_with_dailymidnight(
    idx: pd.DatetimeIndex, partialits_idx: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    # Helper function for intersect_flex.
    # Create mapping.
    normalized = idx.normalize()
    if not tools_freq.is_shorter_than_daily(idx.freq):
        idxmap = pd.Series(normalized, idx)
    else:
        to_prev_day = idx.time >= idx[0].time()
        dailymidnight = normalized.where(to_prev_day, normalized - pd.DateOffset(days=1))
        idxmap = pd.Series(dailymidnight, idx)
    # Trim.
    return idxmap[idxmap.isin(partialits_idx)].index


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
    # Coerce and turn into list (iterable does not necessarily have __len__; list does)
    idxs = [convert_and_validate(idx) for idx in idxs]

    # Trivial cases.
    if len(idxs) == 0:
        raise ValueError("Must specify at least one index.")

    if len(idxs) == 1:
        return (idxs[0],)

    # If we land here, we have at least 2 indices.

    # Assert frequencies equivalent.
    # (Even if we want to ignore the frequency, the frequencies should be compatible,
    # which is (also) tested in the sorted method.)
    short, *_, long = tools_freq.sorted(idx.freq for idx in idxs)
    all_equivalent_freq = tools_freq.up_or_down(short, long) == 0  # compare extremes
    if not ignore_freq and not all_equivalent_freq:
        raise ValueError(
            f"Indices do not have equal (or equivalent) frequencies; got {short} and {long}. Try setting `ignore_freq`."
        )

    # Assert timezones equal.
    unique_tzs = set([idx.tz for idx in idxs])
    all_equal_tz = len(unique_tzs) == 1
    if not ignore_tz and not all_equal_tz:
        raise ValueError(
            f"Indices do not have equal timezones; got {unique_tzs}. Try setting `ignore_tz`."
        )

    # If we land here, we have at least 2 indices with equivalent or ignored freq and equal or ignored tz. But, one or more might be empty.

    if any(idx.empty for idx in idxs):
        return tuple(idx[:0] for idx in idxs)

    # Assert start-of-day equal.
    unique_sods = set(idx[0].time() for idx in idxs)
    all_equal_sod = len(unique_sods) == 1
    if not ignore_startofday and not all_equal_sod:
        raise ValueError(
            f"Indices must have equal start-of-day; got {unique_sods}. Try setting `ignore_startofday`."
        )

    # If we land here, we have at least 2 indices, all not empty, with equivalent or ignored freq, equal or ignored tz, and equal or ignored start-of-day.

    if all_equal_sod:
        # If we land here, we have at least 2 indices, all not empty, with equivalent or ignored freq, equal or ignored tz, and EQUAL start-of-day.

        le_and_ri = [(idx, to_right(idx)) for idx in idxs]  # do conversion before removing .freq

        # Convert to wall time to ignore timezones. NB destroys .freq !
        if not all_equal_tz:
            le_and_ri = [(le.tz_localize(None), ri.tz_localize(None)) for (le, ri) in le_and_ri]

        # If we land here, we have at least 2 indices, all not empty, with (possibly) no frequency, EQUAL tz, and EQUAL start-of-day.

        # Do actual intersection.
        # Find stretch of time present in all indices.
        common_le, common_ri = le_and_ri[0]
        for le, ri in le_and_ri[1:]:
            common_le, common_ri = common_le.intersection(le), common_ri.intersection(ri)
        # If empty: return empty indices.
        if common_le.empty or common_ri.empty:
            return tuple(idx[:0] for idx in idxs)
        # If not empty: return overlapping part in all indices.
        first_left_stamp, last_right_stamp = min(common_le), max(common_ri)
        return tuple(
            idx[(le >= first_left_stamp) & (ri <= last_right_stamp)]
            for idx, (le, ri) in zip(idxs, le_and_ri)
        )

    # If we land here, we have at least 2 indices, all not empty, with equivalent or ignored freq, equal or ignored tz, and UNEQUAL start-of-day.

    # Approach:
    # - Remove time-part to ignore start-of-day.
    # - Call 'self' but with all idxs in daily-or-longer frequency with midnight time-part.
    # - Map back to original idx.

    dailymidnights = [_convert_to_dailymidnight(idx) for idx in idxs]  # all same sod
    partials = intersect_flex(
        dailymidnights, ignore_freq=True, ignore_startofday=False, ignore_tz=ignore_tz
    )
    return tuple(_trim_with_dailymidnight(idx, partial) for idx, partial in zip(idxs, partials))
