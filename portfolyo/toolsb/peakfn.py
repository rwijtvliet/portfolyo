"""Module to work with peak and offpeak periods."""

import datetime as dt
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from . import changefreq as tools_changefreq
from . import freq as tools_freq
from . import index as tools_index
from . import startofday as tools_sod
from .types import BoolTimeSeries, PintTimeSeries

PeakFunction = Callable[[pd.DatetimeIndex], BoolTimeSeries]


def factory(
    peak_left: dt.time | str | dt.timedelta | None = None,
    peak_right: dt.time | str | dt.timedelta | None = None,
    isoweekdays: Iterable[int] | None = None,
) -> PeakFunction:
    """Create function to identify which timestamps in an index are peakhours and which are offpeak.

    Parameters
    ----------
    peak_left, optional (default: midnight)
        Start time of peak period on days that have a peak period (left-bound, incl).
    peak_right, optional (default: midnight of following day)
        End time of peak period on days that have a peak period (right-bound, excl).
    isoweekdays, optional (default: Monday through Friday)
        Which days of the week have a peak period. (1=Monday, 7=Sunday)

    Returns
    -------
        Function that takes a DatetimeIndex input and returns a Series of booleans with that index,
        indicating for each timestamp if it is part of the peak period or not.

    Notes
    -----
    The values for ``peak_left`` and ``peak_right`` are used to determine the longest
    frequency for which the function makes sense. This is the longest frequency of which
    the timestamps are either entirely peak or entirely offpeak.
    - If one of them does not fall on the full hour (but on a full quarter-hour), e.g.
      07:30, the function can only be used for indices with quarterhour frequency (as
      there are some hours which are partly peak and partly offpeak).
    - If they are both full hours, e.g. 08:00, the function can be used for indices with
      quarterhourly and hourly frequency.
    - If they are both None, each day is entirely peak or entirely offpeak. The function
      can be used for indices with a daily frequency or shorter.
    """
    if peak_left is None:
        peak_left = tools_sod.MIDNIGHT
    peak_left = tools_sod.convert(peak_left)
    if peak_right is None:
        peak_right = tools_sod.MIDNIGHT
    peak_right = tools_sod.convert(peak_right)
    if isoweekdays is None:
        isoweekdays = [1, 2, 3, 4, 5]

    # Characterize the input.
    must_check_time = not (peak_left == tools_sod.MIDNIGHT and peak_right == tools_sod.MIDNIGHT)
    weekday_count = sum(wd in isoweekdays for wd in (1, 2, 3, 4, 5, 6, 7))
    must_check_date = 0 < weekday_count < 7
    if not must_check_time and not must_check_date:
        raise ValueError(
            "Input specifies no special cases; all time periods included or all time periods "
            f"excluded; got {peak_left}-{peak_right} on {weekday_count} days of the week."
        )

    # Find longest frequency for which peak and offpeak can be calculated
    if not must_check_time:
        longest_freq = "D"
    elif peak_left.minute == 0 and peak_right.minute == 0:
        longest_freq = "h"
    elif peak_left.minute % 30 == 0 and peak_right.minute % 30 == 0:
        longest_freq = "30min"
    elif peak_left.minute % 15 == 0 and peak_right.minute % 15 == 0:
        longest_freq = "15min"
    else:
        raise ValueError(
            f"Input specifies times that are not 'round' quarter-hours; got {peak_left} and {peak_right}."
        )

    def filter_date(idx: pd.DatetimeIndex) -> np.ndarray:
        return idx.map(lambda ts: ts.isoweekday() in isoweekdays).values.astype(bool)

    def filter_time(idx: pd.DatetimeIndex) -> np.ndarray | bool:
        time_left = idx.time
        time_right = tools_index.to_right(idx).time
        mask = True
        if peak_left != tools_sod.MIDNIGHT:
            cond1 = time_left >= peak_left
            cond2 = time_right > peak_left
            if any(offenders := ~cond1 & cond2):
                raise ValueError(
                    f"Found timestamps that are partly peak and partly offpeak: {idx[offenders]}"
                )
            mask &= cond1
        if peak_right != tools_sod.MIDNIGHT:
            cond1 = time_left < peak_right
            cond2 = time_right <= peak_right
            if any(offenders := cond1 & ~cond2):
                raise ValueError(
                    f"Found timestamps that are partly peak and partly offpeak: {idx[offenders]}"
                )
            mask &= cond1
        return mask

    def peak_fn(idx: pd.DatetimeIndex) -> BoolTimeSeries:
        # Check if function works for this frequency.
        if tools_freq.up_or_down(idx.freq, longest_freq) > 0:
            raise ValueError(
                f"Peak periods can only be calculated for indices with frequency of {longest_freq} or shorter."
            )
        mask = True
        if must_check_time:
            mask &= filter_time(idx)
        if must_check_date:
            mask &= filter_date(idx)
        return pd.Series(mask, idx)

    return peak_fn


def base_duration(idx: pd.DatetimeIndex) -> PintTimeSeries:
    """Duration of base period in each element of a datetimeindex. Alias of .duration.

    See also
    --------
    portfolyo.duration
    """
    return tools_index.duration(idx)


def peak_duration(idx: pd.DatetimeIndex, peak_fn: PeakFunction) -> PintTimeSeries:
    """Duration of peak periods in each element of a datetimeindex.

    Parameters
    ----------
    idx
        Index for which to calculate the durations. May be in any frequency.
    peak_fn
        Function that returns boolean Series indicating if timestamps in index lie in peak period.

    Returns
    -------
        Series with ``idx`` as index, and duration of peak hours during each timeperiod in ``idx``.

    Notes
    -----
    ``peak_fn`` might only work on indices with daily-or-shorter, hourly-or-shorter, or
    quarterhourly-or-shorter frequency. ``idx`` is resampled to account for this; the returned
    Series has the original index.
    """
    eval_idx = idx  # index to evaluate if peak or offpeak
    for eval_freq in ("D", "h", "15min"):  # peakfn must work for one of these.
        if tools_freq.up_or_down(eval_idx.freq, eval_freq) > 0:  # upsampling necessary
            eval_idx = tools_changefreq.index(eval_idx, eval_freq)
        try:
            eval_bool = peak_fn(eval_idx)  # boolean series
        except ValueError:
            pass  # maybe we need to upsample more
        else:
            eval_duration = eval_bool * tools_index.duration(eval_idx)  # pint-series
            return tools_changefreq.summable(eval_duration, idx.freq).rename("duration")

    # Couldn't determine the duration.
    raise ValueError("Couldn't calculate the duration of the peak period for the provided index.")


def offpeak_duration(idx: pd.DatetimeIndex, peak_fn: PeakFunction) -> PintTimeSeries:
    """
    Duration of offpeak periods in each element of a datetimeindex.

    Parameters
    ----------
    idx
        Index for which to calculate the durations.
    peak_fn
        Function that returns boolean Series indicating if timestamps in index lie in peak period.

    Returns
    -------
        Series with ``idx`` as index, and duration of offpeak hours during each timeperiod in ``idx``.

    Notes
    -----
    ``peak_fn`` might only work on indices with daily-or-shorter, hourly-or-shorter, or
    quarterhourly-or-shorter frequency. ``idx`` is resampled to account for this; the returned
    Series has the original index.
    """
    return tools_index.duration(idx) - peak_duration(idx, peak_fn)
