"""Module to work with peak and offpeak periods."""

import datetime as dt
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from . import changefreq as tools_changefreq
from . import duration as tools_duration
from . import freq as tools_freq
from . import right as tools_right

PeakFunction = Callable[[pd.DatetimeIndex], pd.Series]

MIDNIGHT = dt.time(hour=0)


def factory(
    peak_left: dt.time = None,
    peak_right: dt.time = None,
    isoweekdays: Iterable[int] = None,
) -> PeakFunction:
    """Create function to identify which timestamps in an index are peakhours and which are offpeak.

    Parameters
    ----------
    peak_left : dt.time, optional (default: midnight)
        Start time of peak period on days that have a peak period (left-bound, incl).
    peak_right : dt.time, optional (default: midnight of following day)
        End time of peak period on days that have a peak period (right-bound, excl).
    isoweekdays : Iterable[int], optional (default: Monday through Friday)
        Which days of the week have a peak period. (1=Monday, 7=Sunday)

    Returns
    -------
    PeakhourFunction
        That takes a DatetimeIndex input and returns a Series of booleans with that index,
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
        peak_left = MIDNIGHT
    if peak_right is None:
        peak_right = MIDNIGHT
    if isoweekdays is None:
        isoweekdays = [1, 2, 3, 4, 5]

    # Characterize the input.
    must_check_time = not (peak_left == MIDNIGHT and peak_right == MIDNIGHT)
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
    elif peak_left.minute % 15 == 0 and peak_right.minute % 15 == 0:
        longest_freq = "15min"
    else:
        raise ValueError(
            f"Input specifies times that are not 'round' quarter-hours; got {peak_left} and {peak_right}."
        )

    def filter_date(i: pd.DatetimeIndex) -> np.ndarray:
        return i.map(lambda ts: ts.isoweekday() in isoweekdays).values.astype(bool)

    def filter_time(i: pd.DatetimeIndex) -> np.ndarray:
        time_left = i.time
        time_right = tools_right.index(i).time
        mask = True
        if peak_left != MIDNIGHT:
            cond1 = time_left >= peak_left
            cond2 = time_right > peak_left
            if any(offenders := ~cond1 & cond2):
                raise ValueError(
                    f"Found timestamps that are partly peak and partly offpeak: {i[offenders]}"
                )
            mask &= cond1
        if peak_right != MIDNIGHT:
            cond1 = time_left < peak_right
            cond2 = time_right <= peak_right
            if any(offenders := cond1 & ~cond2):
                raise ValueError(
                    f"Found timestamps that are partly peak and partly offpeak: {i[offenders]}"
                )
            mask &= cond1
        return mask

    def peak_fn(i: pd.DatetimeIndex) -> pd.Series:
        # Check if function works for this frequency.
        if tools_freq.up_or_down(i.freq, longest_freq) > 0:
            raise ValueError(
                f"Peak periods can only be calculated for indices with frequency of {longest_freq} or shorter."
            )
        mask = True
        if must_check_time:
            mask &= filter_time(i)
        if must_check_date:
            mask &= filter_date(i)
        return pd.Series(mask, i)

    return peak_fn


def peak_duration(i: pd.DatetimeIndex, peak_fn: PeakFunction) -> pd.Series:
    """
    Duration of peak periods in each element of a datetimeindex.

    Parameters
    ----------
    i : pd.DatetimeIndex
        Index for which to calculate the durations. May be in any frequency.
    peak_fn : PeakFunction
        Function that returns boolean Series indicating if timestamps in index lie in peak period.

    Returns
    -------
    Series
        with ``i`` as index, and duration of peak hours during each timeperiod in ``i``.

    Notes
    -----
    ``peak_fn`` might only work on indices with daily-or-shorter, hourly-or-shorter, or
    quarterhourly-or-shorter frequency. ``i`` is resampled to account for this; the returned
    Series has the original index.
    """
    eval_i = i  # index to evaluate if peak or offpeak
    for eval_freq in ("D", "h", "15min"):
        if tools_freq.up_or_down(eval_i.freq, eval_freq) > 0:  # upsampling necessary
            eval_i = tools_changefreq.index(eval_i, eval_freq)
        try:
            eval_bool = peak_fn(eval_i)  # boolean series
        except ValueError:
            pass  # maybe we need to upsample more
        else:
            eval_duration = eval_bool * tools_duration.index(eval_i)  # pint-series
            return tools_changefreq.summable(eval_duration, i.freq).rename("duration")

    # Couldn't determine the duration.
    raise ValueError(
        "Couldn't calculate the duration of the peak period for the provided index."
    )


def offpeak_duration(i: pd.DatetimeIndex, peak_fn: PeakFunction) -> pd.Series:
    """
    Duration of offpeak periods in each element of a datetimeindex.

    Parameters
    ----------
    i : pd.DatetimeIndex
        Index for which to calculate the durations.
    peak_fn : PeakFunction
        Function that returns boolean Series indicating if timestamps in index lie in peak period.

    Returns
    -------
    Series
        with ``i`` as index, and duration of offpeak hours during each timeperiod in ``i``.

    Notes
    -----
    ``peak_fn`` might only work on indices with daily-or-shorter, hourly-or-shorter, or
    quarterhourly-or-shorter frequency. ``i`` is resampled to account for this.
    """
    return tools_duration.index(i) - peak_duration(i, peak_fn)
