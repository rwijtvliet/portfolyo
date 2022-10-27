"""
Module for doing basic operations on timestamps, timedeltas, and frequencies.
"""

import datetime as dt
from typing import Any, Callable, Tuple, Union

import numpy as np
import pandas as pd
from pytz import AmbiguousTimeError

from .nits import Q_

# Allowed frequencies.
# Perfect containment; a short-frequency time period always entirely falls within a single high-frequency time period.
# AS -> 4 QS; QS -> 3 MS; MS -> 28-31 D; D -> 23-25 H; H -> 4 15T
FREQUENCIES = ["AS", "QS", "MS", "D", "H", "15T"]


def guess_frequency(tdelta: pd.Timedelta) -> str:
    f"""Guess the frequency from a time delta.

    Parameters
    ----------
    tdelta : pd.Timedelta
        Time delta between start and end of period.

    Returns
    -------
    str
        One of {', '.join(FREQUENCIES)}.
    """
    if tdelta == pd.Timedelta(minutes=15):
        return "15T"
    elif tdelta == pd.Timedelta(hours=1):
        return "H"
    elif pd.Timedelta(hours=23) <= tdelta <= pd.Timedelta(hours=25):
        return "D"
    elif pd.Timedelta(days=27) <= tdelta <= pd.Timedelta(days=32):
        return "MS"
    elif pd.Timedelta(days=89) <= tdelta <= pd.Timedelta(days=93):
        return "QS"
    elif pd.Timedelta(days=364) <= tdelta <= pd.Timedelta(days=367):
        return "AS"
    else:
        raise ValueError(
            f"The timedelta ({tdelta}) doesn't seem to be fit to any of the allowed "
            f"frequencies ({', '.join(FREQUENCIES)})."
        )


def ts_right(ts_left: pd.Timestamp, freq: str = None) -> pd.Timestamp:
    """Right-bound timestamp belonging to left-bound timestamp.

    Parameters
    ----------
    ts_left : pd.Timestamp
        Timestamp(s) for which to calculate the right-bound timestamp.
    freq : {'15T' (quarter-hour), 'H' (hour), 'D' (day), 'MS' (month), 'QS' (quarter),
        'AS' (year)}, optional
        Frequency to use in determining the right-bound timestamp.
        If none specified, use ``.freq`` attribute of ``ts_left``.

    Returns
    -------
    pd.Timestamp
        Corresponding right-bound timestamp.

    Examples
    --------
    >>> ts_right(pd.Timestamp('2020-03-01'), 'MS')
    Timestamp('2020-04-01 00:00:00')
    >>> ts_right(pd.Timestamp('2020-03-01', tz='Europe/Berlin'), 'MS')
    Timestamp('2020-04-01 00:00:00+0200', tz='Europe/Berlin')
    >>> ts_right(pd.Timestamp('2020-03-01'), 'AS')
    ValueError
    """
    if freq is None:
        freq = ts_left.freq
    return ts_left + offset(freq)


def floor_ts(
    ts: Union[pd.Timestamp, pd.DatetimeIndex], freq=None, future: int = 0
) -> Union[pd.Timestamp, pd.DatetimeIndex]:
    """Floor timestamp to 'natural' period boundary.

    i.e., find (latest) period start that is on or before the timestamp.

    Parameters
    ----------
    ts : Timestamp or DatetimeIndex.
        Timestamp(s) to floor.
    freq : {'15T' (quarter-hour), 'H' (hour), 'D' (day), 'MS' (month), 'QS' (quarter),
        'AS' (year)}, optional
        What to floor it to, e.g. 'QS' to get start of quarter it's contained in. If
        none specified, use .freq attribute of timestamp.
    future : int, optional (default: 0)
        0 to get latest period start that is ``ts`` or earlier. 1 (-1) to get
        start of period after (before) that. 2 (-2) .. etc.

    Returns
    -------
    Timestamp or DatetimeIndex (same type as ``ts``).
        At begin of period.

    Notes
    -----
    If ``ts`` is exactly at the start of the period, ceil_ts(ts, 0) == floor_ts(ts, 0) == ts.

    Examples
    --------
    >>> floor_ts(pd.Timestamp('2020-04-21 15:42'), 'AS')
    Timestamp('2020-01-01 00:00:00')
    >>> floor_ts(pd.Timestamp('2020-04-21 15:42'), 'MS')
    Timestamp('2020-04-01 00:00:00')
    >>> floor_ts(pd.Timestamp('2020-04-21 15:42'), '15T')
    Timestamp('2020-04-21 15:30:00')
    >>> floor_ts(pd.Timestamp('2020-04-21 15:42', tz='Europe/Berlin'), 'MS')
    Timestamp('2020-04-01 00:00:00+0200', tz='Europe/Berlin')
    >>> floor_ts(pd.Timestamp('2020-04-21 15:42'), 'MS', 2)
    Timestamp('2020-06-01 00:00:00')
    """
    if freq is None:
        freq = ts.freq

    # Rounding to short (< day) frequencies.
    try:
        # Can only infer if it's an index.
        kwargs = {"ambiguous": "infer"} if isinstance(ts, pd.DatetimeIndex) else {}
        if freq == "15T":
            return ts.floor("15T", **kwargs) + pd.Timedelta(minutes=future * 15)
        elif freq == "H":
            return ts.floor("H", **kwargs) + pd.Timedelta(hours=future)
    except AmbiguousTimeError:
        # converting to UTC and then flooring to nearest hour.
        # TODO: this is incorrect for timezones with fractional offset to UTC.
        return floor_ts(ts.tz_convert("UTC"), freq, future).tz_convert(ts.tz)

    # Rounding to longer (>= day) frequencies.
    ts = ts.floor("D")  # make sure we return a midnight value
    if freq == "D":
        return ts + pd.Timedelta(days=future)
    elif freq == "MS":
        return ts + pd.offsets.MonthBegin(1) + pd.offsets.MonthBegin(future - 1)
    elif freq == "QS":
        return (
            ts
            + pd.offsets.QuarterBegin(1, startingMonth=1)
            + pd.offsets.QuarterBegin(future - 1, startingMonth=1)
        )
    elif freq == "AS":
        return ts + pd.offsets.YearBegin(1) + pd.offsets.YearBegin(future - 1)
    else:
        raise ValueError(
            f"Parameter ``freq`` must be one of {', '.join(FREQUENCIES)}; got {freq}."
        )


def ceil_ts(
    ts: Union[pd.Timestamp, pd.DatetimeIndex], freq=None, future: int = 0
) -> Union[pd.Timestamp, pd.DatetimeIndex]:
    """Ceil timestamp to 'natural' period boundary.

    i.e., find (earliest) period start that is on or after the timestamp.

    Parameters
    ----------
    ts : Timestamp or DatetimeIndex.
        Timestamp(s) to ceil.
    freq : {'15T' (quarter-hour), 'H' (hour), 'D' (day), 'MS' (month), 'QS' (quarter),
        'AS' (year)}, optional
        What to ceil it to, e.g. 'QS' to get start of quarter it's contained in. If
        none specified, use .freq attribute of timestamp.
    future : int, optional
        0 (default) to get earliest period start that is ``ts`` or later. 1 (-1) to get
        start of period after (before) that. 2 (-2) .. etc.

    Returns
    -------
    Timestamp or DatetimeIndex (same type as ``ts``).
        At end of period.

    Notes
    -----
    If ``ts`` is exactly at the start of the period, ceil_ts(ts) == floor_ts(ts) == ts.

    Examples
    --------
    >>> ceil_ts(pd.Timestamp('2020-04-21 15:42'), 'AS')
    Timestamp('2021-01-01 00:00:00')
    >>> ceil_ts(pd.Timestamp('2020-04-21 15:42'), 'MS')
    Timestamp('2020-05-01 00:00:00')
    >>> ceil_ts(pd.Timestamp('2020-04-21 15:42'), '15T')
    Timestamp('2020-04-21 15:45:00')
    >>> ceil_ts(pd.Timestamp('2020-04-21 15:42', tz='Europe/Berlin'), 'MS')
    Timestamp('2020-05-01 00:00:00+0200', tz='Europe/Berlin')
    >>> ceil_ts(pd.Timestamp('2020-04-21 15:42'), 'MS', 2)
    Timestamp('2020-07-01 00:00:00')
    """
    if isinstance(ts, pd.DatetimeIndex):
        return pd.DatetimeIndex([ceil_ts(t, freq, future) for t in ts])
    # if ts at start of period, ceil==floor
    offset = 1 if ts != floor_ts(ts, freq, 0) else 0
    return floor_ts(ts, freq, future + offset)


def assert_boundary_ts(ts: Union[pd.Timestamp, pd.DatetimeIndex], freq: str) -> None:
    """Assert that timestamp is a valid start or end timestamp for frequency ``freq``.

    Raise ValueError if not. If pandas.DatetimeIndex is supplied, do assertion for every
    value in the index.

    Parameters
    ----------
    ts : Union[pd.Timestamp, pd.DatetimeIndex]
        Timestamp(s) for which to do the assertion.
    freq : {'15T' (quarter-hour), 'H' (hour), 'D' (day), 'MS' (month), 'QS' (quarter),
        'AS' (year)}

    Examples
    --------
    >>> assert_boundary_ts(pd.Timestamp('2020-02-01'), 'MS')
    None
    >>> assert_boundary_ts(pd.Timestamp('2020-02-01'), 'AS')
    ValueError
    >>> assert_boundary_ts(pd.Timestamp('2020-04-21 15:42'), 'AS')
    ValueError
    """

    if isinstance(ts, pd.DatetimeIndex):
        if (floor_ts(ts, freq) != ts).any():
            raise AssertionError(
                f"Not all values in ``ts`` are a valid boundary timestamp for the frequency {freq}."
            )
        return

    # Assume it's a single timestamp.
    if floor_ts(ts, freq) != ts:
        raise AssertionError(
            f"Timestamp {ts} is not a valid boundary timestamp for the frequency {freq}."
        )


def trim_index(i: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Trim index to only keep full periods.

    Parameters
    ----------
    i : pd.DatetimeIndex
        The (untrimmed) DatetimeIndex
    freq : str
        Frequency to trim to. E.g. 'MS' to only keep full months.

    Returns
    -------
    pd.DatetimeIndex
        Subset of `i`, with same frequency.

    Examples
    --------
    >>> trim_index(pd.date_range('2020-04-21', periods=200, freq='D'), 'MS')
    DatetimeIndex(['2020-05-01', '2020-05-02', ..., '2020-10-31'], dtype='datetime64[ns]', length=184, freq='D')
    >>> trim_index(pd.date_range('2020-04-21', periods=200, freq='D'), 'AS')
    DatetimeIndex([], dtype='datetime64[ns]', freq='D')
    """
    start = ceil_ts(i[0], freq, 0)
    end = floor_ts(ts_right(i[-1]), freq, 0)
    return i[(i >= start) & (i < end)]


def offset(freq: str) -> Union[pd.Timedelta, pd.DateOffset]:
    """Object that can be added to a left-bound timestamp to find corresponding right-bound timestamp

    Parameters
    ----------
    freq : str
        Frequency denoting the length of the time period.

    Returns
    -------
    pd.Timedelta | pd.DateOffset

    Examples
    --------
    >>> offset("H")
    Timedelta('0 days 01:00:00')
    >>> offset("MS")
    <DateOffset: months=1>
    """
    # Get right timestamp for each index value, based on the frequency.
    # . This one breaks for 'MS':
    # (i + pd.DateOffset(nanoseconds=i.freq.nanos))
    # . This drops a value at some DST transitions:
    # (i.shift(1))
    # . This one gives wrong value at DST transitions:
    # i + i.freq

    if freq == "15T":
        return pd.Timedelta(hours=0.25)
    elif freq == "H":
        return pd.Timedelta(hours=1)
    elif freq == "D":
        return pd.DateOffset(days=1)
    elif freq == "MS":
        return pd.DateOffset(months=1)
    elif freq == "QS":
        return pd.DateOffset(months=3)
    elif freq == "AS":
        return pd.DateOffset(years=1)
    else:
        for freq2 in ["MS", "QS"]:  # Edge case: month-/quarterly but starting != Jan.
            if freq_up_or_down(freq2, freq) == 0:
                return offset(freq2)
        raise ValueError(
            f"Parameter ``freq`` must be one of {', '.join(FREQUENCIES)}; got '{freq}'."
        )


def duration(
    ts_left: Union[pd.Timestamp, pd.DatetimeIndex], freq: str = None
) -> Union[Q_, pd.Series]:
    """Duration of a timestamp.

    Parameters
    ----------
    ts_left : Union[pd.Timestamp, pd.DatetimeIndex]
        Timestamp(s) for which to calculate the duration.
    freq : {'15T' (quarter-hour), 'H' (hour), 'D' (day), 'MS' (month), 'QS' (quarter),
        'AS' (year)}, optional
        Frequency to use in determining the duration.
        If none specified, use ``.freq`` attribute of ``ts_left``.

    Returns
    -------
    pint Quantity (if ts_left is Timestamp) or pint-Series (if ts_left is DatetimeIndex).

    Example
    -------
    >>> duration(pd.Timestamp('2020-04-21'), 'D')
    24.0 h
    >>> duration(pd.Timestamp('2020-03-29'), 'D')
    24.0 h
    >>> duration(pd.Timestamp('2020-03-29', tz='Europe/Berlin'), 'D')
    23.0 h
    """
    if freq is None:
        freq = ts_left.freq

    if isinstance(ts_left, pd.DatetimeIndex):
        if freq in ["15T", "H"]:
            # Speed-up things for fixed-duration frequencies.
            h = 1 if freq == "H" else 0.25
            hours = pd.Series(h, ts_left)
        else:
            # Individual calculations for non-fixed-duration frequencies.
            deltas = ts_right(ts_left, freq) - ts_left
            hours = deltas.apply(lambda td: td.total_seconds() / 3600)
        return hours.astype("pint[h]").rename("duration")

    # Assume it's a single timestamp.
    if freq in ["15T", "H"]:
        return Q_(1 if freq == "H" else 0.25, "h")
    else:
        return Q_((ts_right(ts_left, freq) - ts_left).total_seconds() / 3600, "h")


def ts_leftright(left=None, right=None) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Makes 2 timestamps coherent to one another.

    Parameters
    ----------
    left : timestamp, optional
    right : timestamp, optional
        If no values is given for either, the entire next year is given.
        If no value for ts_left is given, the beginning of the year of ts_right is given.
        If no value for ts_right is given, the end of the year of ts_left is given.
        If a value is given for each, they are swapped if their order is incorrect.
        If no time zone is provided for either timestamp, the Europe/Berlin timezone is
        assumed.

    Returns
    -------
    (localized timestamp, localized timestamp)
    """
    # Convert both into timestamps, if possible. None is converted into pd.NaT
    left, right = pd.Timestamp(left), pd.Timestamp(right)

    if right is pd.NaT:
        if left is pd.NaT:
            return ts_leftright(floor_ts(pd.Timestamp.now(), "AS", 1))
        elif left.tz is None:
            return ts_leftright(left.tz_localize("Europe/Berlin"))
        return ts_leftright(left, floor_ts(left, "AS", 1))

    # if we land here, we at least know ts_right.
    if left is pd.NaT:
        back = -1 if right == floor_ts(right, "AS", 0) else 0
        return ts_leftright(floor_ts(right, "AS", back), right)

    # if we land here, we know ts_left and ts_right.
    if right.tz is None:
        if left.tz is None:
            return ts_leftright(left.tz_localize("Europe/Berlin"), right)
        return ts_leftright(left, right.tz_localize(left.tz))

    # if we land here, we know ts_left and localized ts_right.
    if left.tz is None:
        return ts_leftright(left.tz_localize(right.tz), right)

    # if we land here, we know localized ts_left and localized ts_right
    if left > right:
        left, right = right, left

    # return values possibly with distinct timezones. We cannot easily avoid this,
    # because summer- and wintertime are also distinct timezones.
    return left, right


def freq_up_or_down(freq_source, freq_target, common_ts=None) -> int:
    """
    Compare source frequency with target frequency to see if it needs up- or downsampling.

    Upsampling means that the number of values increases - one value in the source
    corresponds to multiple values in the target.

    Parameters
    ----------
    freq_source, freq_target : frequencies to compare.
    common_ts : timestamp, optional
        Timestamp to use as anchor from which to compare the two.

    Returns
    -------
    1 (-1, 0) if source frequency must be upsampled (downsampled, no change) to obtain
        target frequency.

    Notes
    -----
    Arbitrarily using a time point as anchor to calculate the length of the time period
    from. May have influence on the ratio (duration of a month, quarter, year etc are
    influenced by this), but, for most common frequencies, not on which is larger.

    Examples
    --------
    >>> freq_up_or_down('D', 'MS')
    -1
    >>> freq_up_or_down('MS', 'D')
    1
    >>> freq_up_or_down('MS', 'MS')
    0
    """
    standard_common_ts = pd.Timestamp("2020-01-01 0:00")
    backup_common_ts = pd.Timestamp("2020-02-03 04:05:06")
    if common_ts is None:
        common_ts = standard_common_ts
    ts1 = common_ts + pd.tseries.frequencies.to_offset(freq_source)
    ts2 = common_ts + pd.tseries.frequencies.to_offset(freq_target)
    if ts1 > ts2:
        return 1
    elif ts1 < ts2:
        return -1
    if common_ts == standard_common_ts:
        # If they are the same, try with another timestamp.
        return freq_up_or_down(freq_source, freq_target, backup_common_ts)
    return 0  # only if both give the same answer.


def _freq_longestshortest(shortest: bool, *freqs):
    """Determine which frequency denotes the shortest or longest time period."""
    common_ts = pd.Timestamp("2020-01-01")
    ts = [common_ts + pd.tseries.frequencies.to_offset(fr) for fr in freqs]
    i = (np.argmin if shortest else np.argmax)(ts)
    return freqs[i]


def freq_shortest(*freqs) -> Any:
    """Find shortest of several frequencies.

    Parameters
    ----------
    *freqs : str
        Frequencies to compare.

    Returns
    -------
    The shortest of the provided frequencies.

    Examples
    --------
    >>> freq_shortest('MS', 'H', 'AS', 'D')
    'H'
    """
    return _freq_longestshortest(True, *freqs)


def freq_longest(*freqs) -> Any:
    """Find longest of several frequencies.

    Parameters
    ----------
    *freqs : str
        Frequencies to compare.

    Returns
    -------
    The longest of the provided frequencies.

    Examples
    --------
    >>> freq_longest('MS', 'H', 'AS', 'D')
    'AS'
    """
    return _freq_longestshortest(False, *freqs)


def gasday_factory(gasday_start: dt.time) -> Callable[[pd.Timestamp], pd.Timestamp]:
    """Returns function that calculates the start of the "gas day" in which a timestamp falls.

    Necessary as a day is not always defined from midnight to midnight. For gas markets,
    a day (e.g. Jan 10) often starts in the moring, e.g. the left-closed interval
    Jan 10 06:00:00 - Jan 11 06:00:00.

    Parameters
    ----------
    gasday_start : dt.time
        Start of the gas day in market under consideration.

    Returns
    -------
    Callable[[pd.Timestamp], pd.Timestamp]

    """

    def gasday(ts: pd.Timestamp) -> pd.Timestamp:
        """Returns the "gas day" (timestamp at midnight) that a delivery timestamp belongs to.

        Parameters
        ----------
        ts : pd.Timestamp
            Moment of delivery.

        Returns
        -------
        pd.Timestamp
            (Start of) corresponding gas day.

        Examples
        --------
        gasday(pd.Timestamp('2020-01-01 04:00:00')) == pd.Timestamp('2019-12-31 00:00:00')
        gasday(pd.Timestamp('2020-01-01 08:00:00')) == pd.Timestamp('2020-01-01 00:00:00')
        gasday(pd.Timestamp('2020-03-29 08:00:00+02:00 Europe/Berlin')) ==
               pd.Timestamp('2020-03-29 00:00:00+01:00 Europe/Berlin')
        """
        day = ts.floor("D")
        if ts.time() < gasday_start:  # get previous day
            # Must use custom function; dt.timedelta(days=1) is always 24h (incorrect during DST change)
            day = day - offset("D")
        return day

    return gasday


gasday_de = gasday_factory(dt.time(hour=6))
