"""
Tools for dealing with frequencies.
"""

import numpy as np
import pandas as pd

from .types import Series_or_DataFrame


# Allowed frequencies.
ALLOWED_FREQUENCIES_DOCS = "'15min' (=quarterhour), 'h', 'D', 'MS', 'QS' (or 'QS-FEB', 'QS-MAR', etc.), or 'YS' (or 'YS-FEB', 'YS-MAR', etc.)"
ALLOWED_CLASSES = [
    pd.tseries.offsets.YearBegin,
    pd.tseries.offsets.QuarterBegin,
    pd.tseries.offsets.MonthBegin,
    pd.tseries.offsets.Day,
    pd.tseries.offsets.Hour,
    pd.tseries.offsets.Minute,
]
TO_OFFSET = pd.tseries.frequencies.to_offset
SHORTEST_TO_LONGEST = [
    type(TO_OFFSET(freq)) for freq in ["15min", "h", "D", "MS", "QS", "YS"]
]

quarter_matrix = [
    ["QS", "QS-APR", "QS-JUL", "QS-OCT"],
    ["QS-FEB", "QS-MAY", "QS-AUG", "QS-NOV"],
    ["QS-MAR", "QS-JUN", "QS-SEP", "QS-DEC"],
]


def assert_freq_valid(freq: str) -> None:
    """
    Validate if the given frequency string is allowed based on pandas offset objects.

    Parameters:
    freq (str): A string representing a frequency alias (e.g., "YS", "QS", "MS").

    Raises:
    ValueError: If the frequency is not allowed.
    """

    freq_offset = pd.tseries.frequencies.to_offset(freq)
    mro_class = freq_offset.__class__.__mro__[0]

    # Check if the MRO is in the list of allowed MROs
    # have to make sure it's only the first class on the list
    if mro_class not in ALLOWED_CLASSES:
        raise ValueError(f"The passed frequency '{freq}' is not allowed.")

    # Define restricted classes that should have n == 1
    restricted_classes = {
        pd.tseries.offsets.MonthBegin: 1,
        pd.tseries.offsets.Day: 1,
        pd.tseries.offsets.Hour: 1,
        pd.tseries.offsets.Minute: 15,
    }
    allowed_n = restricted_classes.get(type(freq_offset))
    if allowed_n is not None:  # case where freq is not in restricted class
        # Check if freq_offset.n is not None and if it doesn't match allowed_n
        if freq_offset.n is None or freq_offset.n != allowed_n:
            raise ValueError(f"The passed frequency {freq} is not allowed.")


def assert_freq_sufficiently_long(freq, freq_ref, strict: bool = False) -> None:
    """
    Compares ``freq`` and ``freq_ref``, raising an AssertionError if ``freq`` is not long enough.

    Parameters
    ----------
    freq_source, freq_ref : frequencies to compare.
    strict : bool, optional (default: False)
        - If ``strict`` is True, ``freq`` must be strictly longer than ``freq_long``.
        - If False, it may be equally long.

    """
    # freq should start from the beginning of the year
    index_freq = SHORTEST_TO_LONGEST.index(type(TO_OFFSET(freq)))
    index_ref = SHORTEST_TO_LONGEST.index(type(TO_OFFSET(freq_ref)))
    if strict is True:
        if not (index_freq > index_ref):
            raise AssertionError(
                f"The passed frequency is not sufficiently long; passed {freq}, but should be {freq_ref} or longer."
            )
    else:
        if not (index_freq >= index_ref):
            raise AssertionError(
                f"The passed frequency is not sufficiently long; passed {freq}, but should be {freq_ref} or longer."
            )


def up_or_down(freq_source: str, freq_target: str) -> int:
    """
    Compare source frequency with target frequency to see if it needs up- or downsampling.

    Upsampling means that the number of values increases - one value in the source
    corresponds to multiple values in the target.

    Parameters
    ----------
    freq_source, freq_target : frequencies to compare.

    Returns
    -------
    * 1 if source frequency must be upsampled to obtain (i.e, is longer than) target frequency.
    * 0 if source frequency is same as target frequency.
    * -1 if source frequency must be downsampled to obtain (i.e, is shorter than) target frequency.

    Notes
    -----
    If the freq can't be down- or upsampled, throws ValueError.

    Examples
    --------
    >>> freq.up_or_down('D', 'MS')
    -1
    >>> freq.up_or_down('MS', 'D')
    1
    >>> freq.up_or_down('MS', 'MS')
    0
    >>> freq.up_or_down('QS', 'QS-APR')
    ValueError

    """
    restricted_classes = [
        pd._libs.tslibs.offsets.QuarterBegin,
        pd._libs.tslibs.offsets.YearBegin,
    ]
    # Convert freq from str to offset
    freq_source_as_offset = pd.tseries.frequencies.to_offset(freq_source)
    freq_target_as_offset = pd.tseries.frequencies.to_offset(freq_target)

    # Compare if the freq are the same
    if freq_source_as_offset == freq_target_as_offset:
        return 0
    # One of the freq can be in restricted class, but not both
    if not (
        type(freq_source_as_offset) in restricted_classes
        and type(freq_target_as_offset) in restricted_classes
    ):
        try:
            assert_freq_sufficiently_long(freq_source, freq_target, strict=True)
            return 1
        except AssertionError:
            return -1
    # If both in restricted class
    else:
        source_index = restricted_classes.index(type(freq_source_as_offset))
        target_index = restricted_classes.index(type(freq_target_as_offset))
        # the code below describes the case when year and/or quarter starts from the same month group
        # example: JAN,APR,JUl and OCT
        # if we are in the same quadrant (belong to the same month group), we can transfrom one to another
        # better described at https://github.com/rwijtvliet/portfolyo/issues/57
        group_by_month_beginn = (
            freq_source_as_offset.startingMonth
            if source_index == 0
            else freq_source_as_offset.month
        ) % 3 == (
            freq_target_as_offset.startingMonth
            if target_index == 0
            else freq_target_as_offset.month
        ) % 3

        if group_by_month_beginn:
            if source_index > target_index:
                # we are in case AS and QS
                return 1
            elif source_index < target_index:
                # we are in the case QS and AS
                return -1
            elif source_index == 0:
                # we are in the case QS and QS
                return 0

        raise ValueError(
            f"The passed frequency {freq_source} can't be aggregated to {freq_target}."
        )


def assert_freq_equally_long(freq, freq_ref) -> None:
    """
    Compares ``freq`` and ``freq_ref``, raising an AssertionError if ``freq`` is not equally long as ``freq_ref``.

    Parameters
    ----------
    freq_source, freq_ref : frequencies to compare.
    Valid examples
    --------
    >>> freq.assert_freq_equally_long('QS', 'QS')
    or
    >>> freq.assert_freq_equally_long('QS', 'QS-APR')
    or
    >>> freq.assert_freq_equally_long('QS', 'QS-FEB')

    """
    assert_freq_sufficiently_long(freq, freq_ref, strict=False)
    assert_freq_sufficiently_long(freq_ref, freq, strict=False)


def assert_freq_sufficiently_short(freq, freq_ref, strict: bool = False) -> None:
    """
    Compares ``freq`` and ``freq_ref``, raising an AssertionError if ``freq`` is not short enough.

    Parameters
    ----------
    freq_source, freq_ref : frequencies to compare.
    strict : bool, optional (default: False)
        - If ``strict`` is True, ``freq`` must be strictly shorter than ``freq_long``.
        - If False, it may be equally long, or rather, short.

    """
    assert_freq_sufficiently_long(freq_ref, freq, strict)


def _longestshortest(shortest: bool, *freqs: str):
    """Determine which frequency denotes the shortest or longest time period."""
    common_ts = pd.Timestamp("2020-01-01")
    # ts = [common_ts + pd.tseries.frequencies.to_offset(fr) for fr in freqs]
    # Compute the duration each frequency represents
    durations = []
    for fr in freqs:
        offset = pd.tseries.frequencies.to_offset(fr)
        delta = (common_ts + 2 * offset) - (common_ts + offset)  # Actual time span
        durations.append(delta)
    i = (np.argmin if shortest else np.argmax)(durations)
    return freqs[i]


def shortest(*freqs: str) -> str:
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
    >>> freq.shortest('MS', 'h', 'YS', 'D')
    'h'
    """
    return _longestshortest(True, *freqs)


def longest(*freqs: str) -> str:
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
    >>> freq.longest('MS', 'h', 'YS', 'D')
    'YS'
    """
    return _longestshortest(False, *freqs)


def to_offset(freq: str) -> pd.Timedelta | pd.DateOffset:
    """Object that can be added to a left-bound timestamp to find corresponding right-bound timestamp.

    Parameters
    ----------
    freq : str
        Frequency denoting the length of the time period.

    Returns
    -------
    pd.Timedelta | pd.DateOffset

    Examples
    --------
    >>> freq.to_offset("h")
    Timedelta('0 days 01:00:00')
    >>> freq.to_offset("MS")
    <DateOffset: months=1>
    """
    # Convert the frequency string to an offset object
    offset = pd.tseries.frequencies.to_offset(freq)

    # Custom handling for specific simple frequencies
    if isinstance(offset, pd.tseries.offsets.Minute) and offset.n == 15:
        return pd.Timedelta(minutes=15)
    elif isinstance(offset, pd.tseries.offsets.Hour) and offset.n == 1:
        return pd.Timedelta(hours=1)
    elif isinstance(offset, pd.tseries.offsets.Day) and offset.n == 1:
        return pd.DateOffset(days=1)
    elif isinstance(offset, pd.tseries.offsets.MonthBegin) and offset.n == 1:
        return pd.DateOffset(months=1)
    elif isinstance(offset, pd.tseries.offsets.QuarterBegin) and offset.n == 1:
        return pd.DateOffset(months=3)
    elif isinstance(offset, pd.tseries.offsets.YearBegin) and offset.n == 1:
        return pd.DateOffset(years=1)
    else:
        raise ValueError(
            f"Parameter ``freq`` must be one of {ALLOWED_FREQUENCIES_DOCS}; got '{freq}'."
        )


def from_tdelta(tdelta: pd.Timedelta) -> str:
    f"""Guess the frequency from a time delta.

    Parameters
    ----------
    tdelta : pd.Timedelta
        Time delta between start and end of delivery period.

    Returns
    -------
    str
        One of {ALLOWED_FREQUENCIES_DOCS}.
    """
    if tdelta == pd.Timedelta(minutes=15):
        return "15min"
    elif tdelta == pd.Timedelta(hours=1):
        return "h"
    elif pd.Timedelta(hours=23) <= tdelta <= pd.Timedelta(hours=25):
        return "D"
    elif pd.Timedelta(days=27) <= tdelta <= pd.Timedelta(days=32):
        return "MS"
    elif pd.Timedelta(days=89) <= tdelta <= pd.Timedelta(days=93):
        return "QS"
    elif pd.Timedelta(days=364) <= tdelta <= pd.Timedelta(days=367):
        return "YS"
    else:
        raise ValueError(
            f"The timedelta ({tdelta}) doesn't seem to be fit to any of the allowed "
            f"frequencies ({ALLOWED_FREQUENCIES_DOCS})."
        )


def guess_to_index(i: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """ "Try to infer the frequency of the index and set it if possible.
    Parameters
    ----------
    i : pd.DatetimeIndex
    Returns
    -------
    pd.DatetimeIndex
        DatetimeIndex, with the inferred frequency if possible.
    """
    # Find frequency.
    if i.freq:
        return i
    # Freq not set.
    i = i.copy(deep=True)

    try:
        inferred_freq = pd.infer_freq(i)
        for row_index in range(len(quarter_matrix)):  # Loop through the rows
            if (
                inferred_freq in quarter_matrix[row_index]
            ):  # check if inferred_freq is somewhere in this row
                inferred_freq = quarter_matrix[row_index][
                    0
                ]  # set inferred_freq to the first value in the row
        i.freq = inferred_freq

    except ValueError:
        pass  # Couldn't find a frequency, e.g., because there are not enough values
    return i


def guess_to_frame(fr: Series_or_DataFrame) -> Series_or_DataFrame:
    """Try to infer the frequency of the frame's index and set it if possible.

    Parameters
    ----------
    fr : pd.Series or pd.DataFrame

    Returns
    -------
    pd.Series | pd.DataFrame
        Same type as ``fr``, with the inferred frequency if possible.
    """
    # Handle non-datetime-indices.
    if not isinstance(fr.index, pd.DatetimeIndex):
        raise ValueError(
            "The data does not have a datetime index and can therefore not have a frequency."
        )

    if fr.index.freq:
        return fr

    return fr.set_axis(guess_to_index(fr.index), axis=0)


def set_to_frame(fr: Series_or_DataFrame, wanted: str) -> Series_or_DataFrame:
    """Try to force frequency of frame's index.

    Parameters
    ----------
    fr : pd.Series or pd.DataFrame
    wanted : str
        Frequency to set.

    Returns
    -------
    pd.Series | pd.DataFrame
        Same type as ``fr``, with, if possible, a valid value for ``fr.index.freq``.
    """
    # Handle non-datetime-indices.
    if not isinstance(fr.index, pd.DatetimeIndex):
        raise ValueError(
            "The data does not have a datetime index and can therefore not have a frequency."
        )

    # Set frequency.
    i = fr.index.copy(deep=True)
    i.freq = wanted

    return fr.set_axis(i, axis=0)
