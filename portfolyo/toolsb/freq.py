"""
Tools for dealing with frequencies.
"""

import builtins
import functools
from typing import Iterable

import pandas as pd
from pandas.tseries.frequencies import MONTHS, to_offset
from pandas.tseries.offsets import BaseOffset

from . import _decorator as tools_decorator
from .types import Frequencylike

# Developer notes:
#
# We do not support arbitrary frequencies. Instead, we focus on some due to their prevalence in the energy industry.
# * Shorter-than-daily: mainly hour (h) and quarterhour (15min), but we also support some additional ones (1min, 5min, 30min).
#   These are all fixed-length frequencies, i.e., the timedelta between timestamps is constant.
# * Daily or longer: day (D), months (MS), quarters starting in any month (QS-JAN, QS-FEB, ...), years starting in any month (YS-JAN, YS-FEB, ...)
#   These all are non-fixed-length frequencies, with the timedelta between timestamps taking on multiple values.
#
# For the shorter-than-daily frequencies, we do not support all indices. Instead, there are additional requirements, namely,
#   (1) that they start on a full hour and
#   (2) that they contain an integer number of days.
# That way, we ensure that
#   (1) downsampling to days etc. is possible, and
#   (2) we can use the first timestamp as the "start-of-day" of the timeseries.
#
# Terminology when comparing 2 frequencies:
#
# COMPATIBLE vs INCOMPATIBLE.
# Resampling between 2 frequencies is not always possible, e.g., to resample from QS-FEB (quarters starting in Feb) to YS-JAN (years starting in Jan). Direct resampling is not possible, because neither 'neatly fits' inside the other. To still resample, we must take a detour by first resampling to a shorter frequency that neatly divides BOTH, e.g. MS (monthly). This is must be done manually by the user. The frequencies are colled "incompatible".
#
# EQUIVALENT.
# For quarters, we have the situation that several frequencies map 1-to-1 onto each other. For example, 'QS-FEB' describes a quarterly series where one of the quarters starts with February. This means another of its quarters starts in May, and therefore, this frequency is the same as 'QS-MAY'. These frequencies are called "equivalent".
# * This also means that downsampling from quarters to years is possible to 4 distinct yearly frequencies (e.g. 'QS-FEB' can be upsampled to 'YS-FEB', but also 'YS-MAY', which has different periods and different values). And that upsampling from years to quarters is also possible to 4 quarterly frequencies - though these all result in the same timeseries (e.g. 'YS-FEB' can be downsampled to 'QS-FEB' and to 'QS-MAY', though this has the same periods and the same values).


# Allowed frequencies.


_FREQUENCIES: set[BaseOffset] = {
    *(to_offset(freq) for freq in ("min", "5min", "15min", "30min", "h", "D", "MS")),
    *(to_offset(f"QS-{m}") for m in MONTHS),
    *(to_offset(f"YS-{m}") for m in MONTHS),
}

ALLOWED_FREQUENCIES_DOCS = "'min', '5min', '15min', '30min', 'h', 'D', 'MS', 'QS' (or 'QS-FEB', 'QS-MAR', etc.), or 'YS' (or 'YS-FEB', 'YS-MAR', etc.)"


# Subsets of allowed frequencies.
# . One subset.
_SHORTERTHANDAILY: set[BaseOffset] = {
    to_offset(freq) for freq in ("min", "5min", "15min", "30min", "h")
}
# . Three subsets that are mutually exclusive.
_SORTED: tuple[BaseOffset, ...] = tuple(
    to_offset(freq) for freq in ("min", "5min", "15min", "30min", "h", "D", "MS")
)
_QUARTERLY: set[BaseOffset] = set((to_offset(f"QS-{m}") for m in MONTHS))
_YEARLY: set[BaseOffset] = set((to_offset(f"YS-{m}") for m in MONTHS))


# Mappings between frequencies.


@functools.lru_cache()
def _equivalent_freqs(freq: BaseOffset) -> set[BaseOffset]:
    """Return all frequencies that are equivalent (or equal) to ``freq``."""
    if freq in _QUARTERLY:
        return {f for f in _QUARTERLY if f.startingMonth % 3 == freq.startingMonth % 3}
    elif freq in _FREQUENCIES:
        return {freq}
    raise ValueError("Unexpected frequency.")


@functools.lru_cache()
def _downsample_targets(freq: BaseOffset) -> set[BaseOffset]:
    """Return all frequencies that ``freq`` can be downsampled to."""
    if freq in _SORTED:
        pos = _SORTED.index(freq)
        return set(_SORTED[pos + 1 :]) | _QUARTERLY | _YEARLY
    elif freq in _QUARTERLY:
        return set((f for f in _YEARLY if f.month % 3 == freq.startingMonth % 3))
    elif freq in _YEARLY:
        return set()
    raise ValueError("Unexpected frequency.")


# Conversion and validation.
# --------------------------


@functools.lru_cache()
def convert(freq: Frequencylike) -> BaseOffset:
    """Convert argument to correct/expected type."""
    if isinstance(freq, str):
        freq = to_offset(freq)
    return freq


@functools.lru_cache()
def validate(freq: BaseOffset | None) -> None:
    """Validate if argument has necessary properties to be used in portfolio lines."""
    if freq is None:
        raise ValueError("Frequency may not be None.")
    if freq not in _FREQUENCIES:
        raise ValueError(f"Frequency must be one of {ALLOWED_FREQUENCIES_DOCS}.")


coerce = tools_decorator.create_coercedecorator(
    conversion=convert, validation=validate, default_param="freq"
)


# --------------------------


@coerce()
def is_shorter_than_daily(freq: BaseOffset) -> bool:
    """Return True if ``freq`` is shorter than daily, i.e., hourly or shorter. This
    also implies that the frequency is a fixed-length frequency."""
    return freq in _SHORTERTHANDAILY


@coerce("source_freq", "target_freq")
def up_or_down(source_freq: BaseOffset, target_freq: BaseOffset) -> int:
    """See if changing the frequency of an index requires up- or downsampling.

    Upsampling means that the number of values increases - one value in the source
    corresponds to multiple values in the target.

    Parameters
    ----------
    source_freq
        Frequency of the source data.
    target_freq
        Frequency to resample it to.

    Returns
    -------
    * 1 if index must be upsampled to obtain target frequency. E.g. 'D' -> 'MS'.
    * 0 if source frequency is equal (or equivalent) to target frequency. E.g. 'QS-JAN' -> 'QS-APR'.
    * -1 if index must be downsampled to obtain target frequency. E.g. 'MS' -> 'D'.

    Raises
    ------
    ValueError if resampling is not possible because frequencies are incompatible. E.g. 'QS-JAN' -> 'YS-FEB'.
    """
    if target_freq in _downsample_targets(source_freq):
        return -1
    elif source_freq in _downsample_targets(target_freq):
        return 1
    elif target_freq in _equivalent_freqs(source_freq):
        return 0
    raise ValueError(f"Can't (directly) resample from {source_freq} to {target_freq}.")


def sorted(freqs: Iterable[Frequencylike]) -> tuple[BaseOffset, ...]:
    """Sort several frequencies from shortest to longest.

    Parameters
    ----------
    freqs
        Frequencies to sort.

    Returns
    -------
        Sorted frequncies. Equivalent frequencies (e.g. QS-JAN and QS-APR) may appear
        in any order.

    Raises
    ------
    ValueError
        If any pair of frequencies is incompatible (e.g. QS-JAN and QS-FEB, or QS-JAN
        and YS-FEB, or YS-JAN and YS-FEB).

    Examples
    --------
    >>> sorted(['h', 'YS', 'QS'])
    ('h', 'QS', 'YS')
    >>> sorted(['h', 'YS-FEB', 'QS'])
    ValueError
    """
    return tuple(builtins.sorted(freqs, key=functools.cmp_to_key(up_or_down)))


def shortest(freqs: Iterable[Frequencylike]) -> BaseOffset:
    """Find shortest of several frequencies.

    Parameters
    ----------
    *freqs
        Frequencies to compare.

    Returns
    -------
       The shortest of the provided frequencies. If there is a tie between equivalent
       frequencies (e.g. QS-JAN and QS-APR), any one may be returned.

    Raises
    ------
    ValueError
        If any pair of frequencies is incompatible (e.g. QS-JAN and QS-FEB, or QS-JAN
        and YS-FEB, or YS-JAN and YS-FEB).

    Examples
    --------
    >>> shortest(['h', 'YS', 'QS'])
    'h'
    >>> shortest(['h', 'YS-FEB', 'QS'])
    ValueError
    """
    return sorted(set(freqs))[0]


def longest(freqs: Iterable[Frequencylike]) -> BaseOffset:
    """Find longest of several frequencies.

    Parameters
    ----------
    *freqs
        Frequencies to compare.

    Returns
    -------
       The longest of the provided frequencies. If there is a tie between equivalent
       frequencies (e.g. QS-JAN and QS-APR), any one may be returned.

    Raises
    ------
    ValueError
        If any pair of frequencies is incompatible (e.g. QS-JAN and QS-FEB, or QS-JAN
        and YS-FEB, or YS-JAN and YS-FEB).


    Examples
    --------
    >>> longest(['h', 'YS', 'QS'])
    'YS'
    >>> longest(['h', 'YS-FEB', 'QS'])
    ValueError
    """
    return sorted(set(freqs))[-1]


@coerce()
def to_jump(freq: BaseOffset) -> pd.Timedelta | pd.DateOffset:
    """Jump object corresponding to a frequency. Can be added to a left-bound delivery
    period timestamp to get the right-bound timestamp of that delivery period (which
    is the left-bound timestamp of the following delivery period).

    Parameters
    ----------
    freq
        Frequency of delivery period.

    Returns
    -------
        Term that can be (repeatedly) added to / subtracted from a left-bound
        timestamp to find the next / previous left-bound timestamps.

    Notes
    -----
    Only gives correct result if added to a valid left-bound stamp for the frequency.
    If necessary, check if this is the case with `tools.stamp.is_boundary()`, or ensure
    it is the case with `tools.stamp.floor()` or `tools.stamp.ceil()`.

    Examples
    --------
    >>> freq.to_jump("h")
    Timedelta('0 days 01:00:00')
    >>> freq.to_jump("MS")
    <DateOffset: months=1>
    """
    # Custom handling for specific simple frequencies
    if isinstance(freq, pd.tseries.offsets.Minute) and freq.n in (1, 5, 15, 30):
        return pd.Timedelta(minutes=freq.n)
    elif isinstance(freq, pd.tseries.offsets.Hour) and freq.n == 1:
        return pd.Timedelta(hours=1)
    elif isinstance(freq, pd.tseries.offsets.Day) and freq.n == 1:
        return pd.DateOffset(days=1)
    elif isinstance(freq, pd.tseries.offsets.MonthBegin) and freq.n == 1:
        return pd.DateOffset(months=1)
    elif isinstance(freq, pd.tseries.offsets.QuarterBegin) and freq.n == 1:
        return pd.DateOffset(months=3)
    elif isinstance(freq, pd.tseries.offsets.YearBegin) and freq.n == 1:
        return pd.DateOffset(years=1)
    else:  # shouldn't occur due to check decorator
        raise ValueError(
            f"Parameter ``freq`` must be one of {ALLOWED_FREQUENCIES_DOCS}; got '{freq}'."
        )
