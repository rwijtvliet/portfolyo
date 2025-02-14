"""
Tools for dealing with frequencies.
"""

from typing import Callable
import functools
import pandas as pd
import inspect
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import BaseOffset

from portfolyo.tools.types import Frequencylike

# Developer notes:
#
# We do not support arbitrary frequencies. Instead, we focus on some due to their prevalence in the energy industry.
# * Shorter-than-daily: mainly hour (h) and quarterhour (15min), but we also support some additional ones (1min, 5min, 30min).
# * Daily or longer: day (D), months (MS), quarters starting in any month (QS-JAN, QS-FEB, ...), years starting in any month (YS-JAN, YS-FEB, ...)
#
# For the shorter-than-daily frequencies, we do not support all indices. Instead, there are additional requirements, namely,
#   (1) that they start on a full hour and
#   (2) that they contain an integer number of days.
# That way, we ensure that
#   (1) downsampling to days etc. is possible, and
#   (2) we can use the first timestamp as the "start-of-day" of the timeseries.
#
# Resampling between 2 frequencies is not always possible, e.g., to resample from QS-FEB (quarters starting in Feb) to YS-JAN (years starting in Jan). Direct resampling is not possible, because neither 'neatly fits' inside the other. To still resample, we must take a detour by first resampling to a shorter frequency that neatly divides BOTH, e.g. MS (monthly). This is must be done manually by the user. This situation can occur when using common frequencies (as in the example, QS-FEB vs YS-JAN).
#
#
# For quarters, we have the situation that several frequencies map 1-to-1 onto each other. For example, 'QS-FEB' describes a quarterly series where one of the quarters starts with February.
# * This means another of its quarters starts in May, and therefore, this frequency is the same as 'QS-MAY'.
# * This also means that downsampling from quarters to years is possible to 4 distinct yearly frequencies (e.g. 'QS-FEB' can be upsampled to 'YS-FEB', but also 'YS-MAY', which has different periods and different values). And that upsampling from years to quarters is also possible to 4 quarterly frequencies - though these all result in the same timeseries (e.g. 'YS-FEB' can be downsampled to 'QS-FEB' and to 'QS-MAY', though this has the same periods and the same values).

_MONTHS: set[str] = {
    "JAN",
    "APR",
    "JUL",
    "OCT",
    "FEB",
    "MAY",
    "AUG",
    "NOV",
    "MAR",
    "JUN",
    "SEP",
    "DEC",
}


# Allowed frequencies.


_FREQUENCIES: set[BaseOffset] = {
    *(to_offset(freq) for freq in ("min", "5min", "15min", "30min", "h", "D", "MS")),
    *(to_offset(f"QS-{m}") for m in _MONTHS),
    *(to_offset(f"YS-{m}") for m in _MONTHS),
}

# There are a few thing we might want to know about a SINGLE frequency:
# - Is it shorter-than-daily? These are special, because an index with one of these frequencies is not automatically valid (must have integer number of days).
# - What is the equivalent DateOffset or Timedelta that can be added to a left-bound timestamp to find the right-bound timestamp (i.e., the stort of the next timestamp).

# Also, there are a few things we might want to know when COMPARING frequencies:
# - Can we resample from one to the other?


_SHORTERTHANDAILY: set[BaseOffset] = {
    to_offset(freq) for freq in ("min", "5min", "15min", "30min", "h")
}


# Each frequency is element of exactly one of the following containers.

_SORTED: tuple[BaseOffset, ...] = tuple(
    to_offset(freq) for freq in ("min", "5min", "15min", "30min", "h", "D", "MS")
)
_QUARTERS: set[BaseOffset] = set((to_offset(f"QS-{m}") for m in _MONTHS))
_YEARS: set[BaseOffset] = set((to_offset(f"YS-{m}") for m in _MONTHS))


# Create maps to quickly find out which resampling is possible.


def _identityresample_targets(freq: BaseOffset) -> set[BaseOffset]:
    if freq in _QUARTERS:
        return {f for f in _QUARTERS if f.startingMonth % 3 == freq.startingMonth % 3}
    elif freq in _FREQUENCIES:
        return {freq}
    else:
        raise ValueError("Unexpected frequency.")


_IDENTITYRESAMPLE_TARGETS: dict[BaseOffset, set[BaseOffset]] = {
    freq: _identityresample_targets(freq) for freq in _FREQUENCIES
}


def _downsample_targets(freq: BaseOffset) -> set[BaseOffset]:
    if freq in _SORTED:
        pos = _SORTED.index(freq)
        return set(_SORTED[pos + 1 :]) | _QUARTERS | _YEARS
    elif freq in _QUARTERS:
        return set((f for f in _YEARS if f.month % 3 == freq.startingMonth % 3))
    elif freq in _YEARS:
        return set()
    else:
        raise ValueError("Unexpected frequency.")


_DOWNSAMPLE_TARGETS: dict[BaseOffset, set[BaseOffset]] = {
    freq: _downsample_targets(freq) for freq in _FREQUENCIES
}

# Allowed for docs.

DOCS = "'min', '5min', '15min', '30min', 'h', 'D', 'MS', 'QS' (or 'QS-FEB', 'QS-MAR', etc.), or 'YS' (or 'YS-FEB', 'YS-MAR', etc.)"


def assert_freq_valid(freq: BaseOffset | None) -> None:
    f"""Validate if the given frequency is valid.

    Parameters
    ----------
    freq
        Frequency to be checked. Must be one of {DOCS}.

    Raises
    ------
    AssertionError
        If the frequency is not valid.

    Notes
    -----
    If the frequency of an index is valid, the index itself is not automatically valid.
    To check validy of an index, use the ``assert_index_valid()`` function.
    """
    if freq is None:
        raise AssertionError("Frequency may not be None.")

    elif isinstance(freq, str):
        try:
            freq = to_offset(freq)
        except ValueError as e:
            raise AssertionError(
                "Frequency string cannot be turned into a pandas BaseOffset object."
            ) from e

    if freq not in _FREQUENCIES:
        raise AssertionError(f"Frequency must be one of {DOCS}.")


def accept_freqstr(*params) -> Callable:
    """Create decorator that ensures the wrapped function also accepts string-values for the
    selected arguments, and turns them into pandas.DateOffset objects."""

    # Guard clause.
    if not len(params):
        raise ValueError(
            "Provide the name(s) of the parameters that must be turned into DateOffset objects.."
        )

    def decorator(fn: Callable):
        sig = inspect.signature(fn)

        # Guard clause.
        not_found = [param for param in params if param not in sig.parameters]
        if len(not_found):
            raise ValueError(
                f"The following parameters are not part of the function's definition: {', '.join(not_found)}."
            )

        # Change type hints.
        for param in params:
            fn.__annotations__[param] = Frequencylike

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Turn into DateOffset if it is a string.
            for paramname in params:
                if isinstance(arg := bound_args.arguments[paramname], str):
                    bound_args.arguments[paramname] = to_offset(arg)

            # Execute function as normal.
            return fn(*bound_args.args, **bound_args.kwargs)

        return wrapped

    return decorator


@accept_freqstr("freq")
def is_shorter_than_daily(freq: BaseOffset) -> bool:
    if freq in _SHORTERTHANDAILY:
        return True
    elif freq in _FREQUENCIES:
        return False
    raise ValueError(f"Unsupported frequency {freq}. Use one of {DOCS} instead.")


@accept_freqstr("source_freq", "target_freq")
def up_or_down(source_freq: BaseOffset, target_freq: BaseOffset) -> int | None:
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
    * 1 if index must be upsampled to obtain target frequency. E.g. source_freq='D', target_freq='MS'.
    * 0 if source frequency is same as target frequency. E.g. source_freq='QS-JAN', target_freq='QS-APR'.
    * -1 if index must be downsampled to obtain target frequency. E.g. source_freq='MS', target_freq='D'.
    * None if resampling is not possible. E.g. source_freq='QS-JAN', targes_freq='YS-FEB'.
    """
    if target_freq in _DOWNSAMPLE_TARGETS[source_freq]:
        return -1
    elif source_freq in _DOWNSAMPLE_TARGETS[target_freq]:
        return 1
    elif target_freq in _IDENTITYRESAMPLE_TARGETS[source_freq]:
        return 0
    else:
        return None


@accept_freqstr("freq")
def to_jump(freq: BaseOffset) -> pd.Timedelta | pd.DateOffset:
    """Jump object corresponding to a frequency.

    Parameters
    ----------
    freq
        Frequency denoting the length of the delivery period.

    Returns
    -------
        Term that can be (repeatedly) added to / subtracted from a left-bound
        timestamp to find the next / previous left-bound timestamps.

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
    else:
        raise ValueError(f"Parameter ``freq`` must be one of {DOCS}; got '{freq}'.")
