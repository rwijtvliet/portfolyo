"""
Tools for dealing with frequencies.
"""

from typing import Callable
import functools
import pandas as pd
import inspect
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import BaseOffset

# Developer notes:
#
# We try to support a large number of frequencies, but there are some that we focus on due to their prevalence in the energy industry. We call these the "common" frequencies:
# * Daily or longer: day (D), months (MS), quarters starting in any month (QS-JAN, QS-FEB, ...), years starting in any month (YS-JAN, YS-FEB, ...)
# * Shorter-than-daily: hour (h) and any number of minutes that evenly divides an hour, and any number of seconds that evenly divides a minute.
#
# For the shorter-than-daily frequencies, there is an additional requirement for the index, namely, that they end on a day boundary and on an hour boundary. E.g., '15min' is only a "common" frequency if the timestamps are at 0, 15, 30, and 45 min past the hour. A timeseries with an index with timestamps at 3, 18, 33, and 48 min past the hour is not considered to be "common".
#
# Resampling between 2 frequencies is not always possible, e.g., to resample from QS-FEB (quarters starting in Feb) to YS-JAN (years starting in Jan). Direct resampling is not possible, because neither 'neatly fits' inside the other. To still resample, we must take a detour by first resampling to a shorter frequency that neatly divides BOTH, e.g. MS (monthly). This is must be done manually by the user.
# This situation can occur when using common frequencies (as in the example, QS-FEB vs YS-JAN) and is evenmore likely when using uncommon ones (e.g. weekly).
#
#
# We are using sets and dictionaries here, to allow for fast membership testing.
#
#
# For quarters, we have the situation that several frequencies map 1-to-1 onto each other. For example, 'QS-FEB' describes a quarterly series where one of the quarters starts with February.
# * This means another of its quarters starts in May, and therefore, this frequency is the same as 'QS-MAY'. To keep track of this, we create 'qgroups' a, b, and c; each month belongs to exactly one qgroup, and one qgroup belongs to 4 months.
# * This also means that downsampling from quarters to years is possible to 4 distinct yearly frequencies (e.g. 'QS-FEB' can be upsampled to 'YS-FEB', but also 'YS-MAY', which has different periods and different values). And that upsampling from years to quarters is also possible to 4 quarterly frequencies - though these all result in the same timeseries (e.g. 'YS-FEB' can be downsampled to 'QS-FEB', but also to 'QS-MAY', though this has the same periods and the same values).

QGROUPS: dict[str, set[str]] = {  # qgroup: months
    "a": {"JAN", "APR", "JUL", "OCT"},
    "b": {"FEB", "MAY", "AUG", "NOV"},
    "c": {"MAR", "JUN", "SEP", "DEC"},
}
MONTHS: dict[str, str] = {  # month: qgroup
    month: cycle for cycle, months in QGROUPS.items() for month in months
}


# Some common frequencies we will likely encounter.

DAILYORLONGER_FREQUENCIES: set[BaseOffset] = {
    to_offset(freq)
    for freq in ("D", "MS", *(f"{f}-{m}" for f in ["QS", "YS"] for m in MONTHS))
}
SHORTERTHANDAILY_FREQUENCIES: set[BaseOffset] = {
    to_offset(freq) for freq in ("min", "5min", "15min", "30min", "h")
}
FREQUENCIES: set[BaseOffset] = {
    *SHORTERTHANDAILY_FREQUENCIES,
    to_offset("D"),
    to_offset("MS"),
    *(to_offset(f"QS-{m}") for m in MONTHS),
    *(to_offset(f"YS-{m}") for m in MONTHS),
}

# For those common frequencies, create maps to quickly find out which resampling is possible.

_DOWNSAMPLE_TARGETS: dict[BaseOffset, set[BaseOffset]] = {  # source: targets
    to_offset(source): {to_offset(target) for target in targets}
    for source, targets in {
        "min": {"5min"},
        "5min": {"15min"},
        "15min": {"30min"},
        "30min": {"h"},
        "h": {"D"},
        "D": {"MS"},
        "MS": {f"QS-{m}" for m in MONTHS},
        **{f"QS-{m}": {f"YS-{m2}" for m2 in QGROUPS[qg]} for m, qg in MONTHS.items()},
        **{f"YS-{m}": set() for m in MONTHS},
    }.items()
}
DOWNSAMPLE_TARGETS: dict[BaseOffset, set[BaseOffset]] = {}  # source: targets
for source, known_targets in reversed(_DOWNSAMPLE_TARGETS.items()):
    additional_targets = {t2 for t1 in known_targets for t2 in DOWNSAMPLE_TARGETS[t1]}
    DOWNSAMPLE_TARGETS[source] = known_targets | additional_targets

IDENTITYRESAMPLE_TARGETS: dict[BaseOffset, set[BaseOffset]] = {
    to_offset(f"QS-{m}"): {to_offset(f"QS-{m2}") for m2 in QGROUPS[qg] if m2 != m}
    for m, qg in MONTHS.items()
}


# Allowed frequencies.
ALLOWED_FREQUENCIES_DOCS = "'min', '5min', 15min' (=quarterhour), '30min', 'h', 'D', 'MS', 'QS' (or 'QS-FEB', 'QS-MAR', etc.), or 'YS' (or 'YS-FEB', 'YS-MAR', etc.)"


def ensure_dateoffset(*parameters) -> Callable:
    """Create decorator that ensures the wrapped function also accepts string-values for the
    selected arguments, and turns them into pandas.DateOffset objects."""

    # Guard clause.
    if not len(parameters):
        raise ValueError(
            "Provide the name(s) of the parameters that must be turned into DateOffset objects.."
        )

    def decorator(fn):
        sig = inspect.signature(fn)

        # Guard clause.
        not_found = [
            parameter for parameter in parameters if parameter not in sig.parameters
        ]
        if len(not_found):
            raise ValueError(
                f"The following parameters are not part of the function's definition: {', '.join(not_found)}."
            )

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Turn into DateOffset if it is a string.
            for paramname in parameters:
                if isinstance(value := bound_args.arguments[paramname], str):
                    bound_args.arguments[paramname] = to_offset(value)

            # Execute function as normal.
            return fn(*bound_args.args, **bound_args.kwargs)

        return wrapped

    return decorator


def assert_index_valid(i: pd.DatetimeIndex) -> None:
    f"""Validate if the given index has the necessary properties to be used in portfolio lines.

    Parameters
    ----------
    i
        Index to be checked. Frequency must be valid (one of {ALLOWED_FREQUENCIES_DOCS}), and
        for shorter-than-daily frequencies, the index must contain entire days and start at a full
        hour.

    Raises
    ------
    AssertionError
        If the index is not valid.
    """
    assert i.freq is not None, "Frequency must be set."
    assert (
        i.freq in FREQUENCIES
    ), f"Frequency must be one of {ALLOWED_FREQUENCIES_DOCS}."

    start_of_day = i[0].time()
    at_full_hour = start_of_day.minute == start_of_day.second == 0
    assert at_full_hour, "Days must start on the full hour."

    if i.freq in SHORTERTHANDAILY_FREQUENCIES:
        fulldays = (i[-1] + i.freq).time() == start_of_day
        assert (
            fulldays
        ), "Index must contain an integer number of days, i.e., the end time of the final period must equal the start time of the first period."


@ensure_dateoffset("target_freq")
def up_or_down(i: pd.DatetimeIndex, target_freq: BaseOffset) -> int | None:
    """See if changing the frequency of an index requires up- or downsampling.

    Upsampling means that the number of values increases - one value in the source
    corresponds to multiple values in the target.

    Parameters
    ----------
    i
        Index to be resampled.
    target_freq
        Frequency to resample it to.

    Returns
    -------
    * 1 if index must be upsampled to obtain target frequency. E.g. index.freq='D', target_freq='MS'.
    * 0 if index frequency is same as target frequency. E.g. index.freq='QS-JAN', target_freq='QS-APR'.
    * -1 if index must be downsampled to obtain target frequency. E.g. index.freq='MS', target_freq='D'.
    * None if resampling is not possible. E.g. index.freq='QS-JAN', targes_freq='QS-FEB'.
    """
    assert_index_valid(i)

    if target_freq in DOWNSAMPLE_TARGETS[i.freq]:
        return -1
    elif i.freq in DOWNSAMPLE_TARGETS[target_freq]:
        return 1
    elif i.freq == target_freq:
        return 0
    elif target_freq in DOWNSAMPLE_TARGETS.get(i.freq, {}):
        return 0
    else:
        return None
