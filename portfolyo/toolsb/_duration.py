"""Functionality to calculate the duration of (a) delivery period(s)."""

import pandas as pd
from pint import Quantity

from . import _lefttoright as tools_lefttoright
from . import freq as tools_freq
from . import index as tools_index
from . import unit as tools_unit
from .types import PintSeries


@tools_freq.check()
def stamp(stamp: pd.Timestamp, freq: pd.DateOffset) -> Quantity:
    f"""Duration of a delivery period.

    Parameters
    ----------
    stamp
        Left-bound (i.e., start) timestamp of delivery period for which to calculate the duration.
    freq : {tools_freq.ALLOWED_FREQUENCIES_DOCS}
        Frequency of delivery period.

    Returns
    -------
        Duration.

    Example
    -------
    >>> duration(pd.Timestamp('2020-04-21'), 'D')
    24.0 h
    >>> duration(pd.Timestamp('2020-03-29'), 'D')
    24.0 h
    >>> duration(pd.Timestamp('2020-03-29', tz='Europe/Berlin'), 'D')
    23.0 h
    """
    jump = tools_lefttoright.jump(freq)

    if isinstance(jump, pd.Timedelta):
        tdelta = jump  # one timedelta
    else:
        assert freq.is_on_offset(stamp)
        tdelta = (stamp + jump) - stamp  # one timedelta
    hours = tdelta.total_seconds() / 3600
    return tools_unit.Q_(hours, "h")


@tools_index.check()
def index(idx: pd.DatetimeIndex) -> PintSeries:
    """Duration of the delivery periods in a datetime index.

    Parameters
    ----------
    idx
        Index for which to calculate the durations.

    Returns
    -------
        Series, with ``idx`` as index and durations as values.
    """
    jump = tools_lefttoright.jump(idx.freq)
    if isinstance(jump, pd.Timedelta):
        tdelta = jump  # one timedelta
    else:
        tdelta = (idx + jump) - idx  # timedeltaindex
    hours = tdelta.total_seconds() / 3600  # one value or index
    return pd.Series(hours, idx, dtype="pint[h]")
