"""Duration of delivery periods."""

import pandas as pd

from . import freq as tools_freq
from . import right as tools_right
from . import unit as tools_unit


def stamp(ts: pd.Timestamp, freq: str) -> tools_unit.Q_:
    f"""Duration of a timestamp.

    Parameters
    ----------
    ts : pd.Timestamp
        Timestamp for which to calculate the duration.
    freq : {tools_freq.ALLOWED_FREQUENCIES_DOCS}
        Frequency to use in determining the duration.

    Returns
    -------
    pint Quantity.

    Example
    -------
    >>> duration.stamp(pd.Timestamp('2020-04-21'), 'D')
    24.0 h
    >>> duration.stamp(pd.Timestamp('2020-03-29'), 'D')
    24.0 h
    >>> duration.stamp(pd.Timestamp('2020-03-29', tz='Europe/Berlin'), 'D')
    23.0 h
    """
    if freq in ["15min", "h"]:
        h = 1.0 if freq == "h" else 0.25
    else:
        h = (tools_right.stamp(ts, freq) - ts).total_seconds() / 3600
    return tools_unit.Q_(h, "h")


def index(i: pd.DatetimeIndex) -> pd.Series:
    """Duration of the timestamps in an index.

    Parameters
    ----------
    i : DatetimeIndex
        Index for which to calculate the duration.

    Returns
    -------
    pint-Series
        With ``i`` as its index, and the corresponding duration as the values.
    """
    if i.freq in ["15min", "h"]:
        # Speed-up things for fixed-duration frequencies.
        h = 1.0 if i.freq == "h" else 0.25
    else:
        # Individual calculations for non-fixed-duration frequencies.
        h = (tools_right.index(i) - i).map(lambda td: td.total_seconds() / 3600)

    return pd.Series(h, i, dtype="pint[h]").rename("duration")


def frame(fr: pd.Series | pd.DataFrame) -> pd.Series:
    """Duration of the timestamps in the index of a Series or DataFrame.

    Parameters
    ----------
    fr : Series or DataFrame
        Object with index for which to calculate the duration.

    Returns
    -------
    pint-Series
        With ``i`` as its index, and the corresponding duration as the values.
    """
    return index(fr.index)
