"""
Duration of delivery periods.
"""


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
    freq : {{{', '.join(tools_freq.FREQUENCIES)}}}
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
    if freq in ["15T", "H"]:
        h = 1 if freq == "H" else 0.25
    else:
        h = (tools_right.stamp(ts, freq) - ts).total_seconds() / 3600
    return tools_unit.Q_(h, "h")


def index(i: pd.DatetimeIndex, freq: str = None) -> pd.Series:
    f"""Duration of a timestamp.

    Parameters
    ----------
    i : pd.DatetimeIndex
        Index for which to calculate the duration.
    freq : {{{', '.join(tools_freq.FREQUENCIES)}}}, optional
        Frequency to use in determining the duration.
        If none specified, use ``.freq`` attribute of ``i``.

    Returns
    -------
    pint-Series
        With ``i`` as its index, and the corresponding duration as the values.
    """
    freq = freq or i.freq

    if freq in ["15T", "H"]:
        # Speed-up things for fixed-duration frequencies.
        h = 1.0 if freq == "H" else 0.25
        hours = pd.Series(h, i)
    else:
        # Individual calculations for non-fixed-duration frequencies.
        hours = (tools_right.index(i, freq) - i).apply(
            lambda td: td.total_seconds() / 3600
        )
    return hours.astype("pint[h]").rename("duration")
