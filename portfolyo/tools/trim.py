"""
Trim objects to only contain 'full' delivery periods.
"""

from typing import Union

import pandas as pd

from . import ceil as tools_ceil
from . import floor as tools_floor
from . import freq as tools_freq
from . import right as tools_right


def index(i: pd.DatetimeIndex, freq: str, offset_hours: int = 0) -> pd.DatetimeIndex:
    f"""Trim index to only keep full periods of certain frequency.

    Parameters
    ----------
    i : pd.DatetimeIndex
        The (untrimmed) DatetimeIndex
    freq : {{{', '.join(tools_freq.FREQUENCIES)}}}
        Frequency to trim to. E.g. 'MS' to only keep full months.
    offset_hours : int, optional (default: 0)
        Offset of delivery period compared to midnight, in hours. E.g. 6 if delivery
        periods start at 06:00:00.
        Used only when trimming a below-daily index to a daily-or-longer frequency.

    Returns
    -------
    pd.DatetimeIndex
        Subset of ``i``, with same frequency.
    """
    if not i.freq:
        raise ValueError("Index ``i`` does not have a frequency.")
    mask_start = i >= tools_ceil.stamp(i[0], freq, 0, offset_hours)
    mask_end = tools_right.index(i) <= tools_floor.stamp(i[-1], freq, 0, offset_hours)
    return i[mask_start & mask_end]


def frame(
    fr: Union[pd.Series, pd.DataFrame], freq: str, offset_hours: int = 0
) -> Union[pd.Series, pd.DataFrame]:
    f"""Trim index of series or dataframe to only keep full periods of certain frequency.

    Parameters
    ----------
    fr : Union[pd.Series, pd.DataFrame]
        The (untrimmed) pandas series or dataframe.
    freq : {{{', '.join(tools_freq.FREQUENCIES)}}}
        Frequency to trim to. E.g. 'MS' to only keep full months.
    offset_hours : int, optional (default: 0)
        Offset of delivery period compared to midnight, in hours. E.g. 6 if delivery
        periods start at 06:00:00.
        Used only when trimming a below-daily frame to a daily-or-longer frequency.

    Returns
    -------
    Union[pd.Series, pd.DataFrame]
        Subset of ``fr``, with same frequency.
    """
    i = index(fr.index, freq, offset_hours)
    return fr.loc[i]
