from typing import List, Union

import pandas as pd

from ..core.pfline import PfLine
from ..core.pfstate import PfState
from ..tools.intersect import indices_flex


def indexable(
    *frames: Union[pd.Series, pd.DataFrame, PfLine, PfState],
    ignore_freq: bool = False,
    ignore_tz: bool = False,
    ignore_start_of_day: bool = False,
) -> List[Union[pd.Series, pd.DataFrame, PfLine, PfState]]:
    """Intersect several dataframes and/or series.

    Parameters
    ----------
    *frames : pd.Series and/or pd.DataFrame and/or PfLines and/or PfStates
        The frames to intersect.
    ignore_freq: bool, optional (default: False)
        If True, do the intersection even if the frequencies do not match; drop the
        time periods that do not (fully) exist in either of the frames.
    ignore_tz: bool, optional (default: False)
        If True, ignore the timezones; perform the intersection using 'wall time'.
    ignore_start_of_day: bool, optional (default: False)
        If True, perform the intersection even if the frames have a different start-of-day.
        The start-of-day of the original frames is preserved, even if the frequency is shorter
        than daily.

    Returns
    -------
    list of series and/or dataframes
        As input, but trimmed to their intersection.

    Notes
    -----
    The indices must have equal frequency, timezone, start-of-day. Otherwise, an error
    is raised. If there is no overlap, empty frames are returned.
    """
    new_idxs = indices_flex(
        *[fr.index for fr in frames],
        ignore_freq=ignore_freq,
        ignore_tz=ignore_tz,
        ignore_start_of_day=ignore_start_of_day,
    )
    return [fr.loc[idx] for idx, fr in zip(new_idxs, frames)]
