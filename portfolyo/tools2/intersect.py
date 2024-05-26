from typing import List

from ..tools import intersect as tools_intersect
from .types import Indexable


def indexable(
    *objs: Indexable,
    ignore_freq: bool = False,
    ignore_tz: bool = False,
    ignore_start_of_day: bool = False,
) -> List[Indexable]:
    """Intersect several dataframes and/or series.

    Parameters
    ----------
    *objs : pd.Series and/or pd.DataFrame and/or PfLines and/or PfStates
        The indexable objects to intersect.
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
    As input, but trimmed to their intersection.

    Notes
    -----
    The indices must have equal frequency, timezone, start-of-day. Otherwise, an error
    is raised. If there is no overlap, empty frames are returned.
    """
    new_idxs = tools_intersect.indices_flex(
        *[o.index for o in objs],
        ignore_freq=ignore_freq,
        ignore_tz=ignore_tz,
        ignore_start_of_day=ignore_start_of_day,
    )
    return [o.loc[i] for i, o in zip(new_idxs, objs)]
