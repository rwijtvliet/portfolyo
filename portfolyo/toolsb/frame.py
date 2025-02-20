"""Tools for working with pandas series and dataframes."""

import functools
from typing import Any, Iterable, overload

import numpy as np
import pandas as pd

from portfolyo.tools.types import Series_or_DataFrame

from . import freq as tools_freq
from . import index as tools_index


def add_header(
    frame: pd.Series | pd.DataFrame, header: Any, axis: int = 1
) -> pd.Series | pd.DataFrame:
    """Add additional (top-)level to dataframe axis (column or index).

    Parameters
    ----------
    frame
        Series or dataframe to add header to.
    header
        Value to add as uniform top-level column or index value.
    axis, optional (default: 1)
        Which axis to add the level to, columns (1) or index (0)

    Returns
    -------
        Same series or dataframe, but with additional level. Series changed to
        dataframe if axis == 1.
    """
    return pd.concat([frame], keys=[header], axis=axis)


def concat(
    frames: Iterable[pd.Series | pd.DataFrame], axis: int = 0, *args, **kwargs
) -> pd.Series | pd.DataFrame:
    """
    Wrapper for ``pandas.concat``; concatenate pandas objects even if they have
    unequal number of levels on concatenation axis.

    Levels containing empty strings are added from below (when concatenating along
    columns) or right (when concateniting along rows) to match the maximum number
    found in the dataframes.

    Parameters
    ----------
    frames
        Series or dataframes that must be concatenated.
    axis, optional (default: 0)
        Axis along which concatenation must take place.

    Returns
    -------
        Concatenated Series or Dataframe.

    Notes
    -----
    Any arguments and kwarguments are passed onto the ``pandas.concat`` function.

    See also
    --------
    pandas.concat
    """

    def nlevels(fr):
        if axis == 1:
            return 0 if isinstance(fr, pd.Series) else fr.columns.nlevels
        else:
            return fr.index.nlevels

    def add_levels(fr, want: int):
        for _ in range(want - nlevels(fr)):
            fr = add_header(fr, "", axis=axis)  # prepend empty level
            kwargs = {} if isinstance(fr, pd.Series) else {"axis": axis}
            fr = fr.swaplevel(0, -1, **kwargs)  # move empty levels to bottom
        return fr

    want = np.max([nlevels(fr) for fr in frames])
    frames = [add_levels(fr, want) for fr in frames]
    return pd.concat(frames, axis=axis, *args, **kwargs)


@functools.wraps(np.allclose)
def series_allclose(s1: pd.Series, s2: pd.Series, *args, **kwargs) -> bool:
    """Compare if all values in series are equal/close. Works with series that have units."""
    without_units = [not (hasattr(s, "pint")) for s in [s1, s2]]
    if all(without_units):
        return np.allclose(s1, s2, *args, **kwargs)
    elif any(without_units):
        return False
    elif s1.pint.dimensionality != s2.pint.dimensionality:
        return False
    # Both have units, and both have same dimensionality (e.g. 'length'). Check values.
    s1_vals = s1.pint.m
    s2_vals = s2.pint.to(s1.pint.u).pint.m
    return np.allclose(s1_vals, s2_vals, *args, **kwargs)


@overload
def trim(frame: pd.Series, freq: str) -> pd.Series:
    ...


@overload
def trim(frame: pd.DataFrame, freq: str) -> pd.DataFrame:
    ...


def trim(frame: pd.Series | pd.DataFrame, freq: str) -> pd.Series | pd.DataFrame:
    f"""Trim index of series or dataframe to only keep full periods of certain frequency.

    Parameters
    ----------
    frame
        The series or dataframe to trim.
    freq : {tools_freq.ALLOWED_FREQUENCIES_DOCS}
        Delivery period frequency to trim to. E.g. 'MS' to only keep full months.

    Returns
    -------
        Subset of ``frame``, with same frequency.
    """
    return frame.loc[tools_index.trim(frame.index, freq)]


def intersect(*frames: Series_or_DataFrame) -> tuple[Series_or_DataFrame, ...]:
    """Intersect several dataframes and/or series.

    Parameters
    ----------
    *frames
        The series or dataframes to intersect.

    Returns
    -------
        As input, but trimmed to the intersection of their indices.

    Notes
    -----
    The frames' indices must have equivalent frequencies, equal timezones and equal
    start-of-day. Otherwise, an error is raised. If there is no overlap, empty
    frames are returned.
    """
    intersected_idx = tools_index.intersect(*(frame.index for frame in frames))
    return tuple([frame.loc[intersected_idx] for frame in frames])


def intersect_flex(
    *frames: Series_or_DataFrame,
    ignore_freq: bool = False,
    ignore_tz: bool = False,
    ignore_startofday: bool = False,
) -> tuple[Series_or_DataFrame, ...]:
    """Intersect several datetime indices, but allow for more flexibility of ignoring
    certain properties.

    Parameters
    ----------
    *frames
        The series or dataframes to intersect.
    ignore_freq, optional (default: False)
        If True, do the intersection even if the frequencies do not match; drop the
        time periods that do not (fully) exist in either of the frames. The frequencies
        of the original indices are preserved.
    ignore_tz, optional (default: False)
        If True, ignore the timezones; perform the intersection using 'wall time'. The
        timezones of the original indices are preserved.
    ignore_startofday, optional (default: False)
        If True, do the intersection even if the indices have a different start-of-day.
        The start-of-day of the original indices are preserved (even if the frequency is
        shorter than daily).

    Returns
    -------
        As input, but trimmed to the intersection of their indices.

    See also
    --------
    .intersect()
    """
    intersected_idxs = tools_index.intersect_flex(
        *(frame.index for frame in frames),
        ignore_freq=ignore_freq,
        ignore_tz=ignore_tz,
        ignore_startofday=ignore_startofday,
    )
    return tuple([frame.loc[idx] for frame, idx in zip(frames, intersected_idxs)])
