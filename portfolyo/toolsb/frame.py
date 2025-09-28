"""Tools for working with pandas series and dataframes."""

import functools
from typing import Any, Iterable, overload

import numpy as np
import pandas as pd
import pint_pandas

from . import index as tools_index
from .types import Frequencylike, Series_or_Dataframe, TimeSeries_or_TimeDataframe

# Conversion and validation.
# --------------------------


def validate_timeframe(fr: pd.Series | pd.DataFrame) -> None:
    tools_index.validate(fr.index)


def coerce_timeframe(fr: Series_or_Dataframe) -> Series_or_Dataframe:
    fr = fr.set_axis(pd.DatetimeIndex(fr.index), axis=0)
    validate_timeframe(fr)
    return fr


def validate_pintframe(fr: pd.Series | pd.DataFrame) -> None:
    if isinstance(fr, pd.DataFrame):
        for _, s in fr.items():
            validate_timeframe(s)

    # If we are here, fr is a series.
    if not isinstance(fr.dtype, pint_pandas.PintType):
        raise ValueError(f"This is not a pintseries: {fr}.")


def coerce_pintframe(fr: Series_or_Dataframe) -> Series_or_Dataframe:
    if isinstance(fr, pd.DataFrame):
        return pd.DataFrame({col: s for col, s in fr.items()})

    # If we are here, fr is a series.
    if pd.api.types.is_integer_dtype(fr.dtype):
        fr = fr.astype(float).astype("pint[]")
    elif pd.api.types.is_float_dtype(fr.dtype):
        fr = fr.astype("pint[]")
    elif pd.api.types.is_object_dtype(fr.dtype) and isinstance(fr.iloc[0], pint.Quantity):
        units = fr.iloc[0].units
        fr = fr.astype(f"pint[{units}]")  # works if all quantities have same dimension

    validate_pintframe(fr)
    return fr


def coerce_pinttimeframe(fr: Series_or_Dataframe) -> Series_or_Dataframe:
    return coerce_timeframe(coerce_pintframe(fr))


# --------------------------


def _nlevels(fr: pd.Series | pd.DataFrame, axis: int) -> int:
    """Count levels on specified axis."""
    if axis == 0:
        return fr.index.nlevels
    elif isinstance(fr, pd.DataFrame):  # axis == 1
        return fr.columns.nlevels
    else:  # axis == 1 and fr is Series
        return 0


def _add_level(
    fr: pd.Series | pd.DataFrame, levelvalue: Any, axis: int, top: bool = True
) -> pd.Series | pd.DataFrame:
    """Add a level with specified value to top or bottom of specified axis."""
    fr = pd.concat({levelvalue: fr}, axis=axis)  # add (prepend) level. Might turn Series into df
    if top or _nlevels(fr, axis) < 2:  # no need to swap, or impossible to swap
        return fr
    elif isinstance(fr, pd.Series):
        return fr.swaplevel(0, 1)  # move to bottom
    else:  # DataFrame
        return fr.swaplevel(0, 1, axis=axis)  # move to bottom


def _ensure_nlevels(fr: pd.Series | pd.DataFrame, axis: int, want: int) -> pd.Series | pd.DataFrame:
    """Add levels from below/right to reach wanted level count on specified axis."""
    for _ in range(want - _nlevels(fr, axis)):
        fr = _add_level(fr, "", axis=axis, top=False)  # prepend empty level
    return fr


def add_header(
    fr: pd.Series | pd.DataFrame, header: Any, axis: int = 1
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
        Same series or dataframe, but with additional level. Series changed to dataframe if axis == 1.
    """
    return _add_level(fr, header, axis)


@overload
def concat(frames: Iterable[pd.Series], *, axis: int = 0, **kwargs) -> pd.Series | pd.DataFrame:
    # Iterable of Series returns Series if axis==0 and Dataframe if axis==1.
    ...


@overload
def concat(frames: Iterable[pd.Series | pd.DataFrame], *, axis: int = 0, **kwargs) -> pd.DataFrame:
    # If one of iterables is dataframe, returns Dataframe.
    ...


@functools.wraps(pd.concat)
def concat(
    frames: Iterable[pd.Series | pd.DataFrame], *, axis: int = 0, **kwargs
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
    Any kwargs are passed onto the ``pandas.concat`` function.

    See also
    --------
    pandas.concat
    """
    want = np.max([_nlevels(fr, axis) for fr in frames])
    frames = [_ensure_nlevels(fr, axis, want) for fr in frames]
    return pd.concat(frames, axis=axis, **kwargs)


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


def trim(fr: TimeSeries_or_TimeDataframe, freq: Frequencylike) -> TimeSeries_or_TimeDataframe:
    """Trim index of series or dataframe to only keep full periods of certain frequency.

    Parameters
    ----------
    fr
        The series or dataframe to trim.
    freq
        Delivery period frequency to trim to. E.g. 'MS' to only keep full months.

    Returns
    -------
        Subset of ``fr``, with same frequency.
    """
    return fr.loc[tools_index.trim(fr.index, freq)]


def intersect(
    frs: Iterable[TimeSeries_or_TimeDataframe],
) -> tuple[TimeSeries_or_TimeDataframe, ...]:
    """Intersect several dataframes and/or series.

    Parameters
    ----------
    frs
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
    intersected_idx = tools_index.intersect(fr.index for fr in frs)
    return tuple([fr.loc[intersected_idx] for fr in frs])


def intersect_flex(
    frs: TimeSeries_or_TimeDataframe,
    *,
    ignore_freq: bool = False,
    ignore_tz: bool = False,
    ignore_startofday: bool = False,
) -> tuple[TimeSeries_or_TimeDataframe, ...]:
    """Intersect several datetime indices, but allow for more flexibility of ignoring certain
    properties.

    Parameters
    ----------
    frs
        Series or dataframes to intersect.
    ignore_freq, optional (default: False)
        If True, do intersection even if frequencies are not equivalent; drop time periods that do
        not (fully) exist in either of the frame indices. The frequencies of original frames are
        preserved. If frequencies are incompatible, an error is raised.
    ignore_tz, optional (default: False)
        If True, ignore timezones; perform intersection using 'wall time'. The timezones of original
        frames are preserved.
    ignore_startofday, optional (default: False)
        If True, do intersection even if frame indices have a different start-of-day. The
        start-of-day of original frames are preserved (even if frequency is shorter than daily).

    Returns
    -------
        As input, but trimmed to the intersection of their indices.

    See also
    --------
    .intersect()
    """
    intersected_idxs = tools_index.intersect_flex(
        (fr.index for fr in frs),
        ignore_freq=ignore_freq,
        ignore_tz=ignore_tz,
        ignore_startofday=ignore_startofday,
    )
    return tuple([fr.loc[idx] for fr, idx in zip(frs, intersected_idxs)])
