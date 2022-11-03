"""
Tools for working with pandas series and dataframes.
"""

import functools
from typing import Any, Iterable, Union

import numpy as np
import pandas as pd

from . import freq as tools_freq


def set_frequency(
    fr: Union[pd.Series, pd.DataFrame], wanted: str = None, strict: bool = False
) -> Union[pd.Series, pd.DataFrame]:
    """Try to read, infer, or force frequency of frame's index.

    Parameters
    ----------
    fr : pd.Series or pd.DataFrame
    wanted : str, optional
        Frequency to set. If none provided, try to infer.
    strict : bool, optional (default: False)
        If True, raise ValueError if a valid frequency is not found.

    Returns
    -------
    Union[pd.Series, pd.DataFrame]
        Same type as ``fr``, with, if possible, a valid value for ``fr.index.freq``.
    """
    # Handle non-datetime-indices.
    if not isinstance(fr.index, pd.DatetimeIndex):
        raise ValueError(
            "The data does not have a datetime index and can therefore not have a frequency."
        )

    # Find frequency.
    fr = fr.copy()
    if fr.index.freq:
        pass
    elif wanted:
        fr.index.freq = wanted
    else:
        try:
            fr.index.freq = pd.infer_freq(fr.index)
        except ValueError:
            pass  # couldn't find one, e.g. because not enough values

    # Correct if necessary.
    freq = fr.index.freq
    if not freq and strict:  # No frequency found.
        raise ValueError("The data does not seem to have a regular frequency.")
    elif freq and freq not in tools_freq.FREQUENCIES:
        # Edge case: year-/quarterly but starting != Jan.
        if tools_freq.up_or_down(freq, "AS") == 0:
            fr.index.freq = "AS"  # will likely fail
        elif tools_freq.up_or_down(freq, "QS") == 0:
            fr.index.freq = "QS"  # will only succeed if QS-APR, QS-JUL or QS-OCT
        elif strict:
            raise ValueError(
                "The data has a non-allowed frequency. Must be one of "
                f"{', '.join(tools_freq.FREQUENCIES)}; found '{freq}'."
            )
    return fr


def fill_gaps(
    fr: Union[pd.Series, pd.DataFrame], maxgap: int = 2
) -> Union[pd.Series, pd.DataFrame]:
    """Fill gaps in series by linear interpolation.

    Parameters
    ----------
    fr : pd.Series or pd.DataFrame
    maxgap : int, optional
        Maximum number of rows that are filled. Larger gaps are kept. Default: 2.

    Returns
    -------
    pd.Series or pd.DataFrame
        with gaps filled up.
    """
    if isinstance(fr, pd.DataFrame):
        return pd.DataFrame({c: fill_gaps(s, maxgap) for c, s in fr.items()})

    is_gap = fr.isna()
    next_gap = is_gap.shift(-1)
    prev_gap = is_gap.shift(1)
    index_beforegap = fr[~is_gap & next_gap].index
    index_aftergap = fr[~is_gap & prev_gap].index
    # remove orphans at beginning and end
    if index_beforegap.empty:
        return fr
    elif index_beforegap[-1] > index_aftergap[-1]:
        index_beforegap = index_beforegap[:-1]
    if index_aftergap.empty:
        return fr
    elif index_aftergap[0] < index_beforegap[0]:
        index_aftergap = index_aftergap[1:]
    fr = fr.copy()
    for i_before, i_after in zip(index_beforegap, index_aftergap):
        section = fr.loc[i_before:i_after]
        if len(section) > maxgap + 2:
            continue
        x0, y0, x1, y1 = i_before, fr[i_before], i_after, fr[i_after]
        dx, dy = x1 - x0, y1 - y0
        fr.loc[i_before:i_after] = y0 + (section.index - x0) / dx * dy
    return fr


def add_header(df: pd.DataFrame, header: Any, axis: int = 1) -> pd.DataFrame:
    """Add additional (top-)level to dataframe axis (column or index).

    Parameters
    ----------
    df : pd.DataFrame
    header : Any
        Value to add to top or left of index.
    axis : int, optional (default: 1)

    Returns
    -------
    pd.DataFrame
    """
    return pd.concat([df], keys=[header], axis=axis)


def concat(dfs: Iterable[pd.DataFrame], axis: int = 0, *args, **kwargs) -> pd.DataFrame:
    """
    Wrapper for ``pandas.concat``; concatenate pandas objects even if they have
    unequal number of levels on concatenation axis.

    Levels containing empty strings are added from below (when concatenating along
    columns) or right (when concateniting along rows) to match the maximum number
    found in the dataframes.

    Parameters
    ----------
    dfs : Iterable
        Dataframes that must be concatenated.
    axis : int, optional
        Axis along which concatenation must take place. The default is 0.

    Returns
    -------
    pd.DataFrame
        Concatenated Dataframe.

    Notes
    -----
    Any arguments and kwarguments are passed onto the ``pandas.concat`` function.

    See also
    --------
    pandas.concat
    """

    def index(df):
        return df.columns if axis == 1 else df.index

    def add_levels(df):
        need = want - index(df).nlevels
        if need > 0:
            df = pd.concat([df], keys=[("",) * need], axis=axis)  # prepend empty levels
            for i in range(want - need):  # move empty levels to bottom
                df = df.swaplevel(i, i + need, axis=axis)
        return df

    want = np.max([index(df).nlevels for df in dfs])
    dfs = [add_levels(df) for df in dfs]
    return pd.concat(dfs, axis=axis, *args, **kwargs)


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
    s2_vals = s2.astype(f"pint[{s1.pint.u:P}]").pint.m
    return np.allclose(s1_vals, s2_vals, *args, **kwargs)
