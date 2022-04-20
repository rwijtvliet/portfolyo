"""Module with tools to modify and standardize dataframes."""

from . import stamps, zones
from pandas.core.frame import NDFrame
from typing import Iterable, Union, Any
import pandas as pd
import numpy as np
import functools

# TODO: rename 'standardize'


def set_ts_index(
    fr: NDFrame,
    column: str = None,
    bound: str = "left",
    tz_in: str = None,
    continuous: bool = True,
) -> NDFrame:
    """
    Create and add a standardized, continuous timestamp index to a dataframe.

    Parameters
    ----------
    fr : NDFrame
        Pandas series or dataframe.
    column : str, optional
        Column to create the timestamp from. Used only if `fr` is DataFrame; ignored
        otherwise. Use existing index if none specified.
    bound : str, {'left' (default), 'right'}
        If 'left' ('right'), specifies that input timestamps are left-(right-)bound.
    tz_in : str
        Timezone to assume for input dataframe.
        Only relevant for tz-naive input dataframes, that have the local time in a location
        that has DST-transitions.
    continuous : bool, optional (default: True)
        If true, raise error if data has missing values (i.e., gaps).

    Returns
    -------
    NDFrame
        Same type as `fr`, with left-bound timestamp index.
        Column `column` (if applicable) is removed, and index renamed to 'ts_left'.
    """
    # TODO: Move to documentation
    # """
    # Notes
    # -----
    # When only working with non-localized timestamps - meaning, the timezone is not
    # relevant, and every day has 24h (there is no daylight-savings-time) - then ``tz_in``
    # and ``tz_out`` must be set to None (= their default values).
    # When working with localized timestamps, the input data may or may not be localized:
    # - If the input data is localized, use the ``tz_out`` parameter to convert the time-
    # stamps to the correct timezone. The ``tz_in`` parameter is disregarded.
    # - If the input data is not localized, but rather contains local timestamps without
    # timezone information, use the ``tz_in`` parameter to specify in which timezone they
    # should be assumed to take place. If not specified, it is assumed that ``tz_in`` ==
    # ``tz_out``.
    # """
    if column and isinstance(fr, pd.DataFrame):
        fr = fr.set_index(column)
    else:
        fr = fr.copy()  # don't change passed-in fr

    # Make leftbound.

    i = pd.DatetimeIndex(fr.index)  # turn / try to turn into datetime
    if bound == "right":
        i = stamps.make_leftbound(i)
    i.name = "ts_left"
    fr.index = i

    # Set Europe/Berlin timezone and set frequency.
    fr = zones.force_tzaware(fr, "Europe/Berlin", tz_in=tz_in)

    # Check frequency.

    if fr.index.freq is None:
        freq = stamps.guess_frequency((i[1:] - i[:-1]).median())
        fr2 = fr.resample(freq).asfreq()
        # If the new dataframe has additional rows, the original dataframe was not gapless.
        if continuous and len(fr2) > len(fr):
            missing = [i for i in fr2.index if i not in fr.index]
            raise ValueError(
                f"The index of ``fr`` does not have continuous data; missing data for: {missing}."
            )
        fr = fr2

    if fr.index.freq not in stamps.FREQUENCIES:
        for freq2 in ["MS", "QS"]:  # Edge case: month-/quarterly but starting != Jan.
            if stamps.freq_up_or_down(fr.index.freq, freq2) == 0:
                fr.index.freq = freq2
                break
        else:
            raise ValueError(
                f"The data has a non-allowed frequency. Must be one of {', '.join(stamps.FREQUENCIES)}; found {fr.index.freq}."
            )

    # Check boundaries.

    stamps.assert_boundary_ts(fr.index, fr.index.freq)

    # if bound == "left":
    #     pass
    # elif bound.startswith("right"):
    #     if bound == "right":
    #         # At start of DST:
    #         # . Rightbound timestamps (a) contain 3:00 but not 2:00, or (b) vice versa. Try both.
    #         # . (Leftbound timestamps contain 3:00 but not 2:00.)
    #         # At end of DST:
    #         # . Rightbound timestamps contain two timestamps (a) 2:00 or (b) 3:00. Try both.
    #         # . (Leftbound timestamps contain two timestamps 2:00.)
    #         try:
    #             return set_ts_index(fr, None, "right_a", tz_in)
    #         except (NonExistentTimeError, AmbiguousTimeError):
    #             return set_ts_index(fr, None, "right_b", tz_in)
    #     minutes = (fr.index[1] - fr.index[0]).seconds / 60
    #     if bound == "right_a":
    #         fr.loc[fr.index[0] + pd.Timedelta(minutes=-minutes)] = np.nan
    #         fr = pd.concat([fr.iloc[-1:], fr.iloc[:-1]]).shift(-1).dropna()
    #     if bound == "right_b":
    #         fr.index += pd.Timedelta(minutes=-minutes)
    # else:
    #     raise ValueError(
    #         f"Parameter ``bound`` must be 'left' or 'right'; got '{bound}'."
    #     )
    # fr.index.name = "ts_left"

    # # Set frequency.

    # if fr.index.freq is None:
    #     fr.index.freq = pd.infer_freq(fr.index)
    # if fr.index.freq is None:
    #     # (infer_freq does not always work, e.g. during summer-to-wintertime changeover)
    #     tdelta = (fr.index[1:] - fr.index[:-1]).median()
    #     if pd.Timedelta(hours=23) <= tdelta <= pd.Timedelta(hours=25):
    #         freq = "D"
    #     elif pd.Timedelta(days=27) <= tdelta <= pd.Timedelta(days=32):
    #         freq = "MS"
    #     elif pd.Timedelta(days=89) <= tdelta <= pd.Timedelta(days=93):
    #         freq = "QS"
    #     elif pd.Timedelta(days=364) <= tdelta <= pd.Timedelta(days=367):
    #         freq = "AS"
    #     else:
    #         freq = tdelta
    #     fr2 = fr.resample(freq).asfreq()
    #     # If the new dataframe has additional rows, the original dataframe was not gapless.
    #     if continuous and len(fr2) > len(fr):
    #         missing = [i for i in fr2.index if i not in fr.index]
    #         raise ValueError(
    #             f"`fr` does not have continuous data; missing data for: {missing}."
    #         )
    #     fr = fr2

    # # Check if frequency all ok.

    # if fr.index.freq is None:
    #     raise ValueError("Cannot find a frequency in `fr`.")
    # elif fr.index.freq not in stamps.FREQUENCIES:
    #     for freq in ["MS", "QS"]:  # Edge case: month-/quarterly but starting != Jan.
    #         if freq_up_or_down(fr.index.freq, freq) == 0:
    #             fr.index.freq = freq
    #             break
    #     else:
    #         raise ValueError(
    #             f"Found unsupported frequency ({fr.index.freq}). Must be one of: {stamps.FREQUENCIES}."
    #         )

    return fr


def fill_gaps(fr: NDFrame, maxgap: int = 2) -> NDFrame:
    """Fill gaps in series by linear interpolation.

    Parameters
    ----------
    fr : NDFrame
        Pandas Series or DataFrame.
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


def concat(dfs: Iterable, axis: int = 0, *args, **kwargs) -> pd.DataFrame:
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


def wavg(
    fr: Union[pd.Series, pd.DataFrame],
    weights: Union[Iterable, pd.Series, pd.DataFrame] = None,
    axis: int = 0,
) -> Union[pd.Series, float]:
    """
    Weighted average of dataframe.

    Parameters
    ----------
    fr : Union[pd.Series, pd.DataFrame]
        The input values.
    weights : Union[Iterable, pd.Series, pd.DataFrame], optional
        The weights. If provided as a Series, the weights and values are aligned along
        its index. If no weights are provided, the normal (unweighted) average is returned
        instead.
    axis : int, optional
        Calculate each column's average over all rows (if axis==0, default) or
        each row's average over all columns (if axis==1). Ignored for Series.

    Returns
    -------
    Union[pd.Series, float]
        The weighted average. A single float if `fr` is a Series; a Series if
        `fr` is a Dataframe.
    """
    # Orient so that we can always sum over rows.
    if axis == 1:
        if isinstance(fr, pd.DataFrame):
            fr = fr.T
        if isinstance(weights, pd.DataFrame):
            weights = weights.T

    if weights is None:
        # Return normal average.
        return fr.mean(axis=0)

    # Turn weights into Series if it's an iterable.
    if not isinstance(weights, pd.DataFrame) and not isinstance(weights, pd.Series):
        weights = pd.Series(weights, fr.index)

    summed = fr.mul(weights, axis=0).sum(skipna=False)  # float or float-Series
    totalweight = weights.sum()  # float or float-Series
    result = summed / totalweight

    # Correction: if total weight is 0, and all original values are the same, keep the original value.
    correctable = np.isclose(totalweight, 0) & (
        fr.nunique() == 1
    )  # bool or bool-Series
    if isinstance(fr, pd.Series):
        return result if not correctable else fr.iloc[0]
    result[correctable] = fr.iloc[0, :][correctable]
    return result


def trim_frame(fr: NDFrame, freq: str) -> NDFrame:
    """Trim index of frame to only keep full periods of certain frequency.

    Parameters
    ----------
    fr : NDFrame
        The (untrimmed) pandas series or dataframe.
    freq : str
        Frequency to trim to. E.g. 'MS' to only keep full months.

    Returns
    -------
    NDFrame
        Subset of `fr`, with same frequency.
    """
    i = stamps.trim_index(fr.index, freq)
    return fr.loc[i]


# TODO: move to testing folder
@functools.wraps(np.allclose)
def series_allclose(s1, s2, *args, **kwargs):
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
    s2_vals = s2.astype(f"pint[{s1.pint.u}]").pint.m
    return np.allclose(s1_vals, s2_vals, *args, **kwargs)
