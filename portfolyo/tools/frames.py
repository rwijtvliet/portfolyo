"""Module with tools to modify and standardize dataframes."""

from pytz import AmbiguousTimeError, NonExistentTimeError
from . import stamps, zones, nits

from pandas.core.frame import NDFrame
from typing import Iterable, Union, Any
import pandas as pd
import numpy as np
import functools


def standardize(
    fr: NDFrame,
    force: str = None,
    bound: str = "left",
    *,
    tz: str = "Europe/Berlin",
    floating: bool = True,
    index_col: str = None,
    force_freq: str = None,
) -> NDFrame:
    """Standardize a series or dataframe.

    The output has

    Parameters
    ----------
    fr : NDFrame
        Pandas series or dataframe.
    force : {'aware', 'agnostic'}, optional (default: None)
        Force ``fr`` to be timezone aware or timezone agnostic. If None: keep index
        as-is.
    bound : {'left', 'right'}, optional (default: 'left')
        If 'left' ('right'), specifies that input timestamps are left-(right-)bound.
    tz : str, optional (default: "Europe/Berlin")
        The timezone in which to interpret non-localized values. If ``force`` ==
        'aware': also the timezone to localize to. Ignored if ``force`` is None.
    floating : bool, optional (default: True)
        If ``force`` == 'aware': how to convert to ``tz`` if ``fr`` has other timezone.
        Keep local time (``floating`` == True) or keep universal time (``floating`` ==
        False). Ignored if ``force`` == 'agnostic' or None.
    index_col : str, optional
        Column to create the timestamp from. Use existing index if none specified.
        Ignored if ``fr`` is not a DataFrame.
    force_freq : str, optional
        If a frequency cannot be inferred from the data (e.g. due to gaps), it is
        resampled at this frequency. Default: raise Exception.

    Returns
    -------
    NDFrame
        Same type as ``fr``, with a left-bound DatetimeIndex, a valid frequency, and
        wanted timezone.

    Notes
    -----
    It is assumed that we are dealing with "time-averable" data, such as values in [MW]
    or [Eur/MWh]. This is especially important when converting daily (and longer) values
    between a tz-agnostic context and a tz-aware context with DST-transitions.

    See also
    --------
    ``portfolyo.force_tzaware``
    ``portfolyo.force_tzagnostic``
    """
    kwargs = {"tz": tz, "floating": floating, "force_freq": force_freq}

    # Set index.
    if index_col and isinstance(fr, pd.DataFrame):
        fr = fr.set_index(index_col)
    else:
        fr = fr.copy()  # don't change passed-in fr
    fr.index = pd.DatetimeIndex(fr.index)  # turn / try to turn into datetime

    # We want to cover 2 additional cases for convenience sake:
    # a. The user passes a frame that still needs to be localized (--> freq unknown)
    # b. The user passes a frame that is not left-bound (--> freq needed)

    # Make sure it has a frequency, i.e., make sure it is tz-aware or tz-agnostic.
    # Pipeline if frequency not yet found: right -> left -> localize -> tz-aware -> freq
    fr = set_frequency(fr)
    freq_input, tz_input = fr.index.freq, fr.index.tz

    # The data may be right-bound.

    if bound == "right":  # right -> left
        for how in ["A", "B"]:
            try:
                fr_left = fr.set_axis(stamps.right_to_left(fr.index, how))
                return standardize(fr_left, force, "left", **kwargs)
            except ValueError as e:
                if how == "B":
                    raise ValueError("Cannot make this frame left-bound.") from e
                pass

    # Now the data is left-bound.
    # If the frequency is not found, and it is tz-naive, the index may need to be localized.

    if not freq_input and not tz_input and tz:  # left -> tz-aware (try)
        try:
            fr_aware = fr.tz_localize(tz, ambiguous="infer")
        except (AmbiguousTimeError, NonExistentTimeError):
            pass  # fr did not need / cound not be localized. Continue with fr as-is.
        else:
            return standardize(fr_aware, force, "left", **kwargs)

    # All options to infer frequency have been exhausted. One may or may not have been found.
    # Does the user want to force a frequency?

    if (not freq_input) and force_freq:
        # No freq has been found, but user specifies which freq it should be.
        fr_withfreq = fr.asfreq(force_freq)
        return standardize(fr_withfreq, force, "left", tz=tz, floating=floating)

    elif (not freq_input) and (not force_freq):
        # No freq has been bound, and user specifies no freq either.
        raise ValueError(
            "A frequency could not be inferred for this data. Force a frequency (by passing the"
            " ``force_freq`` parameter), or localize the data in advance (with ``fr.tz_localize()``)."
        )

    elif freq_input and force_freq and force_freq != freq_input:
        # Freq has been found, but user specifies it should be a different freq.
        raise ValueError(
            f"This data seems to have a frequency {freq_input}, which is different from the frequency"
            f" the user wants to force on it {force_freq}. Note that the ``force_freq`` parameter is"
            " for filling gaps in the input data. It should not be used for resampling! If the"
            " data has e.g. daily values but you want monthly values, use ``force_freq='D'``, and"
            " pass the return value to one of the functions in the ``portfolyo.changefreq`` module."
        )

    # Now the data has frequency set. It is tz-aware (possibly with wrong tz) or tz-agnostic.

    # Fix timezone.
    if force == "aware":
        fr = zones.force_tzaware(fr, tz, floating=floating)
    elif force == "agnostic" or force == "naive":
        fr = zones.force_tzagnostic(fr)
    elif force is None:  # don't try to fix timezone.
        pass
    else:
        raise ValueError(
            f"Parameter ``force`` must be one of 'aware', 'agnostic'; got {force}."
        )

    # Standardize index name.
    fr.index.name = "ts_left"
    # After standardizing timezone, the frequency should have been set.
    return set_frequency(fr, freq_input, strict=force_freq)


def assert_standardized(fr: NDFrame):
    """Assert that series or dataframe is standardized."""

    freq = fr.index.freq
    if not freq:
        raise AssertionError("Index must have frequency set.")
    if freq not in (freqs := stamps.FREQUENCIES):
        raise AssertionError(
            f"Index frequency must be one of {', '.join(freqs)}; found '{freq}'."
        )
    try:
        stamps.assert_boundary_ts(fr.index, freq)
    except ValueError as e:
        raise AssertionError(
            f"Index values are not (all) at start of a '{freq}'-period."
        ) from e


# def set_ts_index(
#     fr: NDFrame,
#     column: str = None,
#     bound: str = "left",
#     *,
#     continuous: bool = True,
# ) -> NDFrame:
#     """
#     Create/add a standardized timestamp index to a dataframe.

#     Parameters
#     ----------
#     fr : NDFrame
#         Pandas series or dataframe.
#     column : str, optional
#         Column to create the timestamp from. Used only if ``fr`` is DataFrame; ignored
#         otherwise. Use existing index if none specified.
#     bound : str, {'left' (default), 'right'}
#         If 'left' ('right'), specifies that input timestamps are left-(right-)bound.
#     continuous : bool, optional (default: True)
#         If true, raise error if data has missing values (i.e., gaps).

#     Returns
#     -------
#     NDFrame
#         Same type as ``fr``, with left-bound timestamp index.
#         Column ``column`` (if applicable) is removed, and index renamed to 'ts_left'.
#     """

#     # Set index.
#     if column and isinstance(fr, pd.DataFrame):
#         fr = fr.set_index(column)
#     else:
#         fr = fr.copy()  # don't change passed-in fr

#     # Make leftbound.
#     i = pd.DatetimeIndex(fr.index)  # turn / try to turn into datetime
#     if bound == "right":
#         i = stamps.make_leftbound(i)
#     i.name = "ts_left"
#     fr.index = i

#     # Set and check frequency.
#     if fr.index.freq is None:
#         freq = stamps.guess_frequency((i[1:] - i[:-1]).median())
#         fr2 = fr.resample(freq).asfreq()
#         # If the new dataframe has additional rows, the original dataframe was not gapless.
#         if continuous and len(fr2) > len(fr):
#             missing = [i for i in fr2.index if i not in fr.index]
#             raise ValueError(
#                 f"The index of ``fr`` does not have continuous data; missing data for: {missing}."
#             )
#         fr = fr2

#     if fr.index.freq not in stamps.FREQUENCIES:
#         for freq2 in ["MS", "QS"]:  # Edge case: month-/quarterly but starting != Jan.
#             if stamps.freq_up_or_down(fr.index.freq, freq2) == 0:
#                 fr.index.freq = freq2
#                 break
#         else:
#             raise ValueError(
#                 f"The data has a non-allowed frequency. Must be one of {', '.join(stamps.FREQUENCIES)}; found {fr.index.freq}."
#             )

#     # Check boundaries.
#     stamps.assert_boundary_ts(fr.index, fr.index.freq)

#     return fr


def set_frequency(fr: NDFrame, wanted: str = None, strict: bool = False) -> NDFrame:
    """Try to read, infer, and force frequency of frame's index.

    Parameters
    ----------
    fr : NDFrame
        Pandas series or dataframe.
    wanted : str, optional
        Suggestion for the frequency to set, if it cannot be inferred.
    strict : bool, optional (default: False)
        If True, raise ValueError if a valid frequency is not found.

    Returns
    -------
    NDFrame
        Same type as ``fr``, with, if possible, a valid value for ``fr.index.freq``.
    """
    # Find frequency.
    fr = fr.copy()
    if fr.index.freq:
        pass
    elif freq := pd.infer_freq(fr.index):
        fr.index.freq = freq
    elif wanted:
        fr.index.freq = wanted
    # Correct if necessary.
    freq = fr.index.freq
    if not freq and strict:  # No frequency found.
        raise ValueError("The data does not seem to have a regular frequency.")
    elif freq and freq not in stamps.FREQUENCIES:
        # Edge case: month-/quarterly but starting != Jan.
        if stamps.freq_up_or_down(freq, "AS") == 0:
            fr.index.freq = "AS"
        elif stamps.freq_up_or_down(freq, "QS") == 0:
            fr.index.freq = "QS"
        elif strict:
            raise ValueError(
                "The data has a non-allowed frequency. Must be one of "
                f"{', '.join(stamps.FREQUENCIES)}; found '{freq}'."
            )
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
    s2_vals = s2.astype(nits.pintunit(s1.pint.u)).pint.m
    return np.allclose(s1_vals, s2_vals, *args, **kwargs)
