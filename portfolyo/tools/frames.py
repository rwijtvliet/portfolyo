"""Module with tools to modify and standardize dataframes."""

from pint import DimensionalityError
from pytz import AmbiguousTimeError, NonExistentTimeError
from . import stamps, zones, nits

from pandas.core.frame import NDFrame
from typing import Iterable, Any, Union
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


def set_frequency(fr: NDFrame, wanted: str = None, strict: bool = False) -> NDFrame:
    """Try to read, infer, and force frequency of frame's index.

    Parameters
    ----------
    fr : NDFrame
        Pandas series or dataframe.
    wanted : str, optional
        Frequency to set. If none provided, try to infer.
    strict : bool, optional (default: False)
        If True, raise ValueError if a valid frequency is not found.

    Returns
    -------
    NDFrame
        Same type as ``fr``, with, if possible, a valid value for ``fr.index.freq``.
    """
    # Handle non-datetime-indices.
    if not isinstance(fr.index, pd.DatetimeIndex):
        if strict:
            raise ValueError(
                "The data does not have a datetime index and can therefore not have a frequency."
            )
        else:
            return fr

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
    elif freq and freq not in stamps.FREQUENCIES:
        # Edge case: year-/quarterly but starting != Jan.
        if stamps.freq_up_or_down(freq, "AS") == 0:
            fr.index.freq = "AS"  # will likely fail
        elif stamps.freq_up_or_down(freq, "QS") == 0:
            fr.index.freq = "QS"  # will only succeed if QS-APR, QS-JUL or QS-OCT
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


def wavg(
    fr: Union[pd.Series, pd.DataFrame],
    weights: Union[Iterable, pd.Series, pd.DataFrame] = None,
    axis: int = 0,
) -> Union[pd.Series, float]:
    """
    Weighted average of series or dataframe.

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
    if isinstance(fr, pd.DataFrame):
        return _wavg_df(fr, weights, axis)
    elif isinstance(fr, pd.Series):
        return _wavg_series(fr, weights)
    else:
        raise TypeError(
            f"Parameter ``fr`` must be Series or DataFrame; got {type(fr)}."
        )


def _wavg_series(
    s: pd.Series, weights: Union[Iterable, pd.Series] = None
) -> Union[float, nits.Q_]:
    """
    Weighted average of series.

    Parameters
    ----------
    s : pd.Series
        The input values.
    weights : Union[Iterable, pd.Series], optional
        The weights. If provided as a Series, the weights and values are aligned along
        their indices. If no weights are provided, the normal (unweighted) average is
        returned instead.

    Returns
    -------
    Union[float, Quantity]
        The weighted average.

    Notes
    -----
    Will raise Error if values in ``s`` have distinct units.
    """
    # Unweighted average if no weights provided.
    if weights is None:
        return s.mean()

    # Prep: ensure weights is also a Series.
    if not isinstance(weights, pd.Series):
        weights = pd.Series(weights, s.index)

    # Get multiplication factors as floats.
    factors = (weights / weights.sum()).astype(float)
    # Calculate the average.
    result = s.mul(factors).sum(skipna=False)  # float or quantity

    # Special case: if total weight is 0, but all original values are identical, return this value.
    if np.isclose(weights.sum(), 0) and s.nunique() == 1:
        return s.iloc[0]

    return result


def _wavg_df(
    df: pd.DataFrame,
    weights: Union[Iterable, pd.Series, pd.DataFrame] = None,
    axis: int = 0,
) -> pd.Series:
    """
    Weighted average of dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The input values.
    weights : Union[Iterable, pd.Series, pd.DataFrame], optional
        The weights. If provided as a Series, its index are is used for alignment (with
        ``df``'s index if axis==0, or its columns if axis==1). If no weights are provided,
        the normal (unweighted) average is returned instead.
    axis : int, optional (default: 0)
        - if 0, calculate average over all rows (for each column);
        - if 1, calculate average over all columns (for each row).

    Returns
    -------
    pd.Series
        The weighted average.

    Notes
    -----
    Will raise error if axis == 1 and columns have distinct unit-dimensions.
    """
    # Prep: orient so that we can always average over rows.
    if axis == 1:
        df = df.T

    # Do averaging series-by-series.
    if isinstance(weights, pd.DataFrame):
        if axis == 1:
            weights = weights.T
        result = df.apply(lambda s: _wavg_series(s, weights.loc[:, s.name]))
    else:  # weights == series or iterable or None
        result = df.apply(lambda s: _wavg_series(s, weights))

    # Correction: turn series of pint-Quantities into pint-series, if possible.
    if pd.api.types.is_object_dtype(result):
        firstval = result.iloc[0]
        try:
            result = result.astype(f"pint[{firstval.units}]")
        except (AttributeError, DimensionalityError):
            pass

    return result


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
    s2_vals = s2.astype(f"pint[{s1.pint.u:P}]").pint.m
    return np.allclose(s1_vals, s2_vals, *args, **kwargs)
