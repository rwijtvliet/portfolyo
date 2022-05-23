"""Module with tools to modify and standardize dataframes."""

from . import stamps, zones

from pandas.core.frame import NDFrame
from typing import Iterable, Union, Any
import pandas as pd
import numpy as np
import functools


"""
========================
Preprocessing input data
========================

``portfolyo`` mainly works with ``pandas series`` and ``pandas dataframe`` objects as
input data. Let's say we have an object ``fr`` as input data, which may be either.
``portfolyo`` expects and assumes ``fr`` to adhere to certain specifications,
and depending on your data source, some cleanup or manipulation of the input data might
be necessary.

------------------------
Left-bound DatetimeIndex
------------------------
The index of ``fr`` must be a ``pandas.DatetimeIndex``. Each timestamp in the index
describes a *period* in time. (The reason a ``pandas.PeriodIndex`` is not used for
this, is that these cannot handle timezones that use daylight-savings time.)

The timestamps in the index must describe the *start* of the period they describe. E.g.,
if we have hourly values, the timestamp with time '04:00' must describe the hour
starting at 04:00 (i.e., 04:00-05:00) and not the hour ending at 04:00 (i.e.,
03:00-04:00).

These two criteria can be met by passing ``fr`` to the ``portfolyo.standardize_index``
function. If the timestamps are right-bound, the frequency of the data must be known or
be able to be inferred. If it is not (e.g. due to missing values), a guess is made from
the median timedelta between consecutive timestamps.

---------
Frequency
---------
The index must have a frequency (``fr.index.freq``) set; it must be one of the ones in
``portfolyo.FREQUENCIES``. The following abbreviations are used by ``pandas`` and
throughout this package:
* ``15T``: quarterhourly;
* ``H``: hourly;
* ``D``: daily;
* ``MS``: monthly;
* ``QS``: quarterly;
* ``AS``: yearly.

If the frequency is not set, and ``pandas.infer_freq(fr.index)`` returns ``None``, this
is because of one or more of the following reasons:

Too few datapoints
------------------
If there is only one timestamp in the index, e.g., '2020-01-01 0:00', it is impossible
to know if this represents an hour, a day, or the entire year; ``pandas.infer_freq``
needs at least 3 datapoints to infer the frequency. In this case, we can manually set
the frequency with e.g. ``fr.index.freq = 'AS```.

Gaps in data
------------
If the index has gaps - e.g., it has timestamps for April 1, April 2, April 3, and
April 5, the frequency can also not be determined. In this case, April 4 needs to be
inserted. If its value is not known, we can set in to ``numpy.nan`` and use the
``portfolyo.fill_gaps()`` function to do a linear interpolation and replace the ``na``-
values.

Local timestamps
----------------
If the data:
* applies to a timezone which does not have a fixed UTC-offset (e.g. if it observes
daylight-savings time), and
* contains hourly values (or shorter), and
* has timestamps which are not timezone-aware (i.e., do not contain the UTC-offset),
then there will be repeated or missing timestamps during a DST transition, and the
frequency cannot be determined.

E.g. in the 'Europe/Berlin' timezone, an hourly timeseries that is not timezone-aware
will contain the timestamps '2020-03-29 00:00', '01:00', '03:00', ..., as well as
'2020-10-25 00:00', '01:00', '02:00', '02:00', '03:00', ...

In this case, the data must be localized, i.e., the timezone must be set to it. This can
be done with the ``fr.tz_localize()`` method. Alternatively, we can pass any ``fr`` to
the ``portfolyo.standardize_timezone()`` function to make sure it is always converted to
the wanted timezone. Beware though, that this may involve lossy conversions, see ***
Working with timezones***.
"""


def standardize_index(fr: NDFrame, column: str = None, bound: str = "left") -> NDFrame:
    """Standardize the index of a series or dataframe.

    Parameters
    ----------
    fr : NDFrame
        Pandas series or dataframe.
    column : str, optional
        Column to create the timestamp from. Used only if ``fr`` is DataFrame; ignored
        otherwise. Use existing index if none specified.
    bound : str, {'left' (default), 'right'}
        If 'left' ('right'), specifies that input timestamps are left-(right-)bound.

    Returns
    -------
    NDFrame
        Same type as ``fr``, with left-bound timestamp index.
        Column ``column`` (if applicable) is removed, and index renamed to 'ts_left'.
    """
    # Set index.
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

    return fr


def standardize_timezone(
    fr: NDFrame,
    force: str,
    strict: bool = True,
    tz: str = "Europe/Berlin",
    floating: bool = True,
) -> NDFrame:
    """Standardize the timezone of a series or dataframe by changing index and/or
    converting data.

    Parameters
    ----------
    fr : NDFrame
        Pandas series or dataframe.
    force : {'aware', 'agnostic'}
        Force ``fr`` to be timezone aware or timezone agnostic.
    strict : bool, optional (default: True)
        If True, raise ValueError if the frequency of the resulting frame cannot be
        determined.
    tz : str, optional (default: "Europe/Berlin")
        The timezone in which to interpret non-localized values. If force == 'aware':
        also the timezone to localize to.
    floating : bool, optional (default: True)
        If force == 'aware': how to convert to ``tz`` if ``fr`` has other timezone. Keep
        local time (``floating`` == True) or keep universal time (``floating`` == False).
        Ignored for force == 'agnostic'.

    Returns
    -------
    NDFrame
        Same type as ``fr``, with correct timezone.

    Notes
    -----
    It is assumed that we are dealing with "time-averable" data, such as values in [MW]
    or [Eur/MWh]. This is especially important when converting daily (and longer) values
    between a tz-agnostic context and a tz-aware context with DST-transitions.

    See also
    --------
    ``.force_tzaware``
    ``.force_tzagnostic``
    """
    if force == "aware":
        fr = zones.force_tzaware(fr, tz, floating=floating)
    elif force == "agnostic" or force == "naive":
        fr = zones.force_tzagnostic(fr, tz_in=tz)
    else:
        raise ValueError("Parameter ``force`` must be 'aware' or 'agnostic'.")

    # After standardizing timezone, the frequency should be set.
    return set_frequency(fr, strict=strict)


def set_ts_index(
    fr: NDFrame,
    column: str = None,
    bound: str = "left",
    *,
    continuous: bool = True,
) -> NDFrame:
    """
    Create/add a standardized timestamp index to a dataframe.

    Parameters
    ----------
    fr : NDFrame
        Pandas series or dataframe.
    column : str, optional
        Column to create the timestamp from. Used only if ``fr`` is DataFrame; ignored
        otherwise. Use existing index if none specified.
    bound : str, {'left' (default), 'right'}
        If 'left' ('right'), specifies that input timestamps are left-(right-)bound.
    continuous : bool, optional (default: True)
        If true, raise error if data has missing values (i.e., gaps).

    Returns
    -------
    NDFrame
        Same type as ``fr``, with left-bound timestamp index.
        Column ``column`` (if applicable) is removed, and index renamed to 'ts_left'.
    """

    # Set index.
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

    # Set and check frequency.
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

    return fr


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
    s2_vals = s2.astype(f"pint[{s1.pint.u}]").pint.m
    return np.allclose(s1_vals, s2_vals, *args, **kwargs)
