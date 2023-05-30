from typing import Iterable, Mapping, Union

import numpy as np
import pandas as pd
import pint

from . import unit as tools_unit


def general(
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
        return dataframe(fr, weights, axis)
    elif isinstance(fr, pd.Series):
        return series(fr, weights)
    else:
        raise TypeError(
            f"Parameter ``fr`` must be Series or DataFrame; got {type(fr)}."
        )


def series(
    s: pd.Series, weights: Union[Iterable, Mapping, pd.Series] = None
) -> Union[float, tools_unit.Q_]:
    """
    Weighted average of series.

    Parameters
    ----------
    s : pd.Series
        The input values.
    weights : Union[Iterable, Mapping, pd.Series], optional
        The weights. If provided as a Mapping or Series, the weights and values are
        aligned along their indices/keys. If no weights are provided, the normal
        (unweighted) average is returned instead.

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
    weights = weights_as_series(weights, s.index)
    # Keep only relevant section.
    s = s.loc[weights.index]
    # Replace NaN with 0 in locations where it doesn't change the result.
    replaceable = s.isna() & (weights == 0.0)
    s[replaceable] = 0.0

    # If we arrive here, ``s`` only has NaN on locations where weight != 0.

    # Check if ALL weights are 0.
    # In that case, the result is NaN.
    if (weights == 0).all():  # edge case (very uncommon)
        return np.nan

    # If we arrive here, not all weights are 0.

    # Check if ``s`` contains a NaN.
    # In that case, the result is NaN.
    if s.isna().sum() > 0:
        return np.nan

    # For the other rows, we must calculate the wavg.

    # Check if SUM of weights is 0.
    # In that case, the result is NaN in case of non-uniform values.
    weightsum = weights.sum()  # a float or quantity
    if np.isclose(weightsum, 0):  # edge case (more common)
        is_uniform = uniform_series(s)
        return s.iloc[0] if is_uniform else np.nan

    # If we arrive here, the sum of the weights is not zero.

    factors = weights.div(weightsum).astype(float)
    scaled_values = s * factors  # fast, even with quantities
    return sum(scaled_values)


def dataframe_bak(
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
        result = df.apply(lambda s: series(s, weights.loc[:, s.name]))
    else:  # weights == series or iterable or None
        result = df.apply(lambda s: series(s, weights))

    # Correction: turn series of pint-Quantities into pint-series, if possible.
    if pd.api.types.is_object_dtype(result):
        firstval = result.iloc[0]
        try:
            result = result.astype(f"pint[{firstval.units}]")
        except (AttributeError, pint.DimensionalityError):
            pass

    return result


def dataframe(
    df: pd.DataFrame,
    weights: Union[Iterable, Mapping, pd.Series, pd.DataFrame] = None,
    axis: int = 0,
) -> pd.Series:
    """
    Weighted average of dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The input values.
    weights : Union[Iterable, Mapping, pd.Series, pd.DataFrame], optional
        The weights. If provided as a Series or Mapping, its index are is used for
        alignment (with ``df``'s index if axis==0, or its columns if axis==1). If no
        weights are provided, the normal (unweighted) average is returned instead.
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
    # Developer note: it is possible to repeatedly call the `series` function in this
    # same module, which results in a much shorter function (see dataframe_bak). How-
    # ever, the speed penalty is enormous, which is why this elaborate function is used.

    # Prep: orient so that we can always average over columns.
    if axis == 0:
        df = df.T

    # Unweighted average if no weights are provided.
    if weights is None:
        return df.apply(np.mean)  # can't do .mean() if pint-series

    # Do averaging.
    elif isinstance(weights, pd.DataFrame):
        if axis == 0:
            weights = weights.T

        return dataframe_columnwavg_with_weightsdataframe(df, weights)

    else:  # weights == series or iterable
        weights = weights_as_series(weights, df.columns)  # ensure weights is Series
        return dataframe_columnwavg_with_weightsseries(df, weights)


def dataframe_columnwavg_with_weightsdataframe(
    df: pd.DataFrame, weights: pd.DataFrame
) -> pd.Series:
    # Keep only relevant section.
    df = df.loc[weights.index, weights.columns]
    originalindex = df.index
    # Replace NaN with 0 in locations where it doesn't change the result.
    replaceable = df.isna() & (weights == 0.0)
    df[replaceable] = 0.0

    series = []

    # If we arrive here, ``df`` only has NaN on locations where weight != 0.

    # Check, for each row, if ALL weights are 0.
    # In that case, the result in those rows is NaN.
    all0weights = (weights == 0).all(axis=1)  # bool Series
    series.append(pd.Series(np.nan, all0weights.index[all0weights]))
    # Update remaining rows
    df, weights = df[~all0weights], weights[~all0weights]

    # For the other rows, not all weights are 0.

    # Check, for each row, if it contains a NaN.
    # In that case, the result for those rows is NaN.
    hasna = df.isna().sum(axis=1) > 0
    series.append(pd.Series(np.nan, hasna.index[hasna]))
    # Update remaining rows
    df, weights = df[~hasna], weights[~hasna]

    # For the other rows, we must calculate the wavg.

    # Check, for each row, if SUM of weights is 0.
    # In that case, the result is NaN in all rows with a non-uniform value.
    # HACK: workaround for speed
    # weightsum = weights.apply(np.sum, axis=1) # slow
    weightsum = sum(s for _, s in weights.items())  # fast, also for quantities
    sum0weights = weightsum == 0.0  # TODO: use np.isclose? (must allow float and q)
    if sum0weights.any():  # edge case (more common)
        sub_df = df.loc[sum0weights, :]
        is_uniform = uniform_df(sub_df, axis=1)
        # Add resulting value for each row where sum of weights is 0.
        series.append(sub_df.loc[is_uniform, :].iloc[:, 0])
        series.append(pd.Series(np.nan, is_uniform.index[~is_uniform]))
        # Update remaining rows
        df, weights, weightsum = (
            df[~sum0weights],
            weights[~sum0weights],
            weightsum[~sum0weights],
        )

    # For the remaining rows we must calculate the wavg with the formula.

    factors = weights.div(weightsum, axis=0).astype(float)
    scaled_values = df * factors  # fast, even with quantities
    # HACK: wavg_values need to be calculated with workaround for speed
    # wavg_values = scaled_values.sum(axis=1) #doesn't work in current pint-version
    # wavg_values = scaled_values.apply(np.sum, axis=1) #slow
    wavg_values = sum(s for _, s in scaled_values.items())  # fast, also for quantts
    series.append(wavg_values)

    # Every index value of weights is now in exactly one series.
    return concatseries(series, originalindex)


def dataframe_columnwavg_with_weightsseries(
    df: pd.DataFrame, weights: pd.Series
) -> pd.Series:
    originalindex = df.index
    # Keep only relevant section.
    df = df.loc[:, weights.index]
    # Replace NaN with 0 in locations where it doesn't change the result.
    replaceable = df.isna() & (weights == 0.0)
    df[replaceable] = 0.0

    # If we arrive here, ``df`` only has NaN on locations where weight != 0.

    # Check if ALL weights are 0.
    # In that case, the result in all rows is NaN.
    if (weights == 0).all():  # edge case (very uncommon)
        return pd.Series(np.nan, df.index)

    # If we arrive here, not all weights are 0.

    series = []

    # Check, for each row, if it contains a NaN.
    # In that case, the result for those rows is NaN.
    hasna = df.isna().sum(axis=1) > 0
    series.append(pd.Series(np.nan, hasna.index[hasna]))
    # Update remaining rows
    df = df[~hasna]

    # For the other rows, we must calculate the wavg.

    # Check if SUM of weights is 0.
    # In that case, the result is NaN in all rows with a non-uniform value.
    weightsum = weights.sum()  # a float or quantity
    if np.isclose(weightsum, 0):  # edge case (more common)
        is_uniform = uniform_df(df, axis=1)
        # Add resulting value for each row where sum of weights is 0.
        series.append(df.loc[is_uniform, :].iloc[:, 0])
        series.append(pd.Series(np.nan, is_uniform.index[~is_uniform]))
        return pd.concat(series).sort_index()

    # If we arrive here, the sum of the weights is not zero.

    factors = weights.div(weightsum).astype(float)
    scaled_values = df * factors  # fast, even with quantities
    # HACK: wavg_values need to be calculated with workaround for speed (but still quite slow for quantities)
    # wavg_values = scaled_values.sum(axis=1) #doesn't work in current pint-version
    # wavg_values = scaled_values.apply(np.sum, axis=1) #slow
    wavg_values = sum(s for _, s in scaled_values.items())  # fast, also for pint
    series.append(wavg_values)

    # Every index value of weights is now in exactly one series.
    return concatseries(series, originalindex)


def weights_as_series(
    weights: Union[Iterable, Mapping], refindex: Iterable
) -> pd.Series:
    if isinstance(weights, pd.Series):
        return weights
    if isinstance(weights, Mapping):
        return pd.Series(weights)
    if isinstance(weights, Iterable):
        return pd.Series(weights, refindex)
    raise TypeError("``weights`` must be iterable or mapping.")


def uniform_series(series: pd.Series) -> bool:
    for i, val in enumerate(series.values):
        if i == 0:
            theval = val
        elif val != theval:
            return False
    return True


def uniform_df(df: pd.DataFrame, axis: int = 0) -> pd.Series:
    if axis == 1:
        df = df.T
    return pd.Series({col: uniform_series(s) for col, s in df.items()})


def concatseries(series: Iterable[pd.Series], refindex: Iterable = None) -> pd.Series:
    """Concatenate some series, and try to make it a pint-series if possible."""
    dtypes = set()
    for s in series:
        if s.isna().all():
            continue
        dtypes.add(s.dtype)
    if len(dtypes) == 1:
        dtype = dtypes.pop()
        series = [s.astype(dtype) for s in series]
    result = pd.concat(series)

    if not refindex:
        return result.sort_index()
    result = result.loc[refindex]
    if isinstance(refindex, pd.DatetimeIndex) and (freq := refindex.freq):
        result.index.freq = freq
    return result
