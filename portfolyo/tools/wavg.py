from typing import Iterable, Mapping, overload

import numpy as np
import pandas as pd

from . import unit as tools_unit

# Developer notes:
# HACK: for speed:
# meaning: for series and dataframes with pint quantities, .sum() does not work,
# and .apply(np.sum) is really slow. The workaround is fast also for pint quantities.

# Developer notes:
# The following behaviour is wanted in calculating the weighted average:

# weights            values              rule                           result

# sum of weights != 0
# 1, -1, 2           10, 20, 30          "normal"                       (10*1 + 20*-1 + 30*2 ) / (1 + -1 + 2)
# 1, -1, 2           10, NaN, 30         NaN if values include NaN      NaN
# 1, 0, 2            10, NaN, 30         ignore NaN if weight = 0       (10*1 + 30*2) / (1 + 2)
# --> Remove all values for which weight == 0.
# --> If remaining values conain NaN --> result is NaN.
# --> Otherwise, calculate the result normally.

# sum of weights == 0 but not all 0
# 1, 1, -2           10, 20, 30          NaN if values distinct         NaN
# 1, 1, -2           10, 10, 10          value if values identical      10
# 1, -1, 0           10, 10, 30          ignore value if weight = 0     10
# 1, 1, -2           10, 10, NaN         NaN if values include NaN      NaN
# 1, -1, 0           10, 10, NaN         ignore NaN if weight = 0       10
# 1, -1, 0           NaN, NaN, NaN       NaN if values are all NaN      NaN
# --> Remove all values for which weight == 0.
# --> If remaining values contain NaN --> result is NaN
# --> Otherwise, if remaining values are identical --> result is that value
# --> Otherwise, result is Inf.

# all weights are 0
# 0, 0, 0            10, 20, 30          NaN if values distinct         NaN
# 0, 0, 0            10, 10, 10          Value if values identical      10
# 0, 0, 0            10, 10, NaN         NaN if values include NaN      NaN
# --> If values contain NaN --> result is NaN
# --> Otherwise, if values are identical --> result is that value
# --> Otherwise, result is NaN.

RESULT_IF_WEIGHTSUM0_VALUESNOTUNIFORM = np.nan


@overload
def general(
    fr: pd.Series, weights: Iterable | Mapping | pd.Series = None, axis: int = 0
) -> float:
    ...


@overload
def general(
    fr: pd.DataFrame,
    weights: Iterable | Mapping | pd.Series | pd.DataFrame = None,
    axis: int = 0,
) -> pd.Series:
    ...


def general(
    fr: pd.Series | pd.DataFrame,
    weights: Iterable | Mapping | pd.Series | pd.DataFrame = None,
    axis: int = 0,
) -> float | tools_unit.Q_ | pd.Series:
    """
    Weighted average of series or dataframe.

    Parameters
    ----------
    fr : pd.Series | pd.DataFrame
        The input values.
    weights : Iterable | Mapping | pd.Series | pd.DataFrame, optional
        The weights. If provided as a Mapping or Series, the weights and values
        are aligned along its index. If no weights are provided, the normal
        (unweighted) average is returned instead.
    axis : int, optional
        Calculate each column's average over all rows (if axis==0, default) or
        each row's average over all columns (if axis==1). Ignored for Series.

    Returns
    -------
    float | Q_ | pd.Series
        The weighted average. A single float or single Quantitiy if ``fr`` is a Series;
        a Series if ``fr`` is a Dataframe.
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
    s: pd.Series, weights: Iterable | Mapping | pd.Series = None
) -> float | tools_unit.Q_:
    """
    Weighted average of series.

    Parameters
    ----------
    s : pd.Series
        The input values.
    weights : Iterable | Mapping | pd.Series, optional
        The weights. If provided as a Mapping or Series, the weights and values are
        aligned along their indices/keys. If no weights are provided, the normal
        (unweighted) average is returned instead.

    Returns
    -------
    float | Quantity
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
    try:
        s = s.loc[weights.index]
    except KeyError as e:  # more weights than values
        raise ValueError("No values found for one or more weights.") from e

    # Unweighted average if all weights the same but not all 0.
    if weights.nunique() == 1 and not np.isclose(weights.iloc[0], 0):
        return s.mean()

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
        is_uniform = values_areuniform(s)
        return s.iloc[0] if is_uniform else np.nan

    # If we arrive here, the sum of the weights is not zero.

    factors = weights.div(weightsum).astype(float)
    scaled_values = s * factors  # fast, even with quantities
    return sum(scaled_values)


def dataframe(
    df: pd.DataFrame,
    weights: Iterable | Mapping | pd.Series | pd.DataFrame = None,
    axis: int = 0,
) -> pd.Series:
    """
    Weighted average of dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The input values.
    weights : Iterable | Mapping | pd.Series | pd.DataFrame, optional
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
    # same module, which results in a much shorter function. However, the speed penalty
    # is enormous, which is why this elaborate function is used.

    # Unweighted average if no weights are provided.
    if weights is None:
        # Fix possible problems, like distinct units of same dimension
        df = tools_unit.defaultunit(df)
        return df.apply(np.mean, axis=axis)  # can't do .mean() if pint-series

    # Prep: orient so that we can always average over columns.
    if axis == 0:
        df = df.T  # slow, but axis==0 is uncommon
        # HACK: transposing moves unit to element-level, undo here

    # Fix possible problems, also those introduced by .T
    df = tools_unit.defaultunit(df)

    # Do averaging.
    if isinstance(weights, pd.DataFrame):
        if axis == 0:
            weights = weights.T  # slow, but axis==0 is uncommon
            # HACK: transposing moves unit to element-level, undo here
            weights = tools_unit.defaultunit(weights)

        result = dataframe_columnwavg_with_weightsdataframe(df, weights)

    else:  # weights == series or iterable
        weights = weights_as_series(weights, df.columns)  # ensure weights is Series
        result = dataframe_columnwavg_with_weightsseries(df, weights)

    return tools_unit.defaultunit(result)


def dataframe_columnwavg_with_weightsdataframe(
    df: pd.DataFrame, weights: pd.DataFrame
) -> pd.Series:
    # Keep only relevant section.
    try:
        df = df.loc[weights.index, weights.columns]
    except KeyError as e:  # more weights than values
        raise ValueError("No values found for one or more weights.") from e
    originalindex = df.index

    # Create masks and aggregates for weights.
    # . One float/quantity for each row.
    weightssum = sum(s for _, s in weights.items())  # HACK: for speed
    # . One boolean for each row.
    weights_sum0 = weightssum == 0.0  # TODO: use np.isclose?
    # . One boolean for each weight.
    weight_is0 = weights == 0.0  # TODO: use np.isclose?
    # . One boolean for each row.
    weights_all0 = weight_is0.all(axis=1)

    # Handle each case seperately, and combine later.

    series = []

    # "Normal": sum of weights != 0.

    if (mask := ~weights_sum0).any():
        series.extend(
            _dataframe_columnwavg_with_weightssumnot0(
                df[mask], weights[mask], weightssum[mask]
            )
        )

    # Sum of weights == 0 but not all values are 0.

    if (mask := weights_sum0 & ~weights_all0).any():
        series.extend(
            _dataframe_columnwavg_with_weightssum0notall0(df[mask], weights[mask])
        )

    # Each weight has a value of 0.

    if (mask := weights_all0).any():
        series.extend(_dataframe_columnwavg_with_weightsall0(df[mask]))

    # Every index value of weights is now in exactly one series.
    return concatseries(series, originalindex)


def dataframe_columnwavg_with_weightsseries(
    df: pd.DataFrame, weights: pd.Series
) -> pd.Series:
    originalindex = df.index
    # Keep only relevant section.
    try:
        df = df.loc[:, weights.index]
    except KeyError as e:  # more weights than values
        raise ValueError("No values found for one or more weights.") from e

    # Create masks and aggregates for weights.
    # . One fleat/quantity.
    weightssum = sum(w for w in weights.values)
    # . One boolean.
    weights_sum0 = weightssum == 0.0
    # . One boolean for each weight.
    weight_is0 = weights == 0.0
    # . One boolean.
    weights_all0 = weight_is0.all()

    # See which case we have and calculate.

    # "Normal": sum of weights != 0.

    if not weights_sum0:
        series = _dataframe_columnwavg_with_weightssumnot0(df, weights, weightssum)

    # Sum of weights == 0 but not all values are 0.

    elif not weights_all0:
        series = _dataframe_columnwavg_with_weightssum0notall0(df, weights)

    # Each weight has a value of 0.

    else:
        series = _dataframe_columnwavg_with_weightsall0(df)

    # Every index value of weights is now in exactly one series.
    return concatseries(series, originalindex)


def _dataframe_columnwavg_with_weightssumnot0(
    df: pd.DataFrame,
    weights: pd.Series | pd.DataFrame,
    weightssum: float | tools_unit.Q_ | pd.Series,
) -> Iterable[pd.Series]:
    # Calculate the weighted average if sum of weights != 0.
    weight_is0 = weights == 0.0
    value_isna = df.isna()
    df = df.where(~(weight_is0 & value_isna), other=0.0)  # to ignore NaN if allowed
    factors = weights.div(weightssum, axis=0).astype(float)
    scaled_values = df * factors  # fast, even with quantities
    result = sum(s for _, s in scaled_values.items())  # HACK: for speed
    return [result]


def _dataframe_columnwavg_with_weightssum0notall0(
    df: pd.DataFrame, weights: pd.Series | pd.DataFrame
) -> Iterable[pd.Series]:
    # Calculate the weighted average if sum of weights == 0, but not all weights are 0.

    series = []

    if len(df.index) == 0:
        return series

    # Create masks and aggregates.
    weight_is0 = weights == 0.0  # one boolean for each weight
    value_isna = df.isna()  # one boolean for each value

    # Rows containing NaN-values whose weight != 0

    mask = (value_isna & ~weight_is0).any(axis=1)
    series.append(pd.Series(np.nan, df.index[mask]))

    # Remaining rows only have NaN if weight == 0; these elements can be ignored.
    # Ignore ALL values with weight == 0, and check if rows are uniform.

    if isinstance(weights, pd.Series):
        # Keep only remaining rows and keep only columns with weight != 0.
        df = df.loc[~mask, ~weight_is0]
    else:
        # Keep only remaining rows, and replace all values with weight == 0 with NaN.
        df, weight_is0 = df[~mask], weight_is0[~mask]
        df = df.where(~weight_is0, other=np.nan)
    series.append(rowvalue_uniformity(df))
    return series


def _dataframe_columnwavg_with_weightsall0(
    df: pd.DataFrame,
) -> Iterable[pd.Series]:
    # Calculate the weighted average if all weights == 0.

    series = []

    if len(df.index) == 0:
        return series

    # Create masks and aggregates.
    value_isna = df.isna()  # one boolean for each value

    # Rows containing NaN-values

    mask = value_isna.any(axis=1)
    series.append(pd.Series(np.nan, df[mask].index))
    # Keep only remaining rows.
    df = df[~mask]

    # Remaining rows do not have NaN.

    # Check if rows are uniform.
    series.append(rowvalue_uniformity(df))

    return series


def rowvalue_uniformity(df: pd.DataFrame) -> pd.Series:
    """Calculate a value for each row. First discard the NaN. Then see if remaining
    values are identical. If yes, that value is the result. If not, NaN is the result.
    """

    # HACK: we should do these calculations row-by-row, but this is really slow for pint,
    # because the row Series don't have a pint-unit, but are series of pint-quantities.
    # So we use the following column-operations instead to get to the same result.
    # - Replace values whose weight == 0 with NaN.
    # We now want to verify, if all remainig values in a row are identical.
    # - Initially fill buffer with NaN-values.
    # - Observe first column. Ignore NaN. For not-NaN: put in buffer.
    # - Observe second column. Ignore NaN. For not-NaN: if buffer is empty, put in buffer.
    #   If buffer is not empty, compare. If not same, put uniform-flag to False.
    # - Continue for other columns.
    # Uniform flag is now set to False for non-uniform rows. For rows with uniform values
    # or uniform NaN, this value/NaN is found in buffer.
    uniform = pd.Series(True, df.index)
    buffer = pd.Series(np.nan, df.index)  # define to ensure exists even if df empty
    for i, (_, s) in enumerate(df.items()):
        if i == 0:
            # define here to ensure ``values`` has pint dtype if df does too
            to_type = float if pd.api.types.is_integer_dtype(s.dtype) else s.dtype
            buffer = pd.Series(np.nan, s.index).astype(to_type)
        must_compare = s.notna() & buffer.notna()
        if must_compare.any():
            # HACK: cannot use ``uniform & (...)`` because must_compare has partial index, and missing values are set to False
            uniform = ~(~uniform | ~(s[must_compare] == buffer[must_compare]))
        must_replace = s.notna() & buffer.isna()
        if must_replace.any():
            buffer = buffer.fillna(s[must_replace])

    buffer[~uniform] = RESULT_IF_WEIGHTSUM0_VALUESNOTUNIFORM
    return buffer


def weights_as_series(weights: Iterable | Mapping, refindex: Iterable) -> pd.Series:
    # Step 1: turn into Series.
    if isinstance(weights, pd.Series):
        pass
    elif isinstance(weights, Mapping):
        weights = pd.Series(weights)
    elif isinstance(weights, Iterable):
        weights = pd.Series(weights, refindex)
    else:
        raise TypeError("``weights`` must be iterable or mapping.")
    # Step 2: avoid Series of Quantity-objects (convert to pint-series instead).
    return tools_unit.avoid_frame_of_objects(weights)


def values_areuniform(series: pd.Series, mask: Iterable = None) -> bool:
    """Return True if all values in series are same. If mask is provided, only compare
    values where the mask is True. If there are no values to compare, return True."""
    values = series[mask].values if mask is not None else series.values
    for i, val in enumerate(values):
        if i == 0:
            theval = val
        elif val != theval:
            return False
    return True


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

    if refindex is None:
        return result.sort_index()
    result = result.loc[refindex]
    if isinstance(refindex, pd.DatetimeIndex) and (freq := refindex.freq):
        result.index.freq = freq
    return result
