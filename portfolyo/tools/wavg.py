from typing import Iterable, Mapping, overload

import numpy as np
import pandas as pd

from . import freq as tools_freq
from . import unit as tools_unit

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
    fr: pd.Series,
    weights: Iterable | Mapping | pd.Series | None = None,
    axis: int = 0,
) -> float | tools_unit.Q_: ...


@overload
def general(
    fr: pd.DataFrame,
    weights: Iterable | Mapping | pd.Series | pd.DataFrame | None = None,
    axis: int = 0,
) -> pd.Series: ...


def general(
    fr: pd.Series | pd.DataFrame,
    weights: Iterable | Mapping | pd.Series | pd.DataFrame | None = None,
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
    s: pd.Series, weights: Iterable | Mapping | pd.Series | None = None
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
        return s.iloc[0] if is_uniform else RESULT_IF_WEIGHTSUM0_VALUESNOTUNIFORM

    # If we arrive here, the sum of the weights is not zero.

    factors = weights.div(weightsum).astype(float)
    scaled_values = s * factors  # fast, even with quantities
    return sum(scaled_values)


def dataframe(
    df: pd.DataFrame,
    weights: Iterable | Mapping | pd.Series | pd.DataFrame | None = None,
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
        Axis to collapse.
        - if 0, collapse rows to calculate average of columns;
        - if 1, collapse columns to calculate average of rows.

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

    # Prep: orient to always collapse the rows and keep the columns.
    if axis == 1:
        idx_to_keep = df.index  # store now, because .T loses properties (e.g. freq)
        df = df.T
    else:
        idx_to_keep = df.columns
    idx_to_collapse = df.index
    # Fix possible problems, like distinct units of same dimension in the same column.
    df = tools_unit.avoid_frame_of_objects(df)

    # Calculate average of each column.
    # . No weights.
    if weights is None:
        return df.mean().set_axis(idx_to_keep)
    # . Element-specific weights.
    elif isinstance(weights, pd.DataFrame):
        if axis == 1:
            weights = weights.T
        result = dataframe_colwavg_with_weightsdataframe(df, weights)
    # . Row or column-specific weights.
    else:  # weights == series or iterable
        weights = weights_as_series(
            weights, idx_to_collapse
        )  # ensure weights is Series
        result = dataframe_colwavg_with_weightsseries(df, weights)

    # Fix possible issue: lost .freq property on index.
    if isinstance(result.index, pd.DatetimeIndex) and result.index.freq is None:
        result = tools_freq.guess_to_frame(result)
    return tools_unit.avoid_frame_of_objects(result, False)


def dataframe_colwavg_with_weightsdataframe(
    df: pd.DataFrame, weights: pd.DataFrame
) -> pd.Series:
    # Keep only relevant section.
    try:
        df = df.loc[weights.index, weights.columns]
    except KeyError as e:  # more weights than values
        raise ValueError("No values found for one or more weights.") from e

    # Create masks and aggregates for weights.
    # . One float/quantity for each col.
    weightssum = weights.sum()
    # . One boolean for each row.
    weights_sum0 = weightssum == 0.0  # TODO: use np.isclose?
    # . One boolean for each weight.
    weight_is0 = weights == 0.0  # TODO: use np.isclose?
    # . One boolean for each row.
    weights_all0 = weight_is0.all(axis=0)

    # Handle each case seperately, and combine later.

    series = []

    # "Normal": sum of weights != 0.
    if (mask := ~weights_sum0).any():
        series.append(
            _dataframe_colwavg_with_weightssumnot0(
                df.loc[:, mask], weights.loc[:, mask], weightssum[mask]
            )
        )

    # Sum of weights == 0 but not all values are 0.
    if (mask := weights_sum0 & ~weights_all0).any():
        series.append(
            _dataframe_colwavg_with_weightssum0notall0(
                df.loc[:, mask], weights.loc[:, mask]
            )
        )

    # Each weight has a value of 0.
    if (mask := weights_all0).any():
        series.append(_dataframe_colwavg_with_weightsall0(df.loc[:, mask]))

    # Every index value of weights is now in exactly one series.
    return pd.concat(series).sort_index()


def dataframe_colwavg_with_weightsseries(
    df: pd.DataFrame, weights: pd.Series
) -> pd.Series:
    # Keep only relevant section.
    try:
        df = df.loc[weights.index, :]
    except KeyError as e:  # more weights than values
        raise ValueError("No values found for one or more weights.") from e

    # Create masks and aggregates for weights.
    # . One fleat/quantity.
    weightssum = sum(weights.values)
    # . One boolean.
    weights_sum0 = weightssum == 0.0
    # . One boolean for each weight.
    weight_is0 = weights == 0.0
    # . One boolean.
    weights_all0 = weight_is0.all()

    # See which case we have and calculate.

    # "Normal": sum of weights != 0.
    if not weights_sum0:
        return _dataframe_colwavg_with_weightssumnot0(df, weights, weightssum)

    # Sum of weights == 0 but not all values are 0.
    elif not weights_all0:
        return _dataframe_colwavg_with_weightssum0notall0(df, weights)

    # Each weight has a value of 0.
    else:
        return _dataframe_colwavg_with_weightsall0(df)


def _dataframe_colwavg_with_weightssumnot0(
    df: pd.DataFrame,
    weights: pd.Series | pd.DataFrame,
    weightssum: float | tools_unit.Q_ | pd.Series,
) -> pd.Series:
    # Calculate the weighted average if sum of weights != 0.
    weight_is0 = weights == 0.0
    value_isna = df.isna()
    df = df.where(~(weight_is0 & value_isna), other=0.0)  # to ignore NaN if allowed
    factors = weights.div(weightssum).astype(float)
    scaled_values = df.mul(factors, axis=0)  # fast, even with quantities
    result = scaled_values.sum()
    return result


def _dataframe_colwavg_with_weightssum0notall0(
    df: pd.DataFrame, weights: pd.Series | pd.DataFrame
) -> pd.Series:
    # Calculate the weighted average if sum of weights == 0, but not all weights are 0.

    series = []

    if len(df.index) == 0:
        return series

    # Ensure weights is Dataframe.
    if isinstance(weights, pd.Series):
        weights = pd.DataFrame({c: weights for c in df})

    returnvalues = {}
    for c, s in df.items():
        relevantweights = weights[c]
        relevantweights_not0 = relevantweights != 0.0

        # Column contains NaN-values whose weight != 0
        if (s.isna() & relevantweights_not0).any():
            returnvalues[c] = np.nan

        # Column contain no NaN-values, or only where weight == 0. Ignore these elements.
        else:
            relevantseries = s[~s.isna() & relevantweights_not0]
            if not len(relevantseries) or relevantseries.nunique() > 1:
                returnvalues[c] = RESULT_IF_WEIGHTSUM0_VALUESNOTUNIFORM
            else:
                returnvalues[c] = relevantseries.iloc[0]

    return pd.Series(returnvalues)


def _dataframe_colwavg_with_weightsall0(
    df: pd.DataFrame,
) -> pd.Series:
    # Calculate the weighted average if all weights == 0.

    series = []

    if len(df.index) == 0:
        return series

    returnvalues = {}
    for c, s in df.items():
        if s.isna().any():
            returnvalues[c] = np.nan
        elif not len(s) or s.nunique() > 1:
            returnvalues[c] = RESULT_IF_WEIGHTSUM0_VALUESNOTUNIFORM
        else:
            returnvalues[c] = s.iloc[0]

    return pd.Series(returnvalues)


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


def concatseries(series: Iterable[pd.Series]) -> pd.Series:
    """Concatenate some series, and try to make it a pint-series if possible."""
    return tools_unit.avoid_frame_of_objects(pd.concat(series).sort_index(), False)
