from typing import Iterable, Mapping, overload

import numpy as np
import pandas as pd
import pint_pandas
from typing import Callable
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

# Handling units
# --------------
# Mathematically, weights must have uniform dimension (but not identical units).
# Step 1: calculate the sum of the weights.
# Step 2: remove
# step 1) don't consider individual units, but calculate values * weights.
# step 2) calculate


@overload
def general(
    fr: pd.Series,
    weights: Iterable | Mapping | pd.Series | None = None,
    axis: int = 0,
) -> float | tools_unit.Q_:
    ...


@overload
def general(
    fr: pd.DataFrame,
    weights: Iterable | Mapping | pd.Series | pd.DataFrame | None = None,
    axis: int = 0,
) -> pd.Series:
    ...


def general(
    fr,
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
    raise TypeError(f"Parameter ``fr`` must be Series or DataFrame; got {type(fr)}.")


# Various functions to calculate the average from a series.


def unweighted(s: pd.Series) -> float | tools_unit.Q_:
    s = tools_unit.normalize_frame(s)
    return s.mean()


def unweighted_on_subset(keep: pd.Index):
    def unweighted_subset(s: pd.Series) -> float | tools_unit.Q_:
        s = s.loc[keep]
        return unweighted(s)

    return unweighted_subset


def nanunlessuniform(s: pd.Series) -> float | tools_unit.Q_:
    if any(s.isna()):
        return np.nan
    s = tools_unit.normalize_frame(s)
    if not values_are_uniform(s):
        return np.nan
    return s.iloc[0]


def nanunlessuniform_on_subet(keep: pd.Index):
    def nanunlessuniform_subset(s: pd.Series) -> float | tools_unit.Q_:
        s = s.loc[keep]
        return nanunlessuniform(s)

    return nanunlessuniform_subset


def wavg_on_subset(factors: pd.Series):
    def wavg_subset(s: pd.Series) -> float | tools_unit.Q_:
        s = s.loc[factors.index]
        if any(s.isna()):
            return np.nan
        s = tools_unit.normalize_frame(s)
        return sum(s * factors)

    return wavg_subset


def partial(
    weights: Iterable | Mapping | pd.Series | None, refindex: pd.Index
) -> Callable[[pd.Series], float | tools_unit.Q_]:
    """Do analyses and prepare values to quickly calculate the wavg of a series.

    Parameters
    ----------
    weights
        Weights to be used for calculating the average.
    refindex
        Index of the series that will be passed in.

    Returns
    -------
        Function that takes pandas.Series of values and returns the weighted average.
    """
    # Case: no weights provided: unweighted average
    if weights is None:
        return unweighted

    # Prep: ensure weights is also a Series of floats.
    weights = weights_as_floatseries(weights, refindex)

    # Check: is a value available for all weights?
    surplus_weights = [i not in refindex for i in weights.index]
    if any(surplus_weights):  # more weights than values
        raise ValueError(
            f"There are surplus weights (i.e., weights without a corresponding value): {weights.loc[surplus_weights]}."
        )

    # Case: all weights same (and unequal to 0): unweighted average.
    if weights.round(10).nunique() == 1 and not np.isclose(weights.iloc[0], 0.0):
        return unweighted_on_subset(weights.index)

    non0 = ~np.isclose(weights, 0.0)  # bool-array

    # Case: all weights same (and equal to 0): undefined unless uniform.
    if not any(non0):  # edge case (very uncommon)
        return nanunlessuniform

    weights = weights[non0]  # keep relevant weights
    weightsum = weights.sum()  # float
    weightsum0 = np.isclose(weightsum, 0.0)

    # Case: distinct weights, sum equal to 0: undefined unless uniform.
    if weightsum0:
        return nanunlessuniform_on_subet(weights.index)

    # Case: distinct weights, sum unequal to 0: finally, a real weighted avg.
    factors = weights.div(weightsum)
    return wavg_on_subset(factors)


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
    return partial(weights, s.index)(s)


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
    # Prep: orient to always collapse the rows and keep the columns.
    if axis == 1:
        df = df.T

    # Two cases:

    # a) Different set of weights in each column.
    if isinstance(weights, pd.DataFrame):
        if axis == 1:
            weights = weights.T

        if surplusweights := set(weights.columns) - set(df.columns):
            raise ValueError(
                f"No values found for one or more weights: {surplusweights}."
            )

        resultvalues = {}
        for c, weightseries in weights.items():
            resultvalues[c] = partial(weightseries, df.index)(df[c])
        result = pd.Series(resultvalues)

    # b) Same weights applicable to each column. Also applies to None
    else:
        fn = partial(weights, df.index)
        result = pd.Series({c: fn(s) for c, s in df.items()})

    # Transposing loses some properties, like .index.freq
    if axis == 1 and isinstance(result.index, pd.DatetimeIndex):
        result = tools_freq.guess_to_frame(result)
    return tools_unit.normalize_frame(result, False)


def weights_as_floatseries(
    weights: Iterable | Mapping, refindex: Iterable
) -> pd.Series:
    # Step 1: turn into Series.
    if isinstance(weights, pd.Series):
        weightseries = weights
    elif isinstance(weights, Mapping):
        weightseries = pd.Series(weights)
    elif isinstance(weights, Iterable):
        weightseries = pd.Series(weights, refindex)
    else:
        raise TypeError("``weights`` must be iterable or mapping.")
    # Step 2: avoid Series of Quantity-objects (convert to pint-series instead).
    weightseries = tools_unit.normalize_frame(weightseries)
    # Step 3: keep magnitude only.
    if isinstance(weightseries.dtype, pint_pandas.PintType):
        weightseries = weightseries.pint.magnitude
    return weightseries


def values_are_uniform(s: pd.Series) -> bool:
    """Return True if all values in series are same. If there are no values to compare, return True."""
    if isinstance(s.dtype, pint_pandas.PintType):
        s = s.pint.magnitude
    return s.round(10).nunique() <= 1
