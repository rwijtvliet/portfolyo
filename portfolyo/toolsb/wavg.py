from typing import Iterable, Mapping, overload

import numpy as np
import pandas as pd
import pint
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
# --> Otherwise, result is NaN.

# all weights are 0
# 0, 0, 0            10, 20, 30          NaN if values distinct         NaN
# 0, 0, 0            10, 10, 10          Value if values identical      10
# 0, 0, 0            10, 10, NaN         NaN if values include NaN      NaN
# --> If values contain NaN --> result is NaN
# --> Otherwise, if values are identical --> result is that value
# --> Otherwise, result is NaN.

# Note concerning units:
# Units are considered. However, only weights and frames with uniform dimensionality are considered.
# OK:      weights = [3 MW, 0.05 GW], values = [4 Eur/MWh, 10 ctEur/kWh]
# Not OK:  weights = [3 MW, 0.05 Eur/MWh], values = [4 Eur/MWh, 10 MW].


@overload
def general(
    fr: pd.Series,
    weights: Iterable | Mapping | pd.Series | None = None,
    axis: int = 0,
) -> pint.Quantity:
    ...


@overload
def general(
    fr: pd.DataFrame,
    weights: Iterable | Mapping | pd.Series | pd.DataFrame | None = None,
    axis: int = 0,
) -> pd.Series:
    ...


def general(
    fr: pd.Series | pd.DataFrame,
    weights: Iterable | Mapping | pd.Series | pd.DataFrame | None = None,
    axis: int = 0,
) -> float | pint.Quantity | pd.Series:
    """Weighted average of series or dataframe.

    Parameters
    ----------
    fr
        Input values.
    weights, optional
        Weights. If provided as Mapping or Series, weights and values are aligned along indices/keys.
        If no weights are provided, normal (unweighted) average is returned.
    axis, optional
        Axis to collapse.
        - if 0, collapse the rows (i.e., for each column, calculate average over all rows);
        - if 1, collapse the columns (i.e., for each row, calculate average over all columns).
        Ignored if ``fr`` is a Series.

    Returns
    -------
        Weighted average. Single float or single Quantitiy if ``fr`` is a Series; Series if ``fr``
        is a Dataframe.
    """
    if isinstance(fr, pd.DataFrame):
        return dataframe(fr, weights, axis)
    elif isinstance(fr, pd.Series):
        return series(fr, weights)
    else:
        raise TypeError(f"Parameter ``fr`` must be Series or DataFrame; got {type(fr)}.")


@tools_unit.apply_coercion_pintframe("s")
def series(s: pd.Series, weights: Iterable | Mapping | pd.Series | None = None) -> pint.Quantity:
    """Weighted average of series.

    Parameters
    ----------
    s
        Input values.
    weights, optional
        Weights. If provided as Mapping or Series, weights and values are aligned along indices/keys.
        If no weights are provided, normal (unweighted) average is returned.

    Returns
    -------
        Weighted average.

    Notes
    -----
    Will raise Error if values in ``s`` have distinct units.
    """
    units = s.pint.units
    magnitudes = s.pint.magnitude

    if weights is None:
        wavg_float = magnitudes.values.mean()

    else:
        weights = _weights_as_floatseries(weights, s.index)
        magnitudes = magnitudes.loc[weights.index]  # align and remove unneeded values
        wavg_float = _numpy_values1d_weights1d(magnitudes.values, weights.values)

    return wavg_float * units


@tools_unit.apply_coercion_pintframe("df")
def dataframe(
    df: pd.DataFrame,
    weights: Iterable | Mapping | pd.Series | pd.DataFrame | None = None,
    axis: int = 0,
) -> pd.Series:
    """Weighted average of dataframe.

    Parameters
    ----------
    df
        Input values.
    weights, optional
        Weights. If provided as Mapping or Series, weights and values are aligned along indices/keys.
        If no weights are provided, normal (unweighted) average is returned.
    axis, optional
        Axis to collapse.
        - if 0, collapse the rows (i.e., for each column, calculate average over all rows);
        - if 1, collapse the columns (i.e., for each row, calculate average over all columns).

    Returns
    -------
        Weighted average.

    Notes
    -----
    Will raise error if axis == 1 and columns have distinct unit-dimensions.
    """
    weights = _weights_for_2dvalues(weights, df.index, df.columns, axis)
    return _dataframe_axis1(df, weights) if axis == 1 else _dataframe_axis0(df, weights)


def _dataframe_axis1(df: pd.DataFrame, weights: None | pd.Series | pd.DataFrame) -> pd.Series:
    # When collapsing the columns, they all need to have the same unit dimension, e.g. MW and kW.
    # However, the calculations don't work unless the units are actually identical, so do that here.
    # The resulting series will also have this one unit for all values.
    units = df.iloc[0, 0].units
    magnitudes = pd.DataFrame({c: s.pint.to(units).pint.magnitude for c, s in df.items()})

    if weights is None:
        wavg_floatarray = magnitudes.values.mean(axis=1)

    elif isinstance(weights, pd.Series):
        # weights = _weights_as_floatseries(weights, magnitudes.columns)
        magnitudes = magnitudes.loc[:, weights.index]  # align and remove unneeded value cols
        wavg_floatarray = _numpy_values2d_weights1d(magnitudes.values, weights.values, 1)

    else:  # isinstance(weights, pd.DataFrame):
        # weights = _weights_as_floatdf(weights)
        if set(weights.index) != set(magnitudes.index):
            raise ValueError("To reduce the columns, all rows must be present in the weights.")
        weights = weights.loc[magnitudes.index, :]  # align and keep original index order
        magnitudes = magnitudes.loc[:, weights.columns]  # align and remove unneeded value columns
        wavg_floatarray = _numpy_values2d_weights2d(magnitudes.values, weights.values, 1)

    return pd.Series(wavg_floatarray, magnitudes.index).astype(f"pint[{units}]")


def _dataframe_axis0(df: pd.DataFrame, weights: None | pd.Series | pd.DataFrame) -> pd.Series:
    # When collapsing the rows, all columns can have distinct units, we store these to later re-apply them.
    units = df.dtypes.apply(lambda pt: pt.units)
    magnitudes = pd.DataFrame({c: s.pint.magnitude for c, s in df.items()})

    if weights is None:
        wavg_floatarray = magnitudes.values.mean(axis=0)

    elif isinstance(weights, pd.Series):
        # weights = _weights_as_floatseries(weights, magnitudes.index)
        magnitudes = magnitudes.loc[weights.index, :]  # align and remove unneeded value rows
        wavg_floatarray = _numpy_values2d_weights1d(magnitudes.values, weights.values, 0)

    else:  # isinstance(weights, pd.DataFrame)
        if set(weights.columns) != set(magnitudes.columns):
            raise ValueError("To reduce the rows, all columns must be present in the weights.")
        weights = weights.loc[:, magnitudes.columns]  # align and keep original column order
        magnitudes = magnitudes.loc[weights.index, :]  # align and remove unneeded value rows
        wavg_floatarray = _numpy_values2d_weights2d(magnitudes.values, weights.values, 0)

    return pd.Series({c: v * units[c] for c, v in zip(magnitudes.columns, wavg_floatarray)})


def _numpy_values1d_weights1d(values: np.ndarray, weights: np.ndarray) -> float:
    # values = (M)
    # weights = (M)

    if np.any(np.isnan(weights)):
        raise ValueError("Weights must not contain nan-values.")
    if weights.shape != values.shape:
        raise ValueError("Length of weights must match length of values.")

    # Only one of the following 3 cases can apply.

    weight_is0 = np.isclose(weights, 0)
    if np.all(weight_is0):  # case 1: all weights are zero: nan unless all values same
        return _numpy_uniquenonnan_1d(values)
    weights, values = (
        weights[~weight_is0],
        values[~weight_is0],
    )  # keep only nonzero weights/values
    weight_sum = np.sum(weights)
    if np.isclose(weight_sum, 0):  # case2: sum of weights is zero: nan unless all values same
        return _numpy_uniquenonnan_1d(values)
    factors = weights / weight_sum
    scaled_values = values * factors
    return np.sum(scaled_values)  # case 3: sum of weights is not zero: normal wavg calc


def _numpy_values2d_weights1d(values: np.ndarray, weights: np.ndarray, axis: int) -> np.ndarray:
    # values = (MxN) if axis==0, or (NxM) if axis==1
    # weights = (M)

    if axis == 1:
        values = values.T  # so now we can always reduce rows

    # values = (MxN)
    # weights = (M)

    if np.any(np.isnan(weights)):
        raise ValueError("Weights must not contain nan-values.")
    if weights.shape[0] != values.shape[0]:
        raise ValueError("Length of weights must match length of values axis to reduce.")

    # Only one of the following 3 cases can apply, but calculations must still be done per column.
    # Returned array always has size (N).

    weight_is0 = np.isclose(weights, 0)
    if np.all(weight_is0):  # case 1: all weights are zero: nan unless all values same
        return _numpy_uniquenonnan_2d(values)
    weights, values = (
        weights[~weight_is0],
        values[~weight_is0],
    )  # keep only nonzero weights/values
    weight_sum = np.sum(weights)
    if np.isclose(weight_sum, 0):  # case 2: sum of weights is zero: nan unless all values same
        return _numpy_uniquenonnan_2d(values)
    factors = weights / weight_sum
    scaled_values = values * factors[:, np.newaxis]
    return np.sum(scaled_values, axis=0)  # case 3: sum of weights is not zero: normal wavg calc


def _numpy_values2d_weights2d(values: np.ndarray, weights: np.ndarray, axis: int) -> np.ndarray:
    # values = (MxN) if axis==0, or (NxM) if axis==1
    # weights = (MxN) if axis==0, or (NxM) if axis==1

    if axis == 1:
        values = values.T  # so now we can always reduce rows
        weights = weights.T  # so now we can always reduce rows

    # values = weights = (M x N)

    # NOTE: the commented-out code below works and is much easier than the code below it. However,
    # for arrays with many rows, it is horribly slow when reducing the columns, because each row is
    # calculated separately. The longer and more complex code in the remainder of this function
    # takes advantage of parallellization, and is much faster.
    # return np.array(
    #     [
    #         _numpy_values1d_weights1d(values1d, weights1d)
    #         for values1d, weights1d in zip(values.T, weights.T)
    #     ]
    # )

    if np.any(np.isnan(weights)):
        raise ValueError("Weights must not contain nan-values.")
    if weights.shape != values.shape:
        raise ValueError("Shape of weights must match shape of values.")

    # For each column, one of the following 3 cases can apply. We must therefore do all, until all
    # columns are accounted for. Returned array always has size (N).

    result = np.full(values.shape[1], np.nan)  # will be filled
    accounted_for = np.zeros(values.shape[1], dtype=bool)  # will be filled

    weight_is0 = np.isclose(weights, 0)
    cols_all0 = np.all(weight_is0, axis=0)
    if np.any(cols_all0):  # case 1: all weights are zero: nan unless all values same
        result[cols_all0] = _numpy_uniquenonnan_2d(values[:, cols_all0])
        accounted_for |= cols_all0
        if np.all(accounted_for):
            return result

    # weights, values = weights[~weight_is0], values[~weight_is0]  # keep only nonzero weights/values
    weight_sum = np.sum(weights, axis=0)
    cols_sum0 = np.isclose(weight_sum, 0) & ~accounted_for
    if np.any(cols_sum0):  # case 2: sum of weights is zero: nan unless all values same
        result[cols_sum0] = [
            _numpy_uniquenonnan_1d(column[~is0])
            for column, is0 in zip(values.T[cols_sum0], weight_is0.T[cols_sum0], strict=True)
        ]
        accounted_for |= cols_sum0
        if np.all(accounted_for):
            return result

    factors = weights[:, ~accounted_for] / weight_sum[~accounted_for]
    scaled_values = values[:, ~accounted_for] * factors
    result[~accounted_for] = np.sum(
        scaled_values, axis=0
    )  # case 3: sum of weights is not zero: normal wavg calc
    return result


def _numpy_uniquenonnan_1d(values: np.ndarray) -> float:
    if values.size and not np.any(np.isnan(values)) and np.allclose(values, values[0]):
        return values[0]
    else:
        return np.nan


def _numpy_uniquenonnan_2d(values: np.ndarray) -> np.ndarray:
    has_nan = np.any(np.isnan(values), axis=0)
    first_row = values[0, :]
    same_as_first = np.all(np.isclose(values, first_row[np.newaxis, :]), axis=0)
    keep = ~has_nan & same_as_first
    return np.where(keep, first_row, np.nan)


def _weights_for_2dvalues(
    weights: None | pd.Series | pd.DataFrame | Iterable | Mapping,
    refindex: pd.Index | None = None,
    refcolumns: pd.Index | None = None,
    axis: int | None = None,
) -> None | pd.Series | pd.DataFrame:
    # Coece to pintframe to ensure all weights have same unit (or are dimensionless), and then keep only the magnitude.
    if weights is None:
        return None
    elif isinstance(weights, pd.Series):
        return _weights_as_floatseries(weights)
    elif isinstance(weights, pd.DataFrame):
        return _weights_as_floatdf(weights)
    try:
        return _weights_as_floatdf(weights, refindex, refcolumns)
    except (TypeError, ValueError):
        pass
    try:
        return _weights_as_floatseries(weights, refindex if axis == 0 else refcolumns)
    except (TypeError, ValueError):
        pass
    raise ValueError("Can't turn weights into Series or Dataframe.")


def _weights_as_floatseries(
    weights: pd.Series | Iterable | Mapping, refindex: pd.Index | None = None
) -> pd.Series:
    # Step 1: turn into Series.
    if isinstance(weights, pd.Series):
        pass
    elif isinstance(weights, Mapping):
        weights = pd.Series(weights)
    elif isinstance(weights, Iterable):
        weights = pd.Series(weights, refindex)
    else:
        raise TypeError("``weights`` must be a series, a mapping, or another iterable.")
    # Step 2: coece to pintframe to ensure all weights have same unit (or are dimensionless), and then keep only the magnitude.
    return tools_unit.coerce_pintframe_oneunit(weights).pint.magnitude


def _weights_as_floatdf(
    weights: pd.DataFrame | Iterable | Mapping,
    refindex: pd.Index | None = None,
    refcolumns: pd.Index | None = None,
) -> pd.DataFrame:
    # Step 1: turn into Dataframe.
    if isinstance(weights, pd.DataFrame):
        pass
    elif isinstance(weights, Mapping):
        weights = pd.DataFrame(weights)
    elif isinstance(weights, Iterable):
        weights = pd.DataFrame(weights, refindex, refcolumns)
    else:
        raise TypeError("``weights`` must be a dataframe, a mapping, or another iterable.")
    # Step 2: coece to pintframe to ensure all weights have same unit (or are dimensionless), and then keep only the magnitude.
    return pd.DataFrame(
        {c: s.pint.magnitude for c, s in tools_unit.coerce_pintframe_oneunit(weights).items()}
    )
