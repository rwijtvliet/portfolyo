from typing import Iterable, Union

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
    s: pd.Series, weights: Union[Iterable, pd.Series] = None
) -> Union[float, tools_unit.Q_]:
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


def dataframe(
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
