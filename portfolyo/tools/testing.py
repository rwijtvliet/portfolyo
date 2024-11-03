"""Testing of pandas objects, taking into account they may have units."""

import functools
from typing import Any

import numpy as np
import pandas as pd
import pint_pandas
import pint

from . import unit as tools_unit


def assert_value_equal(left: Any, right: Any):
    try:
        if np.isnan(left) and np.isnan(right):
            return
        assert np.isclose(left, right)
    except Exception as e:
        raise AssertionError from e


@functools.wraps(pd.testing.assert_frame_equal)
def assert_frame_equal(left: pd.DataFrame, right: pd.DataFrame, *args, **kwargs):
    # Dataframes equal even if *order* of columns is not the same.
    left = left.sort_index(axis=1)
    right = right.sort_index(axis=1)
    assert set(left.columns) == set(right.columns)

    for (coll, sl), (colr, sr) in zip(left.items(), right.items()):
        # Names must match.
        assert coll == colr
        # Series must match.
        assert_series_equal(sl, sr, *args, **kwargs)


@functools.wraps(pd.testing.assert_series_equal)
def assert_series_equal(left: pd.Series, right: pd.Series, *args, **kwargs):
    if pd.api.types.is_float_dtype(left) or pd.api.types.is_integer_dtype(left):
        leftm = left.replace([np.inf, -np.inf], np.nan)
        rightm = right.replace([np.inf, -np.inf], np.nan)
        pd.testing.assert_series_equal(leftm, rightm, *args, **kwargs)

    elif isinstance(left.dtype, pint_pandas.PintType):
        # Units.
        try:
            right = right.pint.to(left.pint.units)
        except pint.DimensionalityError as e:
            raise AssertionError("Dimensions not equal.") from e
        # Magnitudes.
        leftm = left.pint.magnitude.replace([np.inf, -np.inf], np.nan)
        rightm = right.pint.magnitude.replace([np.inf, -np.inf], np.nan)
        pd.testing.assert_series_equal(leftm, rightm, *args, **kwargs)

    elif pd.api.types.is_object_dtype(left) and isinstance(left.iloc[0], tools_unit.Q_):
        # series of quantities?
        leftm = left.apply(lambda q: q.magnitude).replace([np.inf, -np.inf], np.nan)
        leftu = left.apply(lambda q: q.units)
        rightm = right.apply(lambda q: q.magnitude).replace([np.inf, -np.inf], np.nan)
        rightu = right.apply(lambda q: q.units)
        # TODO: this (incorrectly) raises AssertionError if e.g. 5 MWh is compared with 5000 kWh.
        pd.testing.assert_series_equal(leftm, rightm, *args, **kwargs)
        assert (leftu == rightu).all()

    else:
        # Whatever this is, use normal pandas testing function.
        pd.testing.assert_series_equal(left, right, *args, **kwargs)


assert_index_equal = pd.testing.assert_index_equal


def assert_indices_compatible(left: pd.DatetimeIndex, right: pd.DatetimeIndex):
    if (lf := left.freq) != (rf := right.freq):
        raise AssertionError(f"Indices have unequal frequency: {lf} and {rf}.")
    if (lt := left[0].time()) != (rt := right[0].time()):
        raise AssertionError(f"Indices that have unequal start-of-day; {lt} and {rt}.")
    if (lz := left.tz) != (rz := right.tz):
        raise AssertionError(f"Indices that have unequal timezone: {lz} and {rz}.")


def assert_w_q_compatible(freq: str, w: pd.Series, q: pd.Series):
    """Assert if timeseries with power- and energy-values are consistent."""
    if freq == "15min":
        assert_series_equal(q, w * tools_unit.Q_(0.25, "h"), check_names=False)
    elif freq == "h":
        assert_series_equal(q, w * tools_unit.Q_(1.0, "h"), check_names=False)
    elif freq == "D":
        assert (q >= w * tools_unit.Q_(22.99, "h")).all()
        assert (q <= w * tools_unit.Q_(25.01, "h")).all()
    elif freq == "MS":
        assert (q >= w * 27 * tools_unit.Q_(24.0, "h")).all()
        assert (q <= w * 32 * tools_unit.Q_(24.0, "h")).all()
    elif freq == "QS":
        assert (q >= w * 89 * tools_unit.Q_(24.0, "h")).all()
        assert (q <= w * 93 * tools_unit.Q_(24.0, "h")).all()
    elif freq == "YS":
        assert (q >= w * tools_unit.Q_(8759.9, "h")).all()
        assert (q <= w * tools_unit.Q_(8784.1, "h")).all()
    else:
        raise ValueError(f"Uncaught value for freq: {freq}.")


def assert_p_q_r_compatible(r: pd.Series, p: pd.Series, q: pd.Series):
    """Assert if timeseries with revenue-, power-, and energy-values are consistent."""
    assert_series_equal(r, q * p, check_names=False)
