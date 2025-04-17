"""Testing of pandas objects, taking into account they may have units."""

import functools
from typing import Any

import numpy as np
import pandas as pd
import pint_pandas
import pint
from . import unit as tools_unit
from . import freq as tools_freq


def assert_scalar_equal(left: Any, right: Any):
    try:
        if np.isnan(left) and np.isnan(right):  # np.nan != np.nan, so separate check needed here
            if isinstance(left, pint.Quantity):
                assert left.units == right.units
        else:
            assert np.isclose(
                left, right
            )  # works on Quantities too, even if left=5MW, right=5000kW
    except Exception as e:
        raise AssertionError from e


@functools.wraps(pd.testing.assert_index_equal)
def assert_index_equal(left: pd.Index, right: pd.Index, *args, **kwargs):
    assert isinstance(left, pd.DatetimeIndex) == isinstance(right, pd.DatetimeIndex)
    if isinstance(left, pd.DatetimeIndex):
        if ((left.freq is None) is not (right.freq is None)) or (
            left.freq != right.freq and tools_freq.up_or_down(left.freq, right.freq) != 0
        ):
            raise AssertionError(f"Unequal frequencies. Left: {left.freq}; right: {right.freq}.")
        left, right = left._with_freq(None), right._with_freq(None)
    pd.testing.assert_index_equal(left, right, *args, **kwargs)


@functools.wraps(pd.testing.assert_series_equal)
def assert_series_equal(left: pd.Series, right: pd.Series, *args, **kwargs):
    # Ensure pintseries, if possible.
    left, right = tools_unit.convert_pintframe(left), tools_unit.convert_pintframe(right)

    assert isinstance(left.dtype, pint_pandas.PintType) == isinstance(
        right.dtype, pint_pandas.PintType
    )

    # If we are here, both are pintseries, or both are not pintseries.

    if isinstance(left.dtype, pint_pandas.PintType):
        # For pintseries: make units equal to avoid incorrect assertionerror from pd.testing.
        assert tools_unit.get_basedimty(left) == tools_unit.get_basedimty(right)
        right = right.pint.to(left.pint.units)
        left, right = left.pint.magnitude, right.pint.magnitude

    # Following function works on all series, including series of pint Quantity objects. It only
    # does not work on pintseries, but the preprocessing above takes care of that case by using
    # only the magnitude. Also, even though np.nan != np.nan when comparing scalars, np.nan ==
    # np.nan when using the function below.
    assert_index_equal(left.index, right.index)  # use own index test
    try:
        pd.testing.assert_series_equal(
            left, right, *args, **{**kwargs, "check_index": False, "check_freq": False}
        )
    except TypeError:  # can happen if series of quantities
        for le, ri in zip(left, right):
            assert_scalar_equal(le, ri)


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


def assert_index_compatible(left: pd.DatetimeIndex, right: pd.DatetimeIndex):
    """Assert that indices are compatible, i.e., with equal frequency, start-of-day, and timezone."""
    if (lf := left.freq) != (r := right.freq):
        raise AssertionError(f"Indices have unequal frequency: {lf} and {r}.")
    if (lt := left[0].time()) != (rt := right[0].time()):
        raise AssertionError(f"Indices that have unequal start-of-day; {lt} and {rt}.")
    if (lz := left.tz) != (rz := right.tz):
        raise AssertionError(f"Indices that have unequal timezone; {lz} and {rz}.")


def assert_w_q_compatible(freq: str, w: pd.Series, q: pd.Series):
    """Assert that timeseries with power- and energy-values are consistent."""
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
