"""Testing of pandas objects, taking into account they may have units."""

import functools
from typing import Any

import numpy as np
import pandas as pd
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
    leftm, leftu = tools_unit.split_magn_unit(left)
    rightm, rightu = tools_unit.split_magn_unit(right)

    # Magnitudes must be the same.
    leftm = leftm.replace([np.inf, -np.inf], np.nan)
    rightm = rightm.replace([np.inf, -np.inf], np.nan)
    pd.testing.assert_series_equal(leftm, rightm, *args, **kwargs)

    # Units must be the same.
    if leftu is None:
        assert leftu is rightu
    elif isinstance(leftu, pint.Unit):  # all values share the same unit, leftu is Unit
        assert leftu == rightu
    else:  # each value has its own unit; leftu is Series
        pd.testing.assert_series_equal(leftu, rightu)


assert_index_equal = pd.testing.assert_index_equal


def assert_indices_compatible(left: pd.DatetimeIndex, right: pd.DatetimeIndex):
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
