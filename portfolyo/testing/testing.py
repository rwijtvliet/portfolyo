"""Testing of pandas objects, taking into account they may have units."""

import functools

import numpy as np
import pandas as pd

from .. import tools
from ..tools.unit import Q_


@functools.wraps(pd.testing.assert_frame_equal)
def assert_frame_equal(left: pd.DataFrame, right: pd.DataFrame, *args, **kwargs):
    # Dataframes equal even if *order* of columns is not the same.
    left = tools.unit.drop_units(left).sort_index(axis=1)
    right = tools.unit.drop_units(right).sort_index(axis=1)
    left = left.replace([np.inf, -np.inf], np.nan)
    right = right.replace([np.inf, -np.inf], np.nan)
    pd.testing.assert_frame_equal(left, right, *args, **kwargs)


@functools.wraps(pd.testing.assert_series_equal)
def assert_series_equal(left: pd.Series, right: pd.Series, *args, **kwargs):
    if hasattr(left, "pint") and hasattr(right, "pint"):  # pint-series
        left, right = left.pint.to_base_units(), right.pint.to_base_units()
        assert left.pint.u == right.pint.u
        assert_series_equal(left.pint.m, right.pint.m, *args, **kwargs)

    elif left.dtype == right.dtype == object:  # maybe: series of Quantities
        try:
            lq = [v.to_base_units() for v in left.values]
            rq = [v.to_base_units() for v in right.values]
        except AttributeError as e:
            raise AssertionError from e
        left = pd.Series([q.m for q in lq], left.index)
        right = pd.Series([q.m for q in rq], right.index)
        assert_series_equal(left, right, *args, **kwargs)
        for lu, ru in zip([q.u for q in lq], [q.u for q in rq]):
            assert lu == ru

    else:  # normal series of floats or ints
        left = left.replace([np.inf, -np.inf], np.nan)
        right = right.replace([np.inf, -np.inf], np.nan)
        pd.testing.assert_series_equal(left, right, *args, **kwargs)


assert_index_equal = pd.testing.assert_index_equal


def assert_indices_compatible(left: pd.DatetimeIndex, right: pd.DatetimeIndex):
    if (lf := left.freq) != (r := right.freq):
        raise AssertionError(f"Indices have unequal frequency: {lf} and {r}.")
    if (lf := left[0].time()) != (r := right[0].time()):
        raise AssertionError(f"Indices that have unequal start-of-day; {lf} and {r}.")


def assert_w_q_compatible(freq: str, w: pd.Series, q: pd.Series):
    """Assert if timeseries with power- and energy-values are consistent."""
    if freq == "15T":
        assert_series_equal(q, w * Q_(0.25, "h"), check_names=False)
    elif freq == "H":
        assert_series_equal(q, w * Q_(1.0, "h"), check_names=False)
    elif freq == "D":
        assert (q >= w * Q_(22.99, "h")).all()
        assert (q <= w * Q_(25.01, "h")).all()
    elif freq == "MS":
        assert (q >= w * 27 * Q_(24.0, "h")).all()
        assert (q <= w * 32 * Q_(24.0, "h")).all()
    elif freq == "QS":
        assert (q >= w * 89 * Q_(24.0, "h")).all()
        assert (q <= w * 93 * Q_(24.0, "h")).all()
    elif freq == "AS":
        assert (q >= w * Q_(8759.9, "h")).all()
        assert (q <= w * Q_(8784.1, "h")).all()
    else:
        raise ValueError(f"Uncaught value for freq: {freq}.")


def assert_p_q_r_compatible(r: pd.Series, p: pd.Series, q: pd.Series):
    """Assert if timeseries with revenue-, power-, and energy-values are consistent."""
    assert_series_equal(r, q * p, check_names=False)
