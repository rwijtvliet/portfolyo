"""Testing of pandas objects, taking into account they may have units."""

import functools

import numpy as np
import pandas as pd

from .. import tools


@functools.wraps(pd.testing.assert_frame_equal)
def assert_frame_equal(left, right, *args, **kwargs):
    # Dataframes equal even if *order* of columns is not the same.
    left = tools.unit.drop_units(left).sort_index(axis=1)
    right = tools.unit.drop_units(right).sort_index(axis=1)
    left = left.replace([np.inf, -np.inf], np.nan)
    right = right.replace([np.inf, -np.inf], np.nan)
    pd.testing.assert_frame_equal(left, right, *args, **kwargs)


@functools.wraps(pd.testing.assert_series_equal)
def assert_series_equal(left, right, *args, **kwargs):

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
