"""Testing of pandas objects, taking into account they may have units."""

import functools
from ..tools import nits
import pandas as pd


@functools.wraps(pd.testing.assert_frame_equal)
def assert_frame_equal(left, right, *args, **kwargs):
    # Dataframes equal even if *order* of columns is not the same.
    left = nits.drop_units(left).sort_index(axis=1)
    right = nits.drop_units(right).sort_index(axis=1)
    pd.testing.assert_frame_equal(left, right, *args, **kwargs)


@functools.wraps(pd.testing.assert_series_equal)
def assert_series_equal(left, right, *args, **kwargs):
    # If series of quantities, compare its magnitude (in base units)
    left = nits.drop_units(left)
    right = nits.drop_units(right)
    pd.testing.assert_series_equal(left, right, *args, **kwargs)


assert_index_equal = pd.testing.assert_index_equal
