from typing import Union

import numpy as np
import pandas as pd
import pytest
from numpy import nan
from portfolyo import dev, testing, tools


@pytest.mark.parametrize(
    ("values", "maxgap", "gapvalues"),
    [
        ([1, 2, 3, 4, 25, 7, 8], 1, []),
        ([1, 2, 3, 4, nan, 7, 8], 1, [5.5]),
        ([1, 2, 3, 4, nan, 7, 8], 2, [5.5]),
        ([1, 2, 3, 4, nan, 7, 8], 3, [5.5]),
        ([3, 2, 1, nan, nan, 7, 8], 1, [nan, nan]),
        ([3, 2, 1, nan, nan, 7, 8], 2, [3, 5]),
        ([3, 2, 1, nan, nan, 7, 8], 3, [3, 5]),
        ([nan, 2, 1, nan, nan, 7, nan], 1, [nan, nan, nan, nan]),
        ([nan, 2, 1, nan, nan, 7, nan], 2, [nan, 3, 5, nan]),
    ],
)
@pytest.mark.parametrize(
    ("index", "tol"),
    [
        (range(7), 0),
        (range(-3, 4), 0),
        (pd.date_range("2020", periods=7, freq="D"), 0),
        (pd.date_range("2020", periods=7, freq="M", tz="Europe/Berlin"), 0.04),
    ],
)
def test_fill_gaps(values, index, maxgap, gapvalues, tol):
    """Test if gaps are correctly interpolated."""
    # Test as Series.
    s = pd.Series(values, index)
    s_new = tools.frames.fill_gaps(s, maxgap)
    s[s.isna()] = gapvalues
    pd.testing.assert_series_equal(s_new, s, rtol=tol)
    # Test as DataFrame.
    df = pd.DataFrame({"a": values}, index)
    df_new = tools.frames.fill_gaps(df, maxgap)
    df[df.isna()] = gapvalues
    pd.testing.assert_frame_equal(df_new, df, rtol=tol)


@pytest.mark.parametrize(
    ("df_columns", "header", "expected_columns"),
    [
        (["A"], "B", [("B", "A")]),
        (["A1", "A2"], "B", [("B", "A1"), ("B", "A2")]),
        (pd.MultiIndex.from_tuples([("B", "A")]), "C", [("C", "B", "A")]),
        (
            pd.MultiIndex.from_product([["B"], ["A1", "A2"]]),
            "C",
            [("C", "B", "A1"), ("C", "B", "A2")],
        ),
        (
            pd.MultiIndex.from_tuples([("B1", "A1"), ("B2", "A2")]),
            "C",
            [("C", "B1", "A1"), ("C", "B2", "A2")],
        ),
    ],
)
def test_addheader_tocolumns(df_columns, header, expected_columns):
    """Test if header can be added to the columns of a dataframe."""
    i = dev.get_index()
    df_in = pd.DataFrame(np.random.rand(len(i), len(df_columns)), i, df_columns)
    result_columns = tools.frames.add_header(df_in, header).columns.to_list()
    assert np.array_equal(result_columns, expected_columns)


# TODO: put in ... fixture (?)
test_index_D = dev.get_index("D")
test_index_D_deconstructed = test_index_D.map(lambda ts: (ts.year, ts.month, ts.day))
test_index_H = dev.get_index("H")
test_index_H_deconstructed = test_index_H.map(lambda ts: (ts.year, ts.month, ts.day))


@pytest.mark.parametrize(
    ("df_index", "header", "expected_index"),
    [
        (test_index_D, "test", [("test", i) for i in test_index_D]),
        (
            test_index_D_deconstructed,
            "test",
            [("test", *i) for i in test_index_D_deconstructed],
        ),
        (test_index_H, "test", [("test", i) for i in test_index_H]),
        (
            test_index_H_deconstructed,
            "test",
            [("test", *i) for i in test_index_H_deconstructed],
        ),
    ],
)
def test_addheader_torows(df_index, header, expected_index):
    """Test if header can be added to the rows of a dataframe."""
    df_in = pd.DataFrame(np.random.rand(len(df_index), 2), df_index, ["A", "B"])
    result_index = tools.frames.add_header(df_in, header, axis=0).index.to_list()
    assert np.array_equal(result_index, expected_index)


@pytest.mark.parametrize(
    ("freq", "num", "wanted", "strict", "expected"),
    [
        # D
        # . enough
        ("D", 10, None, False, "D"),
        ("D", 10, None, True, "D"),
        ("D", 10, "MS", False, ValueError),
        ("D", 10, "MS", True, ValueError),
        ("D", 10, "D", False, "D"),
        ("D", 10, "D", True, "D"),
        # . too few
        ("D", 2, None, False, None),
        ("D", 2, None, True, ValueError),
        ("D", 2, "MS", False, ValueError),
        ("D", 2, "MS", True, ValueError),
        ("D", 2, "D", False, "D"),
        ("D", 2, "D", True, "D"),
        # 15T, too few
        ("15T", 2, None, False, None),
        ("15T", 2, None, True, ValueError),
        ("15T", 2, "MS", False, ValueError),
        ("15T", 2, "MS", True, ValueError),
        ("15T", 2, "15T", False, "15T"),
        ("15T", 2, "15T", True, "15T"),
        # invalid freq, not correctable
        # . enough
        ("2D", 10, None, False, "2D"),
        ("2D", 10, None, True, ValueError),
        ("2D", 10, "MS", False, ValueError),
        ("2D", 10, "MS", True, ValueError),
        ("2D", 10, "2D", False, "2D"),
        ("2D", 10, "2D", True, ValueError),
        # . too few
        ("2D", 2, None, False, None),
        ("2D", 2, None, True, ValueError),
        ("2D", 2, "MS", False, ValueError),
        ("2D", 2, "MS", True, ValueError),
        ("2D", 2, "2D", False, "2D"),
        ("2D", 2, "2D", True, ValueError),
        # invalid freq, correctable
        # . enough
        ("QS-APR", 10, None, False, "QS"),
        ("QS-APR", 10, None, True, "QS"),
        ("QS-APR", 10, "MS", False, ValueError),
        ("QS-APR", 10, "MS", True, ValueError),
        ("QS-APR", 10, "QS", False, "QS"),
        ("QS-APR", 10, "QS", True, "QS"),
        # . too few
        ("QS-APR", 2, None, False, None),
        ("QS-APR", 2, None, True, ValueError),
        ("QS-APR", 2, "MS", False, ValueError),
        ("QS-APR", 2, "MS", True, ValueError),
        ("QS-APR", 2, "QS", False, "QS"),
        ("QS-APR", 2, "QS", True, "QS"),
    ],
)
def test_setfreq(
    freq: str, num: int, wanted: str, strict: bool, expected: Union[str, Exception]
):
    # Create frame, with frequency, which is not set.
    fr = pd.Series(np.random.rand(num), pd.date_range("2020", periods=num, freq=freq))
    fr.index.freq = None
    # Test.
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            _ = tools.frames.set_frequency(fr, wanted, strict)
        return
    fr2 = tools.frames.set_frequency(fr, wanted, strict)
    assert fr2.index.freq == expected


# TODO: put in ... fixture (?)
test_values = np.random.rand(len(test_index_D), 10)
test_df_1 = pd.DataFrame(test_values[:, :2], test_index_D, ["A", "B"])
test_df_2 = pd.DataFrame(test_values[:, 2], test_index_D, ["C"])
expect_concat_12 = pd.DataFrame(test_values[:, :3], test_index_D, ["A", "B", "C"])
test_df_3 = pd.DataFrame(test_values[:, 2], test_index_D, pd.Index([("D", "C")]))
expect_concat_13 = pd.DataFrame(
    test_values[:, :3], test_index_D, pd.Index([("A", ""), ("B", ""), ("D", "C")])
)


@pytest.mark.parametrize(
    ("dfs", "axis", "expected"),
    [
        ([test_df_1, test_df_2], 1, expect_concat_12),
        ([test_df_1, test_df_3], 1, expect_concat_13),
    ],
)
def test_concat(dfs, axis, expected):
    """Test if concatenation works as expected."""
    result = tools.frames.concat(dfs, axis)
    testing.assert_frame_equal(result, expected)


vals1 = np.array([1, 2.0, -4.1234, 0])
vals2 = np.array([1, 2.0, -4.1234, 0.5])


@pytest.mark.parametrize(
    ("s1", "s2", "expected_result"),
    [
        (pd.Series(vals1), pd.Series(vals1), True),
        (pd.Series(vals1), pd.Series(vals2), False),
        (pd.Series(vals1), pd.Series(vals1, dtype="pint[MW]"), False),
        (pd.Series(vals1).astype("pint[MW]"), pd.Series(vals1, dtype="pint[MW]"), True),
        (
            pd.Series(vals1 * 1000).astype("pint[kW]"),
            pd.Series(vals1, dtype="pint[MW]"),
            True,
        ),
        (
            pd.Series(vals1 * 1000).astype("pint[MW]"),
            pd.Series(vals1, dtype="pint[MW]"),
            False,
        ),
    ],
)
def test_series_allclose(s1, s2, expected_result):
    """Test if series can be correctly compared, even if they have a unit."""
    assert tools.frames.series_allclose(s1, s2) == expected_result
