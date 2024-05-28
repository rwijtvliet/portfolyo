"""Testing hedge functions using hard-coded hedge values."""

import numpy as np
import pandas as pd
import pytest

from portfolyo import testing, tools


@pytest.mark.parametrize("how", ["vol", "val"])
@pytest.mark.parametrize(
    "w_vals,start,w_expected",
    [
        ([1, 2, 3], "2020-01-01", 2),
        (range(12), "2020-01-01", 5.5),
        ([*[10] * (23 + 8), *[15] * 5], "2020-03-29", 10.69444),
        ([*[10] * (25 + 8), *[15] * 5], "2020-10-25", 10.65789),
    ],
)
def test_onehedge_uniformpricesandduration(w_vals, start, w_expected, how):
    """Test hedge with uniform prices and durations."""
    i = pd.date_range(start, freq="H", periods=len(w_vals), tz="Europe/Berlin")
    df = pd.DataFrame({"w": w_vals, "p": 100.0}, i)
    df["duration"] = tools.duration.index(i)

    result = tools.hedge.one_hedge(df, how=how)
    expected = pd.Series({"w": w_expected, "p": 100.0})
    testing.assert_series_equal(result, expected)


@pytest.mark.parametrize("how", ["vol", "val"])
@pytest.mark.parametrize(
    "w_vals,start,w_expected,tz",
    [
        ([1, 2], "2020-01-01", 1.483333333, None),  # 29 days in Feb
        ([1, 2], "2021-01-01", 1.474576271, None),  # 28 days in Feb
        (range(12), "2020-01-01", 5.51366120, None),  # no DST
        (range(12), "2020-01-01", 5.514458106, "Europe/Berlin"),  # DST
    ],
)
def test_onehedge_uniformprices(w_vals, start, w_expected, how, tz):
    """Test hedge with uniform prices but distinct durations."""
    i = pd.date_range(start, freq="MS", periods=len(w_vals), tz=tz)
    df = pd.DataFrame({"w": w_vals, "p": 100.0}, i)
    df["duration"] = tools.duration.index(i)

    result = tools.hedge.one_hedge(df, how=how)
    expected = pd.Series({"w": w_expected, "p": 100.0})
    testing.assert_series_equal(result, expected)


@pytest.mark.parametrize("how", ["vol", "val"])
@pytest.mark.parametrize(
    "w_vals,p_vals,start,p_expected,w_expected_vol,w_expected_val,tz",
    [
        (
            [1, 2],
            [100, 200],
            "2020-01-01",
            148.333333333,
            1.4833333,
            1.651685393,
            None,
        ),  # 29 days in Feb
        (
            [1, 2],
            [100, 200],
            "2021-01-01",
            147.4576271,
            1.474576271,
            1.643678161,
            None,
        ),  # 28 days in Feb
        (
            np.arange(12),
            np.arange(12) * 100,
            "2020-01-01",
            551.366120,
            5.51366120,
            7.673934589,
            None,
        ),  # no DST
        (
            np.arange(12),
            np.arange(12) * 100,
            "2020-01-01",
            551.4458106,
            5.514458106,
            7.674415244,
            "Europe/Berlin",
        ),  # DST
    ],
)
def test_onehedge(
    w_vals, p_vals, start, w_expected_val, w_expected_vol, p_expected, how, tz
):
    """Test value hedge."""
    i = pd.date_range(start, freq="MS", periods=len(w_vals), tz=tz)
    df = pd.DataFrame({"w": w_vals, "p": p_vals}, i)
    df["duration"] = tools.duration.index(i)

    result = tools.hedge.one_hedge(df, how=how)
    w_expected = w_expected_val if how == "val" else w_expected_vol
    expected = pd.Series({"w": w_expected, "p": p_expected})
    testing.assert_series_equal(result, expected)
