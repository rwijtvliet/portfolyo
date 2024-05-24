from pathlib import Path

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


@pytest.mark.parametrize("withunits", ["units", "nounits"])
@pytest.mark.parametrize("how", ["vol", "val"])
@pytest.mark.parametrize("bpo", ["b", "po"])
@pytest.mark.parametrize("aggfreq", ["MS", "QS", "AS"])
@pytest.mark.parametrize("freq", ["H", "D"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
def test_hedge_fromexcel(tz, freq, aggfreq, bpo, how, withunits):
    """Test if hedge results are correctly calculated, by comparing against previously calculated results."""
    if freq == "D" and bpo == "po":
        pytest.skip("Don't decompose in peak and offpeak if daily values")

    path = Path(__file__).parent / "test_hedge_data.xlsx"
    sheetname = f'{freq}_{"None" if tz is None else tz.replace("/", "")}'
    peak_fn = tools.product.germanpower_peakfn if bpo == "po" else None

    # Input data.
    dfin = pd.read_excel(path, sheetname, header=6, index_col=0, usecols="A,B:C")
    if tz:
        dfin = dfin.tz_localize(tz, ambiguous="infer")
    dfin.index.freq = pd.infer_freq(dfin.index)
    win, pin = dfin.w, dfin.p
    if withunits == "units":
        win = win.astype("pint[MW]")
        pin = pin.astype("pint[Eur/MWh]")

    # Expected output data.
    dfout = pd.read_excel(path, f"{sheetname}_out", header=[3, 4, 5, 6], index_col=0)
    if tz:
        dfout = dfout.tz_localize(tz, ambiguous="infer")
    dfout.index.freq = pd.infer_freq(dfout.index)
    w_expected = dfout[aggfreq][bpo == "po"][how]["w"]
    p_expected = dfout[aggfreq][bpo == "po"].iloc[:, 0]  # price is first column
    if withunits == "units":
        w_expected = w_expected.astype("pint[MW]")
        p_expected = p_expected.astype("pint[Eur/MWh]")

    # Test output data.
    w_result, p_result = tools.hedge.hedge(win, pin, how, peak_fn, aggfreq)

    testing.assert_series_equal(p_result, p_expected, check_names=False)
    testing.assert_series_equal(w_result, w_expected, check_names=False)
