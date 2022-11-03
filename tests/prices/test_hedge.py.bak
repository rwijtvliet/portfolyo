from portfolyo import dev
from portfolyo.prices import hedge
from portfolyo.prices import utils as putils
from portfolyo import testing
from portfolyo.tools import stamps
from pathlib import Path
import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("po", [True, False])
@pytest.mark.parametrize(
    ("values", "start", "result_base", "result_po"),
    [
        ([1, 2, 3], "2020-01-01", 2, (np.nan, 2)),
        (range(12), "2020-01-01", 5.5, (9.5, 3.5)),
        ([*[10] * (23 + 8), *[15] * 5], "2020-03-29", 10.69444, (15, 10)),
        ([*[10] * (25 + 8), *[15] * 5], "2020-10-25", 10.65789, (15, 10)),
    ],
)
def test_basic_volhedge(values, start, result_base, result_po, po):
    """Test if basic volume hedge with single values is calculated correctly."""
    index = pd.date_range(start, freq="H", periods=len(values), tz="Europe/Berlin")
    df = pd.DataFrame({"w": values, "p": 100.0}, index)
    result = hedge._hedge(df, how="vol", po=po)
    if not po:
        # Result for base hedge
        expected_result = pd.Series({"w": result_base, "p": 100.0})
    else:
        # Result for peak/offpeak hedge
        values = {}
        if not np.isnan(result_po[0]):
            values[("peak", "w")] = result_po[0]
            values[("peak", "p")] = 100.0
        if not np.isnan(result_po[1]):
            values[("offpeak", "w")] = result_po[1]
            values[("offpeak", "p")] = 100.0

        expected_result = pd.Series(values)
    testing.assert_series_equal(result.sort_index(), expected_result.sort_index())


@pytest.mark.parametrize("withunit", [True, False])
@pytest.mark.parametrize("how", ["vol", "val"])
@pytest.mark.parametrize("po", [True, False])
@pytest.mark.parametrize("aggfreq", ["MS", "QS", "AS"])
@pytest.mark.parametrize("freq", ["H", "D"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
def test_hedge_fromexcel(tz, freq, aggfreq, po, how, withunit):
    """Test if hedge results are correctly calculated, by comparing against previously calculated results."""
    if freq == "D" and po:
        return  # Only decompose in peak and offpeak if hourly values

    if tz is None:
        return  # TODO: allow hedging with dataframes without timezone information, and remove this block so that they are also checked

    path = Path(__file__).parent / "test_hedge_data.xlsx"
    sheetname = f'{freq}_{"None" if tz is None else tz.replace("/", "")}'

    # Input data.
    df_in = pd.read_excel(path, sheetname, header=6, index_col=0, usecols="A,B:C")
    if tz:
        df_in = df_in.tz_localize(tz, ambiguous="infer")
    df_in.index.freq = pd.infer_freq(df_in.index)
    w_in, p_in = df_in.w, df_in.p
    if withunit:
        w_in = w_in.astype("pint[MW]")
        p_in = p_in.astype("pint[Eur/MWh]")

    # Expected output data.
    df_out = pd.read_excel(path, f"{sheetname}_out", header=[3, 4, 5, 6], index_col=0)
    if tz:
        df_out = df_out.tz_localize(tz, ambiguous="infer")
    df_out.index.freq = pd.infer_freq(df_out.index)
    w_expected = df_out[aggfreq][po][how]["w"]
    p_expected = df_out[aggfreq][po].iloc[:, 0]  # price is first column
    if withunit:
        w_expected = w_expected.astype("pint[MW]")
        p_expected = p_expected.astype("pint[Eur/MWh]")

    # Test output data.
    w_test, p_test = hedge.hedge(w_in, p_in, how, aggfreq, po)

    testing.assert_series_equal(p_test, p_expected, check_names=False)
    testing.assert_series_equal(w_test, w_expected, check_names=False)
