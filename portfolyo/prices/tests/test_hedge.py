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


def _manually_calc_hedgeresults_poFalse(
    w: pd.Series, p: pd.Series, how: str, aggfreq: str
) -> tuple[pd.Series]:
    """
    'Manually' calculate the hedge result for MW and Eur/MWh timeseries ``w`` and ``p``,
    if we *do not* want to split in peak and offpeak values.
    Returns tuple with w_hedge, p_hedge series in original frequency.
    """
    # Calculate the sums.
    dic = {}
    for ts, wval, pval, dur in zip(w.index, w, p, w.index.duration.pint.magnitude):
        aggts = stamps.floor_ts(ts, aggfreq)  # Key: start of delivery period.
        _add_base(dic, aggts, wval, pval, dur)

    # Calculate the values.
    records = _records(dic, how)

    # Turn into long series.
    availkeys = np.array(list(records.keys()))
    keys = availkeys[np.searchsorted(availkeys, w.index, side="right") - 1]
    w_series = _makeseries([records[k]["w"] for k in keys], w.index)
    p_series = _makeseries([records[k]["p"] for k in keys], p.index)
    return w_series, p_series


def _manually_calc_hedgeresults_poTrue(w, p, how, aggfreq) -> dict:
    """
    'Manually' calculate the hedge result for MW and Eur/MWh timeseries ``w`` and ``p``,
    if we *do* want to split in peak and offpeak values.
    Returns tuple with w_hedge, p_hedge series in original frequency.
    """
    # Calculate the sums.
    dic = {}
    for ts, wval, pval, dur in zip(w.index, w, p, w.index.duration.pint.magnitude):
        aggts = stamps.floor_ts(ts, aggfreq)  # Key: start of delivery period.
        _add_peakoffpeak(dic, aggts, wval, pval, dur, putils.is_peak_hour(ts))

    # Calculate the values.
    records = _records(dic, how)

    # Turn into long series.
    availkeys = np.array(list(records.keys()))
    keys = availkeys[np.searchsorted(availkeys, w.index, side="right") - 1]
    inds = [0 if putils.is_peak_hour(ts) else 1 for ts in w.index]
    w_series = _makeseries([records[k]["w"][i] for k, i in zip(keys, inds)], w.index)
    p_series = _makeseries([records[k]["p"][i] for k, i in zip(keys, inds)], w.index)
    return w_series, p_series


def _add_base(dic, key, wval, pval, dur) -> None:
    if key not in dic:
        dic[key] = {"w.d": 0, "d": 0, "w.p.d": 0, "p.d": 0}
    dic[key]["w.d"] += wval * dur
    dic[key]["d"] += dur
    dic[key]["w.p.d"] += pval * wval * dur
    dic[key]["p.d"] += pval * dur


def _add_peakoffpeak(dic, key, wval, pval, dur, ispeak) -> None:
    if key not in dic:
        dic[key] = {
            "w.d": np.array([0.0, 0.0]),
            "d": np.array([0.0, 0.0]),
            "w.p.d": np.array([0.0, 0.0]),
            "p.d": np.array([0.0, 0.0]),
        }
    if ispeak:
        dic[key]["w.d"] += [wval * dur, 0]
        dic[key]["d"] += [dur, 0]
        dic[key]["w.p.d"] += [pval * wval * dur, 0]
        dic[key]["p.d"] += [pval * dur, 0]
    else:
        dic[key]["w.d"] += [0, wval * dur]
        dic[key]["d"] += [0, dur]
        dic[key]["w.p.d"] += [0, pval * wval * dur]
        dic[key]["p.d"] += [0, pval * dur]


def _records(dic, how):
    records = {}
    for aggts, v in dic.items():
        w_hedge = v["w.p.d"] / v["p.d"] if how == "val" else v["w.d"] / v["d"]
        p_hedge = v["p.d"] / v["d"]
        records[aggts] = {"w": w_hedge, "p": p_hedge}
    return records


def _makeseries(quantities, index):
    try:
        unit = quantities[0].units
    except AttributeError:
        return pd.Series(quantities, index)
    magnitudes = [q.to(unit).magnitude for q in quantities]
    return pd.Series(magnitudes, index, dtype=f"pint[{unit}]")


# @pytest.mark.parametrize("freq", ["H", "15T", "D", "MS"])
# def test__w_hedge_bpoFalse(start, freq, count, tz):
#     w, p, ref_results = get_hedgeresults(start, freq, count, tz, False, None)
#     for how in ["vol", "val"]:
#         test_result = hedge._hedge(w, p, how, False)
#         ref_result = ref_results[None][how][0]
#         assert np.isclose(test_result, ref_result)


# @pytest.mark.parametrize("freq", ["H", "15T"])
# def test__w_hedge_bpoTrue(start, freq, count, tz):
#     w, p, ref_results = get_hedgeresults(start, freq, count, tz, True, None)
#     for how in ["vol", "val"]:
#         test_result = hedge._hedge(w, p, how, True).sort_index()
#         ref_bpo = ref_results[None][how]
#         records = {"w_peak": ref_bpo[0], "w_offpeak": ref_bpo[1]}
#         ref_result = pd.Series(records).dropna().sort_index()
#         pd.testing.assert_series_equal(test_result, ref_result)


# @pytest.mark.parametrize("freq", ["H", "15T", "D", "MS"])
# def test_wide_bpoFalse(start, freq_bpoFalse, count, tz, aggfreq, how):
#     freq = freq_bpoFalse
#     w, p, ref_results = get_hedgeresults(start, freq, count, tz, False, aggfreq)
#     test_result = hedge.hedge(w, p, aggfreq, how, False, False).sort_index()
#     records = {ts: values[how][0] for ts, values in ref_results.items()}
#     ref_result = pd.Series(records).sort_index().dropna()
#     pd.testing.assert_series_equal(test_result, ref_result, check_freq=False)


# @pytest.mark.parametrize("freq", ["H", "15T"])
# def test_wide_bpoTrue(start, freq_bpoTrue, count, tz, aggfreq, how):
#     freq = freq_bpoTrue
#     w, p, ref_results = get_hedgeresults(start, freq, count, tz, True, aggfreq)
#     test_result = hedge.hedge(w, p, aggfreq, how, True, False).sort_index()
#     records = {
#         ts: {"w_peak": values[how][0], "w_offpeak": values[how][1]}
#         for ts, values in ref_results.items()
#     }
#     ref_result = pd.DataFrame.from_records(records).sort_index().T.dropna(1)
#     pd.testing.assert_frame_equal(test_result, ref_result, check_freq=False)


@pytest.fixture(params=["2020", "2020-04-15 9:00", "2020-03-29", "2020-10-25"])
def start(request):
    return request.param


@pytest.fixture(params=[2, 24, 100, 1000])
def count(request):
    return request.param


@pytest.fixture(params=[None, "Europe/Berlin"])
def tz(request):
    return request.param


@pytest.fixture(params=["vol", "val"])
def how(request):
    return request.param


@pytest.fixture(params=["MS", "QS", "AS"])
def aggfreq(request):
    return request.param


@pytest.mark.parametrize("freq", ["H", "15T", "D"])
def test_hedge_poFalse(start, freq, count, tz, how, aggfreq):
    """Test if hedge is done correctly, if we don't use peak and offpeak products."""
    if tz is None:
        return  # TODO: allow hedging with dataframes without timezone information, and remove this block so that they are also checked

    i = pd.date_range(start, freq=freq, periods=count, tz=tz)
    w = dev.mockup.w_offtake(i).pint.magnitude
    p = dev.mockup.p_marketprices(i).pint.magnitude

    w_test, p_test = hedge.hedge(w, p, how, aggfreq, False)
    w_expected, p_expected = _manually_calc_hedgeresults_poFalse(w, p, how, aggfreq)

    w_expected = w_expected.loc[w_test.index]
    p_expected = p_expected.loc[p_test.index]

    testing.assert_series_equal(w_test, w_expected, check_names=False)
    testing.assert_series_equal(p_test, p_expected, check_names=False)


@pytest.mark.parametrize("freq", ["H", "15T"])
def test_hegde_bpoTrue(start, freq, count, tz, how, aggfreq):
    """Test if hedge is done correctly, if we do use peak and offpeak products."""
    if tz is None:
        return  # TODO: allow hedging with dataframes without timezone information, and remove this block so that they are also checked

    i = pd.date_range(start, freq=freq, periods=count, tz=tz)
    w = dev.mockup.w_offtake(i).pint.magnitude
    p = dev.mockup.p_marketprices(i).pint.magnitude

    w_test, p_test = hedge.hedge(w, p, how, aggfreq, True)
    w_expected, p_expected = _manually_calc_hedgeresults_poTrue(w, p, how, aggfreq)

    w_expected = w_expected.loc[w_test.index]
    p_expected = p_expected.loc[p_test.index]

    testing.assert_series_equal(w_test, w_expected, check_names=False)
    testing.assert_series_equal(p_test, p_expected, check_names=False)
