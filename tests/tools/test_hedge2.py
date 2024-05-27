"""Testing hedge functions using pre-calculated hedge values from excel."""

from pathlib import Path

import pandas as pd
import pytest

from portfolyo import testing, tools

PATH = Path(__file__).parent / "test_hedge_data.xlsx"


def sheetname(freq, tz):
    return f'{freq}_{"None" if tz is None else tz.replace("/", "")}'


@pytest.fixture(
    scope="session",
    params=[pytest.param(True, id="withunits"), pytest.param(False, id="withoutunits")],
)
def has_units(request):
    return request.param


@pytest.fixture(scope="session", params=["H", "D"])
def freq(request):
    return request.param


@pytest.fixture(scope="session", params=[None, "Europe/Berlin"])
def tz(request):
    return request.param


@pytest.fixture(scope="session", params=["vol", "val"])
def how(request):
    return request.param


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(None, id="base"),
        pytest.param(tools.product.germanpower_peakfn, id="peakoffpeakgerman"),
    ],
)
def peak_fn(request):
    return request.param


@pytest.fixture(scope="session", params=["MS", "QS", "AS"])
def aggfreq(request):
    return request.param


@pytest.fixture
def dfin(freq: str, tz: str) -> pd.DataFrame:
    df = pd.read_excel(
        PATH, sheetname(freq, tz), header=6, index_col=0, usecols="A,B:C"
    )
    if tz:
        df = df.tz_localize(tz, ambiguous="infer")
    df.index.freq = pd.infer_freq(df.index)
    return df


@pytest.fixture
def dfexp(freq: str, tz: str) -> pd.DataFrame:
    df = pd.read_excel(
        PATH, f"{sheetname(freq, tz)}_out", header=[3, 4, 5, 6], index_col=0
    )
    if tz:
        df = df.tz_localize(tz, ambiguous="infer")
    df.index.freq = pd.infer_freq(df.index)
    return df


@pytest.fixture
def win(dfin: pd.DataFrame, has_units: bool) -> pd.Series:
    s = dfin["w"]
    return s if not has_units else s.astype("pint[MW]")


@pytest.fixture
def pin(dfin: pd.DataFrame, has_units: bool) -> pd.Series:
    s = dfin["p"]
    return s if not has_units else s.astype("pint[Eur/MWh]")


@pytest.fixture
def wexp(
    dfexp: pd.DataFrame,
    has_units: bool,
    aggfreq: str,
    peak_fn: tools.peakfn.PeakFunction,
    how: str,
) -> pd.Series:
    if dfexp.index.freq == "D" and peak_fn is not None:
        pytest.skip("Don't decompose in peak and offpeak if daily values")
    s = dfexp[aggfreq][peak_fn is not None][how]["w"]
    return s if not has_units else s.astype("pint[MW]")


@pytest.fixture
def pexp(
    dfexp: pd.DataFrame,
    has_units: bool,
    aggfreq: str,
    peak_fn: tools.peakfn.PeakFunction,
) -> pd.Series:
    s = dfexp[aggfreq][peak_fn is not None].iloc[:, 0]  # price is first column
    return s if not has_units else s.astype("pint[Eur/MWh]")


def test_hedge_fromexcel(
    win: pd.Series,
    pin: pd.Series,
    wexp: pd.Series,
    pexp: pd.Series,
    how: str,
    aggfreq: str,
    peak_fn: tools.peakfn.PeakFunction,
):
    """Test if hedge results are correctly calculated, by comparing against previously calculated results."""
    wresult, presult = tools.hedge.hedge(win, pin, how, peak_fn, aggfreq)
    testing.assert_series_equal(wresult, wexp, check_names=False)
    testing.assert_series_equal(presult, pexp, check_names=False)
