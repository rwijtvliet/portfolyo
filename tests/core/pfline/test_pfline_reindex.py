import portfolyo as pf
import numpy as np
import pytest
import pandas as pd

idx = pd.date_range("2020", freq="MS", periods=3)
idx2 = pd.date_range("2020-02", freq="MS", periods=3)
s = pd.Series([10, 20, 30], idx)
s2 = pd.Series([20, 30, 0], idx2)


@pytest.mark.parametrize(
    "pfl,idx,expected",
    [
        (pf.PfLine(s.astype("pint[MWh]")), idx2, pf.PfLine(s2.astype("pint[MWh]"))),
        (pf.PfLine(s.astype("pint[MW]")), idx2, pf.PfLine(s2.astype("pint[MW]"))),
        (pf.PfLine(s.astype("pint[Wh]")), idx2, pf.PfLine(s2.astype("pint[Wh]"))),
    ],
)
def test_pfl_reindex(pfl: pf.PfLine, idx: pd.DatetimeIndex, expected: pf.PfLine):
    assert pfl.reindex(idx) == expected


KIND_AND_UNITS = {
    pf.Kind.VOLUME: ["MWh", "tce/h"],
    pf.Kind.PRICE: ["Eur/MWh", "ctEur/tce"],
    pf.Kind.REVENUE: ["Eur", "kEur"],
}


@pytest.fixture(scope="session", params=["h", "15min", "D", "MS", "QS", "YS", "YS-APR"])
def freq(request):
    return request.param


@pytest.fixture(scope="session", params=[None, "Europe/Berlin", "Asia/Kolkata"])
def tz(request):
    return request.param


@pytest.fixture(scope="session", params=[((10, 20, 30), (20, 30, 0))])
def floats_inout(request):
    return request.param


@pytest.fixture(scope="session", params=pf.Kind)
def kind(request):
    return request.param


@pytest.fixture(scope="session", params=[0, 1])
def units(request, kind):
    # returns 1 unit (non-complete) or 2 units (complete).
    i = request.param
    if kind is pf.Kind.COMPLETE:
        return (KIND_AND_UNITS[pf.Kind.VOLUME][i], KIND_AND_UNITS[pf.Kind.REVENUE][i])
    else:
        return (KIND_AND_UNITS[kind][i],)


@pytest.fixture
def valuecount(freq):
    if freq == "h":
        return 96  # must be full number of days
    if freq == "15min":
        return 96 * 4  # must be full number of days
    return 3


@pytest.fixture
def diff(freq):
    if freq == "h":
        return 24  # must be full number of days
    if freq == "15min":
        return 96  # must be full number of days
    return 1


@pytest.fixture
def index_in(freq, tz, valuecount):
    return pd.date_range("2020", freq=freq, periods=valuecount, tz=tz)


@pytest.fixture
def index_out(freq, tz, valuecount, diff):
    return pd.date_range("2020", freq=freq, periods=valuecount + diff, tz=tz)[diff:]


@pytest.fixture
def floats_in(valuecount):
    return np.arange(1, valuecount + 1)  # 1, 2, 3, ..., valuecount


@pytest.fixture
def floats_out(valuecount, diff):
    return np.array(
        [*np.arange(diff + 1, valuecount + 1), *np.zeros(diff)]
    )  # 1+diff, 2+diff, ..., valuecount, 0, 0, .., 0


@pytest.fixture
def pfl_in(floats_in, index_in, units):
    data = [
        pd.Series(floats_in * (i * 10 + 1), index_in).astype(f"pint[{u}]")
        for i, u in enumerate(units)
    ]
    return pf.PfLine(*data)


@pytest.fixture
def pfl_out(floats_out, index_out, units):
    data = [
        pd.Series(floats_out * (i * 10 + 1), index_out).astype(f"pint[{u}]")
        for i, u in enumerate(units)
    ]
    return pf.PfLine(*data)


@pytest.fixture
def pfl(pfl_in):
    return pfl_in


@pytest.fixture
def index(index_out):
    return index_out


@pytest.fixture
def expected(pfl_out):
    return pfl_out


def test_reindex_flatpfline(
    pfl: pf.PfLine, index: pd.DatetimeIndex, expected: pf.PfLine
):
    result = pfl.reindex(index)
    assert result == expected
