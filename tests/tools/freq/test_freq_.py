import pandas as pd
import pytest

from portfolyo import toolsb


@pytest.fixture(params=["str", "dateoffset"])
def _freq_as(request) -> str:
    return request.param


@pytest.fixture(params=toolsb.freq._FREQUENCIES)
def _freq(request) -> pd.DateOffset:
    return request.param


def freq(_freq, _freq_as) -> str | pd.DateOffset:
    return _freq if _freq_as == "dateoffset" else _freq.name


@pytest.fixture(params=["2h", "5D", "3MS", "30sec"])
def freq_nok(request, _freq_as) -> str | pd.DateOffset:
    freq = request.param
    return freq if _freq_as == "dateoffset" else freq.name


@pytest.fixture(params=["00:00", "03:00", "06:00"])
def startofday(request) -> str:
    return request.param


@pytest.fixture(params=["Europe/Berlin", "India/Kolkata", None])
def timezone(request) -> str | None:
    return request.param
