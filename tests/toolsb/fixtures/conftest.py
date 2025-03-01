import pandas as pd
import pytest

from portfolyo import toolsb


@pytest.fixture(params=["str", "dateoffset"])
def _freq_as(request) -> str:
    return request.param


@pytest.fixture(params=toolsb.freq._FREQUENCIES)
def _freq(request) -> pd.DateOffset:  # for use in other fixtures
    return request.param


@pytest.fixture(
    params=[pd.tseries.frequencies.to_offset(freq) for freq in ("2min", "4h", "7D", "3MS")]
)
def _freq_nok(request) -> pd.DateOffset:  # for use in other fixtures
    return request.param


def freq(_freq, _freq_as) -> str | pd.DateOffset:  # for use as user input
    return _freq if _freq_as == "dateoffset" else _freq.freqstr


def freq_nok(_freq_nok, _freq_as) -> str | pd.DateOffset:  # for use as user input
    return _freq_nok if _freq_as == "dateoffset" else _freq_nok.freqstr


@pytest.fixture(params=["00:00", "06:00", "15:00"])
def startofday(request) -> str:
    return request.param


@pytest.fixture(params=["Europe/Berlin", "Asia/Kolkata", None])
def tz(request) -> str | None:
    return request.param
