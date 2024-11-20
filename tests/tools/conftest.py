import pytest


@pytest.fixture(params=[None, "Europe/Berlin", "Asia/Kolkata"])
def tz(request) -> str | None:
    return request.param


@pytest.fixture(params=["00:00", "06:00"])
def startofday(request) -> str:
    return request.param


@pytest.fixture(params=["15min", "h", "D", "MS", "QS", "YS"])
def freq(request) -> str:
    return request.param


@pytest.fixture(params=["30sec", "5min", "MS-FEB", "QS-FEB", "YS-FEB"])
def uncommon_freq(request) -> str:
    """Uncommon, but allowed, freq."""
    return request.param


@pytest.fixture(params=["45min", "2h"])
def disallowed_freq(request) -> str:
    return request.param
