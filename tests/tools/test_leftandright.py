import datetime as dt
from typing import Iterable

import pandas as pd
import pytest

from portfolyo import tools


@pytest.mark.parametrize("tz_param", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
def test_leftandright_nonespecified(tz_param: str, starttime: str):
    """Test if start and end of interval are correctly calculated, if none is specified."""
    # None specified, so tz parameter and set_hours should be used in both return values.
    nextyear = pd.Timestamp.today().year + 1
    starthour = 6 if starttime == "06:00" else 0
    start_of_day = dt.time(hour=starthour)
    expected = (
        pd.Timestamp(year=nextyear, month=1, day=1, hour=starthour, tz=tz_param),
        pd.Timestamp(year=nextyear + 1, month=1, day=1, hour=starthour, tz=tz_param),
    )
    result = tools.leftandright.stamps(None, None, tz_param, start_of_day)
    for a, b in zip(result, expected):
        assert a == b


@pytest.mark.parametrize("tz_param", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("tz_specified", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
@pytest.mark.parametrize(
    ("tss", "expected_tss"),
    [
        (("2020-01-01", None), ("2020-01-01", "2021-01-01")),
        ((None, "2020-02-02"), ("2020-01-01", "2020-02-02")),
        (("2020-03-03 06:00", None), ("2020-03-03 06:00", "2021-01-01 06:00")),
        ((None, "2021-10-09 06:00"), ("2021-01-01 06:00", "2021-10-09 06:00")),
    ],
)
def test_leftandright_onespecified_astimestamp(
    tss: tuple, expected_tss: tuple, tz_specified: str, tz_param: str, starttime: str
):
    """Test if start and end of interval are correctly calculated, if one is specified
    as a timestamp."""
    # One specified, so tz parameter and start_of_day should be ignored.
    # There should be no timezone errors and no swapping is necessary.
    tss = [pd.Timestamp(ts, tz=tz_specified) for ts in tss]  # one will be NaT
    start_of_day = dt.time(hour=6) if starttime == "06:00" else None
    expected = [pd.Timestamp(ts, tz=tz_specified) for ts in expected_tss]
    result = tools.leftandright.stamps(*tss, tz_param, start_of_day)

    for a, b in zip(result, expected):
        assert a == b


@pytest.mark.parametrize("tz_param", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("tss,starttime,expected_tss"),
    [
        (("2020-01-01", None), None, ("2020-01-01", "2021-01-01")),
        (("2020-01-01", None), "00:00", ("2020-01-01", "2021-01-01")),
        (("2020-01-01", None), "06:00", ("2020-01-01 06:00", "2021-01-01 06:00")),
        ((None, "2020-02-02"), None, ("2020-01-01", "2020-02-02")),
        ((None, "2020-02-02"), "00:00", ("2020-01-01", "2020-02-02")),
        ((None, "2020-02-02"), "06:00", ("2020-01-01 06:00", "2020-02-02 06:00")),
        # starttime should be ignored, because already present in timestamp.
        (("2020-03-03 06:00", None), None, ("2020-03-03 06:00", "2021-01-01 06:00")),
        (("2020-03-03 06:00", None), "00:00", ("2020-03-03 06:00", "2021-01-01 06:00")),
        (("2020-03-03 06:00", None), "06:00", ("2020-03-03 06:00", "2021-01-01 06:00")),
        ((None, "2021-10-09 06:00"), None, ("2021-01-01 06:00", "2021-10-09 06:00")),
        ((None, "2021-10-09 06:00"), "00:00", ("2021-01-01 06:00", "2021-10-09 06:00")),
        ((None, "2021-10-09 06:00"), "06:00", ("2021-01-01 06:00", "2021-10-09 06:00")),
    ],
)
def test_leftandright_onespecified_asstring(
    tss: tuple, expected_tss: tuple, tz_param: str, starttime: str
):
    """Test if start and end of interval are correctly calculated, if one is specified
    as a timestamp."""
    # One specified, but as string. So tz parameter and start_of_day should be used.
    # There should be no timezone errors and no swapping is necessary
    start_of_day = {"06:00": dt.time(hour=6), "00:00": dt.time(hour=0), None: None}[
        starttime
    ]
    expected = [pd.Timestamp(ts, tz=tz_param) for ts in expected_tss]
    result = tools.leftandright.stamps(*tss, tz_param, start_of_day)

    for a, b in zip(result, expected):
        assert a == b


@pytest.mark.parametrize("tz_param", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    "tzs",
    [
        (None, None),
        ("Europe/Berlin", "Europe/Berlin"),
        (None, "Europe/Berlin"),
        ("Europe/Berlin", "Asia/Kolkata"),
    ],
)
@pytest.mark.parametrize("starttime_param", ["00:00", "06:00"])
@pytest.mark.parametrize(
    "times",
    [("00:00", "00:00"), ("00:00", "06:00"), ("06:00", "00:00"), ("06:00", "06:00")],
)
@pytest.mark.parametrize(
    ("dates", "expected_dates"),
    [
        (("2020-01-01", "2020-02-02"), ("2020-01-01", "2020-02-02")),
        (("2020-02-02", "2020-01-01"), ("2020-01-01", "2020-02-02")),
        (("2020-01-01", "2020-01-01"), ("2020-01-01", "2020-01-01")),
        (("2021-10-09", "2020-03-03"), ("2020-03-03", "2021-10-09")),
        (("2020-03-03", "2021-10-09"), ("2020-03-03", "2021-10-09")),
    ],
)
def test_leftandright_bothspecified(
    dates: tuple,
    times: Iterable[str],
    expected_dates: tuple,
    tzs: Iterable[str],
    tz_param: str,
    starttime_param: str,
):
    """Test if start and end of interval are correctly calculated, if both are specified."""
    # Both specified, so tz parameter and start_of_day should be ignored
    # There should be a timezone error if they are unequal, and there should be swapping if necessary.
    tss = [pd.Timestamp(f"{d} {time}", tz=tz) for d, tz, time in zip(dates, tzs, times)]
    start_of_day = dt.time(hour=6) if starttime_param == "06:00" else None

    if tzs[0] != tzs[1] or times[0] != times[1]:
        with pytest.raises(ValueError):
            _ = tools.leftandright.stamps(*tss, tz_param, start_of_day)
        return

    expected = [pd.Timestamp(f"{d} {times[0]}", tz=tzs[0]) for d in expected_dates]
    result = tools.leftandright.stamps(*tss, tz_param, start_of_day)

    for a, b in zip(result, expected):
        assert a == b
