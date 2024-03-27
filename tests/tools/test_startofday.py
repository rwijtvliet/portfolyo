import datetime as dt

import pandas as pd
import pytest

from portfolyo import testing, tools

ERROR_SOD_DICT = {"hour": 6, "minute": 1}  # error
SOD_STRS = ["00:00", "06:00", "15:30"]
SODS = [(0, 0), (6, 0), (15, 30)]
SOD_DICTS = [
    {"hour": 0, "minute": 0},
    {"hour": 6, "minute": 0},
    {"hour": 15, "minute": 30},
    ERROR_SOD_DICT,
]

SEED = 3


STARTDATE_AND_FREQ = [
    ("2020-01-01", "H"),
    ("2020-01-01", "D"),
    ("2020-01-01", "MS"),
    ("2020-03-28", "D"),
    ("2020-03-01", "MS"),
    ("2020-10-25", "D"),
    ("2020-10-01", "MS"),
]


def create_start_of_day(hour, minute, returntype):
    if returntype == "time":
        return dt.time(hour=hour, minute=minute)
    if returntype == "str":
        return f"{hour:02}:{minute:02}:00"
    if returntype == "timedelta":
        return dt.timedelta(hours=hour, minutes=minute)


@pytest.mark.parametrize(
    "i,hour,minute",
    [
        # Midnight
        # . No timezone
        (
            pd.date_range(
                "2020-01-01", "2021-01-01", freq="MS", inclusive="left", tz=None
            ),
            0,
            0,
        ),
        (
            pd.date_range(
                "2020-01-01", "2021-01-01", freq="D", inclusive="left", tz=None
            ),
            0,
            0,
        ),
        (
            pd.date_range(
                "2020-01-01", "2021-01-01", freq="H", inclusive="left", tz=None
            ),
            0,
            0,
        ),
        # . Europe/Berlin
        (
            pd.date_range(
                "2020-01-01",
                "2021-01-01",
                freq="MS",
                inclusive="left",
                tz="Europe/Berlin",
            ),
            0,
            0,
        ),
        (
            pd.date_range(
                "2020-01-01",
                "2021-01-01",
                freq="D",
                inclusive="left",
                tz="Europe/Berlin",
            ),
            0,
            0,
        ),
        (
            pd.date_range(
                "2020-01-01",
                "2021-01-01",
                freq="H",
                inclusive="left",
                tz="Europe/Berlin",
            ),
            0,
            0,
        ),
        # 06:00
        # . No timezone
        (
            pd.date_range(
                "2020-01-01 06:00",
                "2021-01-01 06:00",
                freq="MS",
                inclusive="left",
                tz=None,
            ),
            6,
            0,
        ),
        (
            pd.date_range(
                "2020-01-01 06:00",
                "2021-01-01 06:00",
                freq="D",
                inclusive="left",
                tz=None,
            ),
            6,
            0,
        ),
        (
            pd.date_range(
                "2020-01-01 06:00",
                "2021-01-01 06:00",
                freq="H",
                inclusive="left",
                tz=None,
            ),
            6,
            0,
        ),
        # . Europe/Berlin
        (
            pd.date_range(
                "2020-01-01 06:00",
                "2021-01-01 06:00",
                freq="MS",
                inclusive="left",
                tz="Europe/Berlin",
            ),
            6,
            0,
        ),
        (
            pd.date_range(
                "2020-01-01 06:30",
                "2021-01-01 06:30",
                freq="D",
                inclusive="left",
                tz="Europe/Berlin",
            ),
            6,
            30,
        ),
        (
            pd.date_range(
                "2020-01-01 06:30",
                "2021-01-01 06:30",
                freq="H",
                inclusive="left",
                tz="Europe/Berlin",
            ),
            6,
            30,
        ),
    ],
)
@pytest.mark.parametrize("returntype", ["time", "timedelta", "str"])
def test_get_startofday(i: pd.DatetimeIndex, hour: int, minute: int, returntype: str):
    """Test if start-of-day can be correctly gotten from index."""
    expected = create_start_of_day(hour, minute, returntype)
    result = tools.startofday.get(i, returntype)
    assert result == expected


@pytest.mark.parametrize(
    "i,hour,minute,expected",
    [
        # Midnight
        # . No timezone
        (
            pd.date_range(
                "2020-01-01", "2021-01-01", freq="MS", inclusive="left", tz=None
            ),
            0,
            0,
            None,
        ),
        (
            pd.date_range(
                "2020-01-01", "2021-01-01", freq="D", inclusive="left", tz=None
            ),
            0,
            0,
            None,
        ),
        (
            pd.date_range(
                "2020-01-01", "2021-01-01", freq="H", inclusive="left", tz=None
            ),
            0,
            0,
            None,
        ),
        # . Europe/Berlin
        (
            pd.date_range(
                "2020-01-01",
                "2021-01-01",
                freq="MS",
                inclusive="left",
                tz="Europe/Berlin",
            ),
            0,
            0,
            None,
        ),
        (
            pd.date_range(
                "2020-01-01",
                "2021-01-01",
                freq="D",
                inclusive="left",
                tz="Europe/Berlin",
            ),
            0,
            0,
            None,
        ),
        (
            pd.date_range(
                "2020-01-01",
                "2021-01-01",
                freq="H",
                inclusive="left",
                tz="Europe/Berlin",
            ),
            0,
            0,
            None,
        ),
        # 06:00
        # . No timezone
        (
            pd.date_range(
                "2020-01-01", "2021-01-01", freq="MS", inclusive="left", tz=None
            ),
            6,
            0,
            pd.date_range(
                "2020-01-01 06:00",
                "2021-01-01 06:00",
                freq="MS",
                inclusive="left",
                tz=None,
            ),
        ),
        (
            pd.date_range(
                "2020-01-01", "2021-01-01", freq="D", inclusive="left", tz=None
            ),
            6,
            0,
            pd.date_range(
                "2020-01-01 06:00",
                "2021-01-01 06:00",
                freq="D",
                inclusive="left",
                tz=None,
            ),
        ),
        (
            pd.date_range(
                "2020-01-01", "2021-01-01", freq="H", inclusive="left", tz=None
            ),
            6,
            0,
            pd.date_range(
                "2020-01-01 06:00",
                "2020-12-31 06:00",
                freq="H",
                inclusive="left",
                tz=None,
            ),
        ),
        # . Europe/Berlin
        (
            pd.date_range(
                "2020-01-01",
                "2021-01-01",
                freq="MS",
                inclusive="left",
                tz="Europe/Berlin",
            ),
            6,
            0,
            pd.date_range(
                "2020-01-01 06:00",
                "2021-01-01 06:00",
                freq="MS",
                inclusive="left",
                tz="Europe/Berlin",
            ),
        ),
        (
            pd.date_range(
                "2020-01-01",
                "2021-01-01",
                freq="D",
                inclusive="left",
                tz="Europe/Berlin",
            ),
            6,
            0,
            pd.date_range(
                "2020-01-01 06:00",
                "2021-01-01 06:00",
                freq="D",
                inclusive="left",
                tz="Europe/Berlin",
            ),
        ),
        (
            pd.date_range(
                "2020-01-01",
                "2021-01-01",
                freq="H",
                inclusive="left",
                tz="Europe/Berlin",
            ),
            6,
            0,
            pd.date_range(
                "2020-01-01 06:00",
                "2020-12-31 06:00",
                freq="H",
                inclusive="left",
                tz="Europe/Berlin",
            ),
        ),
        # 06:03
        # . No timezone
        (
            pd.date_range(
                "2020-01-01", "2021-01-01", freq="MS", inclusive="left", tz=None
            ),
            6,
            3,
            Exception,
        ),
        (
            pd.date_range(
                "2020-01-01", "2021-01-01", freq="D", inclusive="left", tz=None
            ),
            6,
            3,
            Exception,
        ),
        (
            pd.date_range(
                "2020-01-01", "2021-01-01", freq="H", inclusive="left", tz=None
            ),
            6,
            3,
            Exception,
        ),
        # . Europe/Berlin
        (
            pd.date_range(
                "2020-01-01",
                "2021-01-01",
                freq="MS",
                inclusive="left",
                tz="Europe/Berlin",
            ),
            6,
            3,
            Exception,
        ),
        (
            pd.date_range(
                "2020-01-01",
                "2021-01-01",
                freq="D",
                inclusive="left",
                tz="Europe/Berlin",
            ),
            6,
            3,
            Exception,
        ),
        (
            pd.date_range(
                "2020-01-01",
                "2021-01-01",
                freq="H",
                inclusive="left",
                tz="Europe/Berlin",
            ),
            6,
            3,
            Exception,
        ),
    ],
)
def test_set_startofday(
    i: pd.DatetimeIndex,
    hour: int,
    minute: int,
    expected: pd.DatetimeIndex,
):
    """Test if start-of-day can be correctly set to index."""
    if expected is None:
        expected = i
    sod = dt.time(hour=hour, minute=minute)
    # Error case.
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            _ = tools.startofday.set(i, sod)
        return
    # Normal case.
    result = tools.startofday.set(i, sod)
    testing.assert_index_equal(result, expected)
