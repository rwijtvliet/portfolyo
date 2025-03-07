import dataclasses

import pandas as pd
import pint
import pytest
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import BaseOffset

from portfolyo import tools


@pytest.fixture(scope="class")
def _stamp_duration(year, monthday, freq_asstr, stamp_on_freqboundary, tz) -> float | int:
    if not stamp_on_freqboundary:
        pytest.skip("Only calculate duration for stamps on bounday.")

    # Fixed durations.
    duration = {"min": 1 / 60, "5min": 5 / 60, "15min": 15 / 60, "h": 1}.get(freq_asstr)
    if duration is not None:
        return duration

    # Duration depends on date.

    if monthday == "01-01":
        if freq_asstr.startswith("YS"):
            return 8760 if year != 2020 else 8784
        elif freq_asstr.startswith("QS"):
            return (31 + (28 if year != 2020 else 29) + 31) * 24 - (
                1 if tz == "Europe/Berlin" else 0
            )
        elif freq_asstr == "MS":
            return 31 * 24
        elif freq_asstr == "D":
            return 24
    elif monthday == "02-01":
        if freq_asstr.startswith("YS"):
            return 8760 if year != 2020 else 8784
        elif freq_asstr.startswith("QS"):
            return ((28 if year != 2020 else 29) + 31 + 30) * 24 - (
                1 if tz == "Europe/Berlin" else 0
            )
        elif freq_asstr == "MS":
            return (28 if year != 2020 else 29) * 24
        elif freq_asstr == "D":
            return 24
    elif monthday == "04-21":
        if freq_asstr == "D":
            return 24
    raise ValueError()


@pytest.fixture(scope="class")
def stamp_duration(_stamp_duration) -> pint.Quantity:
    return tools.unit.Q_(_stamp_duration, "h")


# ---


@dataclasses.dataclass
class _Case1:
    stamp: str
    freqstr: str
    right: str
    duration: float | int


@pytest.fixture(
    scope="class",
    params=[
        ("2020", "min", "2020-01-01 00:01", 1 / 60),
        ("2020", "5min", "2020-01-01 00:05", 5 / 60),
        ("2020", "15min", "2020-01-01 00:15", 15 / 60),
        ("2020", "h", "2020-01-01 01:00", 1),
        ("2020", "D", "2020-01-02", 24),
        ("2020", "MS", "2020-02", 31 * 24),
        ("2020", "QS-JAN", "2020-04", 91 * 24),  # no DST
        ("2020", "QS-APR", "2020-04", 91 * 24),
        ("2020", "QS-JUL", "2020-04", 91 * 24),
        ("2020", "QS-OCT", "2020-04", 91 * 24),
        ("2020", "YS", "2021", 8784),
        ("2020-05", "min", "2020-05-01 00:01", 1 / 60),
        ("2020-05", "5min", "2020-05-01 00:05", 5 / 60),
        ("2020-05", "15min", "2020-05-01 00:15", 15 / 60),
        ("2020-05", "h", "2020-05-01 01:00", 1),
        ("2020-05", "D", "2020-05-02", 24),
        ("2020-05", "MS", "2020-06", 31 * 24),
        ("2020-05", "QS-FEB", "2020-08", 92 * 24),
        ("2020-05", "QS-MAY", "2020-08", 92 * 24),
        ("2020-05", "QS-AUG", "2020-08", 92 * 24),
        ("2020-05", "QS-NOV", "2020-08", 92 * 24),
        ("2020-05", "YS-MAY", "2021-05", 8760),
        ("2020-04-21 15:00", "min", "2020-04-21 15:01", 1 / 60),
        ("2020-04-21 15:00", "5min", "2020-04-21 15:05", 5 / 60),
        ("2020-04-21 15:00", "15min", "2020-04-21 15:15", 15 / 60),
        ("2020-04-21 15:00", "h", "2020-04-21 16:00", 1),
    ],
)
def _case1(request) -> _Case1:
    return _Case1(*request.param)


@pytest.fixture(scope="class")
def case1_stamp(_case1: _Case1) -> pd.Timestamp:
    return pd.Timestamp(_case1.stamp)


@pytest.fixture(scope="class")
def case1_freq(_case1: _Case1) -> BaseOffset:
    return to_offset(_case1.freqstr)


@pytest.fixture(scope="class")
def case1_right(_case1: _Case1) -> pd.Timestamp:
    return pd.Timestamp(_case1.right)


@pytest.fixture(scope="class")
def case1_duration(_case1: _Case1) -> pint.Quantity:
    return tools.unit.Q_(_case1.duration, "h")


# DST ---


@dataclasses.dataclass
class _Case2:
    stamp: str
    freqstr: str
    right: str
    duration: float | int


@pytest.fixture(
    scope="class",
    params=[
        ("2020-03-28", "D", "2020-03-29", 24),
        ("2020-03-28 01:00", "D", "2020-03-29 01:00", 24),
        ("2020-03-28 03:00", "D", "2020-03-29 03:00", 23),
        ("2020-03-29", "D", "2020-03-30", 23),
        ("2020-03-29 01:00", "D", "2020-03-30 01:00", 23),
        ("2020-03-29 03:00", "D", "2020-03-30 03:00", 24),
        ("2020-10-25", "D", "2020-10-26", 25),
        ("2020-10-25 01:00", "D", "2020-10-26 01:00", 25),
        ("2020-10-25 03:00", "D", "2020-10-26 03:00", 24),
        ("2020-03", "MS", "2020-04", 31 * 24 - 1),
        ("2020-10", "MS", "2020-11", 31 * 24 + 1),
        ("2020-01", "QS-JAN", "2020-04", 91 * 24 - 1),
        ("2020-10", "QS-JAN", "2021-01", 92 * 24 + 1),
    ],
)
def _case2(request) -> _Case2:
    return _Case2(*request.param)


@pytest.fixture(scope="class")
def case2_stamp(_case2: _Case2) -> pd.Timestamp:
    return pd.Timestamp(_case2.stamp, tz="Europe/Berlin")


@pytest.fixture(scope="class")
def case2_freq(_case2: _Case2) -> BaseOffset:
    return to_offset(_case2.freqstr)


@pytest.fixture(scope="class")
def case2_right(_case2: _Case2) -> pd.Timestamp:
    return pd.Timestamp(_case2.right, tz="Europe/Berlin")


@pytest.fixture(scope="class")
def case2_duration(_case2: _Case2) -> pint.Quantity:
    return tools.unit.Q_(_case2.duration, "h")


# ---


@dataclasses.dataclass
class _Case3:
    stamp: str
    freqstr: str
    sodstr: str
    floored: str
    ceiled: str


@pytest.fixture(
    scope="class",
    params=[
        ("2020-04-21 12:34:56", "min", "00:00", "2020-04-21 12:34", "2020-04-21 12:35"),
        ("2020-04-21 12:34:56", "min", "06:00", "2020-04-21 12:34", "2020-04-21 12:35"),
        ("2020-04-21 12:34:56", "min", "15:00", "2020-04-21 12:34", "2020-04-21 12:35"),
        ("2020-04-21 12:34:56", "15min", "00:00", "2020-04-21 12:30", "2020-04-21 12:45"),
        ("2020-04-21 12:34:56", "15min", "06:00", "2020-04-21 12:30", "2020-04-21 12:45"),
        ("2020-04-21 12:34:56", "15min", "15:00", "2020-04-21 12:30", "2020-04-21 12:45"),
        ("2020-04-21 12:34:56", "h", "00:00", "2020-04-21 12:00", "2020-04-21 13:00"),
        ("2020-04-21 12:34:56", "h", "06:00", "2020-04-21 12:00", "2020-04-21 13:00"),
        ("2020-04-21 12:34:56", "h", "15:00", "2020-04-21 12:00", "2020-04-21 13:00"),
        ("2020-04-21 12:34:56", "D", "00:00", "2020-04-21 00:00", "2020-04-22 00:00"),
        ("2020-04-21 12:34:56", "D", "15:00", "2020-04-20 15:00", "2020-04-21 15:00"),
        ("2020-04-21 12:34:56", "MS", "00:00", "2020-04-01 00:00", "2020-05-01 00:00"),
        ("2020-04-21 12:34:56", "MS", "15:00", "2020-04-01 15:00", "2020-05-01 15:00"),
        ("2020-04-21 12:34:56", "QS", "00:00", "2020-04-01 00:00", "2020-07-01 00:00"),
        ("2020-04-21 12:34:56", "QS", "15:00", "2020-04-01 15:00", "2020-07-01 15:00"),
        ("2020-04-21 12:34:56", "QS-FEB", "00:00", "2020-02-01 00:00", "2020-05-01 00:00"),
        ("2020-04-21 12:34:56", "QS-FEB", "15:00", "2020-02-01 15:00", "2020-05-01 15:00"),
        ("2020-04-21 12:34:56", "QS-APR", "00:00", "2020-04-01 00:00", "2020-07-01 00:00"),
        ("2020-04-21 12:34:56", "QS-APR", "15:00", "2020-04-01 15:00", "2020-07-01 15:00"),
        ("2020-04-21 12:34:56", "YS", "00:00", "2020-01-01 00:00", "2021-01-01 00:00"),
        ("2020-04-21 12:34:56", "YS", "15:00", "2020-01-01 15:00", "2021-01-01 15:00"),
        ("2020-04-21 12:34:56", "YS-FEB", "00:00", "2020-02-01 00:00", "2021-02-01 00:00"),
        ("2020-04-21 12:34:56", "YS-FEB", "15:00", "2020-02-01 15:00", "2021-02-01 15:00"),
    ],
)
def _case3(request) -> _Case3:
    return _Case3(*request.param)


@pytest.fixture(scope="class")
def case3_stamp(_case3: _Case3, tz) -> pd.Timestamp:
    return pd.Timestamp(_case3.stamp, tz=tz)


@pytest.fixture(scope="class")
def case3_freq(_case3: _Case3) -> BaseOffset:
    return to_offset(_case3.freqstr)


@pytest.fixture(scope="class")
def case3_sodstr(_case3: _Case3) -> str:
    return _case3.sodstr


@pytest.fixture(scope="class")
def case3_floored(_case3: _Case3, tz) -> pd.Timestamp:
    return pd.Timestamp(_case3.floored, tz=tz)


@pytest.fixture(scope="class")
def case3_ceiled(_case3: _Case3, tz) -> pd.Timestamp:
    return pd.Timestamp(_case3.ceiled, tz=tz)


# ---


@dataclasses.dataclass
class _Case4:
    date: str
    time1: str
    time2: str


@pytest.fixture(
    scope="class",
    params=[
        ("2020-01-01", "00:00", "00:00"),
        ("2020-01-01", "00:00", "14:00"),
        ("2020-01-01", "00:00", "14:35"),
        ("2020-04-21", "21:34", "00:00"),
        ("2020-04-21", "21:34", "14:00"),
        ("2020-04-21", "21:34", "14:35"),
        ("2020-03-29", "01:00", "00:00"),  # dst start in Europe
        ("2020-03-29", "03:00", "00:00"),
        ("2020-03-29", "01:00", "04:00"),
        ("2020-03-29", "03:00", "04:00"),
        ("2020-10-25", "01:00", "00:00"),  # dst end in Europe
        ("2020-10-25", "03:00", "00:00"),
        ("2020-10-25", "01:00", "04:00"),
        ("2020-10-25", "03:00", "04:00"),
    ],
)
def _case4(request) -> _Case4:
    return _Case4(*request.param)


@pytest.fixture(scope="class")
def case4_stamp(_case4: _Case4, tz) -> pd.Timestamp:
    return pd.Timestamp(f"{_case4.date} {_case4.time1}", tz=tz)


@pytest.fixture(scope="class")
def case4_newsodstr(_case4: _Case4) -> str:
    return _case4.time2


@pytest.fixture(scope="class")
def case4_newstamp(_case4: _Case4, tz) -> pd.Timestamp:
    return pd.Timestamp(f"{_case4.date} {_case4.time2}", tz=tz)
