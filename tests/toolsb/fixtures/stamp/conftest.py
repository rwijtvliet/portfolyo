import pandas as pd
import pint
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import BaseOffset
import pytest
from portfolyo import toolsb


@pytest.fixture(scope="session")
def stamp_duration(year, monthday, freq_asstr, stamp_on_freqboundary, tz) -> pint.Quantity:
    if not stamp_on_freqboundary:
        pytest.skip("Only calculate duration for stamps on bounday.")

    # Fixed durations.
    duration = {"min": 1 / 60, "5min": 5 / 60, "15min": 15 / 60, "h": 1}.get(freq_asstr)
    if duration is not None:
        return toolsb.unit.Q_(duration, "h")

    # Duration depends on date.
    if monthday == "01-01":
        if freq_asstr.startswith("YS"):
            return toolsb.unit.Q_(8760 if year != 2020 else 8784, "h")
        elif freq_asstr.startswith("QS"):
            return toolsb.unit.Q_(
                (31 + (28 if year != 2020 else 29) + 31) * 24 - (1 if tz == "Europe/Berlin" else 0),
                "h",
            )
        elif freq_asstr == "MS":
            return toolsb.unit.Q_(31 * 24, "h")
        elif freq_asstr == "D":
            return toolsb.unit.Q_(24, "h")
    elif monthday == "02-01":
        if freq_asstr.startswith("YS"):
            return toolsb.unit.Q_(8760 if year != 2020 else 8784, "h")
        elif freq_asstr.startswith("QS"):
            return toolsb.unit.Q_(
                ((28 if year != 2020 else 29) + 31 + 30) * 24 - (1 if tz == "Europe/Berlin" else 0),
                "h",
            )
        elif freq_asstr == "MS":
            return toolsb.unit.Q_((28 if year != 2020 else 29) * 24, "h")
        elif freq_asstr == "D":
            return toolsb.unit.Q_(24, "h")
    elif monthday == "04-21":
        if freq_asstr == "D":
            return toolsb.unit.Q_(24, "h")
    raise ValueError()


@pytest.fixture(
    scope="session",
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
def _case1_stampfreqrightduration(request) -> tuple[str, str, str, float | int]:
    return request.param


@pytest.fixture(scope="session")
def case1_stamp(_case1_stampfreqrightduration) -> pd.Timestamp:
    return pd.Timestamp(_case1_stampfreqrightduration[0])


@pytest.fixture(scope="session")
def case1_freq(_case1_stampfreqrightduration) -> BaseOffset:
    return to_offset(_case1_stampfreqrightduration[1])


@pytest.fixture(scope="session")
def case1_right(_case1_stampfreqrightduration) -> pd.Timestamp:
    return pd.Timestamp(_case1_stampfreqrightduration[2])


@pytest.fixture(scope="session")
def case1_duration(_case1_stampfreqrightduration) -> pint.Quantity:
    return toolsb.unit.Q_(_case1_stampfreqrightduration[3], "h")


@pytest.fixture(
    scope="session",
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
def _case2_stampfreqrightduration(request) -> tuple[str, str, str, float | int]:
    return request.param


@pytest.fixture(scope="session")
def case2_stamp(_case2_stampfreqrightduration) -> pd.Timestamp:
    return pd.Timestamp(_case2_stampfreqrightduration[0], tz="Europe/Berlin")


@pytest.fixture(scope="session")
def case2_freq(_case2_stampfreqrightduration) -> BaseOffset:
    return to_offset(_case2_stampfreqrightduration[1])


@pytest.fixture(scope="session")
def case2_right(_case2_stampfreqrightduration) -> pd.Timestamp:
    return pd.Timestamp(_case2_stampfreqrightduration[2], tz="Europe/Berlin")


@pytest.fixture(scope="session")
def case2_duration(_case2_stampfreqrightduration) -> pint.Quantity:
    return toolsb.unit.Q_(_case2_stampfreqrightduration[3], "h")
