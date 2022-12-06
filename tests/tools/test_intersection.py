from typing import Iterable, Union

import pandas as pd
import pytest

from portfolyo import testing, tools

COMMON_END = "2022-02-02"

TESTCASES = [  # startdates, freq, expected_startdate
    # One starts at first day of year.
    (("2020-01-01", "2020-01-20"), "15T", "2020-01-20"),
    (("2020-01-01", "2020-01-20"), "15T", "2020-01-20"),
    (("2020-01-01", "2020-01-20"), "H", "2020-01-20"),
    (("2020-01-01", "2020-01-20"), "H", "2020-01-20"),
    (("2020-01-01", "2020-01-20"), "D", "2020-01-20"),
    (("2020-01-01", "2020-01-20"), "D", "2020-01-20"),
    (("2020-01-01", "2020-03-01"), "MS", "2020-03-01"),
    (("2020-01-01", "2020-03-01"), "MS", "2020-03-01"),
    (("2020-01-01", "2020-04-01"), "QS", "2020-04-01"),
    (("2020-01-01", "2020-04-01"), "QS", "2020-04-01"),
    (("2020-01-01", "2021-01-01"), "AS", "2021-01-01"),
    (("2020-01-01", "2021-01-01"), "AS", "2021-01-01"),
    # Both start in middle of year.
    (("2020-04-21", "2020-06-20"), "15T", "2020-06-20"),
    (("2020-04-21", "2020-06-20"), "15T", "2020-06-20"),
    (("2020-04-21", "2020-06-20"), "H", "2020-06-20"),
    (("2020-04-21", "2020-06-20"), "H", "2020-06-20"),
    (("2020-04-21", "2020-06-20"), "D", "2020-06-20"),
    (("2020-04-21", "2020-06-20"), "D", "2020-06-20"),
]


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
@pytest.mark.parametrize(("startdates", "freq", "expected_startdate"), TESTCASES)
def test_intersection(
    startdates: Iterable[str],
    starttime: str,
    tz: str,
    freq: str,
    expected_startdate: str,
):
    """Test if intersection of indices gives correct result."""
    idxs = [idx(startdate, starttime, tz, freq) for startdate in startdates]
    do_test_intersection(idxs, expected_startdate, starttime, tz, freq)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
@pytest.mark.parametrize(("startdates", "freq", "expected_startdate"), TESTCASES)
def test_intersection_distinctfreq(
    startdates: Iterable[str],
    starttime: str,
    tz: str,
    freq: str,
    expected_startdate: str,
):
    """Test if intersection of indices with distinct frequencies gives correct result."""
    otherfreq = "H" if freq == "D" else "D"
    idxs = [
        idx(startdates[0], starttime, tz, freq),
        idx(startdates[1], starttime, tz, otherfreq),
    ]
    do_test_intersection(idxs, ValueError)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
@pytest.mark.parametrize(("startdates", "freq", "expected_startdate"), TESTCASES)
def test_intersection_distincttz(
    startdates: Iterable[str],
    starttime: str,
    tz: str,
    freq: str,
    expected_startdate: str,
):
    """Test if intersection of indices with distinct timezones gives correct result."""
    othertz = None if tz == "Europe/Berlin" else "Europe/Berlin"
    idxs = [
        idx(startdates[0], starttime, tz, freq),
        idx(startdates[1], starttime, othertz, freq),
    ]
    do_test_intersection(idxs, ValueError)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
@pytest.mark.parametrize(("startdates", "freq", "expected_startdate"), TESTCASES)
def test_intersection_distinctstartofday(
    startdates: Iterable[str],
    starttime: str,
    tz: str,
    freq: str,
    expected_startdate: str,
):
    """Test if intersection of indices with distinct frequencies gives correct result."""
    otherstarttime = "00:00" if starttime == "06:00" else "06:00"
    idxs = [
        idx(startdates[0], starttime, tz, freq),
        idx(startdates[1], otherstarttime, tz, freq),
    ]
    do_test_intersection(idxs, ValueError)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("freq", tools.freq.FREQUENCIES)
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
def test_intersection_nooverlap(tz: str, freq: str, starttime: str):
    """Test if intersection of non-overlapping indices gives correct result."""
    idxs = [
        idx("2020-01-01", starttime, tz, freq, "2022-01-01"),
        idx("2023-01-01", starttime, tz, freq, "2025-01-01"),
    ]
    do_test_intersection(idxs, None, "", tz, freq)


def idx(
    startdate: str, starttime: str, tz: str, freq: str, enddate: str = COMMON_END
) -> pd.DatetimeIndex:
    # Empty index.
    if startdate is None:
        return pd.DatetimeIndex([], freq=freq, tz=tz)
    # Normal index.
    ts_start = pd.Timestamp(f"{startdate} {starttime}", tz=tz)
    ts_end = pd.Timestamp(f"{enddate} {starttime}", tz=tz)
    return pd.date_range(ts_start, ts_end, freq=freq, inclusive="left")


def do_test_intersection(
    idxs: Iterable[pd.DatetimeIndex],
    expected_startdate: Union[str, Exception],
    expected_starttime: str = None,
    expected_tz: str = None,
    expected_freq: str = None,
):
    # Error case.
    if type(expected_startdate) is type and issubclass(expected_startdate, Exception):
        with pytest.raises(expected_startdate):
            tools.intersect.index(*idxs)
        return
    # Normal case.
    result = tools.intersect.index(*idxs)
    expected = idx(expected_startdate, expected_starttime, expected_tz, expected_freq)
    testing.assert_index_equal(result, expected)
