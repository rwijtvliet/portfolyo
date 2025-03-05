from typing import Iterable

import pandas as pd
import pytest

from portfolyo import testing, tools

COMMON_END = "2022-02-02"

TESTCASES = [  # startdates, freq, expected_startdate
    # One starts at first day of year.
    (("2020-01-01", "2020-01-20"), "15min", "2020-01-20"),
    (("2020-01-01", "2020-01-20"), "15min", "2020-01-20"),
    (("2020-01-01", "2020-01-20"), "h", "2020-01-20"),
    (("2020-01-01", "2020-01-20"), "h", "2020-01-20"),
    (("2020-01-01", "2020-01-20"), "D", "2020-01-20"),
    (("2020-01-01", "2020-01-20"), "D", "2020-01-20"),
    (("2020-01-01", "2020-03-01"), "MS", "2020-03-01"),
    (("2020-01-01", "2020-03-01"), "MS", "2020-03-01"),
    (("2020-01-01", "2020-04-01"), "QS", "2020-04-01"),
    (("2020-01-01", "2020-04-01"), "QS", "2020-04-01"),
    (("2020-01-01", "2021-01-01"), "YS", "2021-01-01"),
    (("2020-01-01", "2021-01-01"), "YS", "2021-01-01"),
    # Both start in middle of year.
    (("2020-04-21", "2020-06-20"), "15min", "2020-06-20"),
    (("2020-04-21", "2020-06-20"), "15min", "2020-06-20"),
    (("2020-04-21", "2020-06-20"), "h", "2020-06-20"),
    (("2020-04-21", "2020-06-20"), "h", "2020-06-20"),
    (("2020-04-21", "2020-06-20"), "D", "2020-06-20"),
    (("2020-04-21", "2020-06-20"), "D", "2020-06-20"),
    (("2020-03-01", "2021-06-01"), "QS-MAR", "2021-06-01"),
    (("2020-04-01", "2021-04-01"), "YS-APR", "2021-04-01"),
]

TESTCASES_3 = [  # first_period, second_period, freqs, result1, result2
    # One starts at first day of year.
    (
        ("2020-01-01", "2023-01-01"),
        ("2020-01-20", "2023-01-01"),
        ("15min", "h"),
        ("2020-01-20", "2023-01-01"),
        ("2020-01-20", "2023-01-01"),
    ),
    (
        ("2020-01-01", "2023-01-01"),
        ("2020-01-20", "2023-01-01"),
        ("15min", "D"),
        ("2020-01-20", "2023-01-01"),
        ("2020-01-20", "2023-01-01"),
    ),
    (
        ("2022-04-01", "2023-01-01"),
        ("2021-02-01", "2023-01-01"),
        ("h", "MS"),
        ("2022-04-01", "2023-01-01"),
        ("2022-04-01", "2023-01-01"),
    ),
    (
        ("2020-01-01", "2023-01-01"),
        ("2020-04-01", "2023-01-01"),
        ("h", "QS"),
        ("2020-04-01", "2023-01-01"),
        ("2020-04-01", "2023-01-01"),
    ),
    (
        ("2020-01-01", "2023-01-01"),
        ("2021-01-01", "2023-01-01"),
        ("D", "YS"),
        ("2021-01-01", "2023-01-01"),
        ("2021-01-01", "2023-01-01"),
    ),
    # Both start in middle of year.
    (
        ("2020-04-21", "2023-01-01"),
        ("2020-06-20", "2023-01-01"),
        ("15min", "h"),
        ("2020-06-20", "2023-01-01"),
        ("2020-06-20", "2023-01-01"),
    ),
    (
        ("2020-04-21", "2023-01-01"),
        ("2020-06-20", "2023-01-01"),
        ("15min", "D"),
        ("2020-06-20", "2023-01-01"),
        ("2020-06-20", "2023-01-01"),
    ),
    (
        ("2020-04-21", "2023-01-01"),
        ("2020-07-01", "2023-01-01"),
        ("h", "MS"),
        ("2020-07-01", "2023-01-01"),
        ("2020-07-01", "2023-01-01"),
    ),
    (
        ("2020-04-21", "2023-01-01"),
        ("2020-07-01", "2023-01-01"),
        ("h", "QS"),
        ("2020-07-01", "2023-01-01"),
        ("2020-07-01", "2023-01-01"),
    ),
    (
        ("2020-04-21", "2023-01-01"),
        ("2021-01-01", "2023-01-01"),
        ("D", "YS"),
        ("2021-01-01", "2023-01-01"),
        ("2021-01-01", "2023-01-01"),
    ),
    (
        ("2020-03-21", "2022-04-01"),
        ("2020-04-01", "2022-04-01"),
        ("D", "YS-APR"),
        ("2020-04-01", "2022-04-01"),
        ("2020-04-01", "2022-04-01"),
    ),
    # the start and end dates of the resulting intersection are not the same for first and second index
    (
        ("2020-03-01", "2021-02-01"),
        ("2020-05-01", "2021-02-01"),
        ("MS", "QS-MAY"),
        ("2020-05-01", "2021-02-01"),
        ("2020-05-01", "2021-02-01"),
    ),
    (
        ("2020-03-01", "2021-04-01"),
        ("2020-04-01", "2021-03-01"),
        ("MS", "YS-APR"),
        ("2020-04-01", "2021-04-01"),
        ("2020-04-01", "2021-03-01"),
    ),
    (
        ("2020-01-01", "2021-04-01"),
        ("2020-04-01", "2021-03-01"),
        ("QS", "YS-APR"),
        ("2020-04-01", "2021-02-01"),
        ("2020-04-01", "2021-03-01"),
    ),
    # add test case with empty index
]


def get_idx(
    startdate: str,
    starttime: str,
    tz: str,
    freq: str,
    enddate: str,
) -> pd.DatetimeIndex:
    # Empty index.
    if startdate is None:
        return pd.DatetimeIndex([], freq=freq, tz=tz)
    # Normal index.
    ts_start = pd.Timestamp(f"{startdate} {starttime}", tz=tz)
    ts_end = pd.Timestamp(f"{enddate} {starttime}", tz=tz)
    return pd.date_range(ts_start, ts_end, freq=freq, inclusive="left")


# NOTE: ignore_start_of_day gives ValueError if we have freq == "15min" or freq == "h"
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(("startdates", "freq", "expected_startdate"), TESTCASES)
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
# @pytest.mark.parametrize("indexorframe", ["idx", "fr"])
def test_intersect_flex_ignore_start_of_day(
    # indexorframe: str,
    startdates: Iterable[str],
    starttime: str,
    tz: str,
    freq: str,
    expected_startdate: str,
):
    otherstarttime = "00:00" if starttime == "06:00" else "06:00"
    idxs = [
        get_idx(
            startdates[0],
            starttime,
            tz,
            freq,
            COMMON_END,
        ),
        get_idx(
            startdates[1],
            otherstarttime,
            tz,
            freq,
            COMMON_END,
        ),
    ]
    do_test_intersect(
        "idx",
        idxs,
        expected_startdate_1=(ValueError if freq == "15min" or freq == "h" else expected_startdate),
        expected_startdate_2=expected_startdate,
        expected_tz=tz,
        expected_freq=freq,
        expected_starttime=starttime,
        expected_otherstarttime=otherstarttime,
        expected_othertz=tz,
        expected_otherfreq=freq,
        enddate_1=COMMON_END,
        enddate_2=COMMON_END,
        ignore_start_of_day=True,
    )


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
@pytest.mark.parametrize(("startdates", "freq", "expected_startdate"), TESTCASES)
# @pytest.mark.parametrize("indexorframe", ["idx", "fr"])
def test_intersect_flex_ignore_tz(
    # indexorframe: str,
    startdates: Iterable[str],
    starttime: str,
    tz: str,
    freq: str,
    expected_startdate: str,
):
    othertz = None if tz == "Europe/Berlin" else "Europe/Berlin"
    idxs = [
        get_idx(startdates[0], starttime, tz, freq, COMMON_END),
        get_idx(startdates[1], starttime, othertz, freq, COMMON_END),
    ]
    do_test_intersect(
        "idx",
        idxs,
        expected_startdate_1=expected_startdate,
        expected_startdate_2=expected_startdate,
        expected_tz=tz,
        expected_freq=freq,
        expected_starttime=starttime,
        expected_otherstarttime=starttime,
        expected_othertz=othertz,
        expected_otherfreq=freq,
        enddate_1=COMMON_END,
        enddate_2=COMMON_END,
        ignore_tz=True,
    )


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("first_period", "second_period", "freq", "result1_dates", "result2_dates"),
    TESTCASES_3,
)
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
def test_intersect_flex_ignore_freq(
    # indexorframe: str,
    first_period: Iterable[str],
    second_period: Iterable[str],
    starttime: str,
    tz: str,
    freq: Iterable[str],
    result1_dates: Iterable[str],
    result2_dates: Iterable[str],
):
    """Test if intersection of indices with distinct frequencies gives correct result."""

    idxs = [
        get_idx(first_period[0], starttime, tz, freq[0], first_period[1]),
        get_idx(second_period[0], starttime, tz, freq[1], second_period[1]),
    ]
    do_test_intersect(
        "idx",
        idxs,
        expected_startdate_1=result1_dates[0],
        expected_startdate_2=result2_dates[0],
        expected_tz=tz,
        expected_freq=freq[0],
        expected_starttime=starttime,
        expected_otherstarttime=starttime,
        expected_othertz=tz,
        expected_otherfreq=freq[1],
        enddate_1=result1_dates[1],
        enddate_2=result2_dates[1],
        ignore_freq=True,
    )


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("first_period", "second_period", "freq", "result1_dates", "result2_dates"),
    TESTCASES_3,
)
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
def test_ignore_all(  # indexorframe: str,
    # indexorframe: str,
    first_period: Iterable[str],
    second_period: Iterable[str],
    starttime: str,
    tz: str,
    freq: Iterable[str],
    result1_dates: Iterable[str],
    result2_dates: Iterable[str],
):
    otherstarttime = "00:00" if starttime == "06:00" else "06:00"
    othertz = None if tz == "Europe/Berlin" else "Europe/Berlin"
    idxs = [
        get_idx(first_period[0], starttime, tz, freq[0], first_period[1]),
        get_idx(second_period[0], otherstarttime, othertz, freq[1], second_period[1]),
    ]
    do_test_intersect(
        "idx",
        idxs,
        expected_startdate_1=(
            ValueError
            if freq[0] == "15min" or freq[0] == "h" or freq[1] == "15min" or freq[1] == "h"
            else result1_dates[0]
        ),
        expected_startdate_2=result2_dates[0],
        expected_tz=tz,
        expected_freq=freq[0],
        expected_starttime=starttime,
        expected_otherstarttime=otherstarttime,
        expected_othertz=othertz,
        expected_otherfreq=freq[1],
        enddate_1=result1_dates[1],
        enddate_2=result2_dates[1],
        ignore_freq=True,
        ignore_start_of_day=True,
        ignore_tz=True,
    )


def do_test_intersect(
    indexorframe: str,
    idxs: Iterable[pd.DatetimeIndex],
    expected_startdate_1: str | Exception,
    expected_startdate_2: str,
    expected_starttime: str = None,
    expected_tz: str = None,
    expected_freq: str = None,
    expected_otherstarttime: str = None,
    expected_othertz: str = None,
    expected_otherfreq: str = None,
    enddate_1: str = None,
    enddate_2: str = None,
    ignore_start_of_day: bool = False,
    ignore_tz: bool = False,
    ignore_freq: bool = False,
):
    if indexorframe == "idx":
        do_test_intersect_index(
            idxs,
            expected_startdate_1,
            expected_startdate_2,
            expected_starttime,
            expected_tz,
            expected_freq,
            expected_otherstarttime,
            expected_othertz,
            expected_otherfreq,
            enddate_1,
            enddate_2,
            ignore_start_of_day,
            ignore_tz,
            ignore_freq,
        )


def do_test_intersect_index(
    idxs: Iterable[pd.DatetimeIndex],
    expected_startdate_1: str,
    expected_startdate_2: str,
    expected_starttime: str = None,
    expected_tz: str = None,
    expected_freq: str = None,
    expected_otherstarttime: str = None,
    expected_othertz: str = None,
    expected_otherfreq: str = None,
    enddate_1: str = None,
    enddate_2: str = None,
    ignore_start_of_day: bool = False,
    ignore_tz: bool = False,
    ignore_freq: bool = False,
):
    # Error case.
    if isinstance(expected_startdate_1, type) and issubclass(expected_startdate_1, Exception):
        with pytest.raises(expected_startdate_1):
            tools.intersect.indices_flex(
                *idxs,
                ignore_start_of_day=False,
                ignore_tz=False,
                ignore_freq=ignore_freq,
            )
        return
    # Normal case.
    out_a, out_b = tools.intersect.indices_flex(
        *idxs,
        ignore_start_of_day=ignore_start_of_day,
        ignore_tz=ignore_tz,
        ignore_freq=ignore_freq,
    )
    expected_a = get_idx(
        expected_startdate_1,
        expected_starttime,
        expected_tz,
        expected_freq,
        enddate_1,
    )
    expected_b = get_idx(
        expected_startdate_2,
        expected_otherstarttime,
        expected_othertz,
        expected_otherfreq,
        enddate_2,
    )
    testing.assert_index_equal(out_a, expected_a)
    testing.assert_index_equal(out_b, expected_b)


def test_intersect_flex_dst():
    """Test if intersection keeps working if DST-boundary is right at end."""
    i1 = pd.date_range("2020", "2020-03-29", freq="D", tz="Europe/Berlin")
    i2 = pd.date_range("2020", "2020-03-30", freq="D", tz="Europe/Berlin")

    expected = pd.date_range("2020", "2020-03-29", freq="D", tz="Europe/Berlin")

    result1, result2 = tools.intersect.indices_flex(i1, i2)
    testing.assert_index_equal(result1, expected)
    testing.assert_index_equal(result2, expected)
