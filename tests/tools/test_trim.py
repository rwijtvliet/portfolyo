import pandas as pd
import pytest

from portfolyo import testing, tools

TESTCASES_ALMOSTFULL = [  # start, end, freq, trimfreq, tr_start, tr_end
    # Trimming below-daily index to below-daily freq.
    ("2020-01-01 00", "2022-01-01 15", "15T", "15T", "2020-01-01 00", "2022-01-01 15"),
    ("2020-01-01 00", "2022-01-01 15", "15T", "H", "2020-01-01 00", "2022-01-01 15"),
    ("2020-01-01 00", "2022-01-01 15", "H", "15T", "2020-01-01 00", "2022-01-01 15"),
    ("2020-01-01 00", "2022-01-01 15", "H", "H", "2020-01-01 00", "2022-01-01 15"),
    ("2020-01-01 06", "2022-01-01 15", "15T", "15T", "2020-01-01 06", "2022-01-01 15"),
    ("2020-01-01 06", "2022-01-01 15", "15T", "H", "2020-01-01 06", "2022-01-01 15"),
    ("2020-01-01 06", "2022-01-01 15", "H", "15T", "2020-01-01 06", "2022-01-01 15"),
    ("2020-01-01 06", "2022-01-01 15", "H", "H", "2020-01-01 06", "2022-01-01 15"),
    ("2020-01-01 06", "2022-01-01 04", "15T", "15T", "2020-01-01 06", "2022-01-01 04"),
    ("2020-01-01 06", "2022-01-01 04", "15T", "H", "2020-01-01 06", "2022-01-01 04"),
    ("2020-01-01 06", "2022-01-01 04", "H", "15T", "2020-01-01 06", "2022-01-01 04"),
    ("2020-01-01 06", "2022-01-01 04", "H", "H", "2020-01-01 06", "2022-01-01 04"),
    # Trimming below-daily index to daily-or-longer freq.
    ("2020-01-01 00", "2022-01-01 15", "15T", "D", "2020-01-01 00", "2022-01-01 00"),
    ("2020-01-01 00", "2022-01-01 15", "15T", "MS", "2020-01-01 00", "2022-01-01 00"),
    ("2020-01-01 00", "2022-01-01 15", "15T", "QS", "2020-01-01 00", "2022-01-01 00"),
    ("2020-01-01 00", "2022-01-01 15", "15T", "AS", "2020-01-01 00", "2022-01-01 00"),
    ("2020-01-01 00", "2022-01-01 15", "H", "D", "2020-01-01 00", "2022-01-01 00"),
    ("2020-01-01 00", "2022-01-01 15", "H", "MS", "2020-01-01 00", "2022-01-01 00"),
    ("2020-01-01 00", "2022-01-01 15", "H", "QS", "2020-01-01 00", "2022-01-01 00"),
    ("2020-01-01 00", "2022-01-01 15", "H", "AS", "2020-01-01 00", "2022-01-01 00"),
    ("2020-01-01 06", "2022-01-01 15", "15T", "D", "2020-01-01 06", "2022-01-01 06"),
    ("2020-01-01 06", "2022-01-01 15", "15T", "MS", "2020-01-01 06", "2022-01-01 06"),
    ("2020-01-01 06", "2022-01-01 15", "15T", "QS", "2020-01-01 06", "2022-01-01 06"),
    ("2020-01-01 06", "2022-01-01 15", "15T", "AS", "2020-01-01 06", "2022-01-01 06"),
    ("2020-01-01 06", "2022-01-01 15", "H", "D", "2020-01-01 06", "2022-01-01 06"),
    ("2020-01-01 06", "2022-01-01 15", "H", "MS", "2020-01-01 06", "2022-01-01 06"),
    ("2020-01-01 06", "2022-01-01 15", "H", "QS", "2020-01-01 06", "2022-01-01 06"),
    ("2020-01-01 06", "2022-01-01 15", "H", "AS", "2020-01-01 06", "2022-01-01 06"),
    ("2020-01-01 06", "2022-01-01 04", "15T", "D", "2020-01-01 06", "2021-12-31 06"),
    ("2020-01-01 06", "2022-01-01 04", "15T", "MS", "2020-01-01 06", "2021-12-01 06"),
    ("2020-01-01 06", "2022-01-01 04", "15T", "QS", "2020-01-01 06", "2021-10-01 06"),
    ("2020-01-01 06", "2022-01-01 04", "15T", "AS", "2020-01-01 06", "2021-01-01 06"),
    ("2020-01-01 06", "2022-01-01 04", "H", "D", "2020-01-01 06", "2021-12-31 06"),
    ("2020-01-01 06", "2022-01-01 04", "H", "MS", "2020-01-01 06", "2021-12-01 06"),
    ("2020-01-01 06", "2022-01-01 04", "H", "QS", "2020-01-01 06", "2021-10-01 06"),
    ("2020-01-01 06", "2022-01-01 04", "H", "AS", "2020-01-01 06", "2021-01-01 06"),
]

TESTCASES_MIDYEAR = [  # start, end, freq, trimfreq, tr_start, tr_end
    # Trimming below-daily index to below-daily freq.
    ("2020-04-21 00", "2022-12-15 15", "15T", "15T", "2020-04-21 00", "2022-12-15 15"),
    ("2020-04-21 00", "2022-12-15 15", "15T", "H", "2020-04-21 00", "2022-12-15 15"),
    ("2020-04-21 00", "2022-12-15 15", "H", "15T", "2020-04-21 00", "2022-12-15 15"),
    ("2020-04-21 00", "2022-12-15 15", "H", "H", "2020-04-21 00", "2022-12-15 15"),
    ("2020-04-21 06", "2022-12-15 15", "15T", "15T", "2020-04-21 06", "2022-12-15 15"),
    ("2020-04-21 06", "2022-12-15 15", "15T", "H", "2020-04-21 06", "2022-12-15 15"),
    ("2020-04-21 06", "2022-12-15 15", "H", "15T", "2020-04-21 06", "2022-12-15 15"),
    ("2020-04-21 06", "2022-12-15 15", "H", "H", "2020-04-21 06", "2022-12-15 15"),
    ("2020-04-21 06", "2022-12-15 04", "15T", "15T", "2020-04-21 06", "2022-12-15 04"),
    ("2020-04-21 06", "2022-12-15 04", "15T", "H", "2020-04-21 06", "2022-12-15 04"),
    ("2020-04-21 06", "2022-12-15 04", "H", "15T", "2020-04-21 06", "2022-12-15 04"),
    ("2020-04-21 06", "2022-12-15 04", "H", "H", "2020-04-21 06", "2022-12-15 04"),
    # Trimming below-daily index to daily-or-longer freq.
    ("2020-04-21 00", "2022-12-15 15", "15T", "D", "2020-04-21 00", "2022-12-15 00"),
    ("2020-04-21 00", "2022-12-15 15", "15T", "MS", "2020-05-01 00", "2022-12-01 00"),
    ("2020-04-21 00", "2022-12-15 15", "15T", "QS", "2020-07-01 00", "2022-10-01 00"),
    ("2020-04-21 00", "2022-12-15 15", "15T", "AS", "2021-01-01 00", "2022-01-01 00"),
    ("2020-04-21 00", "2022-12-15 15", "H", "D", "2020-04-21 00", "2022-12-15 00"),
    ("2020-04-21 00", "2022-12-15 15", "H", "MS", "2020-05-01 00", "2022-12-01 00"),
    ("2020-04-21 00", "2022-12-15 15", "H", "QS", "2020-07-01 00", "2022-10-01 00"),
    ("2020-04-21 00", "2022-12-15 15", "H", "AS", "2021-01-01 00", "2022-01-01 00"),
    ("2020-04-21 06", "2022-12-15 15", "15T", "D", "2020-04-21 06", "2022-12-15 06"),
    ("2020-04-21 06", "2022-12-15 15", "15T", "MS", "2020-05-01 06", "2022-12-01 06"),
    ("2020-04-21 06", "2022-12-15 15", "15T", "QS", "2020-07-01 06", "2022-10-01 06"),
    ("2020-04-21 06", "2022-12-15 15", "15T", "AS", "2021-01-01 06", "2022-01-01 06"),
    ("2020-04-21 06", "2022-12-15 15", "H", "D", "2020-04-21 06", "2022-12-15 06"),
    ("2020-04-21 06", "2022-12-15 15", "H", "MS", "2020-05-01 06", "2022-12-01 06"),
    ("2020-04-21 06", "2022-12-15 15", "H", "QS", "2020-07-01 06", "2022-10-01 06"),
    ("2020-04-21 06", "2022-12-15 15", "H", "AS", "2021-01-01 06", "2022-01-01 06"),
    ("2020-04-21 06", "2022-12-15 04", "15T", "D", "2020-04-21 06", "2022-12-14 06"),
    ("2020-04-21 06", "2022-12-15 04", "15T", "MS", "2020-05-01 06", "2022-12-01 06"),
    ("2020-04-21 06", "2022-12-15 04", "15T", "QS", "2020-07-01 06", "2022-10-01 06"),
    ("2020-04-21 06", "2022-12-15 04", "15T", "AS", "2021-01-01 06", "2022-01-01 06"),
    ("2020-04-21 06", "2022-12-15 04", "H", "D", "2020-04-21 06", "2022-12-14 06"),
    ("2020-04-21 06", "2022-12-15 04", "H", "MS", "2020-05-01 06", "2022-12-01 06"),
    ("2020-04-21 06", "2022-12-15 04", "H", "QS", "2020-07-01 06", "2022-10-01 06"),
    ("2020-04-21 06", "2022-12-15 04", "H", "AS", "2021-01-01 06", "2022-01-01 06"),
    # Trimming daily-or-longer index to shorter-than-daily freq.
    ("2020-04-21 00", "2022-12-15 00", "D", "15T", "2020-04-21 00", "2022-12-15 00"),
    ("2020-04-21 00", "2022-12-15 00", "D", "H", "2020-04-21 00", "2022-12-15 00"),
    ("2020-04-01 00", "2022-12-01 00", "MS", "15T", "2020-04-01 00", "2022-12-01 00"),
    ("2020-04-01 00", "2022-12-01 00", "MS", "H", "2020-04-01 00", "2022-12-01 00"),
    ("2020-04-01 00", "2022-10-01 00", "QS", "15T", "2020-04-01 00", "2022-10-01 00"),
    ("2020-04-01 00", "2022-10-01 00", "QS", "H", "2020-04-01 00", "2022-10-01 00"),
    ("2020-01-01 00", "2022-01-01 00", "AS", "15T", "2020-01-01 00", "2022-01-01 00"),
    ("2020-01-01 00", "2022-01-01 00", "AS", "H", "2020-01-01 00", "2022-01-01 00"),
    ("2020-04-21 06", "2022-12-15 06", "D", "15T", "2020-04-21 06", "2022-12-15 06"),
    ("2020-04-21 06", "2022-12-15 06", "D", "H", "2020-04-21 06", "2022-12-15 06"),
    ("2020-04-01 06", "2022-12-01 06", "MS", "15T", "2020-04-01 06", "2022-12-01 06"),
    ("2020-04-01 06", "2022-12-01 06", "MS", "H", "2020-04-01 06", "2022-12-01 06"),
    ("2020-04-01 06", "2022-10-01 06", "QS", "15T", "2020-04-01 06", "2022-10-01 06"),
    ("2020-04-01 06", "2022-10-01 06", "QS", "H", "2020-04-01 06", "2022-10-01 06"),
    ("2020-01-01 06", "2022-01-01 06", "AS", "15T", "2020-01-01 06", "2022-01-01 06"),
    ("2020-01-01 06", "2022-01-01 06", "AS", "H", "2020-01-01 06", "2022-01-01 06"),
    # Trimming daily-or-longer index to daily-or-longer freq.
    ("2020-04-21 00", "2022-12-15 00", "D", "D", "2020-04-21 00", "2022-12-15 00"),
    ("2020-04-21 00", "2022-12-15 00", "D", "MS", "2020-05-01 00", "2022-12-01 00"),
    ("2020-04-21 00", "2022-12-15 00", "D", "QS", "2020-07-01 00", "2022-10-01 00"),
    ("2020-04-21 00", "2022-12-15 00", "D", "AS", "2021-01-01 00", "2022-01-01 00"),
    ("2020-04-01 00", "2022-12-01 00", "MS", "D", "2020-04-01 00", "2022-12-01 00"),
    ("2020-04-01 00", "2022-12-01 00", "MS", "MS", "2020-04-01 00", "2022-12-01 00"),
    ("2020-04-01 00", "2022-12-01 00", "MS", "QS", "2020-04-01 00", "2022-10-01 00"),
    ("2020-04-01 00", "2022-12-01 00", "MS", "AS", "2021-01-01 00", "2022-01-01 00"),
    ("2020-04-01 00", "2022-10-01 00", "QS", "D", "2020-04-01 00", "2022-10-01 00"),
    ("2020-04-01 00", "2022-10-01 00", "QS", "MS", "2020-04-01 00", "2022-10-01 00"),
    ("2020-04-01 00", "2022-10-01 00", "QS", "QS", "2020-04-01 00", "2022-10-01 00"),
    ("2020-04-01 00", "2022-10-01 00", "QS", "AS", "2021-01-01 00", "2022-01-01 00"),
    ("2020-01-01 00", "2022-01-01 00", "AS", "D", "2020-01-01 00", "2022-01-01 00"),
    ("2020-01-01 00", "2022-01-01 00", "AS", "MS", "2020-01-01 00", "2022-01-01 00"),
    ("2020-01-01 00", "2022-01-01 00", "AS", "QS", "2020-01-01 00", "2022-01-01 00"),
    ("2020-01-01 00", "2022-01-01 00", "AS", "AS", "2020-01-01 00", "2022-01-01 00"),
    ("2020-04-21 06", "2022-12-15 06", "D", "D", "2020-04-21 06", "2022-12-15 06"),
    ("2020-04-21 06", "2022-12-15 06", "D", "MS", "2020-05-01 06", "2022-12-01 06"),
    ("2020-04-21 06", "2022-12-15 06", "D", "QS", "2020-07-01 06", "2022-10-01 06"),
    ("2020-04-21 06", "2022-12-15 06", "D", "AS", "2021-01-01 06", "2022-01-01 06"),
    ("2020-04-01 06", "2022-12-01 06", "MS", "D", "2020-04-01 06", "2022-12-01 06"),
    ("2020-04-01 06", "2022-12-01 06", "MS", "MS", "2020-04-01 06", "2022-12-01 06"),
    ("2020-04-01 06", "2022-12-01 06", "MS", "QS", "2020-04-01 06", "2022-10-01 06"),
    ("2020-04-01 06", "2022-12-01 06", "MS", "AS", "2021-01-01 06", "2022-01-01 06"),
    ("2020-04-01 06", "2022-10-01 06", "QS", "D", "2020-04-01 06", "2022-10-01 06"),
    ("2020-04-01 06", "2022-10-01 06", "QS", "MS", "2020-04-01 06", "2022-10-01 06"),
    ("2020-04-01 06", "2022-10-01 06", "QS", "QS", "2020-04-01 06", "2022-10-01 06"),
    ("2020-04-01 06", "2022-10-01 06", "QS", "AS", "2021-01-01 06", "2022-01-01 06"),
    ("2020-01-01 06", "2022-01-01 06", "AS", "D", "2020-01-01 06", "2022-01-01 06"),
    ("2020-01-01 06", "2022-01-01 06", "AS", "MS", "2020-01-01 06", "2022-01-01 06"),
    ("2020-01-01 06", "2022-01-01 06", "AS", "QS", "2020-01-01 06", "2022-01-01 06"),
    ("2020-01-01 06", "2022-01-01 06", "AS", "AS", "2020-01-01 06", "2022-01-01 06"),
]


@pytest.mark.parametrize("indexorframe", ["index", "s", "s_unit", "df", "df_unit"])
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("freq", ["15T", "H", "D", "MS", "QS", "AS"])
@pytest.mark.parametrize("trimfreq", ["15T", "H", "D", "MS", "QS", "AS"])
def test_trim_notrimming(
    indexorframe: str, freq: str, tz: str, trimfreq: str, starttime: str
):
    """Test if no trimming is done when it is not necessary."""
    start = tr_start = f"2020-01-01 {starttime}"
    end = tr_end = f"2022-01-01 {starttime}"
    do_test_general(indexorframe, start, end, freq, tz, trimfreq, tr_start, tr_end)


@pytest.mark.parametrize("indexorframe", ["index", "s", "s_unit", "df", "df_unit"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("start", "end", "freq", "trimfreq", "tr_start", "tr_end"), TESTCASES_MIDYEAR
)
def test_trim_midyear(
    indexorframe: str,
    start: str,
    end: str,
    freq: str,
    tz: str,
    trimfreq: str,
    tr_start: str,
    tr_end: str,
):
    """Test if no trimming is done when it is not necessary."""
    start, end, tr_start, tr_end = (f"{ts}:00" for ts in (start, end, tr_start, tr_end))
    do_test_general(indexorframe, start, end, freq, tz, trimfreq, tr_start, tr_end)


@pytest.mark.parametrize("indexorframe", ["index", "s", "s_unit", "df", "df_unit"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("start", "end", "freq", "trimfreq", "tr_start", "tr_end"), TESTCASES_ALMOSTFULL
)
def test_trim_almostfull(
    indexorframe: str,
    start: str,
    end: str,
    freq: str,
    tz: str,
    trimfreq: str,
    tr_start: str,
    tr_end: str,
):
    """Test if no trimming is done when it is not necessary."""
    start, end, tr_start, tr_end = (f"{ts}:00" for ts in (start, end, tr_start, tr_end))
    do_test_general(indexorframe, start, end, freq, tz, trimfreq, tr_start, tr_end)


@pytest.mark.parametrize("indexorframe", ["index", "s", "s_unit", "df", "df_unit"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
def test_trim_short2short(indexorframe: str, tz: str):
    """Test if indices with below-daily frequencies are correctly trimmed."""
    # If input and output are below-daily, the offset parameter to the trim function should be ignored.
    start, tr_start = "2020-01-05 04:15:00", "2020-01-05 05:00"
    end, tr_end = "2022-12-21 04:15:00", "2022-12-21 04:00"
    do_test_general(indexorframe, start, end, "15T", tz, "H", tr_start, tr_end)


@pytest.mark.parametrize("indexorframe", ["index", "s", "s_unit", "df", "df_unit"])
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("trimfreq", "tr_start", "tr_end"),
    [
        ("MS", "2020-02-01", "2022-12-01"),
        ("QS", "2020-04-01", "2022-10-01"),
        ("AS", "2021-01-01", "2022-01-01"),
    ],
)
def test_trim_long2long(
    indexorframe: str,
    tr_start: str,
    tr_end: str,
    tz: str,
    trimfreq: str,
    starttime: str,
):
    """Test if indices with daily-or-longer frequencies are correctly trimmed."""
    # If input is daily-or-longer, and has offset, the offset parameter to the trim function should be ignored.
    start, tr_start = f"2020-01-05 {starttime}", f"{tr_start} {starttime}"
    end, tr_end = f"2022-12-21 {starttime}", f"{tr_end} {starttime}"
    do_test_general(indexorframe, start, end, "D", tz, trimfreq, tr_start, tr_end)


@pytest.mark.parametrize("indexorframe", ["index", "s", "s_unit", "df", "df_unit"])
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("freq", "start", "end"),
    [
        ("MS", "2020-02-01", "2022-12-01"),
        ("QS", "2020-04-01", "2022-10-01"),
        ("AS", "2021-01-01", "2022-01-01"),
    ],
)
@pytest.mark.parametrize("trimfreq", ["H", "15T"])
def test_trim_long2short(
    indexorframe: str,
    start: str,
    end: str,
    freq: str,
    tz: str,
    trimfreq: str,
    starttime: str,
):
    """Test if indices with daily-or-longer frequency are correctly trimmed in complex cases."""
    # If trimfrequency is below-daily, the offset parameter to the trim function should be ignored.
    start = tr_start = f"{start} {starttime}"
    end = tr_end = f"{end} {starttime}"
    do_test_general(indexorframe, start, end, freq, tz, trimfreq, tr_start, tr_end)


@pytest.mark.parametrize("indexorframe", ["index", "s", "s_unit", "df", "df_unit"])
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("trimfreq", "tr_startdate", "tr_enddate"),
    [
        ("D", "2020-01-05", "2022-12-21"),
        ("MS", "2020-02-01", "2022-12-01"),
        ("QS", "2020-04-01", "2022-10-01"),
        ("AS", "2021-01-01", "2022-01-01"),
    ],
)
def test_trim_short2long(
    indexorframe: str,
    tr_startdate: str,
    tr_enddate: str,
    trimfreq: str,
    tz,
    starttime: str,
):
    """Test if indices with below-daily frequency are correctly trimmed."""
    # Only in case a below-daily index is trimmed to a daily-or-longer frequency is the offset parameter used.
    start, tr_start = f"2020-01-05 {starttime}", f"{tr_startdate} {starttime}"
    end, tr_end = "2022-12-21 23:00:00", f"{tr_enddate} {starttime}"
    do_test_general(indexorframe, start, end, "H", tz, trimfreq, tr_start, tr_end)


def do_test_general(
    indexorframe: str,
    start: str,
    end: str,
    freq: str,
    tz: str,
    trimfreq: str,
    tr_start: str,
    tr_end: str,
):
    i = pd.date_range(start, end, freq=freq, inclusive="left", tz=tz)
    if not tr_start:
        i_expected = pd.DatetimeIndex([], freq=freq, tz=tz)
    else:
        i_expected = pd.date_range(tr_start, tr_end, freq=freq, inclusive="left", tz=tz)

    if indexorframe == "index":
        do_test_index(i, trimfreq, i_expected)
    elif indexorframe == "s":
        do_test_series(i, trimfreq, i_expected, False)
    elif indexorframe == "s_unit":
        do_test_series(i, trimfreq, i_expected, True)
    elif indexorframe == "df":
        do_test_dataframe(i, trimfreq, i_expected, False)
    elif indexorframe == "df_unit":
        do_test_dataframe(i, trimfreq, i_expected, True)


def do_test_index(i: pd.DatetimeIndex, trimfreq: str, expected: pd.DatetimeIndex):
    result = tools.trim.index(i, trimfreq)
    testing.assert_index_equal(result, expected)


def do_test_series(
    i: pd.DatetimeIndex, trimfreq: str, i_expected: pd.DatetimeIndex, with_units: bool
):
    dtype = "pint[MW]" if with_units else float
    fr = pd.Series(range(len(i)), i, dtype=dtype)
    expected = expected_series(i, i_expected, dtype)

    result = tools.trim.frame(fr, trimfreq)
    testing.assert_series_equal(result, expected)


def do_test_dataframe(
    i: pd.DatetimeIndex, trimfreq: str, i_expected: pd.DatetimeIndex, with_units: bool
):
    dtype = "pint[MW]" if with_units else float
    fr = pd.DataFrame({"a": range(len(i))}, i, dtype=dtype)
    expected = pd.DataFrame({"a": expected_series(i, i_expected, dtype)})

    result = tools.trim.frame(fr, trimfreq)
    testing.assert_frame_equal(result, expected)


def expected_series(i, i_expected, dtype):
    for num, ts in enumerate(i):
        if ts == i_expected[0]:
            break
    else:
        raise ValueError
    return pd.Series(range(num, num + len(i_expected)), i_expected, dtype=dtype)
