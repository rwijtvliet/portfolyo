import pandas as pd
import pytest

from portfolyo import testing, tools

TESTCASES = [  # start, periods, freq, expected_start
    # Natural-boundary.
    # Natural-boundary timestamps; without DST-transition.
    ("2020-02-01 1:00", 24, "h", "2020-02-01"),
    ("2020-02-01 0:15", 96, "15min", "2020-02-01"),
    ("2020-02-02", 28, "D", "2020-02-01"),
    # Natural-boundary timestamps; with DST-start.
    ("2020-03-29 1:00", 24, "h", "2020-03-29"),
    ("2020-03-29 4:00", 24, "h", "2020-03-29 3:00"),
    ("2020-03-29 0:15", 96, "15min", "2020-03-29"),
    ("2020-03-29 0:30", 96, "15min", "2020-03-29 0:15"),
    ("2020-03-29 1:30", 96, "15min", "2020-03-29 1:15"),
    ("2020-03-29 3:30", 96, "15min", "2020-03-29 3:15"),
    ("2020-03-29 3:15", 96, "15min", "2020-03-29 3:00"),
    ("2020-03-02", 31, "D", "2020-03-01"),
    # Natural-boundary timestamps; with DST-end.
    ("2020-10-25 1:00", 24, "h", "2020-10-25"),
    ("2020-10-25 4:00", 24, "h", "2020-10-25 3:00"),
    ("2020-10-25 0:15", 96, "15min", "2020-10-25"),
    ("2020-10-25 0:30", 96, "15min", "2020-10-25 0:15"),
    ("2020-10-25 1:30", 96, "15min", "2020-10-25 1:15"),
    ("2020-10-25 3:30", 96, "15min", "2020-10-25 3:15"),
    ("2020-10-25 3:15", 96, "15min", "2020-10-25 3:00"),
    ("2020-10-02", 31, "D", "2020-10-01"),
    # Natural-boundary timestamps; with DST-start and DST-end.
    ("2020-04-01", 12, "MS", "2020-03-01"),
    ("2020-07-01", 12, "MS", "2020-06-01"),
    ("2020-04-01", 4, "QS", "2020-01-01"),
    ("2020-07-01", 4, "QS", "2020-04-01"),
    ("2020-04-01", 4, "QS-APR", "2020-01-01"),
    ("2020-07-01", 4, "QS-APR", "2020-04-01"),
    ("2020-05-01", 4, "QS-FEB", "2020-02-01"),
    ("2020-08-01", 4, "QS-FEB", "2020-05-01"),
    # Unnatural-boundary.
    # Unnatural-boundary timestamps; without DST-transition.
    ("2020-02-01 01:30", 24, "h", "2020-02-01 00:30"),
    ("2020-02-01 01:32", 96, "15min", "2020-02-01 01:17"),
    ("2020-02-02 06:00", 5, "D", "2020-02-01 06:00"),
    # Unnatural-boundary timestamps; with DST-start.
    ("2020-03-29 1:30", 24, "h", "2020-03-29 0:30"),
    ("2020-03-29 4:30", 24, "h", "2020-03-29 3:30"),
    ("2020-03-29 0:17", 96, "15min", "2020-03-29 0:02"),
    ("2020-03-29 0:32", 96, "15min", "2020-03-29 0:17"),
    ("2020-03-29 1:32", 96, "15min", "2020-03-29 1:17"),
    ("2020-03-29 3:32", 96, "15min", "2020-03-29 3:17"),
    ("2020-03-29 3:17", 96, "15min", "2020-03-29 3:02"),
    ("2020-03-02 06:00", 31, "D", "2020-03-01 06:00"),
    # Natural-boundary timestamps; with DST-end.
    ("2020-10-25 1:30", 24, "h", "2020-10-25 0:30"),
    ("2020-10-25 4:30", 24, "h", "2020-10-25 3:30"),
    ("2020-10-25 0:17", 96, "15min", "2020-10-25 0:02"),
    ("2020-10-25 0:32", 96, "15min", "2020-10-25 0:17"),
    ("2020-10-25 1:32", 96, "15min", "2020-10-25 1:17"),
    ("2020-10-25 3:32", 96, "15min", "2020-10-25 3:17"),
    ("2020-10-25 3:17", 96, "15min", "2020-10-25 3:02"),
    ("2020-10-02 06:00", 31, "D", "2020-10-01 06:00"),
    # Natural-boundary timestamps; with DST-start and DST-end.
    ("2020-04-01 06:00", 12, "MS", "2020-03-01 06:00"),
    ("2020-07-01 06:00", 12, "MS", "2020-06-01 06:00"),
    ("2020-04-01 06:00", 4, "QS", "2020-01-01 06:00"),
    ("2020-07-01 06:00", 4, "QS", "2020-04-01 06:00"),
]

TESTCASES_DST_AWARE = [  # start, periods, freq, expected_start"
    # Index with DST-start.
    ("2020-03-29 1:00", 24, "h", "2020-03-29"),
    ("2020-03-29 3:00", 24, "h", "2020-03-29 1:00"),
    ("2020-03-29 4:00", 24, "h", "2020-03-29 3:00"),
    ("2020-03-29 0:15", 96, "15min", "2020-03-29"),
    ("2020-03-29 1:30", 96, "15min", "2020-03-29 1:15"),
    ("2020-03-29 1:45", 96, "15min", "2020-03-29 1:30"),
    ("2020-03-29 3:00", 96, "15min", "2020-03-29 1:45"),
    ("2020-03-29 3:15", 96, "15min", "2020-03-29 3:00"),
    # Index with DST-end.
    ("2020-10-25 1:00", 24, "h", "2020-10-25"),
    ("2020-10-25 2:00+0200", 24, "h", "2020-10-25 1:00"),
    ("2020-10-25 2:00+0100", 24, "h", "2020-10-25 2:00+0200"),
    ("2020-10-25 3:00", 24, "h", "2020-10-25 2:00+0100"),
    ("2020-10-25 4:00", 24, "h", "2020-10-25 3:00"),
    ("2020-10-25 0:15", 96, "15min", "2020-10-25"),
    ("2020-10-25 0:30", 96, "15min", "2020-10-25 0:15"),
    ("2020-10-25 1:30", 96, "15min", "2020-10-25 1:15"),
    ("2020-10-25 2:15+0200", 96, "15min", "2020-10-25 2:00+0200"),
    ("2020-10-25 2:30+0200", 96, "15min", "2020-10-25 2:15+0200"),
    ("2020-10-25 2:00+0100", 96, "15min", "2020-10-25 2:45+0200"),
    ("2020-10-25 2:15+0100", 96, "15min", "2020-10-25 2:00+0100"),
    ("2020-10-25 2:30+0100", 96, "15min", "2020-10-25 2:15+0100"),
    ("2020-10-25 1:30", 96, "15min", "2020-10-25 1:15"),
    ("2020-10-25 3:30", 96, "15min", "2020-10-25 3:15"),
    ("2020-10-25 3:15", 96, "15min", "2020-10-25 3:00"),
]

TESTCASES_DST_NAIVE = [  # date, times, expected_times
    (
        "2020-03-29",
        ["01:00:00", "03:00:00", "04:00:00", "05:00:00"],
        ["00:00:00", "02:00:00", "03:00:00", "04:00:00"],
    ),
    (
        "2020-10-25",
        ["01:00:00", "02:00:00", "02:00:00", "03:00:00"],
        ["00:00:00", "01:00:00", "01:00:00", "02:00:00"],
    ),
]


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("remove_freq", ["remove", "keep"])
@pytest.mark.parametrize(("start", "periods", "freq", "expected_start"), TESTCASES)
def test_righttoleft(
    start: str, periods: int, freq: str, expected_start: str, tz: str, remove_freq: str
):
    """Test if index of rightbound timestamps can be make leftbound."""
    # Input.
    i = pd.date_range(start, periods=periods, freq=freq, tz=tz)
    if remove_freq == "remove":
        i.freq = None

    # Expected output.
    if remove_freq == "remove" and freq == "QS-APR":
        freq_expected = "QS"  # if freq removed, expect 'base case'
    else:
        freq_expected = freq
    expected = pd.date_range(expected_start, periods=periods, freq=freq_expected, tz=tz)

    # Actual output.
    result = tools.righttoleft.index(i)
    testing.assert_index_equal(result, expected)


@pytest.mark.parametrize("remove_freq", ["remove", "keep"])
@pytest.mark.parametrize(
    ("start", "periods", "freq", "expected_start"), TESTCASES_DST_AWARE
)
def test_righttoleft_dst_tzaware(start, periods, freq, expected_start, remove_freq):
    """Test if index of rightbound timestamps can be make leftbound, across DST changeover."""
    i = pd.date_range(
        pd.Timestamp(start, tz="Europe/Berlin"), periods=periods, freq=freq
    )
    expected = pd.date_range(
        pd.Timestamp(expected_start, tz="Europe/Berlin"), periods=periods, freq=freq
    )
    if remove_freq == "remove":
        i.freq = None
    result = tools.righttoleft.index(i)
    testing.assert_index_equal(result, expected)


@pytest.mark.parametrize(("date", "times", "expected_times"), TESTCASES_DST_NAIVE)
def test_righttoleft_dst_tznaive(date, times, expected_times):
    i = pd.DatetimeIndex(f"{date} {time}" for time in times)
    expected = pd.DatetimeIndex(f"{date} {time}" for time in expected_times)
    result = tools.righttoleft.index(i)
    testing.assert_index_equal(result, expected)
