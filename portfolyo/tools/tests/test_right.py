import pandas as pd
import pytest
from portfolyo import testing, tools

TESTCASES_NODST = [
    ("2020", "15T", "2020-01-01 00:15"),
    ("2020", "H", "2020-01-01 01:00"),
    ("2020", "D", "2020-01-02"),
    ("2020", "MS", "2020-02-01"),
    ("2020", "QS", "2020-04"),
    ("2020", "AS", "2021"),
    ("2020-04-21", "15T", "2020-04-21 00:15"),
    ("2020-04-21", "H", "2020-04-21 01:00"),
    ("2020-04-21", "D", "2020-04-22"),
    ("2020-04-21 15:00", "15T", "2020-04-21 15:15"),
    ("2020-04-21 15:00", "H", "2020-04-21 16:00"),
]

TESTCASES_DST = [
    ("2020-03-29 01:45", "15T", "2020-03-29 03:00"),
    ("2020-03-29 01:00", "H", "2020-03-29 03:00"),
    ("2020-03-29 01:00", "D", "2020-03-30 01:00"),
    ("2020-03-29 03:00", "D", "2020-03-30 03:00"),
    ("2020-10-25 01:45", "15T", "2020-10-25 02:00"),
    ("2020-10-25 02:45+0200", "15T", "2020-10-25 02:00+0100"),
    ("2020-10-25 02:45+0100", "15T", "2020-10-25 03:00+0100"),
    ("2020-10-25 03:45", "15T", "2020-10-25 04:00"),
    ("2020-10-25 01:00", "H", "2020-10-25 02:00"),
    ("2020-10-25 02:00+0200", "H", "2020-10-25 02:00+0100"),
    ("2020-10-25 02:00+0100", "H", "2020-10-25 03:00"),
    ("2020-10-25 03:00", "H", "2020-10-25 04:00"),
    ("2020", "H", "2020-01-01 01:00"),
    ("2020-10-25 01:00", "D", "2020-10-26 01:00"),
    ("2020-10-25 02:00+0200", "D", "2020-10-26 02:00"),
    ("2020-10-25 02:00+0100", "D", "2020-10-26 02:00"),
    ("2020-10-25 03:00", "D", "2020-10-26 03:00"),
]


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(("ts", "freq", "expected_ts_right"), TESTCASES_NODST)
def test_right_stamp_nodst(ts, freq, tz, expected_ts_right):
    """Test if right timestamp is correctly calculated for timestamp without dst-transition."""
    stamptest(ts, freq, tz, expected_ts_right)


@pytest.mark.parametrize(("ts", "freq", "expected_ts_right"), TESTCASES_DST)
def test_right_stamp_dst(ts, freq, expected_ts_right):
    """Test if right timestamp is correctly calculated for timestamp with dst-transition."""
    stamptest(ts, freq, "Europe/Berlin", expected_ts_right)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("freq_as", ["attr", "param"])
@pytest.mark.parametrize(("ts", "freq", "expected_ts_right"), TESTCASES_NODST)
def test_right_index_nodst(ts, freq, tz, expected_ts_right):
    """Test if right timestamp is correctly calculated for index without dst-transition."""
    indextest(ts, freq, tz, expected_ts_right)


@pytest.mark.parametrize("freq_as", ["attr", "param"])
@pytest.mark.parametrize(("ts", "freq", "expected_ts_right"), TESTCASES_DST)
def test_right_index_dst(ts, freq, expected_ts_right):
    """Test if right timestamp is correctly calculated for index with dst-transition."""
    indextest(ts, freq, "Europe/Berlin", expected_ts_right)


def stamptest(ts, freq, tz, expected_ts_right):
    ts = pd.Timestamp(ts, tz=tz)
    result = tools.right.stamp(ts, freq)
    expected = pd.Timestamp(expected_ts_right, tz=tz)
    assert result == expected


def indextest(ts, freq, tz, freq_as, expected_ts_right):
    i = pd.date_range(ts, freq=freq, periods=100, tz=tz)
    if freq_as == "attr":
        result = tools.right.index(i)
    else:
        i.freq = None
        result = tools.right.index(i, freq)
    expected_right = pd.date_range(expected_ts_right, freq=freq, periods=100, tz=tz)
    expected = pd.Series(expected_right, i)
    testing.assert_series_equal(result, expected)
