import pandas as pd
import pytest

from portfolyo import testing, tools

# Here, only test that right-bound side is correctly calculated for VALID indices and
# timestamps. DO NOT check if error is raised for INvalid indices.

TESTCASES = [  # ts, freq, expected_ts_right
    # First day of X and start of day.
    ("2020-01-01", "15T", "2020-01-01 00:15"),
    ("2020-01-01", "H", "2020-01-01 01:00"),
    ("2020-01-01", "D", "2020-01-02"),
    ("2020-01-01", "MS", "2020-02-01"),
    ("2020-01-01", "QS", "2020-04-01"),
    ("2020-01-01", "AS", "2021-01-01"),
    # First day of X but not start of day.
    ("2020-01-01 06:00", "15T", "2020-01-01 06:15"),
    ("2020-01-01 06:00", "H", "2020-01-01 07:00"),
    ("2020-01-01 06:00", "D", "2020-01-02 06:00"),
    ("2020-01-01 06:00", "MS", "2020-02-01 06:00"),
    ("2020-01-01 06:00", "QS", "2020-04-01 06:00"),
    ("2020-01-01 06:00", "AS", "2021-01-01 06:00"),
    # Not first day of X but start of day.
    ("2020-04-01", "15T", "2020-04-01 00:15"),
    ("2020-04-01", "H", "2020-04-01 01:00"),
    ("2020-04-01", "D", "2020-04-02"),
    ("2020-04-01", "MS", "2020-05-01"),
    ("2020-04-01", "QS", "2020-07-01"),
    ("2020-04-21", "15T", "2020-04-21 00:15"),
    ("2020-04-21", "H", "2020-04-21 01:00"),
    ("2020-04-21", "D", "2020-04-22"),
    # Not first day of X and not start of day.
    ("2020-04-01 06:00", "15T", "2020-04-01 06:15"),
    ("2020-04-01 06:00", "H", "2020-04-01 07:00"),
    ("2020-04-01 06:00", "D", "2020-04-02 06:00"),
    ("2020-04-01 06:00", "MS", "2020-05-01 06:00"),
    ("2020-04-01 06:00", "QS", "2020-07-01 06:00"),
    ("2020-04-21 06:00", "15T", "2020-04-21 06:15"),
    ("2020-04-21 06:00", "H", "2020-04-21 07:00"),
    ("2020-04-21 06:00", "D", "2020-04-22 06:00"),
]

TESTCASES_DST = [  # ts, freq, expected_ts_right, periods
    # Start of DST.
    ("2020-03-29 01:00", "15T", "2020-03-29 01:15", 92 + 9 * 96),
    ("2020-03-29 01:00", "H", "2020-03-29 03:00", 23 + 9 * 24),
    ("2020-03-29 00:00", "D", "2020-03-30 00:00", None),
    ("2020-03-29 01:00", "D", "2020-03-30 01:00", None),
    ("2020-03-29 03:00", "D", "2020-03-30 03:00", None),
    ("2020-03-29 06:00", "D", "2020-03-30 06:00", None),
    ("2020-03-01 00:00", "MS", "2020-04-01 00:00", None),
    ("2020-03-01 06:00", "MS", "2020-04-01 06:00", None),
    ("2020-01-01 00:00", "QS", "2020-04-01 00:00", None),
    ("2020-01-01 06:00", "QS", "2020-04-01 06:00", None),
    # End of DST.
    ("2020-10-25 01:00", "15T", "2020-10-25 01:15+0200", 100 + 9 * 96),
    ("2020-10-25 02:00+0200", "15T", "2020-10-25 02:15+0200", 100 + 9 * 96),
    ("2020-10-25 02:00+0100", "15T", "2020-10-25 02:15+0100", 10 * 96),
    ("2020-10-25 03:00", "15T", "2020-10-25 03:15", 10 * 96),
    ("2020-10-25 01:00", "H", "2020-10-25 02:00+0200", 25 + 9 * 24),
    ("2020-10-25 02:00+0200", "H", "2020-10-25 02:00+0100", 25 + 9 * 24),
    ("2020-10-25 02:00+0100", "H", "2020-10-25 03:00", 10 * 24),
    ("2020-10-25 03:00", "H", "2020-10-25 04:00", 10 * 24),
    ("2020-10-25 00:00", "D", "2020-10-26 00:00", None),
    ("2020-10-25 01:00", "D", "2020-10-26 01:00", None),
    ("2020-10-25 03:00", "D", "2020-10-26 03:00", None),
    ("2020-10-25 06:00", "D", "2020-10-26 06:00", None),
    ("2020-10-01 00:00", "MS", "2020-11-01 00:00", None),
    ("2020-10-01 06:00", "MS", "2020-11-01 06:00", None),
    ("2020-10-01 00:00", "QS", "2021-01-01 00:00", None),
    ("2020-10-01 06:00", "QS", "2021-01-01 06:00", None),
]

TESTCASES_15T_STAMPONLY = [  # ts, expected_ts_right
    # Start of DST.
    ("2020-03-29 00:45", "2020-03-29 01:00"),
    ("2020-03-29 01:45", "2020-03-29 03:00"),
    ("2020-03-29 03:45", "2020-03-29 04:00"),
    # End of DST.
    ("2020-10-25 01:45", "2020-10-25 02:00+0200"),
    ("2020-10-25 02:45+0200", "2020-10-25 02:00+0100"),
    ("2020-10-25 02:45+0100", "2020-10-25 03:00+0100"),
    ("2020-10-25 03:45", "2020-10-25 04:00"),
]


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(("ts", "freq", "expected_ts_right"), TESTCASES)
def test_right_index(ts: str, freq: str, tz: str, expected_ts_right: str):
    """Test if right timestamp is correctly calculated for index without dst-transition."""
    do_test_index(ts, freq, tz, expected_ts_right, 96)  # always integer # of days


@pytest.mark.parametrize(("ts", "freq", "expected_ts_right", "periods"), TESTCASES_DST)
def test_right_index_dst(ts: str, freq: str, expected_ts_right: str, periods: int):
    """Test if right timestamp is correctly calculated for index with dst-transition."""
    periods = periods or 10
    do_test_index(ts, freq, "Europe/Berlin", expected_ts_right, periods)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(("ts", "freq", "expected_ts_right"), TESTCASES)
def test_right_stamp(ts: str, freq: str, tz: str, expected_ts_right: str):
    """Test if right timestamp is correctly calculated for timestamp without dst-transition."""
    do_test_stamp(ts, freq, tz, expected_ts_right)


@pytest.mark.parametrize(("ts", "freq", "expected_ts_right", "periods"), TESTCASES_DST)
def test_right_stamp_dst(ts: str, freq: str, expected_ts_right: str, periods: int):
    """Test if right timestamp is correctly calculated for timestamp with dst-transition."""
    do_test_stamp(ts, freq, "Europe/Berlin", expected_ts_right)


@pytest.mark.parametrize(("ts", "expected_ts_right"), TESTCASES_15T_STAMPONLY)
def test_right_stamp_15T(ts: str, expected_ts_right: str):
    """Test if right timestamp is correctly calculated for timestamp at non-round hour."""
    do_test_stamp(ts, "15T", "Europe/Berlin", expected_ts_right)


def do_test_index(ts, freq, tz, expected_ts_right, periods):
    # Must specify timezone in timestamp (instead of index) to avoid confusion at DST-changeover.
    i = pd.date_range(pd.Timestamp(ts, tz=tz), freq=freq, periods=periods)
    result = tools.right.index(i)
    expected = pd.date_range(
        pd.Timestamp(expected_ts_right, tz=tz), freq=freq, periods=periods, name="right"
    )
    testing.assert_index_equal(result, expected)
    tools.standardize.assert_index_standardized(result, __right=True)


def do_test_stamp(ts, freq, tz, expected_ts_right):
    ts = pd.Timestamp(ts, tz=tz)
    result = tools.right.stamp(ts, freq)
    expected = pd.Timestamp(expected_ts_right, tz=tz)
    assert result == expected
