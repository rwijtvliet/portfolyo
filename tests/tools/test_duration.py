import pandas as pd
import pytest

from portfolyo import testing, tools

TESTCASES = [  # ts, freq, expected_hours
    # First day of X and start of day.
    ("2020-01-01", "15T", (0.25, 0.25)),
    ("2020-01-01", "H", (1, 1)),
    ("2020-01-01", "D", (24, 24)),
    ("2020-01-01", "MS", (744, 696)),
    ("2020-01-01", "AS", (8784, 8760)),
    # First day of X but not start of day.
    ("2020-01-01 06:00", "15T", (0.25, 0.25)),
    ("2020-01-01 06:00", "H", (1, 1)),
    ("2020-01-01 06:00", "D", (24, 24)),
    ("2020-01-01 06:00", "MS", (744, 696)),
    ("2020-01-01 06:00", "AS", (8784, 8760)),
    # Not first day of X but start of day.
    ("2020-04-01", "15T", (0.25, 0.25)),
    ("2020-04-01", "H", (1, 1)),
    ("2020-04-01", "D", (24, 24)),
    ("2020-04-01", "MS", (720, 744)),
    ("2020-04-01", "QS", (2184, 2208)),
    ("2020-04-21", "15T", (0.25, 0.25)),
    ("2020-04-21", "H", (1, 1)),
    ("2020-04-21", "D", (24, 24)),
    # Not first day of X and not start of day.
    ("2020-04-01 06:00", "15T", (0.25, 0.25)),
    ("2020-04-01 06:00", "H", (1, 1)),
    ("2020-04-01 06:00", "D", (24, 24)),
    ("2020-04-01 06:00", "MS", (720, 744)),
    ("2020-04-01 06:00", "QS", (2184, 2208)),
    ("2020-04-21 06:00", "15T", (0.25, 0.25)),
    ("2020-04-21 06:00", "H", (1, 1)),
    ("2020-04-21 06:00", "D", (24, 24)),
]

TESTCASES_DST = [  # ts, freq, expected_hours
    # Start of DST.
    ("2020-03-29 01:00", "15T", (0.25, 0.25)),
    ("2020-03-29 01:00", "H", (1, 1)),
    ("2020-03-29 00:00", "D", (23, 24)),
    ("2020-03-29 01:00", "D", (23, 24)),
    ("2020-03-29 03:00", "D", (24, 24)),
    ("2020-03-29 06:00", "D", (24, 24)),
    ("2020-03-28 01:00", "D", (24, 23)),
    ("2020-03-28 03:00", "D", (23, 24)),
    ("2020-03-28 06:00", "D", (23, 24)),
    ("2020-03-01 00:00", "MS", (743, 720)),
    ("2020-03-01 06:00", "MS", (743, 720)),
    ("2020-01-01 00:00", "QS", (2183, 2184)),
    ("2020-01-01 06:00", "QS", (2183, 2184)),
    # End of DST.
    ("2020-10-25 01:00", "15T", (0.25, 0.25)),
    ("2020-10-25 02:00+0200", "15T", (0.25, 0.25)),
    ("2020-10-25 02:00+0100", "15T", (0.25, 0.25)),
    ("2020-10-25 03:00", "15T", (0.25, 0.25)),
    ("2020-10-25 01:00", "H", (1, 1)),
    ("2020-10-25 02:00+0200", "H", (1, 1)),
    ("2020-10-25 02:00+0100", "H", (1, 1)),
    ("2020-10-25 03:00", "H", (1, 1)),
    ("2020-10-25 00:00", "D", (25, 24)),
    ("2020-10-25 01:00", "D", (25, 24)),
    ("2020-10-25 03:00", "D", (24, 24)),
    ("2020-10-25 06:00", "D", (24, 24)),
    ("2020-10-24 01:00", "D", (24, 25)),
    ("2020-10-24 03:00", "D", (25, 24)),
    ("2020-10-24 06:00", "D", (25, 24)),
    ("2020-10-01 00:00", "MS", (745, 720)),
    ("2020-10-01 06:00", "MS", (745, 720)),
    ("2020-10-01 00:00", "QS", (2209, 2159)),
    ("2020-10-01 06:00", "QS", (2209, 2159)),
]


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(("ts", "freq", "expected_hours"), TESTCASES)
def test_duration_index(ts: str, freq: str, tz: str, expected_hours: str):
    """Test if duration is correctly calculated for index."""
    do_test_index(ts, freq, tz, expected_hours)


@pytest.mark.parametrize(("ts", "freq", "expected_hours"), TESTCASES_DST)
def test_duration_index_dst(ts: str, freq: str, expected_hours: str):
    """Test if duration is correctly calculated for index with dst-transition."""
    do_test_index(ts, freq, "Europe/Berlin", expected_hours)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(("ts", "freq", "expected_hours"), TESTCASES)
def test_duration_stamp(ts, freq, expected_hours, tz):
    """Test if duration in correctly calculated for timestamps."""
    do_test_stamp(ts, freq, tz, expected_hours)


@pytest.mark.parametrize(("ts", "freq", "expected_hours"), TESTCASES)
def test_duration_stamp_dst(ts, freq, expected_hours):
    """Test if duration in correctly calculated for timestamps with dst-transition."""
    do_test_stamp(ts, freq, "Europe/Berlin", expected_hours)


def do_test_index(ts, freq, tz, expected_hours):
    # Must specify timezone in timestamp (instead of index) to avoid confusion at DST-changeover.
    i = pd.date_range(pd.Timestamp(ts, tz=tz), freq=freq, periods=2)
    result = tools.duration.index(i)
    expected = pd.Series(expected_hours, i, dtype="pint[h]", name="duration")
    testing.assert_series_equal(result, expected)


def do_test_stamp(ts, freq, tz, expected_hours):
    ts = pd.Timestamp(ts, tz=tz)
    result = tools.duration.stamp(ts, freq)
    expected = tools.unit.Q_(expected_hours[0], "h")
    assert result == expected
