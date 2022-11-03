import pandas as pd
import pytest
from portfolyo import testing, tools

TESTCASES = [
    (pd.Timestamp("2020-01-01"), "D", (24, 24)),
    (pd.Timestamp("2020-01-01"), "MS", (744, 696)),
    (pd.Timestamp("2020-01-01"), "QS", (2184, 2184)),
    (pd.Timestamp("2020-01-01"), "AS", (8784, 8760)),
    (pd.Timestamp("2020-03-01"), "D", (24, 24)),
    (pd.Timestamp("2020-03-01"), "MS", (744, 720)),
    (pd.Timestamp("2020-03-29"), "D", (24, 24)),
    (pd.Timestamp("2020-01-01", tz="Europe/Berlin"), "D", (24, 24)),
    (pd.Timestamp("2020-01-01", tz="Europe/Berlin"), "MS", (744, 696)),
    (pd.Timestamp("2020-01-01", tz="Europe/Berlin"), "QS", (2183, 2184)),
    (pd.Timestamp("2020-01-01", tz="Europe/Berlin"), "AS", (8784, 8760)),
    (pd.Timestamp("2020-03-01", tz="Europe/Berlin"), "D", (24, 24)),
    (pd.Timestamp("2020-03-01", tz="Europe/Berlin"), "MS", (743, 720)),
    (pd.Timestamp("2020-03-29", tz="Europe/Berlin"), "D", (23, 24)),
]


@pytest.mark.parametrize(("ts", "freq", "hours"), TESTCASES)
def test_duration_stamp(ts, freq, hours):
    """Test if duration in correctly calculated."""
    assert tools.duration(ts, freq) == tools.unit.Q_(hours[0], "h")


@pytest.mark.parametrize(("ts", "freq", "hours"), TESTCASES)
@pytest.mark.parametrize("freq_as", ["attr", "param"])
def test_duration_index(ts, freq, hours, freq_as):
    """Test if duration in correctly calculated."""
    i = pd.date_range(ts, freq=freq, periods=2)
    expected = pd.Series(hours, i)
    if freq_as == "attr":
        result = tools.duration.index(i)
    else:
        i.freq = None
        result = tools.duration.index(i, freq)
    testing.assert_series_equal(result, expected)
