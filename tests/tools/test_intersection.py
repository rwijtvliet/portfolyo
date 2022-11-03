import pandas as pd
import pytest
from portfolyo import testing, tools


@pytest.mark.parametrize(
    ("idxs", "expected"),
    [
        # Days, with and without timezone.
        (
            [
                pd.date_range("2020", freq="D", periods=31),
                pd.date_range("2020-01-20", freq="D", periods=40),
            ],
            pd.date_range("2020-01-20", freq="D", periods=12),
        ),
        (
            [
                pd.date_range("2020", freq="D", periods=31, tz="Europe/Berlin"),
                pd.date_range("2020-01-20", freq="D", periods=40, tz="Europe/Berlin"),
            ],
            pd.date_range("2020-01-20", freq="D", periods=12, tz="Europe/Berlin"),
        ),
        # Error: incompatible timezones.
        (
            [
                pd.date_range("2020", freq="D", periods=31),
                pd.date_range("2020-01-20", freq="D", periods=40, tz="Europe/Berlin"),
            ],
            None,
        ),
        # Error: distinct frequencies.
        (
            [
                pd.date_range("2020", freq="H", periods=31),
                pd.date_range("2020-01-20", freq="D", periods=40),
            ],
            None,
        ),
        # No overlap.
        (
            [
                pd.date_range("2020", freq="H", periods=24),
                pd.date_range("2020-01-20", freq="H", periods=72),
            ],
            pd.date_range("2020", freq="H", periods=0),
        ),
        # Months, with and without timezone.
        (
            [
                pd.date_range("2020", freq="MS", periods=31),
                pd.date_range("2020-05-01", freq="MS", periods=40),
            ],
            pd.date_range("2020-05-01", freq="MS", periods=27),
        ),
        (
            [
                pd.date_range("2020", freq="MS", periods=31, tz="Europe/Berlin"),
                pd.date_range("2020-05-01", freq="MS", periods=40, tz="Europe/Berlin"),
            ],
            pd.date_range("2020-05-01", freq="MS", periods=27, tz="Europe/Berlin"),
        ),
        # Test if names retained.
        (
            [
                pd.date_range("2020", freq="MS", periods=31, name="ts_left"),
                pd.date_range("2020-05-01", freq="MS", periods=40),
            ],
            pd.date_range("2020-05-01", freq="MS", periods=27, name="ts_left"),
        ),
        # DST.
        (
            [
                pd.date_range("2020-03-28", freq="H", periods=71, tz="Europe/Berlin"),
                pd.date_range("2020-03-29", freq="H", periods=71, tz="Europe/Berlin"),
            ],
            pd.date_range("2020-03-29", freq="H", periods=47, tz="Europe/Berlin"),
        ),
        (
            [
                pd.date_range("2020-03-28", freq="H", periods=71, tz="Europe/Berlin"),
                pd.date_range("2020-03-30", freq="H", periods=72, tz="Europe/Berlin"),
            ],
            pd.date_range("2020-03-30", freq="H", periods=24, tz="Europe/Berlin"),
        ),
        (
            [
                pd.date_range("2020-03-28", freq="H", periods=72),
                pd.date_range("2020-03-29", freq="H", periods=72),
            ],
            pd.date_range("2020-03-29", freq="H", periods=48),
        ),
        (
            [
                pd.date_range("2020-03-28", freq="H", periods=72),
                pd.date_range("2020-03-30", freq="H", periods=72),
            ],
            pd.date_range("2020-03-30", freq="H", periods=24),
        ),
        (
            [
                pd.date_range("2020-10-24", freq="H", periods=73, tz="Europe/Berlin"),
                pd.date_range("2020-10-25", freq="H", periods=73, tz="Europe/Berlin"),
            ],
            pd.date_range("2020-10-25", freq="H", periods=49, tz="Europe/Berlin"),
        ),
        (
            [
                pd.date_range("2020-10-24", freq="H", periods=73, tz="Europe/Berlin"),
                pd.date_range("2020-10-26", freq="H", periods=72, tz="Europe/Berlin"),
            ],
            pd.date_range("2020-10-26", freq="H", periods=24, tz="Europe/Berlin"),
        ),
        (
            [
                pd.date_range("2020-10-24", freq="H", periods=72),
                pd.date_range("2020-10-25", freq="H", periods=72),
            ],
            pd.date_range("2020-10-25", freq="H", periods=48),
        ),
        (
            [
                pd.date_range("2020-10-24", freq="H", periods=72),
                pd.date_range("2020-10-26", freq="H", periods=72),
            ],
            pd.date_range("2020-10-26", freq="H", periods=24),
        ),
        # Distinct timezones.
        (
            [
                pd.date_range("2020-01-01", freq="15T", periods=96, tz="Europe/Berlin"),
                pd.date_range("2020-01-01", freq="15T", periods=96, tz="Asia/Kolkata"),
            ],
            pd.date_range("2020-01-01", freq="15T", periods=78, tz="Europe/Berlin"),
        ),
        (
            [
                pd.date_range("2020-01-01", freq="15T", periods=96, tz="Asia/Kolkata"),
                pd.date_range("2020-01-01", freq="15T", periods=96, tz="Europe/Berlin"),
            ],
            pd.date_range(
                "2020-01-01 04:30", freq="15T", periods=78, tz="Asia/Kolkata"
            ),
        ),
        (
            [
                pd.date_range("2020-01-01", freq="H", periods=24, tz="Asia/Kolkata"),
                pd.date_range("2020-01-01", freq="H", periods=24, tz="Europe/Berlin"),
            ],
            pd.date_range("2020-01-01", freq="H", periods=0, tz="Asia/Kolkata"),
        ),
    ],
)
def test_intersection(idxs, expected):
    """Test if intersection works correctly."""
    if expected is None:
        with pytest.raises(ValueError):
            _ = tools.intersection.index(*idxs)
    else:
        result = tools.intersection.index(*idxs)
        testing.assert_index_equal(result, expected)
