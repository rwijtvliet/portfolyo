import datetime as dt
from typing import Any

import pandas as pd
import pytest

from portfolyo.tools import stamp

ts_midnight = "2020-02-03"
ts_withtime = "2020-02-03 15:20"


@pytest.mark.parametrize(
    "ts",
    [
        pd.Timestamp(ts_withtime, tz=None),
        pd.Timestamp(ts_withtime, tz="Europe/Berlin"),
        pd.Timestamp(ts_midnight, tz=None),
        pd.Timestamp(ts_midnight, tz="Europe/Berlin"),
    ],
)
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("start_of_day", [None, dt.time(hour=6), dt.time(hour=0)])
def test_create_nochange(ts: Any, tz: str, start_of_day: dt.time):
    """Test if stamps that should remain unchanged, remain unchanged."""
    assert stamp.create(ts, tz, start_of_day) == ts


@pytest.mark.parametrize(
    "ts,tz,start_of_day,expected",
    [
        (ts_midnight, None, None, pd.Timestamp(ts_midnight, tz=None)),
        (
            ts_withtime,
            None,
            None,
            pd.Timestamp(ts_withtime, tz=None),
        ),
        (
            ts_midnight,
            None,
            dt.time(hour=6),
            pd.Timestamp("2020-02-03 06:00", tz=None),
        ),
        (
            ts_withtime,
            None,
            dt.time(hour=6),
            pd.Timestamp(ts_withtime, tz=None),
        ),
        (
            ts_midnight,
            "Europe/Berlin",
            None,
            pd.Timestamp(ts_midnight, tz="Europe/Berlin"),
        ),
        (
            ts_withtime,
            "Europe/Berlin",
            None,
            pd.Timestamp(ts_withtime, tz="Europe/Berlin"),
        ),
        (
            ts_midnight,
            "Europe/Berlin",
            dt.time(hour=6),
            pd.Timestamp("2020-02-03 06:00", tz="Europe/Berlin"),
        ),
        (
            ts_withtime,
            "Europe/Berlin",
            dt.time(hour=6),
            pd.Timestamp(ts_withtime, tz="Europe/Berlin"),
        ),
    ],
)
@pytest.mark.parametrize("instance", ["string", "datetime"])
def test_create_string(
    ts: Any, tz: str, start_of_day: dt.time, instance: str, expected: pd.Timestamp
):
    """Test if timestamp is correctly created from a string."""
    if instance == "datetime":
        ts = dt.datetime.fromisoformat(ts)
    assert stamp.create(ts, tz, start_of_day) == expected
