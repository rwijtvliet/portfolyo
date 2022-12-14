import datetime as dt

import pandas as pd
import pytest

from portfolyo import tools
from portfolyo.prices import utils

# TODO: where are the hedge and conversion tests?? --> check git history

TESTCASES_ISPEAK = [  # ts, ispeak
    ("2020-01-01 01:00", False),
    ("2020-01-01 07:00", False),
    ("2020-01-01 08:00", True),
    ("2020-01-01 19:00", True),
    ("2020-01-01 20:00", False),
    ("2020-01-03 07:59", False),
    ("2020-01-03 08:00", True),
    ("2020-01-03 19:59", True),
    ("2020-01-03 20:00", False),
    ("2020-01-04 07:59", False),
    ("2020-01-04 08:00", False),
    ("2020-01-04 19:59", False),
    ("2020-01-04 20:00", False),
    ("2020-01-05 07:59", False),
    ("2020-01-05 08:00", False),
    ("2020-01-05 19:59", False),
    ("2020-01-05 20:00", False),
    ("2020-03-29 01:00", False),
    ("2020-03-29 03:00", False),
    ("2020-10-25 01:00", False),
    ("2020-10-25 03:00", False),
]


@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize(("ts", "ispeak"), TESTCASES_ISPEAK)
def test_is_peak_hour(ts, tz, ispeak):
    """Test if individual timestamps are correctly identified as laying inside the peak
    hour periods."""
    ts = pd.Timestamp(ts, tz=tz)
    assert utils.is_peak_hour(ts) == ispeak


@pytest.mark.parametrize(("tz", "mar_b_corr"), [(None, 0), ("Europe/Berlin", -1)])
@pytest.mark.parametrize("month", [1, 2, 3])
@pytest.mark.parametrize("freq", ["D", "MS", "QS", "AS"])
@pytest.mark.parametrize(
    ("year", "bp", "jan_1_weekday", "jan_p", "feb_p", "mar_p"),
    [
        (2020, (366, 262), True, 23, 20, 22),
        (2021, (365, 261), True, 21, 20, 23),
        (2022, (365, 260), False, 21, 20, 23),
    ],
)
def test_duration(
    year, bp, jan_1_weekday, jan_p, feb_p, mar_p, tz, mar_b_corr, freq, month
):
    """Test if the correct number of base, peak and offpeak hours are calculated.
    bp = tuple with number of base days and days with peak hours; jan_p, feb_p, mar_p =
    number of days with peak hours in each month; mar_b_corr = correction to number of
    base HOURS."""

    if tz is None:
        return  # TODO: make sure the duration_base, duration_peak etc functions accept timezone-agnostic data

    if month > 1 and freq != "MS":
        return

    start = pd.Timestamp(f"{year}-{month}", tz=tz)

    # Expected values.
    if freq == "AS":
        b = 24 * bp[0]
        p = 12 * bp[1]
    elif freq == "QS":
        b = 24 * (31 + 28 + 31 + (bp[0] - 365)) + mar_b_corr
        p = 12 * (jan_p + feb_p + mar_p)
    elif freq == "MS":
        if month == 1:
            b = 24 * 31
            p = 12 * jan_p
        elif month == 2:
            b = 24 * (28 + (bp[0] - 365))
            p = 12 * feb_p
        else:  # month == 3:
            b = 24 * 31 + mar_b_corr
            p = 12 * mar_p
    else:  # freq == 'D'
        b = 24
        p = 12 * jan_1_weekday
    b = tools.unit.Q_(b, "hours")
    p = tools.unit.Q_(p, "hours")
    o = b - p

    # Test values.
    assert b == utils.duration_base(start, freq)
    assert p == utils.duration_peak(start, freq)
    assert o == utils.duration_offpeak(start, freq)


@pytest.mark.parametrize(
    ("ts_trade_date", "period_type", "period_start", "expected_left"),
    [
        ("2020-1-1", "d", 0, "2020-1-1"),
        ("2020-1-1", "m", 0, "2020-1-1"),
        ("2020-1-1", "q", 0, "2020-1-1"),
        ("2020-1-1", "s", 0, "2019-10-1"),
        ("2020-1-1", "a", 0, "2020-1-1"),
        ("2020-1-1", "d", 1, "2020-1-2"),
        ("2020-1-1", "m", 1, "2020-2-1"),
        ("2020-1-1", "q", 1, "2020-4-1"),
        ("2020-1-1", "s", 1, "2020-4-1"),
        ("2020-1-1", "a", 1, "2021-1-1"),
        ("2020-1-1", "d", 3, "2020-1-4"),
        ("2020-1-1", "m", 3, "2020-4-1"),
        ("2020-1-1", "q", 3, "2020-10-1"),
        ("2020-1-1", "s", 3, "2021-4-1"),
        ("2020-1-1", "a", 3, "2023-1-1"),
        ("2020-1-31", "d", 0, "2020-1-31"),
        ("2020-1-31", "m", 0, "2020-1-1"),
        ("2020-1-31", "q", 0, "2020-1-1"),
        ("2020-1-31", "s", 0, "2019-10-1"),
        ("2020-1-31", "a", 0, "2020-1-1"),
        ("2020-1-31", "d", 1, "2020-2-1"),
        ("2020-1-31", "m", 1, "2020-2-1"),
        ("2020-1-31", "q", 1, "2020-4-1"),
        ("2020-1-31", "s", 1, "2020-4-1"),
        ("2020-1-31", "a", 1, "2021-1-1"),
        ("2020-1-31", "d", 3, "2020-2-3"),
        ("2020-1-31", "m", 3, "2020-4-1"),
        ("2020-1-31", "q", 3, "2020-10-1"),
        ("2020-1-31", "s", 3, "2021-4-1"),
        ("2020-1-31", "a", 3, "2023-1-1"),
        ("2020-2-14", "s", 0, "2019-10-1"),
        ("2020-3-14", "s", 0, "2019-10-1"),
        ("2020-4-14", "s", 0, "2020-4-1"),
        ("2020-5-14", "s", 0, "2020-4-1"),
        ("2020-6-14", "s", 0, "2020-4-1"),
        ("2020-7-14", "s", 0, "2020-4-1"),
        ("2020-8-14", "s", 0, "2020-4-1"),
        ("2020-9-14", "s", 0, "2020-4-1"),
        ("2020-10-14", "s", 0, "2020-10-1"),
        ("2020-11-14", "s", 0, "2020-10-1"),
        ("2020-12-14", "s", 0, "2020-10-1"),
        ("2020-2-14", "s", 1, "2020-4-1"),
        ("2020-3-14", "s", 1, "2020-4-1"),
        ("2020-4-14", "s", 1, "2020-10-1"),
        ("2020-5-14", "s", 1, "2020-10-1"),
        ("2020-6-14", "s", 1, "2020-10-1"),
        ("2020-7-14", "s", 1, "2020-10-1"),
        ("2020-8-14", "s", 1, "2020-10-1"),
        ("2020-9-14", "s", 1, "2020-10-1"),
        ("2020-10-14", "s", 1, "2021-4-1"),
        ("2020-11-14", "s", 1, "2021-4-1"),
        ("2020-12-14", "s", 1, "2021-4-1"),
        ("2020-2-14", "s", 3, "2021-4-1"),
        ("2020-3-14", "s", 3, "2021-4-1"),
        ("2020-4-14", "s", 3, "2021-10-1"),
        ("2020-5-14", "s", 3, "2021-10-1"),
        ("2020-6-14", "s", 3, "2021-10-1"),
        ("2020-7-14", "s", 3, "2021-10-1"),
        ("2020-8-14", "s", 3, "2021-10-1"),
        ("2020-9-14", "s", 3, "2021-10-1"),
        ("2020-10-14", "s", 3, "2022-4-1"),
        ("2020-11-14", "s", 3, "2022-4-1"),
        ("2020-12-14", "s", 3, "2022-4-1"),
    ],
)
@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
def test_deliveryperiod(
    ts_trade_date: str,
    period_type: str,
    period_start: str,
    tz: str,
    expected_left: str,
    starttime: str,
):
    ts_trade = pd.Timestamp(f"{ts_trade_date} 12:00", tz=tz)
    if starttime == "06:00":
        start_of_day = dt.time(hour=6)
    else:
        start_of_day = dt.time(hour=0)
    expected_left = pd.Timestamp(f"{expected_left} {starttime}", tz=tz)
    ts_deliv = utils.delivery_period(ts_trade, period_type, period_start, start_of_day)
    assert ts_deliv[0] == expected_left
    try:
        add = {"m": 1, "q": 3, "s": 6, "a": 12}[period_type]
        assert ts_deliv[1] == expected_left + pd.offsets.MonthBegin(add)
    except KeyError:
        assert ts_deliv[1] == expected_left + dt.timedelta(1)
