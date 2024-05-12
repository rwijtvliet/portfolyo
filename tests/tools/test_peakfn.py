import datetime as dt
from typing import Iterable

import pandas as pd
import pytest

from portfolyo import tools

# 'count' is the number of timestamps in the index that should be in peak period.
# 'stretch' are the index positions of the first and the last timestamp of a continuous
#    part of peak period within the index.


def index(start: str, end: str, freq: str, tz: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq=freq, inclusive="left", tz=tz)


f_germanpower = tools.peakfn.factory(dt.time(hour=8), dt.time(hour=20))
TESTCASES_GERMANPOWER = [  # end, freq, count, stretch
    ("2020-01-08", "15T", 5 * 12 * 4, (32, 79)),
    ("2020-01-08", "H", 5 * 12, (8, 19)),
    ("2020-04-01", "15T", 65 * 12 * 4, (-64, -17)),
    ("2020-04-01", "H", 65 * 12, (-16, -5)),
    ("2021", "15T", 262 * 12 * 4, (-64, -17)),
    ("2021", "H", 262 * 12, (-16, -5)),
    ("2021", "D", ValueError, None),
    ("2021", "MS", ValueError, None),
    ("2021", "QS", ValueError, None),
    ("2021", "AS", ValueError, None),
]

f_everyday_13half = tools.peakfn.factory(
    dt.time(hour=8), dt.time(hour=21, minute=30), [1, 2, 3, 4, 5, 6, 7]
)
TESTCASES_13HALF = [  # end, freq, count, stretch
    ("2020-01-08", "15T", 7 * 13.5 * 4, (32, 85)),
    ("2020-04-01", "15T", 91 * 13.5 * 4, (-64, -11)),
    ("2021", "15T", 366 * 13.5 * 4, (-64, -11)),
    ("2021", "H", ValueError, None),
    ("2021", "D", ValueError, None),
    ("2021", "MS", ValueError, None),
    ("2021", "QS", ValueError, None),
    ("2021", "AS", ValueError, None),
]

f_workingdays_full = tools.peakfn.factory(None, None, [1, 2, 3, 4, 5])
TESTCASES_WORKINGDAYS = [  # end, freq, count, stretch
    ("2020-01-08", "15T", 5 * 24 * 4, (0, 72 * 4 - 1)),
    ("2020-01-08", "H", 5 * 24, (0, 72 - 1)),
    ("2020-01-08", "D", 5, (0, 3 - 1)),
    ("2020-04-01", "15T", 65 * 24 * 4, (0, 72 * 4 - 1)),  # avoid DST transition
    ("2020-04-01", "H", 65 * 24, (0, 72 - 1)),  # avoid DST transition
    ("2020-04-01", "D", 65, (-9, -4 - 1)),
    ("2021", "15T", 262 * 24 * 4, (-11 * 24 * 4, -6 * 24 * 4 - 1)),
    ("2021", "H", 262 * 24, (-11 * 24, -6 * 24 - 1)),
    ("2021", "D", 262, (-11, -6 - 1)),
    ("2021", "MS", ValueError, None),
    ("2021", "QS", ValueError, None),
    ("2021", "AS", ValueError, None),
]

f_everyday_until6 = tools.peakfn.factory(None, dt.time(hour=6), [1, 2, 3, 4, 5, 6, 7])
TESTCASES_EVERYDAY6 = [  # month, freq, tz, count, stretch
    (1, "15T", None, 31 * 6 * 4, (24 * 4, 30 * 4 - 1)),
    (1, "15T", "Europe/Berlin", 31 * 6 * 4, (24 * 4, 30 * 4 - 1)),
    (1, "15T", "Asia/Kolkata", 31 * 6 * 4, (24 * 4, 30 * 4 - 1)),
    (1, "H", None, 31 * 6, (24, 30 - 1)),
    (1, "H", "Europe/Berlin", 31 * 6, (24, 30 - 1)),
    (1, "H", "Asia/Kolkata", 31 * 6, (24, 30 - 1)),
    (1, "D", None, ValueError, None),
    (1, "D", "Europe/Berlin", ValueError, None),
    (1, "D", "Asia/Kolkata", ValueError, None),
    (1, "MS", None, ValueError, None),
    (1, "MS", "Europe/Berlin", ValueError, None),
    (1, "MS", "Asia/Kolkata", ValueError, None),
    (1, "QS", None, ValueError, None),
    (1, "QS", "Europe/Berlin", ValueError, None),
    (1, "QS", "Asia/Kolkata", ValueError, None),
    (1, "AS", None, ValueError, None),
    (1, "AS", "Europe/Berlin", ValueError, None),
    (1, "AS", "Asia/Kolkata", ValueError, None),
    (3, "15T", None, 31 * 6 * 4, (-72 * 4, -66 * 4 - 1)),
    (3, "15T", "Europe/Berlin", (31 * 6 - 1) * 4, (-71 * 4, -66 * 4 - 1)),
    (3, "15T", "Asia/Kolkata", 31 * 6 * 4, (-72 * 4, -66 * 4 - 1)),
    (3, "H", None, 31 * 6, (-72, -66 - 1)),
    (3, "H", "Europe/Berlin", 31 * 6 - 1, (-71, -66 - 1)),  # dst start; one hour less
    (3, "H", "Asia/Kolkata", 31 * 6, (-72, -66 - 1)),
    (10, "15T", None, 31 * 6 * 4, (-168 * 4, -162 * 4 - 1)),
    (10, "15T", "Europe/Berlin", (31 * 6 + 1) * 4, (-169 * 4, -162 * 4 - 1)),
    (10, "15T", "Asia/Kolkata", 31 * 6 * 4, (-168 * 4, -162 * 4 - 1)),
    (10, "H", None, 31 * 6, (-168, -162 - 1)),
    (10, "H", "Europe/Berlin", 31 * 6 + 1, (-169, -162 - 1)),  # dst end; one hour more
    (10, "H", "Asia/Kolkata", 31 * 6, (-168, -162 - 1)),
]


@pytest.mark.parametrize(
    ("peak_left", "peak_right", "isoweekdays", "expected"),
    [
        (None, None, [1, 2, 3, 4, 5, 6, 7], ValueError),
        (dt.time(hour=0), dt.time(hour=0), [1, 2, 3, 4, 5, 6, 7], ValueError),
        (dt.time(hour=0), dt.time(hour=0), [1, 2, 3, 4, 5, 6], None),
        (dt.time(hour=1), None, [1, 2, 3, 4, 5, 6, 7], None),
    ],
)
def test_functioncreation(
    peak_left: dt.time, peak_right: dt.time, isoweekdays: Iterable[int], expected
):
    """Test if an error is raised when creating impossible functions."""
    if type(expected) is type and issubclass(expected, Exception):
        with pytest.raises(expected):
            tools.peakfn.factory(peak_left, peak_right, isoweekdays)
    else:
        tools.peakfn.factory(peak_left, peak_right, isoweekdays)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(("end", "freq", "count", "stretch"), TESTCASES_GERMANPOWER)
def test_peakfn_germanpower(
    end: str, freq: str, tz: str, count: int, stretch: Iterable[int]
):
    """Test if the peak periods are correctly calculated."""
    # Not influenced by timezone, because dst change always during offpeak-hours.
    i = index("2020", end, freq, tz)
    do_test(i, f_germanpower, count, stretch)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(("end", "freq", "count", "stretch"), TESTCASES_13HALF)
def test_peakfn_everyday13half(
    end: str, freq: str, tz: str, count: int, stretch: Iterable[int]
):
    """Test if the peak periods are correctly calculated."""
    # Not influenced by timezone, because dst change always during offpeak-hours.
    i = index("2020", end, freq, tz)
    do_test(i, f_everyday_13half, count, stretch)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(("end", "freq", "count", "stretch"), TESTCASES_WORKINGDAYS)
def test_peakfn_workingdaysfull(
    end: str, freq: str, tz: str, count: int, stretch: Iterable[int]
):
    """Test if the peak periods are correctly calculated."""
    # Not influenced by timezone, because dst change always during offpeak-hours.
    i = index("2020", end, freq, tz)
    do_test(i, f_workingdays_full, count, stretch)


@pytest.mark.parametrize(
    ("month", "freq", "tz", "count", "stretch"), TESTCASES_EVERYDAY6
)
def test_peakfn_everydayuntil6(
    month: int, tz: str, freq: str, count: int, stretch: Iterable[int]
):
    """Test if the peak periods are correctly calculated."""
    i = index(f"2020-{month}", f"2020-{month+1}", freq, tz)
    do_test(i, f_everyday_until6, count, stretch)


def do_test(
    i: pd.DatetimeIndex,
    f: tools.peakfn.PeakFunction,
    count: int,
    stretch: Iterable[int] = None,
):
    # Error case.
    if type(count) is type and issubclass(count, Exception):
        with pytest.raises(count):
            f(i)
        return

    # Non-error-case.
    peak = f(i)
    assert peak.sum() == count
    if stretch:
        first, last = stretch
        assert all(peak.iloc[first:last])
        if first != 0:
            assert not peak.iloc[first - 1]
        if last != -1:
            assert not peak.iloc[last + 1]


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
    i = pd.date_range(start, freq=freq, periods=1)

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
    expected_b = pd.Series([float(b)], i, dtype="pint[h]", name="duration")
    expected_p = pd.Series([float(p)], i, dtype="pint[h]", name="duration")
    expected_o = expected_b - expected_p

    result_b = tools.peakfn.base_duration(i)
    result_p = tools.peakfn.peak_duration(i, f_germanpower)
    result_o = tools.peakfn.offpeak_duration(i, f_germanpower)

    # Test values.
    tools.testing.assert_series_equal(expected_b, result_b)
    tools.testing.assert_series_equal(expected_p, result_p)
    tools.testing.assert_series_equal(expected_o, result_o)
