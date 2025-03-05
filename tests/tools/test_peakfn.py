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
    ("2020-01-08", "15min", 5 * 12 * 4, (32, 79)),
    ("2020-01-08", "h", 5 * 12, (8, 19)),
    ("2020-04-01", "15min", 65 * 12 * 4, (-64, -17)),
    ("2020-04-01", "h", 65 * 12, (-16, -5)),
    ("2021", "15min", 262 * 12 * 4, (-64, -17)),
    ("2021", "h", 262 * 12, (-16, -5)),
    ("2021", "D", ValueError, None),
    ("2021", "MS", ValueError, None),
    ("2021", "QS", ValueError, None),
    ("2021", "YS", ValueError, None),
    ("2021", "YS-APR", ValueError, None),
]

f_everyday_13half = tools.peakfn.factory(
    dt.time(hour=8), dt.time(hour=21, minute=30), [1, 2, 3, 4, 5, 6, 7]
)
TESTCASES_13HALF = [  # end, freq, count, stretch
    ("2020-01-08", "15min", 7 * 13.5 * 4, (32, 85)),
    ("2020-04-01", "15min", 91 * 13.5 * 4, (-64, -11)),
    ("2021", "15min", 366 * 13.5 * 4, (-64, -11)),
    ("2021", "h", ValueError, None),
    ("2021", "D", ValueError, None),
    ("2021", "MS", ValueError, None),
    ("2021", "QS", ValueError, None),
    ("2021", "YS", ValueError, None),
    ("2021", "YS-APR", ValueError, None),
]

f_workingdays_full = tools.peakfn.factory(None, None, [1, 2, 3, 4, 5])
TESTCASES_WORKINGDAYS = [  # end, freq, count, stretch
    ("2020-01-08", "15min", 5 * 24 * 4, (0, 72 * 4 - 1)),
    ("2020-01-08", "h", 5 * 24, (0, 72 - 1)),
    ("2020-01-08", "D", 5, (0, 3 - 1)),
    ("2020-04-01", "15min", 65 * 24 * 4, (0, 72 * 4 - 1)),  # avoid DST transition
    ("2020-04-01", "h", 65 * 24, (0, 72 - 1)),  # avoid DST transition
    ("2020-04-01", "D", 65, (-9, -4 - 1)),
    ("2021", "15min", 262 * 24 * 4, (-11 * 24 * 4, -6 * 24 * 4 - 1)),
    ("2021", "h", 262 * 24, (-11 * 24, -6 * 24 - 1)),
    ("2021", "D", 262, (-11, -6 - 1)),
    ("2021", "MS", ValueError, None),
    ("2021", "QS", ValueError, None),
    ("2021", "YS", ValueError, None),
    ("2021", "YS-APR", ValueError, None),
]

f_everyday_until6 = tools.peakfn.factory(None, dt.time(hour=6), [1, 2, 3, 4, 5, 6, 7])
TESTCASES_EVERYDAY6 = [  # month, freq, tz, count, stretch
    (1, "15min", None, 31 * 6 * 4, (24 * 4, 30 * 4 - 1)),
    (1, "15min", "Europe/Berlin", 31 * 6 * 4, (24 * 4, 30 * 4 - 1)),
    (1, "15min", "Asia/Kolkata", 31 * 6 * 4, (24 * 4, 30 * 4 - 1)),
    (1, "h", None, 31 * 6, (24, 30 - 1)),
    (1, "h", "Europe/Berlin", 31 * 6, (24, 30 - 1)),
    (1, "h", "Asia/Kolkata", 31 * 6, (24, 30 - 1)),
    (1, "D", None, ValueError, None),
    (1, "D", "Europe/Berlin", ValueError, None),
    (1, "D", "Asia/Kolkata", ValueError, None),
    (1, "MS", None, ValueError, None),
    (1, "MS", "Europe/Berlin", ValueError, None),
    (1, "MS", "Asia/Kolkata", ValueError, None),
    (1, "QS", None, ValueError, None),
    (1, "QS", "Europe/Berlin", ValueError, None),
    (1, "QS", "Asia/Kolkata", ValueError, None),
    (1, "YS", None, ValueError, None),
    (1, "YS", "Europe/Berlin", ValueError, None),
    (1, "YS", "Asia/Kolkata", ValueError, None),
    (4, "YS-APR", "Asia/Kolkata", ValueError, None),
    (4, "YS-APR", "Europe/Berlin", ValueError, None),
    (4, "YS-APR", None, ValueError, None),
    (3, "15min", None, 31 * 6 * 4, (-72 * 4, -66 * 4 - 1)),
    (3, "15min", "Europe/Berlin", (31 * 6 - 1) * 4, (-71 * 4, -66 * 4 - 1)),
    (3, "15min", "Asia/Kolkata", 31 * 6 * 4, (-72 * 4, -66 * 4 - 1)),
    (3, "h", None, 31 * 6, (-72, -66 - 1)),
    (3, "h", "Europe/Berlin", 31 * 6 - 1, (-71, -66 - 1)),  # dst start; one hour less
    (3, "h", "Asia/Kolkata", 31 * 6, (-72, -66 - 1)),
    (10, "15min", None, 31 * 6 * 4, (-168 * 4, -162 * 4 - 1)),
    (10, "15min", "Europe/Berlin", (31 * 6 + 1) * 4, (-169 * 4, -162 * 4 - 1)),
    (10, "15min", "Asia/Kolkata", 31 * 6 * 4, (-168 * 4, -162 * 4 - 1)),
    (10, "h", None, 31 * 6, (-168, -162 - 1)),
    (10, "h", "Europe/Berlin", 31 * 6 + 1, (-169, -162 - 1)),  # dst end; one hour more
    (10, "h", "Asia/Kolkata", 31 * 6, (-168, -162 - 1)),
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
def test_peakfn_germanpower(end: str, freq: str, tz: str, count: int, stretch: Iterable[int]):
    """Test if the peak periods are correctly calculated."""
    # Not influenced by timezone, because dst change always during offpeak-hours.
    i = index("2020", end, freq, tz)
    do_test(i, f_germanpower, count, stretch)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(("end", "freq", "count", "stretch"), TESTCASES_13HALF)
def test_peakfn_everyday13half(end: str, freq: str, tz: str, count: int, stretch: Iterable[int]):
    """Test if the peak periods are correctly calculated."""
    # Not influenced by timezone, because dst change always during offpeak-hours.
    i = index("2020", end, freq, tz)
    do_test(i, f_everyday_13half, count, stretch)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(("end", "freq", "count", "stretch"), TESTCASES_WORKINGDAYS)
def test_peakfn_workingdaysfull(end: str, freq: str, tz: str, count: int, stretch: Iterable[int]):
    """Test if the peak periods are correctly calculated."""
    # Not influenced by timezone, because dst change always during offpeak-hours.
    i = index("2020", end, freq, tz)
    do_test(i, f_workingdays_full, count, stretch)


@pytest.mark.parametrize(("month", "freq", "tz", "count", "stretch"), TESTCASES_EVERYDAY6)
def test_peakfn_everydayuntil6(month: int, tz: str, freq: str, count: int, stretch: Iterable[int]):
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
@pytest.mark.parametrize("freq", ["D", "MS", "QS", "YS"])
@pytest.mark.parametrize(
    ("year", "bp", "jan_1_weekday", "jan_p", "feb_p", "mar_p"),
    [
        (2020, (366, 262), True, 23, 20, 22),
        (2021, (365, 261), True, 21, 20, 23),
        (2022, (365, 260), False, 21, 20, 23),
    ],
)
def test_peakduration_longfreqs(
    year, bp, jan_1_weekday, jan_p, feb_p, mar_p, tz, mar_b_corr, freq, month
):
    """Test if the correct number of base, peak and offpeak hours are calculated.
    bp = tuple with number of base days and days with peak hours; jan_p, feb_p, mar_p =
    number of days with peak hours in each month; mar_b_corr = correction to number of
    base HOURS."""

    if month > 1 and freq != "MS":
        pytest.skip("Feb and Mar are only checked on month-level here.")

    start = pd.Timestamp(f"{year}-{month}", tz=tz)
    i = pd.date_range(start, freq=freq, periods=1)

    # Expected values.
    if freq == "YS":
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

    result_b = tools.duration.index(i)
    result_p = tools.peakfn.peak_duration(i, f_germanpower)
    result_o = tools.peakfn.offpeak_duration(i, f_germanpower)

    # Test values.
    tools.testing.assert_series_equal(expected_b, result_b)
    tools.testing.assert_series_equal(expected_p, result_p)
    tools.testing.assert_series_equal(expected_o, result_o)


@pytest.mark.parametrize(
    "i,duration,peakduration",
    [
        # Quarters
        (
            pd.date_range("2020", freq="QS", periods=2, tz=None),
            [(31 + 29 + 31) * 24, (30 + 31 + 30) * 24],
            [(23 + 20 + 22) * 12, (22 + 21 + 22) * 12],
        ),
        (
            pd.date_range("2020", freq="QS", periods=2, tz="Europe/Berlin"),
            [(31 + 29 + 31) * 24 - 1, (30 + 31 + 30) * 24],
            [(23 + 20 + 22) * 12, (22 + 21 + 22) * 12],
        ),
        # Months
        (
            pd.date_range("2020", freq="MS", periods=3, tz=None),
            [31 * 24, 29 * 24, 31 * 24],
            [23 * 12, 20 * 12, 22 * 12],
        ),
        (
            pd.date_range("2020", freq="MS", periods=3, tz="Europe/Berlin"),
            [31 * 24, 29 * 24, 31 * 24 - 1],
            [23 * 12, 20 * 12, 22 * 12],
        ),
        # Days
        (
            pd.date_range("2020", freq="D", periods=7, tz=None),
            [24] * 7,
            [12, 12, 12, 0, 0, 12, 12],
        ),
        # . End-of-March: DST (if observed in tz)
        (
            pd.date_range("2020-03-25", freq="D", periods=7, tz=None),
            [24, 24, 24, 24, 24, 24, 24],
            [12, 12, 12, 0, 0, 12, 12],
        ),
        (
            pd.date_range("2020-03-25", freq="D", periods=7, tz="Europe/Berlin"),
            [24, 24, 24, 24, 23, 24, 24],
            [12, 12, 12, 0, 0, 12, 12],
        ),
        # Hours
        (
            pd.date_range("2020", freq="h", periods=48, tz=None),
            [1] * 48,
            [*[0] * 8, *[1] * 12, *[0] * 12, *[1] * 12, *[0] * 4],
        ),
        # . End-of-March: DST (if observed in tz)
        (
            pd.date_range("2020-03-29", freq="h", periods=48, tz=None),
            [1] * 48,
            [*[0] * 32, *[1] * 12, *[0] * 4],
        ),
        (
            pd.date_range("2020-03-29", freq="h", periods=47, tz="Europe/Berlin"),
            [1] * 47,
            [*[0] * 31, *[1] * 12, *[0] * 4],
        ),
        # Quarterhours
        (
            pd.date_range("2020", freq="15min", periods=192, tz=None),
            [0.25] * 192,
            [*[0] * 32, *[0.25] * 48, *[0] * 48, *[0.25] * 48, *[0] * 16],
        ),
    ],
)
@pytest.mark.parametrize("what", ["base", "peak", "offpeak"])
def test_peakduration_allfreqs(
    i: pd.DatetimeIndex, duration: Iterable, peakduration: Iterable, what: str
):
    """Test if peak duration is correctly calculated for all frequencies."""
    if what == "duration":
        values = duration
        result = tools.duration.index(i)
    elif what == "peak":
        values = peakduration
        result = tools.peakfn.peak_duration(i, f_germanpower)
    else:  # 'offpeak'
        values = (b - p for b, p in zip(duration, peakduration))
        result = tools.peakfn.offpeak_duration(i, f_germanpower)

    expected = pd.Series(values, i, dtype=float).astype("pint[h]").rename("duration")
    tools.testing.assert_series_equal(result, expected)
