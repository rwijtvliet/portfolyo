import datetime as dt
from typing import Iterable

import pandas as pd
import pytest

from portfolyo import tools


def index(start: str, end: str, freq: str, tz: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq=freq, inclusive="left", tz=tz)


f_germanpower = tools.peakperiod.factory(dt.time(hour=8), dt.time(hour=20))
f_everyday13half = tools.peakperiod.factory(
    dt.time(hour=8), dt.time(hour=21, minute=30), [1, 2, 3, 4, 5, 6, 7]
)
f_fullworkingdays = tools.peakperiod(None, None, [1, 2, 3, 4, 5])
f_everyday_until6 = tools.peakperiod(None, dt.time(hour=6), [1, 2, 3, 4, 5, 6, 7])


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("end", "freq", "expected_peak"),
    [
        ("2020-01-08", "15T", 5 * 12 * 4),
        ("2020-01-08", "H", 5 * 12),
        ("2020-04-01", "15T", 65 * 12 * 4),
        ("2020-04-01", "H", 65 * 12),
        ("2021", "15T", 254 * 12 * 4),
        ("2021", "H", 254 * 12),
        ("2021", "D", ValueError),
        ("2021", "MS", ValueError),
        ("2021", "QS", ValueError),
        ("2021", "AS", ValueError),
    ],
)
def test_peakperiod_germanpower_2020(end: str, freq: str, tz: str, expected_peak: int):
    """Test if the peak periods are correctly calculated."""
    # Not influenced by timezone, because dst change always during offpeak-hours.
    i = index("2020", end, freq, tz)
    test(i, f_germanpower, expected_peak)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("end", "freq", "expected_peak"),
    [
        ("2020-01-08", "15T", 7 * 13.5 * 4),
        ("2020-04-01", "15T", 91 * 13.5 * 4),
        ("2021", "15T", 366 * 13.5 * 4),
        ("2021", "H", ValueError),
        ("2021", "D", ValueError),
        ("2021", "MS", ValueError),
        ("2021", "QS", ValueError),
        ("2021", "AS", ValueError),
    ],
)
def test_peakperiod_everyday13half_2020(
    end: str, freq: str, tz: str, expected_peak: int
):
    """Test if the peak periods are correctly calculated."""
    # Not influenced by timezone, because dst change always during offpeak-hours.
    i = index("2020", end, freq, tz)
    test(i, f_everyday13half, expected_peak)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("end", "freq", "expected_peak"),
    [
        ("2020-01-08", "15T", 5 * 24 * 4),
        ("2020-01-08", "H", 5 * 24),
        ("2020-01-08", "D", 5),
        ("2020-04-01", "15T", 65 * 24 * 4),
        ("2020-04-01", "H", 65 * 24),
        ("2020-04-01", "D", 65),
        ("2021", "15T", 254 * 24 * 4),
        ("2021", "H", 254 * 24),
        ("2021", "D", 254),
        ("2021", "MS", ValueError),
        ("2021", "QS", ValueError),
        ("2021", "AS", ValueError),
    ],
)
def test_peakperiod_fullworkingdays_2020(
    end: str, freq: str, tz: str, expected_peak: int
):
    """Test if the peak periods are correctly calculated."""
    # Not influenced by timezone, because dst change always during offpeak-hours.
    i = index("2020", end, freq, tz)
    test(i, f_fullworkingdays, expected_peak)


@pytest.mark.parametrize(
    ("month", "freq", "tz", "expected_peak"),
    [
        (1, "15T", None, 31 * 6 * 4),
        (1, "15T", "Europe/Berlin", 31 * 6 * 4),
        (1, "15T", "Asia/Kolkata", 31 * 6 * 4),
        (1, "H", None, 31 * 6),
        (1, "H", "Europe/Berlin", 31 * 6),
        (1, "H", "Asia/Kolkata", 31 * 6),
        (1, "D", None, ValueError),
        (1, "D", "Europe/Berlin", ValueError),
        (1, "D", "Asia/Kolkata", ValueError),
        (1, "MS", None, ValueError),
        (1, "MS", "Europe/Berlin", ValueError),
        (1, "MS", "Asia/Kolkata", ValueError),
        (1, "QS", None, ValueError),
        (1, "QS", "Europe/Berlin", ValueError),
        (1, "QS", "Asia/Kolkata", ValueError),
        (1, "AS", None, ValueError),
        (1, "AS", "Europe/Berlin", ValueError),
        (1, "AS", "Asia/Kolkata", ValueError),
        (3, "15T", None, 31 * 6 * 4),
        (3, "15T", "Europe/Berlin", (31 * 6 - 1) * 4),
        (3, "15T", "Asia/Kolkata", 31 * 6 * 4),
        (3, "H", None, 31 * 6),
        (3, "H", "Europe/Berlin", 31 * 6 - 1),
        (3, "H", "Asia/Kolkata", 31 * 6),
        (10, "15T", None, 31 * 6 * 4),
        (10, "15T", "Europe/Berlin", (31 * 6 + 1) * 4),
        (10, "15T", "Asia/Kolkata", 31 * 6 * 4),
        (10, "H", None, 31 * 6),
        (10, "H", "Europe/Berlin", 31 * 6 + 1),
        (10, "H", "Asia/Kolkata", 31 * 6),
    ],
)
def test_peakperiod_everydayuntil6_2020(
    month: int, tz: str, freq: str, expected_peak: int
):
    """Test if the peak periods are correctly calculated."""
    i = index(f"2020-{month}", f"2020-{month+1}", freq, tz)
    test(i, f_germanpower, expected_peak)


@pytest.mark.parametrize(
    ("peak_left", "peak_right", "isoweekdays", "expected"),
    [
        (None, None, [1, 2, 3, 4, 5, 6, 7], ValueError),
        (dt.time(hour=0), dt.time(hour=0), [1, 2, 3, 4, 5, 6, 7], ValueError),
        (dt.time(hour=0), dt.time(hour=0), [1, 2, 3, 4, 5, 6], None),
        (dt.time(hour=1), None, [1, 2, 3, 4, 5, 6, 7], None),
    ],
)
def test_incorrectfunction(
    peak_left: dt.time, peak_right: dt.time, isoweekdays: Iterable[int], expected
):
    """Test if an error is raised when creating impossible functions."""
    if type(expected) is type and issubclass(expected, Exception):
        with pytest.raises(expected):
            tools.peakperiod.factory(peak_left, peak_right, isoweekdays)
    else:
        tools.peakperiod.factory(peak_left, peak_right, isoweekdays)


def test(i: pd.DatetimeIndex, f: tools.peakperiod.PeakFunction, expected_peak: int):
    if type(expected_peak) is type and issubclass(expected_peak, Exception):
        with pytest.raises(expected_peak):
            f(i)
    else:
        assert f(i).sum() == expected_peak
