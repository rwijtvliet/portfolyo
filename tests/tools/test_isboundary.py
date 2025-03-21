import datetime as dt
from typing import Iterable

import pandas as pd
import pytest

from portfolyo import testing, tools

TESTCASES = [  # date, i_freq, periods, freq, expected_repeat
    ("2020-01-01", "15min", 2 * 366 * 24 * 4, "15min", 1),
    ("2020-01-01", "15min", 2 * 366 * 24 * 4, "h", 4),
    ("2020-01-01", "15min", 2 * 31 * 24 * 4, "D", 24 * 4),  # >2MS
    ("2020-01-01", "15min", 2 * 31 * 24 * 4, "MS", (31 * 24 * 4, 29 * 24 * 4)),  # >2MS
    (
        "2020-01-01",
        "15min",
        2 * 366 * 24 * 4,
        "YS",
        (366 * 24 * 4, 365 * 24 * 4),
    ),  # >2AS
    ("2020-01-01", "h", 2 * 366 * 24, "15min", 1),
    ("2020-01-01", "h", 2 * 366 * 24, "h", 1),
    ("2020-01-01", "h", 2 * 31 * 24, "D", 24),  # >2MS
    ("2020-01-01", "h", 2 * 31 * 24, "MS", (31 * 24, 29 * 24)),  # >2MS
    ("2020-01-01", "h", 2 * 366 * 24, "YS", (366 * 24, 365 * 24)),  # >2AS
    ("2020-01-01", "D", 2 * 366, "15min", 1),
    ("2020-01-01", "D", 2 * 366, "h", 1),
    ("2020-01-01", "D", 2 * 366, "D", 1),
    ("2020-01-01", "D", 2 * 31, "MS", (31, 29)),  # >2MS
    ("2020-01-01", "D", 6 * 31, "QS", (91, 91)),  # >2QS
    ("2020-01-01", "D", 2 * 366, "YS", (366, 365)),  # >2AS
    ("2020-01-01", "MS", 2 * 12, "15min", 1),
    ("2020-01-01", "MS", 2 * 12, "h", 1),
    ("2020-01-01", "MS", 2 * 12, "D", 1),
    ("2020-01-01", "MS", 2 * 12, "MS", 1),
    ("2020-01-01", "MS", 2 * 3, "QS", 3),  # >2QS
    ("2020-01-01", "MS", 2 * 12, "YS", 12),  # >2AS
    ("2020-01-01", "QS", 2 * 4, "15min", 1),
    ("2020-01-01", "QS", 2 * 4, "h", 1),
    ("2020-01-01", "QS", 2 * 4, "D", 1),
    ("2020-01-01", "QS", 2 * 4, "MS", 1),
    ("2020-01-01", "QS", 2 * 4, "QS", 1),
    ("2020-01-01", "QS", 2 * 4, "YS", 4),  # >2AS
    ("2020-01-01", "YS", 2, "15min", 1),
    ("2020-01-01", "YS", 2, "h", 1),
    ("2020-01-01", "YS", 2, "D", 1),
    ("2020-01-01", "YS", 2, "MS", 1),
    ("2020-01-01", "YS", 2, "QS", 1),
    ("2020-01-01", "YS", 2, "YS", 1),
    ("2020-02-01", "YS-FEB", 2, "YS-FEB", 1),
    ("2020-02-01", "QS-FEB", 2 * 4, "YS-FEB", 4),
]

TESTCASES_DST = [  # ts, i_freq, periods, freq, expected_repeat
    # Start of DST, 2 days.
    ("2020-03-28 00:00", "15min", 2 * 24 * 4 - 4, "15min", 1),
    ("2020-03-28 00:00", "15min", 2 * 24 * 4 - 4, "h", 4),
    ("2020-03-28 00:00", "15min", 2 * 24 * 4 - 4, "D", (24 * 4, 23 * 4)),
    ("2020-03-28 00:00", "15min", 2 * 24 * 4 - 4, "MS", None),  # None at month start
    ("2020-03-28 00:00", "15min", 2 * 24 * 4 - 4, "QS", None),  # None at quarter start
    ("2020-03-28 00:00", "15min", 2 * 24 * 4 - 4, "YS", None),  # None at year start
    ("2020-03-28 00:00", "h", 2 * 24 - 1, "15min", 1),
    ("2020-03-28 00:00", "h", 2 * 24 - 1, "h", 1),
    ("2020-03-28 00:00", "h", 2 * 24 - 1, "D", (24, 23)),
    ("2020-03-28 00:00", "h", 2 * 24 - 1, "MS", None),  # None at month start
    ("2020-03-28 00:00", "h", 2 * 24 - 1, "QS", None),  # None at quarter start
    ("2020-03-28 00:00", "h", 2 * 24 - 1, "YS", None),  # None at year start
    ("2020-03-28 00:00", "D", 2, "15min", 1),
    ("2020-03-28 00:00", "D", 2, "h", 1),
    ("2020-03-28 00:00", "D", 2, "D", 1),
    ("2020-03-28 06:00", "15min", 2 * 24 * 4 - 4, "15min", 1),
    ("2020-03-28 06:00", "15min", 2 * 24 * 4 - 4, "h", 4),
    ("2020-03-28 06:00", "15min", 2 * 24 * 4 - 4, "D", (23 * 4, 24 * 4)),
    ("2020-03-28 06:00", "15min", 2 * 24 * 4 - 4, "MS", None),  # None at month start
    ("2020-03-28 06:00", "15min", 2 * 24 * 4 - 4, "QS", None),  # None at quarter start.
    ("2020-03-28 06:00", "15min", 2 * 24 * 4 - 4, "YS", None),  # None at year start
    ("2020-03-28 06:00", "h", 2 * 24 - 1, "15min", 1),
    ("2020-03-28 06:00", "h", 2 * 24 - 1, "h", 1),
    ("2020-03-28 06:00", "h", 2 * 24 - 1, "D", (23, 24)),
    ("2020-03-28 06:00", "h", 2 * 24 - 1, "MS", None),  # None at month start
    ("2020-03-28 06:00", "h", 2 * 24 - 1, "QS", None),  # None at quarter start
    ("2020-03-28 06:00", "h", 2 * 24 - 1, "YS", None),  # None at year start
    ("2020-03-28 06:00", "D", 2, "15min", 1),
    ("2020-03-28 06:00", "D", 2, "h", 1),
    ("2020-03-28 06:00", "D", 2, "D", 1),
    # Start of DST, entire month.
    ("2020-03-01 00:00", "15min", 31 * 24 * 4 - 4, "15min", 1),
    ("2020-03-01 00:00", "15min", 31 * 24 * 4 - 4, "h", 4),
    ("2020-03-01 00:00", "15min", 31 * 24 * 4 - 4, "D", (*[96] * 28, 92, *[96] * 2)),
    ("2020-03-01 00:00", "15min", 31 * 24 * 4 - 4, "MS", 31 * 24 * 4 - 4),
    ("2020-03-01 00:00", "15min", 31 * 24 * 4 - 4, "YS", None),
    ("2020-03-01 00:00", "h", 31 * 24 - 1, "15min", 1),
    ("2020-03-01 00:00", "h", 31 * 24 - 1, "h", 1),
    ("2020-03-01 00:00", "h", 31 * 24 - 1, "D", (*[24] * 28, 23, *[24] * 2)),
    ("2020-03-01 00:00", "h", 31 * 24 - 1, "MS", 31 * 24 - 1),
    ("2020-03-01 00:00", "h", 31 * 24 - 1, "YS", None),
    ("2020-03-01 00:00", "D", 31, "15min", 1),
    ("2020-03-01 00:00", "D", 31, "h", 1),
    ("2020-03-01 00:00", "D", 31, "D", 1),
    ("2020-03-01 00:00", "D", 31, "MS", 31),
    ("2020-03-01 00:00", "D", 31, "YS", None),
    ("2020-03-01 00:00", "MS", 1, "15min", 1),
    ("2020-03-01 00:00", "MS", 1, "h", 1),
    ("2020-03-01 00:00", "MS", 1, "D", 1),
    ("2020-03-01 00:00", "MS", 1, "MS", 1),
    ("2020-03-01 00:00", "MS", 1, "YS", None),
    ("2020-03-01 06:00", "15min", 31 * 24 * 4 - 4, "15min", 1),
    ("2020-03-01 06:00", "15min", 31 * 24 * 4 - 4, "h", 4),
    ("2020-03-01 06:00", "15min", 31 * 24 * 4 - 4, "D", (*[96] * 27, 92, *[96] * 3)),
    ("2020-03-01 06:00", "15min", 31 * 24 * 4 - 4, "MS", 31 * 24 * 4 - 4),
    ("2020-03-01 06:00", "15min", 31 * 24 * 4 - 4, "YS", None),
    ("2020-03-01 06:00", "h", 31 * 24 - 1, "15min", 1),
    ("2020-03-01 06:00", "h", 31 * 24 - 1, "h", 1),
    ("2020-03-01 06:00", "h", 31 * 24 - 1, "D", (*[24] * 27, 23, *[24] * 3)),
    ("2020-03-01 06:00", "h", 31 * 24 - 1, "MS", 31 * 24 - 1),
    ("2020-03-01 06:00", "h", 31 * 24 - 1, "YS", None),
    ("2020-03-01 06:00", "D", 31, "15min", 1),
    ("2020-03-01 06:00", "D", 31, "h", 1),
    ("2020-03-01 06:00", "D", 31, "D", 1),
    ("2020-03-01 06:00", "D", 31, "MS", 31),
    ("2020-03-01 06:00", "D", 31, "YS", None),
    ("2020-03-01 06:00", "MS", 1, "15min", 1),
    ("2020-03-01 06:00", "MS", 1, "h", 1),
    ("2020-03-01 06:00", "MS", 1, "D", 1),
    ("2020-03-01 06:00", "MS", 1, "MS", 1),
    ("2020-03-01 06:00", "MS", 1, "YS", None),
    # End of DST, 2 days.
    ("2020-10-24 00:00", "15min", 2 * 24 * 4 + 4, "15min", 1),
    ("2020-10-24 00:00", "15min", 2 * 24 * 4 + 4, "h", 4),
    ("2020-10-24 00:00", "15min", 2 * 24 * 4 + 4, "D", (24 * 4, 25 * 4)),
    ("2020-10-24 00:00", "15min", 2 * 24 * 4 + 4, "MS", None),  # None at month start
    ("2020-10-24 00:00", "15min", 2 * 24 * 4 + 4, "QS", None),  # None at quarter start
    ("2020-10-24 00:00", "15min", 2 * 24 * 4 + 4, "YS", None),  # None at year start
    ("2020-10-24 00:00", "h", 2 * 24 + 1, "15min", 1),
    ("2020-10-24 00:00", "h", 2 * 24 + 1, "h", 1),
    ("2020-10-24 00:00", "h", 2 * 24 + 1, "D", (24, 25)),
    ("2020-10-24 00:00", "h", 2 * 24 + 1, "MS", None),  # None at month start
    ("2020-10-24 00:00", "h", 2 * 24 + 1, "QS", None),  # None at quarter start
    ("2020-10-24 00:00", "h", 2 * 24 + 1, "YS", None),  # None at year start
    ("2020-10-24 00:00", "D", 2 * 24 + 1, "15min", 1),
    ("2020-10-24 00:00", "D", 2 * 24 + 1, "h", 1),
    ("2020-10-24 00:00", "D", 2 * 24 + 1, "D", 1),
    ("2020-10-24 06:00", "15min", 2 * 24 * 4 + 4, "15min", 1),
    ("2020-10-24 06:00", "15min", 2 * 24 * 4 + 4, "h", 4),
    ("2020-10-24 06:00", "15min", 2 * 24 * 4 + 4, "D", (25 * 4, 24 * 4)),
    ("2020-10-24 06:00", "15min", 2 * 24 * 4 + 4, "MS", None),  # None at month start
    ("2020-10-24 06:00", "15min", 2 * 24 * 4 + 4, "QS", None),  # None at quarter start
    ("2020-10-24 06:00", "15min", 2 * 24 * 4 + 4, "YS", None),  # None at year start
    ("2020-10-24 06:00", "h", 2 * 24 + 1, "15min", 1),
    ("2020-10-24 06:00", "h", 2 * 24 + 1, "h", 1),
    ("2020-10-24 06:00", "h", 2 * 24 + 1, "D", (25, 24)),
    ("2020-10-24 06:00", "h", 2 * 24 + 1, "MS", None),  # None at month start
    ("2020-10-24 06:00", "h", 2 * 24 + 1, "QS", None),  # None at quarter start
    ("2020-10-24 06:00", "h", 2 * 24 + 1, "YS", None),  # None at year start
    ("2020-10-24 06:00", "D", 2, "15min", 1),
    ("2020-10-24 06:00", "D", 2, "h", 1),
    ("2020-10-24 06:00", "D", 2, "D", 1),
    # End of DST, entire month.
    ("2020-10-01 00:00", "15min", 31 * 24 * 4 + 4, "15min", 1),
    ("2020-10-01 00:00", "15min", 31 * 24 * 4 + 4, "h", 4),
    ("2020-10-01 00:00", "15min", 31 * 24 * 4 + 4, "D", (*[96] * 24, 100, *[96] * 6)),
    ("2020-10-01 00:00", "15min", 31 * 24 * 4 + 4, "MS", 31 * 24 * 4 + 4),
    ("2020-10-01 00:00", "15min", 31 * 24 * 4 + 4, "YS", None),
    ("2020-10-01 00:00", "h", 31 * 24 + 1, "15min", 1),
    ("2020-10-01 00:00", "h", 31 * 24 + 1, "h", 1),
    ("2020-10-01 00:00", "h", 31 * 24 + 1, "D", (*[24] * 24, 25, *[24] * 6)),
    ("2020-10-01 00:00", "h", 31 * 24 + 1, "MS", 31 * 24 + 1),
    ("2020-10-01 00:00", "h", 31 * 24 + 1, "YS", None),
    ("2020-10-01 00:00", "D", 31, "15min", 1),
    ("2020-10-01 00:00", "D", 31, "h", 1),
    ("2020-10-01 00:00", "D", 31, "D", 1),
    ("2020-10-01 00:00", "D", 31, "MS", 31),
    ("2020-10-01 00:00", "D", 31, "YS", None),
    ("2020-10-01 00:00", "MS", 1, "15min", 1),
    ("2020-10-01 00:00", "MS", 1, "h", 1),
    ("2020-10-01 00:00", "MS", 1, "D", 1),
    ("2020-10-01 00:00", "MS", 1, "MS", 1),
    ("2020-10-01 00:00", "MS", 1, "YS", None),
    ("2020-10-01 06:00", "15min", 31 * 24 * 4 + 4, "15min", 1),
    ("2020-10-01 06:00", "15min", 31 * 24 * 4 + 4, "h", 4),
    ("2020-10-01 06:00", "15min", 31 * 24 * 4 + 4, "D", (*[96] * 23, 100, *[96] * 7)),
    ("2020-10-01 06:00", "15min", 31 * 24 * 4 + 4, "MS", 31 * 24 * 4 + 4),
    ("2020-10-01 06:00", "15min", 31 * 24 * 4 + 4, "YS", None),
    ("2020-10-01 06:00", "h", 31 * 24 + 1, "15min", 1),
    ("2020-10-01 06:00", "h", 31 * 24 + 1, "h", 1),
    ("2020-10-01 06:00", "h", 31 * 24 + 1, "D", (*[24] * 23, 25, *[24] * 7)),
    ("2020-10-01 06:00", "h", 31 * 24 + 1, "MS", 31 * 24 + 1),
    ("2020-10-01 06:00", "h", 31 * 24 + 1, "YS", None),
    ("2020-10-01 06:00", "D", 31, "15min", 1),
    ("2020-10-01 06:00", "D", 31, "h", 1),
    ("2020-10-01 06:00", "D", 31, "D", 1),
    ("2020-10-01 06:00", "D", 31, "MS", 31),
    ("2020-10-01 06:00", "D", 31, "YS", None),
    ("2020-10-01 06:00", "MS", 1, "15min", 1),
    ("2020-10-01 06:00", "MS", 1, "h", 1),
    ("2020-10-01 06:00", "MS", 1, "D", 1),
    ("2020-10-01 06:00", "MS", 1, "MS", 1),
    ("2020-10-01 06:00", "MS", 1, "YS", None),
]

TESTCASES_MIDYEAR = [  # date, i_freq, periods, freq, expected_repeat,
    ("2019-12-15", "15min", 2 * 366 * 24 * 4, "15min", 1),
    ("2019-12-15", "15min", 2 * 366 * 24 * 4, "h", 4),
    ("2019-12-15", "15min", 2 * 31 * 24 * 4, "D", 24 * 4),  # >2MS
    ("2019-12-15", "h", 2 * 366 * 24, "15min", 1),
    ("2019-12-15", "h", 2 * 366 * 24, "h", 1),
    ("2019-12-15", "h", 2 * 31 * 24, "D", 24),  # >2MS
    ("2019-12-15", "D", 2 * 366, "15min", 1),
    ("2019-12-15", "D", 2 * 366, "h", 1),
    ("2019-12-15", "D", 2 * 366, "D", 1),
    ("2019-12-01", "MS", 2 * 12, "15min", 1),
    ("2019-12-01", "MS", 2 * 12, "h", 1),
    ("2019-12-01", "MS", 2 * 12, "D", 1),
    ("2019-12-01", "MS", 2 * 12, "MS", 1),
    ("2019-10-01", "QS", 2 * 4, "15min", 1),
    ("2019-10-01", "QS", 2 * 4, "h", 1),
    ("2019-10-01", "QS", 2 * 4, "D", 1),
    ("2019-10-01", "QS", 2 * 4, "MS", 1),
    ("2019-10-01", "QS", 2 * 4, "QS", 1),
    ("2019-02-01", "QS-FEB", 2 * 4, "QS-FEB", 1),
    ("2019-10-01", "QS-OCT", 2 * 4, "MS", 1),
]

TESTCASES_LEADINGZERO = [  # date, i_freq, periods, freq, expected_repeat, leading_zeros
    ("2019-12-15", "15min", 2 * 31 * 24 * 4, "MS", 31 * 24 * 4, 17 * 24 * 4),
    ("2019-12-15", "15min", 2 * 366 * 24 * 4, "YS", 366 * 24 * 4, 17 * 24 * 4),
    ("2019-12-15", "h", 2 * 31 * 24, "MS", 31 * 24, 17 * 24),
    ("2019-12-15", "h", 2 * 366 * 24, "YS", 366 * 24, 17 * 24),
    ("2019-12-15", "D", 2 * 31, "MS", 31, 17),
    ("2019-12-15", "D", 6 * 31, "QS", 91, 17),
    ("2019-12-15", "D", 2 * 366, "YS", 366, 17),  # >2AS
    ("2019-12-01", "MS", 2 * 3, "QS", 3, 1),  # >2QS
    ("2019-12-01", "MS", 2 * 12, "YS", 12, 1),  # >2AS
    ("2019-10-01", "QS", 2 * 4, "YS", 4, 1),  # >2AS
    # DST start.
    # ("2020-03-29 00:00", "D", 2*24, "MS", None), #None at month start
    # ("2020-03-29 00:00", "D", 2*24, "QS", None), #None at quarter start
    # ("2020-03-29 00:00", "D", 2*24, "YS", None), #None at year start
    # ("2020-03-01 00:00", "15min", 31*24*4-4, "QS", 31*24*4-4),
    # ("2020-03-01 00:00", "h", 31*24-1, "QS", 31*24*4-4),
    # ("2020-03-01 00:00", "D", 31, "QS", 31*24*4-4),
    # ("2020-03-01 00:00", "MS", 1, "QS", 31*24*4-4),
    # ("2020-03-01 06:00", "15min", 31*24*4-4, "QS", 31*24*4-4),
    # ("2020-03-01 06:00", "h", 31*24-1, "QS", 31*24*4-4),
    # ("2020-03-01 06:00", "D", 31, "QS", 31*24*4-4),
    # ("2020-03-01 06:00", "MS", 1, "QS", 31*24*4-4),
    # ("2020-10-01 00:00", "15min", 31*24*4+4, "QS", 31*24*4-4),
    # ("2020-10-01 00:00", "h", 31*24+1, "QS", 31*24+1*4-4),
    # ("2020-10-01 00:00", "D", 31, "QS", 31*24*4-4),
    # ("2020-10-01 00:00", "MS", 1, "QS", 31*24*4-4),
    # ("2020-10-01 06:00", "15min", 31*24*4+4, "QS", 31*24*4+4-4),
    # ("2020-10-01 06:00", "h", 31*24+1, "QS", 31*24+1*4-4),
    # ("2020-10-01 06:00", "D", 31, "QS", 31*24+1*4-4),
    # ("2020-10-01 06:00", "MS", 1, "QS", 31*24*4-4),
]


@pytest.mark.parametrize("start_time", ["00:00", "06:00"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("date", "i_freq", "periods", "freq", "expected_repeat"), TESTCASES
)
def test_isboundary_index(
    date: str,
    start_time: str,
    i_freq: str,
    periods: int,
    tz: str,
    freq: str,
    expected_repeat: int | Iterable[int],
):
    """Test if boundary timestamps are correctly identified in index."""
    ts = f"{date} {start_time}"
    do_test_index(ts, i_freq, periods, tz, freq, expected_repeat, 0)


@pytest.mark.parametrize(
    ("ts", "i_freq", "periods", "freq", "expected_repeat"), TESTCASES_DST
)
def test_isboundary_index_dst(
    ts: str,
    i_freq: str,
    periods: int,
    freq: str,
    expected_repeat: int | Iterable[int],
):
    """Test if boundary timestamps are correctly identified in index during dst-transition."""
    do_test_index(ts, i_freq, periods, "Europe/Berlin", freq, expected_repeat, 0)


@pytest.mark.parametrize("start_time", ["00:00", "06:00"])
@pytest.mark.parametrize(
    ("date", "i_freq", "periods", "freq", "expected_repeat"), TESTCASES_MIDYEAR
)
def test_isboundary_index_midyear(
    date: str,
    start_time: str,
    i_freq: str,
    periods: int,
    freq: str,
    expected_repeat: int | Iterable[int],
):
    """Test if boundary timestamps are correctly identified in index when we don't start at beginning of year."""
    ts = f"{date} {start_time}"
    do_test_index(ts, i_freq, periods, "Europe/Berlin", freq, expected_repeat, 0)


@pytest.mark.parametrize("start_time", ["00:00", "06:00"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("date", "i_freq", "periods", "freq", "expected_repeat", "leading_zeros"),
    TESTCASES_LEADINGZERO,
)
def test_isboundary_index_leadingzero(
    date: str,
    start_time: str,
    i_freq: str,
    periods: int,
    tz: str,
    freq: str,
    expected_repeat: int | Iterable[int],
    leading_zeros: int,
):
    """Test if boundary timestamps are correctly identified in index when we first have some zeros."""
    ts = f"{date} {start_time}"
    do_test_index(ts, i_freq, periods, tz, freq, expected_repeat, leading_zeros)


def do_test_index(ts, i_freq, periods, tz, freq, expected_repeat, leading_zeros):
    # Input.
    first = pd.Timestamp(ts, tz=tz)
    i = pd.date_range(first, freq=i_freq, periods=periods + 1)
    # Output.
    result = tools.isboundary.index(i, freq)
    # Expected output.
    expected_values = [False] * leading_zeros
    if expected_repeat is None:
        expected_values.extend([False] * (len(i) - len(expected_values)))
    elif isinstance(expected_repeat, Iterable):
        for er in expected_repeat:
            expected_values.extend([True, *[False] * (er - 1)])
        expected_values.extend([True, *[False] * (len(i) - len(expected_values) - 1)])
    else:
        expected_values.extend(
            [num % expected_repeat == 0 for num in range(len(i) - len(expected_values))]
        )
    expected = pd.Series(expected_values, i, name="isboundary")
    # Assert.
    testing.assert_series_equal(result, expected)


TESTCASES_STAMP = [  # ts, freq, offset_hours, expected
    ("2020", "15min", 0, True),
    ("2020", "15min", 6, True),
    ("2020", "h", 0, True),
    ("2020", "h", 6, True),
    ("2020", "D", 0, True),
    ("2020", "D", 6, False),
    ("2020", "MS", 0, True),
    ("2020", "MS", 6, False),
    ("2020", "YS", 0, True),
    ("2020", "YS", 6, False),
    ("2020-04-01", "15min", 0, True),
    ("2020-04-01", "15min", 6, True),
    ("2020-04-01", "h", 0, True),
    ("2020-04-01", "h", 6, True),
    ("2020-04-01", "D", 0, True),
    ("2020-04-01", "D", 6, False),
    ("2020-04-01", "MS", 0, True),
    ("2020-04-01", "MS", 6, False),
    ("2020-04-01", "YS", 0, False),
    ("2020-04-01", "YS", 6, False),
    ("2020-01-01 15:00", "15min", 0, True),
    ("2020-01-01 15:00", "15min", 6, True),
    ("2020-01-01 15:00", "h", 0, True),
    ("2020-01-01 15:00", "h", 6, True),
    ("2020-01-01 15:00", "D", 0, False),
    ("2020-01-01 15:00", "D", 6, False),
    ("2020-01-01 15:00", "MS", 0, False),
    ("2020-01-01 15:00", "MS", 6, False),
    ("2020-01-01 15:00", "YS", 0, False),
    ("2020-01-01 15:00", "YS", 6, False),
    ("2020-01-01 15:45", "15min", 0, True),
    ("2020-01-01 15:45", "15min", 6, True),
    ("2020-01-01 15:45", "h", 0, False),
    ("2020-01-01 15:45", "h", 6, False),
    ("2020-01-01 15:45", "D", 0, False),
    ("2020-01-01 15:45", "D", 6, False),
    ("2020-01-01 15:45", "MS", 0, False),
    ("2020-01-01 15:45", "MS", 6, False),
    ("2020-01-01 15:45", "YS", 0, False),
    ("2020-01-01 15:45", "YS", 6, False),
    ("2020-01-01 06:00", "15min", 0, True),
    ("2020-01-01 06:00", "15min", 6, True),
    ("2020-01-01 06:00", "h", 0, True),
    ("2020-01-01 06:00", "h", 6, True),
    ("2020-01-01 06:00", "D", 0, False),
    ("2020-01-01 06:00", "D", 6, True),
    ("2020-01-01 06:00", "MS", 0, False),
    ("2020-01-01 06:00", "MS", 6, True),
    ("2020-01-01 06:00", "YS", 0, False),
    ("2020-01-01 06:00", "YS", 6, True),
    ("2020-04-21 06:00", "15min", 0, True),
    ("2020-04-21 06:00", "15min", 6, True),
    ("2020-04-21 06:00", "h", 0, True),
    ("2020-04-21 06:00", "h", 6, True),
    ("2020-04-21 06:00", "D", 0, False),
    ("2020-04-21 06:00", "D", 6, True),
    ("2020-04-21 06:00", "MS", 0, False),
    ("2020-04-21 06:00", "MS", 6, False),
    ("2020-04-21 06:00", "YS", 0, False),
    ("2020-04-21 06:00", "YS", 6, False),
]


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(("ts", "freq", "offset_hours", "expected"), TESTCASES_STAMP)
def test_isboundary_stamp(
    ts: str, tz: str, freq: str, offset_hours: int, expected: bool
):
    """Test if boundary timestamps are correctly identified."""
    ts = pd.Timestamp(ts, tz=tz)
    start_of_day = dt.time(hour=offset_hours)
    result = tools.isboundary.stamp(ts, freq, start_of_day)
    assert result == expected
