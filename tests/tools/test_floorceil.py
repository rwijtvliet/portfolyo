import pandas as pd
import pytest

from portfolyo import testing, tools

TESTCASES_NODST = [  # "ts", "fut", "freq", "floored", "ceiled"
    ("2020", 0, "D", "2020", "2020"),
    ("2020", 0, "MS", "2020", "2020"),
    ("2020", 0, "QS", "2020", "2020"),
    ("2020", 0, "AS", "2020", "2020"),
    ("2020", 1, "D", "2020-01-02", "2020-01-02"),
    ("2020", 1, "MS", "2020-02", "2020-02"),
    ("2020", 1, "QS", "2020-04", "2020-04"),
    ("2020", 1, "AS", "2021", "2021"),
    ("2020", -1, "D", "2019-12-31", "2019-12-31"),
    ("2020", -1, "MS", "2019-12", "2019-12"),
    ("2020", -1, "QS", "2019-10", "2019-10"),
    ("2020", -1, "AS", "2019", "2019"),
    ("2020-02-01", 0, "D", "2020-02-01", "2020-02-01"),
    ("2020-02-01", 0, "MS", "2020-02-01", "2020-02-01"),
    ("2020-02-01", 0, "QS", "2020-01-01", "2020-04-01"),
    ("2020-02-01", 0, "AS", "2020", "2021"),
    ("2020-01-01 23:55", 0, "D", "2020", "2020-01-02"),
    ("2020-01-24 1:32", 0, "MS", "2020", "2020-02"),
    ("2020-03-03 3:33", 0, "QS", "2020", "2020-04"),
    ("2020-10-11 12:34", 0, "AS", "2020", "2021"),
    ("2020-01-01 23:55", 1, "D", "2020-01-02", "2020-01-03"),
    ("2020-01-24 1:32", 1, "MS", "2020-02", "2020-03"),
    ("2020-03-03 3:33", 1, "QS", "2020-04", "2020-07"),
    ("2020-10-11 12:34", 1, "AS", "2021", "2022"),
    ("2020-01-01 23:55", -1, "D", "2019-12-31", "2020-01-01"),
    ("2020-01-24 1:32", -1, "MS", "2019-12", "2020"),
    ("2020-03-03 3:33", -1, "QS", "2019-10", "2020"),
    ("2020-10-11 12:34", -1, "AS", "2019", "2020"),
    ("2020-03-29 00:00", 0, "H", "2020-03-29 00:00", "2020-03-29 00:00"),
    ("2020-10-25 00:00", 0, "H", "2020-10-25 00:00", "2020-10-25 00:00"),
]

TESTCASES_DST = [  # "ts", "tz", "freq", "floored", "ceiled"
    ("2020-04-21 15:25", None, "H", "2020-04-21 15:00", "2020-04-21 16:00"),
    ("2020-04-21 15:25", "Europe/Berlin", "H", "2020-04-21 15:00", "2020-04-21 16:00"),
    (
        "2020-04-21 15:25+0200",
        "Europe/Berlin",
        "H",
        "2020-04-21 15:00+0200",
        "2020-04-21 16:00+0200",
    ),
    ("2020-04-21 15:25", "Asia/Kolkata", "H", "2020-04-21 15:00", "2020-04-21 16:00"),
    ("2020-03-29 01:50", None, "15T", "2020-03-29 01:45", "2020-03-29 02:00"),
    ("2020-03-29 03:05", None, "15T", "2020-03-29 03:00", "2020-03-29 03:15"),
    (
        "2020-03-29 01:50+0100",
        "Europe/Berlin",
        "15T",
        "2020-03-29 01:45+0100",
        "2020-03-29 03:00+0200",
    ),
    (
        "2020-03-29 03:05+0200",
        "Europe/Berlin",
        "15T",
        "2020-03-29 03:00+0200",
        "2020-03-29 03:15+0200",
    ),
    (
        "2020-03-29 01:50",
        "Europe/Berlin",
        "15T",
        "2020-03-29 01:45",
        "2020-03-29 03:00",
    ),
    ("2020-03-29 03:05", None, "15T", "2020-03-29 03:00", "2020-03-29 03:15"),
    ("2020-10-25 02:50", None, "15T", "2020-10-25 02:45", "2020-10-25 03:00"),
    ("2020-10-25 02:05", None, "15T", "2020-10-25 02:00", "2020-10-25 02:15"),
    (
        "2020-10-25 02:50+0200",
        "Europe/Berlin",
        "15T",
        "2020-10-25 02:45+0200",
        "2020-10-25 02:00+0100",
    ),
    (
        "2020-10-25 02:05+0200",
        "Europe/Berlin",
        "15T",
        "2020-10-25 02:00+0200",
        "2020-10-25 02:15+0200",
    ),
    (
        "2020-10-25 02:50+0100",
        "Europe/Berlin",
        "15T",
        "2020-10-25 02:45+0100",
        "2020-10-25 03:00+0100",
    ),
    (
        "2020-10-25 02:05+0100",
        "Europe/Berlin",
        "15T",
        "2020-10-25 02:00+0100",
        "2020-10-25 02:15+0100",
    ),
    (
        "2020-10-25 02:30+0200",
        "Europe/Berlin",
        "H",
        "2020-10-25 02:00+0200",
        "2020-10-25 02:00+0100",
    ),
    (
        "2020-10-25 02:30+0100",
        "Europe/Berlin",
        "H",
        "2020-10-25 02:00+0100",
        "2020-10-25 03:00+0100",
    ),
]

TESTCASES_NONNATURAL = [  # "ts", "freq", "offset_hours", "floored", "ceiled"
    ("2020-04-21 15:25", "H", 6, "2020-04-21 15:00", "2020-04-21 16:00"),
    ("2020-04-21 15:25", "15T", 6, "2020-04-21 15:15", "2020-04-21 15:30"),
    ("2020-03-29 01:40", "15T", 6, "2020-03-29 01:30", "2020-03-29 01:45"),
    ("2020-03-29 03:05", "15T", 6, "2020-03-29 03:00", "2020-03-29 03:15"),
    ("2020-04-21 15:25", "D", 6, "2020-04-21 06:00", "2020-04-22 06:00"),
    ("2020-03-28 01:40", "D", 6, "2020-03-27 06:00", "2020-03-28 06:00"),
    ("2020-03-29 01:40", "D", 6, "2020-03-28 06:00", "2020-03-29 06:00"),
    ("2020-03-30 01:40", "D", 6, "2020-03-29 06:00", "2020-03-30 06:00"),
    ("2020-03-28 03:40", "D", 6, "2020-03-27 06:00", "2020-03-28 06:00"),
    ("2020-03-29 03:40", "D", 6, "2020-03-28 06:00", "2020-03-29 06:00"),
    ("2020-03-30 03:40", "D", 6, "2020-03-29 06:00", "2020-03-30 06:00"),
    ("2020-10-24 01:40", "D", 6, "2020-10-23 06:00", "2020-10-24 06:00"),
    ("2020-10-25 01:40", "D", 6, "2020-10-24 06:00", "2020-10-25 06:00"),
    ("2020-10-26 01:40", "D", 6, "2020-10-25 06:00", "2020-10-26 06:00"),
    ("2020-10-24 03:40", "D", 6, "2020-10-23 06:00", "2020-10-24 06:00"),
    ("2020-10-25 03:40", "D", 6, "2020-10-24 06:00", "2020-10-25 06:00"),
    ("2020-10-26 03:40", "D", 6, "2020-10-25 06:00", "2020-10-26 06:00"),
    ("2020-04-01 05:25", "MS", 6, "2020-03-01 06:00", "2020-04-01 06:00"),
    ("2020-04-21 15:25", "MS", 6, "2020-04-01 06:00", "2020-05-01 06:00"),
    ("2020-03-28 01:40", "MS", 6, "2020-03-01 06:00", "2020-04-01 06:00"),
    ("2020-03-29 01:40", "MS", 6, "2020-03-01 06:00", "2020-04-01 06:00"),
    ("2020-03-30 01:40", "MS", 6, "2020-03-01 06:00", "2020-04-01 06:00"),
    ("2020-10-24 01:40", "MS", 6, "2020-10-01 06:00", "2020-11-01 06:00"),
    ("2020-10-25 01:40", "MS", 6, "2020-10-01 06:00", "2020-11-01 06:00"),
    ("2020-10-26 01:40", "MS", 6, "2020-10-01 06:00", "2020-11-01 06:00"),
    ("2020-04-01 05:25", "QS", 6, "2020-01-01 06:00", "2020-04-01 06:00"),
    ("2020-04-21 15:25", "QS", 6, "2020-04-01 06:00", "2020-05-01 06:00"),
    ("2020-03-28 01:40", "QS", 6, "2020-01-01 06:00", "2020-04-01 06:00"),
    ("2020-03-29 01:40", "QS", 6, "2020-01-01 06:00", "2020-04-01 06:00"),
    ("2020-03-30 01:40", "QS", 6, "2020-01-01 06:00", "2020-04-01 06:00"),
    ("2020-10-24 01:40", "QS", 6, "2020-10-01 06:00", "2020-11-01 06:00"),
    ("2020-10-25 01:40", "QS", 6, "2020-10-01 06:00", "2020-11-01 06:00"),
    ("2020-10-26 01:40", "QS", 6, "2020-10-01 06:00", "2020-11-01 06:00"),
]

# TODO: remove repetition in test_floorceil_1, _2, _3


@pytest.mark.parametrize("stamporindex", ["stamp", "index"])
@pytest.mark.parametrize("function", ["floor", "ceil"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(("ts", "fut", "freq", "floored", "ceiled"), TESTCASES_NODST)
def test_floorceil_1(
    stamporindex: str,
    function: str,
    ts: str,
    tz: str,
    fut: int,
    freq: str,
    floored: str,
    ceiled: str,
):
    """Test flooring and ceiling without dst-transition."""
    if stamporindex == "stamp" and function == "floor":
        stamptest(tools.floor.stamp, ts, freq, fut, 0, tz, floored)
    elif stamporindex == "stamp" and function == "ceil":
        stamptest(tools.ceil.stamp, ts, freq, fut, 0, tz, ceiled)
    elif stamporindex == "index" and function == "floor":
        indextest(tools.floor.index, ts, freq, fut, 0, tz, floored)
    elif stamporindex == "index" and function == "ceil":
        indextest(tools.ceil.index, ts, freq, fut, 0, tz, ceiled)
    else:
        raise ValueError


@pytest.mark.parametrize("stamporindex", ["stamp", "index"])
@pytest.mark.parametrize("function", ["floor", "ceil"])
@pytest.mark.parametrize(("ts", "tz", "freq", "floored", "ceiled"), TESTCASES_DST)
def test_floorceil_2(
    stamporindex: str,
    function: str,
    ts: str,
    tz: str,
    freq: str,
    floored: str,
    ceiled: str,
):
    """Test flooring and ceiling with dst-transition."""
    if stamporindex == "stamp" and function == "floor":
        stamptest(tools.floor.stamp, ts, freq, 0, 0, tz, floored)
    elif stamporindex == "stamp" and function == "ceil":
        stamptest(tools.ceil.stamp, ts, freq, 0, 0, tz, ceiled)
    elif stamporindex == "index" and function == "floor":
        indextest(tools.floor.index, ts, freq, 0, 0, tz, floored)
    elif stamporindex == "index" and function == "ceil":
        indextest(tools.ceil.index, ts, freq, 0, 0, tz, ceiled)
    else:
        raise ValueError


@pytest.mark.parametrize("stamporindex", ["stamp", "index"])
@pytest.mark.parametrize("function", ["floor", "ceil"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("ts", "freq", "offset_hours", "floored", "ceiled"), TESTCASES_NONNATURAL
)
def test_floorceil_3(
    stamporindex: str,
    function: str,
    ts: str,
    tz: str,
    freq: str,
    offset_hours: int,
    floored: str,
    ceiled: str,
):
    """Test flooring and ceiling with dst-transition."""
    if stamporindex == "stamp" and function == "floor":
        stamptest(tools.floor.stamp, ts, freq, 0, offset_hours, tz, floored)
    elif stamporindex == "stamp" and function == "ceil":
        stamptest(tools.ceil.stamp, ts, freq, 0, offset_hours, tz, ceiled)
    elif stamporindex == "index" and function == "floor":
        indextest(tools.floor.index, ts, freq, 0, offset_hours, tz, floored)
    elif stamporindex == "index" and function == "ceil":
        indextest(tools.ceil.index, ts, freq, 0, offset_hours, tz, ceiled)
    else:
        raise ValueError


def stamptest(fn, ts, freq, fut, offset_hours, tz, expected):
    ts = pd.Timestamp(ts, tz=tz)
    expected = pd.Timestamp(expected, tz=tz)
    assert fn(ts, freq, fut, offset_hours) == expected


def indextest(fn, ts, freq, fut, offset_hours, tz, expected):
    ts = pd.Timestamp(ts, tz=tz)
    i = pd.date_range(
        ts,
        periods=10,
        freq=freq,
    )  # causes rounding of ts
    i += ts - i[0]  # undoes the rounding
    result = fn(i, freq, fut, offset_hours)
    result.freq = None  # disregard checking frequencies here
    expected = pd.date_range(expected, periods=10, freq=freq, tz=tz)
    expected.freq = None  # disregard checking frequencies here
    testing.assert_index_equal(result, expected)
