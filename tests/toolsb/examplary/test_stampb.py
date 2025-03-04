import pandas as pd
import pytest

from portfolyo import toolsb, tools

LEFT_FREQ_RIGHT_DURATION = [
    ("2020", "min", "2020-01-01 00:01", 1 / 60),
    ("2020", "5min", "2020-01-01 00:05", 5 / 60),
    ("2020", "15min", "2020-01-01 00:15", 15 / 60),
    ("2020", "h", "2020-01-01 01:00", 1),
    ("2020", "D", "2020-01-02", 24),
    ("2020", "MS", "2020-02", 31 * 24),
    ("2020", "QS-JAN", "2020-04", 91 * 24),
    ("2020", "QS-APR", "2020-04", 91 * 24),
    ("2020", "QS-JUL", "2020-04", 91 * 24),
    ("2020", "QS-OCT", "2020-04", 91 * 24),
    ("2020", "YS", "2021", 8784),
    ("2020-05", "min", "2020-05-01 00:01", 1 / 60),
    ("2020-05", "5min", "2020-05-01 00:05", 5 / 60),
    ("2020-05", "15min", "2020-05-01 00:15", 15 / 60),
    ("2020-05", "h", "2020-05-01 01:00", 1),
    ("2020-05", "D", "2020-05-02", 24),
    ("2020-05", "MS", "2020-06", 31 * 24),
    ("2020-05", "QS-FEB", "2020-08", 92 * 24),
    ("2020-05", "QS-MAY", "2020-08", 92 * 24),
    ("2020-05", "QS-AUG", "2020-08", 92 * 24),
    ("2020-05", "QS-NOV", "2020-08", 92 * 24),
    ("2020-05", "YS-MAY", "2021-05", 8760),
    ("2020-04-21 15:00", "min", "2020-04-21 15:01", 1 / 60),
    ("2020-04-21 15:00", "5min", "2020-04-21 15:05", 5 / 60),
    ("2020-04-21 15:00", "15min", "2020-04-21 15:15", 15 / 60),
    ("2020-04-21 15:00", "h", "2020-04-21 16:00", 1),
]


@pytest.mark.parametrize(
    "stamp,freq,right",
    [
        (pd.Timestamp(left), freq, pd.Timestamp(right))
        for (left, freq, right, _) in LEFT_FREQ_RIGHT_DURATION
    ],
)
def test_stamp_toright(stamp, freq, right):
    assert toolsb.stamp.to_right(stamp, freq) == right


@pytest.mark.parametrize(
    "stamp,freq,duration",
    [
        (pd.Timestamp(left), freq, tools.unit.Q_(duration, "h"))
        for (left, freq, _, duration) in LEFT_FREQ_RIGHT_DURATION
    ],
)
def test_stamp_duration(stamp, freq, duration):
    assert toolsb.stamp.duration(stamp, freq) == duration


STAMP_STARTOFDAY1_STARTOFDAY2 = [
    ("2020-01-01", "00:00", "00:00"),
    ("2020-01-01", "00:00", "14:00"),
    ("2020-01-01", "00:00", "14:35"),
    ("2020-04-21", "21:34", "00:00"),
    ("2020-04-21", "21:34", "14:00"),
    ("2020-04-21", "21:34", "14:35"),
    ("2020-03-29", "01:00", "00:00"),  # dst start in Europe
    ("2020-03-29", "03:00", "00:00"),
    ("2020-03-29", "01:00", "04:00"),
    ("2020-03-29", "03:00", "04:00"),
    ("2020-10-25", "01:00", "00:00"),  # dst end in Europe
    ("2020-10-25", "03:00", "00:00"),
    ("2020-10-25", "01:00", "04:00"),
    ("2020-10-25", "03:00", "04:00"),
]


@pytest.mark.parametrize(
    "stamp,startofday,withreplacedtime",
    [
        (
            pd.Timestamp(f"{stamp} {startofday1}", tz=tz),
            toolsb.startofday.convert(startofday2),
            pd.Timestamp(f"{stamp} {startofday2}", tz=tz),
        )
        for (stamp, startofday1, startofday2) in STAMP_STARTOFDAY1_STARTOFDAY2
        for tz in [None, "Europe/Berlin", "Asia/Kolkata"]
    ],
)
def test_stamp_replacetime(stamp, startofday, withreplacedtime):
    assert toolsb.stamp.replace_time(stamp, startofday) == withreplacedtime


_STAMP_STARTOFDAYS_ISBOUNDARY_FREQS = {
    "2020-01-01 00:00": {
        ("00:00",): {
            True: ("min", "15min", "h", "D", "MS", "QS", "QS-APR", "YS"),
            False: ("QS-FEB", "YS-FEB", "YS-APR"),
        },
        ("06:00",): {
            True: ("min", "15min", "h"),
            False: ("D", "MS", "QS", "QS-FEB", "QS-APR", "YS", "YS-FEB", "YS-APR"),
        },
    },
    "2020-01-01 06:30": {
        ("00:00", "06:00"): {
            True: ("min", "15min"),
            False: ("h", "D", "MS", "QS", "QS-FEB", "QS-APR", "YS", "YS-FEB", "YS-APR"),
        },
    },
    "2020-04-21 00:00": {
        ("00:00",): {
            True: ("min", "15min", "h", "D"),
            False: ("MS", "QS", "QS-FEB", "QS-APR", "YS", "YS-FEB", "YS-APR"),
        },
        ("06:00",): {
            True: ("min", "15min", "h"),
            False: ("D", "MS", "QS", "QS-FEB", "QS-APR", "YS", "YS-FEB", "YS-APR"),
        },
    },
}
STAMP_FREQ_STARTOFDAY_ISBOUNDARY = [
    ("2020-01-01 00:00:00", "min", "00:00", True),
    ("2020-01-01 00:00:00", "15min", "00:00", True),
    ("2020-01-01 00:00:00", "h", "00:00", True),
    ("2020-01-01 00:00:00", "D", "00:00", True),
    ("2020-01-01 00:00:00", "MS", "00:00", True),
    ("2020-01-01 00:00:00", "QS", "00:00", True),
    ("2020-01-01 00:00:00", "QS-APR", "00:00", True),
    ("2020-01-01 00:00:00", "YS", "00:00", True),
    ("2020-01-01 00:00:00", "QS-FEB", "00:00", False),
    ("2020-01-01 00:00:00", "YS-FEB", "00:00", False),
    ("2020-01-01 00:00:00", "YS-APR", "00:00", False),
    ("2020-01-01 00:00:00", "min", "06:00", True),
    ("2020-01-01 00:00:00", "15min", "06:00", True),
    ("2020-01-01 00:00:00", "h", "06:00", True),
    ("2020-01-01 00:00:00", "D", "06:00", False),
    ("2020-01-01 00:00:00", "MS", "06:00", False),
    ("2020-01-01 00:00:00", "QS", "06:00", False),
    ("2020-01-01 00:00:00", "QS-FEB", "06:00", False),
    ("2020-01-01 00:00:00", "QS-APR", "06:00", False),
    ("2020-01-01 00:00:00", "YS", "06:00", False),
    ("2020-01-01 00:00:00", "YS-FEB", "06:00", False),
    ("2020-01-01 00:00:00", "YS-APR", "06:00", False),
    ("2020-01-01 06:30:00", "min", "00:00", True),
    ("2020-01-01 06:30:00", "15min", "00:00", True),
    ("2020-01-01 06:30:00", "h", "00:00", False),
    ("2020-01-01 06:30:00", "D", "00:00", False),
    ("2020-01-01 06:30:00", "MS", "00:00", False),
    ("2020-01-01 06:30:00", "QS", "00:00", False),
    ("2020-01-01 06:30:00", "QS-FEB", "00:00", False),
    ("2020-01-01 06:30:00", "QS-APR", "00:00", False),
    ("2020-01-01 06:30:00", "YS", "00:00", False),
    ("2020-01-01 06:30:00", "YS-FEB", "00:00", False),
    ("2020-01-01 06:30:00", "YS-APR", "00:00", False),
    ("2020-01-01 06:30:00", "min", "06:00", True),
    ("2020-01-01 06:30:00", "15min", "06:00", True),
    ("2020-01-01 06:30:00", "h", "06:00", False),
    ("2020-01-01 06:30:00", "D", "06:00", False),
    ("2020-01-01 06:30:00", "MS", "06:00", False),
    ("2020-01-01 06:30:00", "QS", "06:00", False),
    ("2020-01-01 06:30:00", "QS-FEB", "06:00", False),
    ("2020-01-01 06:30:00", "QS-APR", "06:00", False),
    ("2020-01-01 06:30:00", "YS", "06:00", False),
    ("2020-01-01 06:30:00", "YS-FEB", "06:00", False),
    ("2020-01-01 06:30:00", "YS-APR", "06:00", False),
    ("2020-04-21 00:00:00", "min", "00:00", True),
    ("2020-04-21 00:00:00", "15min", "00:00", True),
    ("2020-04-21 00:00:00", "h", "00:00", True),
    ("2020-04-21 00:00:00", "D", "00:00", True),
    ("2020-04-21 00:00:00", "MS", "00:00", False),
    ("2020-04-21 00:00:00", "QS", "00:00", False),
    ("2020-04-21 00:00:00", "QS-FEB", "00:00", False),
    ("2020-04-21 00:00:00", "QS-APR", "00:00", False),
    ("2020-04-21 00:00:00", "YS", "00:00", False),
    ("2020-04-21 00:00:00", "YS-FEB", "00:00", False),
    ("2020-04-21 00:00:00", "YS-APR", "00:00", False),
    ("2020-04-21 00:00:00", "min", "06:00", True),
    ("2020-04-21 00:00:00", "15min", "06:00", True),
    ("2020-04-21 00:00:00", "h", "06:00", True),
    ("2020-04-21 00:00:00", "D", "06:00", False),
    ("2020-04-21 00:00:00", "MS", "06:00", False),
    ("2020-04-21 00:00:00", "QS", "06:00", False),
    ("2020-04-21 00:00:00", "QS-FEB", "06:00", False),
    ("2020-04-21 00:00:00", "QS-APR", "06:00", False),
    ("2020-04-21 00:00:00", "YS", "06:00", False),
    ("2020-04-21 00:00:00", "YS-FEB", "06:00", False),
    ("2020-04-21 00:00:00", "YS-APR", "06:00", False),
]


@pytest.mark.parametrize(
    "stamp,freq,startofday,isboundary",
    [
        (pd.Timestamp(stamp), freq, startofday, isboundary)
        for (stamp, freq, startofday, isboundary) in STAMP_FREQ_STARTOFDAY_ISBOUNDARY
    ],
)
def test_stamp_isboundary(stamp, freq, startofday, isboundary):
    assert toolsb.stamp.is_boundary(stamp, freq, startofday) == isboundary


@pytest.mark.parametrize(
    "stamp,freq,startofday,rounded",
    [
        (pd.Timestamp(stamp), freq, startofday, pd.Timestamp(stamp))
        for (stamp, freq, startofday, isboundary) in STAMP_FREQ_STARTOFDAY_ISBOUNDARY
        if isboundary
    ],
)
@pytest.mark.parametrize("testfn", ["floor", "ceil"])
def test_stamp_floorceil_isboundary(stamp, freq, startofday, testfn, rounded):
    fn = toolsb.stamp.floor if testfn == "floor" else toolsb.stamp.ceil
    assert fn(stamp, freq, startofday) == rounded


STAMP_FREQ_STARTOFDAYS_FLOORED_CEILED = [
    ("2020-04-21 12:34:56", "min", "00:00", "2020-04-21 12:34", "2020-04-21 12:35"),
    ("2020-04-21 12:34:56", "min", "06:00", "2020-04-21 12:34", "2020-04-21 12:35"),
    ("2020-04-21 12:34:56", "min", "15:00", "2020-04-21 12:34", "2020-04-21 12:35"),
    ("2020-04-21 12:34:56", "15min", "00:00", "2020-04-21 12:30", "2020-04-21 12:45"),
    ("2020-04-21 12:34:56", "15min", "06:00", "2020-04-21 12:30", "2020-04-21 12:45"),
    ("2020-04-21 12:34:56", "15min", "15:00", "2020-04-21 12:30", "2020-04-21 12:45"),
    ("2020-04-21 12:34:56", "h", "00:00", "2020-04-21 12:00", "2020-04-21 13:00"),
    ("2020-04-21 12:34:56", "h", "06:00", "2020-04-21 12:00", "2020-04-21 13:00"),
    ("2020-04-21 12:34:56", "h", "15:00", "2020-04-21 12:00", "2020-04-21 13:00"),
    ("2020-04-21 12:34:56", "D", "00:00", "2020-04-21 00:00", "2020-04-22 00:00"),
    ("2020-04-21 12:34:56", "D", "15:00", "2020-04-20 15:00", "2020-04-21 15:00"),
    ("2020-04-21 12:34:56", "MS", "00:00", "2020-04-01 00:00", "2020-05-01 00:00"),
    ("2020-04-21 12:34:56", "MS", "15:00", "2020-04-01 15:00", "2020-05-01 15:00"),
    ("2020-04-21 12:34:56", "QS", "00:00", "2020-04-01 00:00", "2020-07-01 00:00"),
    ("2020-04-21 12:34:56", "QS", "15:00", "2020-04-01 15:00", "2020-07-01 15:00"),
    ("2020-04-21 12:34:56", "QS-FEB", "00:00", "2020-02-01 00:00", "2020-05-01 00:00"),
    ("2020-04-21 12:34:56", "QS-FEB", "15:00", "2020-02-01 15:00", "2020-05-01 15:00"),
    ("2020-04-21 12:34:56", "QS-APR", "00:00", "2020-04-01 00:00", "2020-07-01 00:00"),
    ("2020-04-21 12:34:56", "QS-APR", "15:00", "2020-04-01 15:00", "2020-07-01 15:00"),
    ("2020-04-21 12:34:56", "YS", "00:00", "2020-01-01 00:00", "2021-01-01 00:00"),
    ("2020-04-21 12:34:56", "YS", "15:00", "2020-01-01 15:00", "2021-01-01 15:00"),
    ("2020-04-21 12:34:56", "YS-FEB", "00:00", "2020-02-01 00:00", "2021-02-01 00:00"),
    ("2020-04-21 12:34:56", "YS-FEB", "15:00", "2020-02-01 15:00", "2021-02-01 15:00"),
]


@pytest.mark.parametrize(
    "stamp,freq,startofday,floored,ceiled",
    [
        (
            pd.Timestamp(stamp),
            freq,
            startofday,
            pd.Timestamp(floored),
            pd.Timestamp(ceiled),
        )
        for (
            stamp,
            freq,
            startofday,
            floored,
            ceiled,
        ) in STAMP_FREQ_STARTOFDAYS_FLOORED_CEILED
    ],
)
@pytest.mark.parametrize("testfn", ["floor", "ceil"])
def test_stamp_floorceil_isnotboundary(stamp, freq, startofday, testfn, floored, ceiled):
    if testfn == "floor":
        fn, expected = toolsb.stamp.floor, floored
    else:
        fn, expected = toolsb.stamp.ceil, ceiled
    assert fn(stamp, freq, startofday) == expected
