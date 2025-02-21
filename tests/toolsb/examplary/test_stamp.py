import pandas as pd
import pytest

from portfolyo import toolsb

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
    (pd.Timestamp(stamp), freq, startofday, isboundary)
    for stamp, d1 in _STAMP_STARTOFDAYS_ISBOUNDARY_FREQS.items()
    for startofdays, d2 in d1.items()
    for startofday in startofdays
    for isboundary, freqs in d2.items()
    for freq in freqs
]


@pytest.mark.parametrize(
    "stamp,freq,startofday,isboundary", STAMP_FREQ_STARTOFDAY_ISBOUNDARY
)
def test_stamp_isboundary(stamp, freq, startofday, isboundary):
    assert toolsb.stamp.is_boundary(stamp, freq, startofday) == isboundary


STAMP_FREQ_STARTOFDAY_ROUNDED = [
    (stamp, freq, startofday, stamp)
    for (stamp, freq, startofday, isboundary) in STAMP_FREQ_STARTOFDAY_ISBOUNDARY
    if isboundary
]


@pytest.mark.parametrize("stamp,freq,startofday,rounded", STAMP_FREQ_STARTOFDAY_ROUNDED)
@pytest.mark.parametrize("fn", ["floor", "ceil"])
def test_stamp_floorceil_isboundary(stamp, freq, startofday, fn, rounded):
    if fn == "floor":
        assert toolsb.stamp.floor(stamp, freq, startofday) == rounded
    else:
        assert toolsb.stamp.ceil(stamp, freq, startofday) == rounded


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
@pytest.mark.parametrize("fn", ["floor", "ceil"])
def test_stamp_floorceil_isnotboundary(stamp, freq, startofday, fn, floored, ceiled):
    if fn == "floor":
        assert toolsb.stamp.floor(stamp, freq, startofday) == floored
    else:
        assert toolsb.stamp.ceil(stamp, freq, startofday) == ceiled
