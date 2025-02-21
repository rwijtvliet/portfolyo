import datetime as dt

import pytest

from portfolyo import toolsb

TIME_AND_STRING = [
    (dt.time(hour=0), "00:00"),
    (dt.time(hour=6), "06:00"),
    (dt.time(hour=6, minute=39, second=51), "06:39:51"),
]

TIME_AND_TDELTA = [
    (dt.time(hour=0), dt.timedelta(seconds=0)),
    (dt.time(hour=6), dt.timedelta(hours=6)),
    (dt.time(hour=6), dt.timedelta(minutes=360)),
    (
        dt.time(hour=6, minute=39, second=51),
        dt.timedelta(hours=6, minutes=39, seconds=51),
    ),
]


@pytest.mark.parametrize(
    "input,expected",
    [(dt.time(hour=0), None), *TIME_AND_STRING, *TIME_AND_TDELTA],
)
def test_conversion(input, expected):
    assert toolsb.startofday.convert(input) == expected


@pytest.mark.parametrize(
    "input,isvalid",
    [
        (dt.time(hour=0), True),
        (dt.time(hour=6), True),
        (dt.time(hour=6, minute=30), False),
        (dt.time(hour=6, minute=39, second=51), False),
    ],
)
def test_validation(input, isvalid):
    if isvalid:
        toolsb.startofday.validate(input)
    else:
        with pytest.raises(ValueError):
            toolsb.startofday.validate(input)


@pytest.mark.parametrize("input,expected", TIME_AND_STRING)
def test_tostring(input, expected):
    assert toolsb.startofday.to_string(input) == expected


@pytest.mark.parametrize("input,expected", TIME_AND_TDELTA)
def test_totdelta(input, expected):
    assert toolsb.startofday.to_tdelta(input) == expected
