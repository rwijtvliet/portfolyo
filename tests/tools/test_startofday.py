import pytest
import datetime as dt
from portfolyo import tools, testing, dev


ERROR_SOD_DICT = {"hour": 6, "minute": 1}  # error
SOD_DICTS = [
    {"hour": 0, "minute": 0},
    {"hour": 6, "minute": 0},
    {"hour": 15, "minute": 30},
    ERROR_SOD_DICT,
]

SEED = 3


STARTDATE_AND_FREQ = [
    ("2020-01-01", "H"),
    ("2020-01-01", "D"),
    ("2020-01-01", "MS"),
    ("2020-03-28", "D"),
    ("2020-03-01", "MS"),
    ("2020-10-25", "D"),
    ("2020-10-01", "MS"),
]


def create_start_of_day(hour, minute, returntype):
    if returntype == "time":
        return dt.time(hour=hour, minute=minute)
    if returntype == "str":
        return f"{hour:02}:{minute:02}:00"
    if returntype == "timedelta":
        return dt.timedelta(hours=hour, minutes=minute)


@pytest.mark.parametrize("startdate,freq", STARTDATE_AND_FREQ)
@pytest.mark.parametrize("sod_dict", SOD_DICTS)
@pytest.mark.parametrize("returntype", ["time", "str", "timedelta"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
def test_get_startofday(startdate, freq, tz, sod_dict, returntype):
    """Test if start-of-day can be correctly gotten from index."""
    i = dev.get_index(freq, tz, startdate, 10, dt.time(**sod_dict), _seed=SEED)
    result = tools.startofday.get(i, returntype)
    expected = create_start_of_day(**sod_dict, returntype=returntype)
    assert result == expected


@pytest.mark.parametrize("startdate,freq", STARTDATE_AND_FREQ)
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("sod_dict_in", SOD_DICTS)
@pytest.mark.parametrize("sod_dict_out", SOD_DICTS)
def test_set_startofday(startdate, freq, tz, sod_dict_in, sod_dict_out):
    """Test if start of day can be correctly set to an index."""
    periods = {"H": 240, "D": 10, "MS": 3}[freq]
    start_of_day_in = create_start_of_day(**sod_dict_in, returntype="time")
    start_of_day_out = create_start_of_day(**sod_dict_out, returntype="time")
    i = dev.get_index(freq, tz, startdate, periods, start_of_day_in, _seed=SEED)
    if freq == "H" or sod_dict_out == ERROR_SOD_DICT:
        with pytest.raises(ValueError):
            _ = tools.startofday.set(i, start_of_day_out)
        return

    result = tools.startofday.set(i, start_of_day_out)
    if start_of_day_in == start_of_day_out:
        expected = i
    else:
        expected = dev.get_index(
            freq, tz, startdate, periods, start_of_day_out, _seed=SEED
        )
    testing.assert_index_equal(result, expected)
