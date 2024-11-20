import pandas as pd
import pytest
import datetime as dt
from portfolyo import testing, tools


@pytest.fixture(params=["2020-01-01", "2020-04-01"])
def startdate(request) -> str:
    return request.param


@pytest.fixture
def startts(startdate, startofday, tz) -> pd.Timestamp:
    return pd.Timestamp(f"{startdate} {startofday}", tz=tz)


@pytest.fixture
def periods(freq) -> int:
    # 10 days or 3 months, quarters, years
    return {"15min": 960, "h": 240, "D": 10}.get(freq, 3)


@pytest.fixture
def index(startts, freq, periods) -> pd.DatetimeIndex:
    return pd.date_range(startts, freq=freq, periods=periods)


@pytest.fixture
def startts_right(startdate, startofday, freq, tz) -> pd.Timestamp:
    time = dt.time.fromisoformat(startofday)
    date = dt.date.fromisoformat(startdate)
    if freq == "15min":
        time = time.replace(minute=time.minute + 15)
    elif freq == "h":
        time = time.replace(hour=time.hour + 1)
    elif freq == "D":
        date = date.replace(day=date.day + 1)
    elif freq == "MS":
        date = date.replace(month=date.month + 1)
    elif freq == "QS":
        date = date.replace(month=date.month + 3)
    elif freq == "YS":
        date = date.replace(year=date.year + 1)
    return pd.Timestamp(f"{date} {time}", tz=tz)


@pytest.fixture
def index_right(startts_right, freq, periods):
    return pd.date_range(startts_right, freq=freq, periods=periods, name="right")


@pytest.fixture
def expect_standardized_index_right(freq):
    return freq != "15min"


def test_right_index(index, index_right, expect_standardized_index_right):
    """Test if right index is correctly calculated for index without dst-transition."""
    result = tools.right.index(index)
    expected = index_right
    testing.assert_index_equal(result, expected)
    if expect_standardized_index_right:
        testing.assert_index_standardized(result)


def test_right_stamp(startts, startts_right, freq):
    """Test if right timestamp is correctly calculated for timestamp without dst-transition."""
    result = tools.right.stamp(startts, freq)
    expected = startts_right
    assert result == expected


# DST

# . General


@pytest.fixture(params=["15min", "h", "D"])
def freq_dst(request):
    return request.param


# . Start of dst (winter -> summer).


@pytest.fixture(params=["2020-03-29"])
def startdate_dst1(request) -> dt.date:
    return dt.date.fromisoformat(request.param)


@pytest.fixture(params=["00:00", "01:00", "03:00"])
def startofday_dst1(request) -> str:
    return request.param


@pytest.fixture
def startts_dst1(startdate_dst1, startofday_dst1) -> pd.Timestamp:
    return pd.Timestamp(f"{startdate_dst1} {startofday_dst1}", tz="Europe/Berlin")


@pytest.fixture
def periods_dst1(startofday_dst1, freq_dst) -> int:
    # 10 days
    if freq_dst == "15min":
        day1 = 92 if startofday_dst1 in ["00:00", "01:00"] else 96
        return day1 + 9 * 96
    elif freq_dst == "h":
        day1 = 23 if startofday_dst1 in ["00:00", "01:00"] else 24
        return day1 + 9 * 24
    elif freq_dst == "D":
        return 10
    raise ValueError()


@pytest.fixture
def index_dst1(startts_dst1, freq_dst, periods_dst1) -> pd.DatetimeIndex:
    return pd.date_range(startts_dst1, freq=freq_dst, periods=periods_dst1)


@pytest.fixture
def startts_right_dst1(startdate_dst1, startofday_dst1, freq_dst) -> pd.Timestamp:
    time = dt.time.fromisoformat(startofday_dst1)
    if freq_dst == "15min":
        time = time.replace(minute=time.minute + 15)
    elif freq_dst == "h":
        delta_hours = 2 if startofday_dst1 == "01:00" else 1
        time = time.replace(hour=time.hour + delta_hours)
    elif freq_dst == "D":
        startdate_dst1 = startdate_dst1.replace(day=startdate_dst1.day + 1)
    return pd.Timestamp(f"{startdate_dst1} {time}", tz="Europe/Berlin")


@pytest.fixture
def index_right_dst1(startts_right_dst1, freq_dst, periods_dst1):
    return pd.date_range(
        startts_right_dst1, freq=freq_dst, periods=periods_dst1, name="right"
    )


@pytest.fixture
def expect_standardized_index_right_dst1(freq_dst, startofday_dst1):
    return not (freq_dst == "15min" or (freq_dst == "h" and startofday_dst1 == "01:00"))


def test_right_index_dst1(
    index_dst1, index_right_dst1, expect_standardized_index_right_dst1
):
    """Test if right index is correctly calculated for index with winter->summer dst-transition."""
    result = tools.right.index(index_dst1)
    expected = index_right_dst1
    testing.assert_index_equal(result, expected)
    if expect_standardized_index_right_dst1:
        testing.assert_index_standardized(result)


def test_right_stamp_dst1(startts_dst1, startts_right_dst1, freq_dst):
    """Test if right timestamp is correctly calculated for timestamp with winter->summer dst-transition."""
    result = tools.right.stamp(startts_dst1, freq_dst)
    expected = startts_right_dst1
    assert result == expected


# . End of dst (summer -> winter).


@pytest.fixture(params=["2020-10-25"])
def startdate_dst2(request) -> dt.date:
    return dt.date.fromisoformat(request.param)


@pytest.fixture(params=["00:00", "01:00", "02:00+0200", "02:00+0100", "03:00"])
def startofday_dst2(request) -> str:
    return request.param


@pytest.fixture
def startts_dst2(startdate_dst2, startofday_dst2) -> pd.Timestamp:
    return pd.Timestamp(f"{startdate_dst2} {startofday_dst2}", tz="Europe/Berlin")


@pytest.fixture
def periods_dst2(startofday_dst2, freq_dst) -> int:
    # 10 days
    if freq_dst == "15min":
        day1 = 100 if startofday_dst2 in ["00:00", "01:00", "02:00+0200"] else 96
        return day1 + 9 * 96
    elif freq_dst == "h":
        day1 = 25 if startofday_dst2 in ["00:00", "01:00", "02:00+0200"] else 24
        return day1 + 9 * 24
    elif freq_dst == "D":
        return 10
    raise ValueError()


@pytest.fixture
def index_dst2(
    startts_dst2, freq_dst, periods_dst2, startofday_dst2
) -> pd.DatetimeIndex:
    if startofday_dst2 in ["02:00+0200", "02:00+0100"] and freq_dst == "D":
        pytest.skip(
            "Pandas thinks this is ambiguous, even though we have '+0100' or '+0200' in the timestamp."
        )
    return pd.date_range(startts_dst2, freq=freq_dst, periods=periods_dst2)


@pytest.fixture
def startts_right_dst2(startdate_dst2, startofday_dst2, freq_dst) -> pd.Timestamp:
    if freq_dst == "15min":
        timestr = startofday_dst2.replace(":00", ":15")
    elif freq_dst == "h":
        timestr = {
            "00:00": "01:00",
            "01:00": "02:00+0200",
            "02:00+0200": "02:00+0100",
            "02:00+0100": "03:00",
            "03:00": "04:00",
        }[startofday_dst2]
    else:  # freq_dst == "D":
        if startofday_dst2 in ["02:00+0200", "02:00+0100"]:
            pytest.skip("Unclear what correct outcome is in this situation.")
        timestr = startofday_dst2
        startdate_dst2 = startdate_dst2.replace(day=startdate_dst2.day + 1)
    return pd.Timestamp(f"{startdate_dst2} {timestr}", tz="Europe/Berlin")


@pytest.fixture
def index_right_dst2(startts_right_dst2, freq_dst, periods_dst2):
    return pd.date_range(
        startts_right_dst2, freq=freq_dst, periods=periods_dst2, name="right"
    )


@pytest.fixture
def expect_standardized_index_right_dst2(freq_dst, startofday_dst2):
    return not (
        freq_dst == "15min" or (freq_dst == "h" and startofday_dst2 == "02:00+0200")
    )


def test_right_index_dst2(
    index_dst2, index_right_dst2, expect_standardized_index_right_dst2
):
    """Test if right index is correctly calculated for index with summer->winter dst-transition."""
    result = tools.right.index(index_dst2)
    expected = index_right_dst2
    testing.assert_index_equal(result, expected)
    if expect_standardized_index_right_dst2:
        testing.assert_index_standardized(result)


def test_right_stamp_dst2(startts_dst2, startts_right_dst2, freq_dst):
    """Test if right timestamp is correctly calculated for index with summer->winter dst-transition."""
    result = tools.right.stamp(startts_dst2, freq_dst)
    expected = startts_right_dst2
    assert result == expected
