import pandas as pd
import pytest
from portfolyo import testing, tools


@pytest.fixture(params=["2020-01-01", "2020-04-01"])
def startdate(request) -> str:
    return request.param


@pytest.fixture
def startts(startdate, startofday, tz) -> pd.Timestamp:
    return pd.Timestamp(f"{startdate} {startofday}", tz=tz)


@pytest.fixture
def index(startts, freq) -> pd.DatetimeIndex:
    # Must specify timezone in timestamp (instead of index) to avoid confusion at DST-changeover.
    return pd.date_range(startts, freq=freq, periods=2)


@pytest.fixture
def durations(freq, startdate) -> tuple[float | int, float | int]:
    """Duration in hours of first two periods in index."""
    if freq == "15min":
        return (0.25, 0.25)
    elif freq == "h":
        return (1, 1)
    elif freq == "D":
        return (24, 24)
    elif freq == "MS":
        return (744, 696) if startdate == "2020-01-01" else (720, 744)
    elif freq == "QS":
        if startdate == "2020-01-01":
            pytest.skip("Don't include DST start/end in this test.")
        return (2184, 2208)  # 2020-04-01
    elif freq == "YS":
        if startdate == "2020-04-01":
            pytest.skip("Only test years starting in Jan.")
        return (8784, 8760)
    raise ValueError()


@pytest.fixture
def index_durations(durations, index) -> pd.Series:
    floats = pd.Series(durations, index, dtype=float)
    return floats.astype("pint[h]").rename("duration")


@pytest.fixture
def expected_duration_stamp(durations) -> tools.unit.Q_:
    return tools.unit.Q_(float(durations[0]), "h")


def test_index_duration(index, index_durations):
    """Test if duration of index timestamps is correctly calculated."""
    result = tools.duration.index(index)
    testing.assert_series_equal(result, index_durations)


def test_duration_stamp(startts, freq, expected_duration_stamp):
    """Test if duration in correctly calculated for timestamps."""
    result = tools.duration.stamp(startts, freq)
    assert result == expected_duration_stamp


# DST

# . General


@pytest.fixture(params=["15min", "h", "D"])
def freq_dst(request):
    return request.param


# . Start of dst (winter -> summer).


@pytest.fixture(params=["2020-03-28", "2020-03-29"])
def startdate_dst1(request) -> str:
    return request.param


@pytest.fixture(params=["00:00", "01:00", "03:00"])
def startofday_dst1(request) -> str:
    return request.param


@pytest.fixture
def startts_dst1(startdate_dst1, startofday_dst1) -> pd.Timestamp:
    return pd.Timestamp(f"{startdate_dst1} {startofday_dst1}", tz="Europe/Berlin")


@pytest.fixture
def index_dst1(startts_dst1, freq_dst) -> pd.DatetimeIndex:
    return pd.date_range(startts_dst1, freq=freq_dst, periods=2)


@pytest.fixture
def durations_dst1(
    startofday_dst1, startdate_dst1, freq_dst
) -> tuple[float | int, float | int]:
    if freq_dst == "15min":
        return (0.25, 0.25)
    elif freq_dst == "h":
        return (1, 1)
    elif freq_dst == "D":
        if startdate_dst1 == "2020-03-29":
            return (24, 24) if startofday_dst1 == "03:00" else (23, 24)
        else:  # 2020-03-28
            return (23, 24) if startofday_dst1 == "03:00" else (24, 23)
    else:
        pytest.skip("Don't test DST with longer-than-daily frequency here.")


@pytest.fixture
def index_durations_dst1(durations_dst1, index_dst1) -> pd.Series:
    floats = pd.Series(durations_dst1, index_dst1, dtype=float)
    return floats.astype("pint[h]").rename("duration")


@pytest.fixture
def stamp_duration_dst1(durations_dst1) -> tools.unit.Q_:
    return tools.unit.Q_(float(durations_dst1[0]), "h")


def test_duration_index_dst1(index_dst1, index_durations_dst1):
    """Test if duration of index timestamps is correctly calculated for winter->summer dst-transition."""
    result = tools.duration.index(index_dst1)
    testing.assert_series_equal(result, index_durations_dst1)


def test_duration_stamp_dst1(startts_dst1, freq_dst, stamp_duration_dst1):
    """Test if duration in correctly calculated for timestamps winter->summer dst-
    transition."""
    result = tools.duration.stamp(startts_dst1, freq_dst)
    assert result == stamp_duration_dst1


# . End of dst (summer -> winter).


@pytest.fixture(params=["2020-10-24", "2020-10-25"])
def startdate_dst2(request) -> str:
    return request.param


@pytest.fixture(params=["00:00", "01:00", "02:00+0200", "02:00+0100", "03:00"])
def startofday_dst2(request) -> str:
    return request.param


@pytest.fixture
def startts_dst2(startdate_dst2, startofday_dst2) -> pd.Timestamp:
    if startdate_dst2 == "2020-10-24" and startofday_dst2 == "02:00+0100":
        pytest.skip("This timestamp does not exist.")
    elif startdate_dst2 == "2020-10-24" and startofday_dst2 == "02:00+0200":
        pytest.skip("This testcase is too ambiguous.")
    elif startdate_dst2 == "2020-10-25" and startofday_dst2 in [
        "02:00+0200",
        "02:00+0100",
    ]:
        pytest.skip(
            "Pandas thinks this is ambiguous, even though we have '+0100' or '+0200' in the timestamp."
        )
    return pd.Timestamp(f"{startdate_dst2} {startofday_dst2}", tz="Europe/Berlin")


@pytest.fixture
def index_dst2(startts_dst2, freq_dst) -> pd.DatetimeIndex:
    return pd.date_range(startts_dst2, freq=freq_dst, periods=2)


@pytest.fixture
def durations_dst2(
    startofday_dst2, startdate_dst2, freq_dst
) -> tuple[float | int, float | int]:
    if freq_dst == "15min":
        return (0.25, 0.25)
    elif freq_dst == "h":
        return (1, 1)
    elif freq_dst == "D":
        if startdate_dst2 == "2020-10-25":
            return (24, 24) if startofday_dst2 in ["03:00", "02:00+0100"] else (25, 24)
        else:  # 2020-10-24
            return (25, 24) if startofday_dst2 == "03:00" else (24, 25)


@pytest.fixture
def index_durations_dst2(durations_dst2, index_dst2) -> pd.Series:
    floats = pd.Series(durations_dst2, index_dst2, dtype=float)
    return floats.astype("pint[h]").rename("duration")


@pytest.fixture
def stamp_duration_dst2(durations_dst2) -> tools.unit.Q_:
    return tools.unit.Q_(float(durations_dst2[0]), "h")


def test_duration_index_dst2(index_dst2, index_durations_dst2):
    """Test if duration of index timestamps is correctly calculated for summer->winter dst-transition."""
    result = tools.duration.index(index_dst2)
    testing.assert_series_equal(result, index_durations_dst2)


def test_duration_stamp_dst2(startts_dst2, freq_dst, stamp_duration_dst2):
    """Test if duration in correctly calculated for timestamps summer->winter dst-
    transition."""
    result = tools.duration.stamp(startts_dst2, freq_dst)
    assert result == stamp_duration_dst2


# DST, monthly values.


@pytest.fixture(params=["MS", "QS"])
def freq_dst3(request) -> str:
    return request.param


@pytest.fixture(params=["w->s", "s->w"])
def when_dst3(request) -> str:
    return request.param


@pytest.fixture(params=["00:00", "01:00", "06:00"])
def startofday_dst3(request) -> str:
    return request.param


@pytest.fixture
def startdate_dst3(freq_dst3, when_dst3) -> str:
    if when_dst3 == "w->s":
        return "2020-03-01" if freq_dst3 == "MS" else "2020-01-01"
    return "2020-10-01"


@pytest.fixture
def startts_dst3(startdate_dst3, startofday_dst3) -> pd.Timestamp:
    return pd.Timestamp(f"{startdate_dst3} {startofday_dst3}", tz="Europe/Berlin")


@pytest.fixture
def index_dst3(startts_dst3, freq_dst3) -> pd.DatetimeIndex:
    return pd.date_range(startts_dst3, freq=freq_dst3, periods=2)


@pytest.fixture
def durations_dst3(when_dst3, freq_dst3) -> tuple[float | int, float | int]:
    if when_dst3 == "w->s":
        return (743, 720) if freq_dst3 == "MS" else (2183, 2184)
    else:
        return (745, 720) if freq_dst3 == "MS" else (2209, 2159)


@pytest.fixture
def index_durations_dst3(durations_dst3, index_dst3) -> pd.Series:
    floats = pd.Series(durations_dst3, index_dst3, dtype=float)
    return floats.astype("pint[h]").rename("duration")


@pytest.fixture
def stamp_duration_dst3(durations_dst3) -> tools.unit.Q_:
    return tools.unit.Q_(float(durations_dst3[0]), "h")


def test_duration_index_dst3(index_dst3, index_durations_dst3):
    """Test if duration of index timestamps is correctly calculated for dst-transition on monthly/quarterly level."""
    result = tools.duration.index(index_dst3)
    testing.assert_series_equal(result, index_durations_dst3)


def test_duration_stamp_dst3(startts_dst3, freq_dst3, stamp_duration_dst3):
    """Test if duration in correctly calculated for dst-
    transition on monthly/quarterly level."""
    result = tools.duration.stamp(startts_dst3, freq_dst3)
    assert result == stamp_duration_dst3
