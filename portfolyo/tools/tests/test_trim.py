import pandas as pd
import pytest
from portfolyo import testing, tools


@pytest.mark.parametrize("indexorframe", ["index", "s", "s_unit", "df", "df_unit"])
@pytest.mark.parametrize("offset_hours", [0, 6])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("freq", tools.freq.FREQUENCIES)
@pytest.mark.parametrize("trimfreq", tools.freq.FREQUENCIES)
def test_trimindex_notrimming(
    indexorframe: str, freq: str, tz: str, trimfreq: str, offset_hours: int
):
    """Test if no trimming is done when it is not necessary."""
    start = f"2020-01-01 {offset_hours:02}:00:00"
    end = f"2021-01-01 {offset_hours:02}:00:00"
    test_general(indexorframe, start, end, freq, tz, trimfreq, offset_hours, start, end)


@pytest.mark.parametrize("indexorframe", ["index", "s", "s_unit", "df", "df_unit"])
@pytest.mark.parametrize("offset_hours_ignore", [0, 2, 6])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
def test_trimindex_short2short(indexorframe: str, tz: str, offset_hours_ignore: int):
    """Test if indices with below-daily frequencies are correctly trimmed."""
    # If input and output are below-daily, the offset parameter to the trim function should be ignored.
    start, tr_start = "2020-01-05 04:15:00", "2020-01-05 05:00"
    end, tr_end = "2022-12-21 04:15:00", "2022-12-21 04:00"
    test_general(
        indexorframe,
        start,
        end,
        "15T",
        tz,
        "H",
        offset_hours_ignore,
        tr_start,
        tr_end,
    )


@pytest.mark.parametrize("indexorframe", ["index", "s", "s_unit", "df", "df_unit"])
@pytest.mark.parametrize("offset_hours", [0, 6])
@pytest.mark.parametrize("offset_hours_ignore", [0, 2, 6])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("trimfreq", "tr_start", "tr_end"),
    [
        ("MS", "2020-02-01", "2022-12-01"),
        ("QS", "2020-04-01", "2022-10-01"),
        ("AS", "2021-01-01", "2022-01-01"),
    ],
)
def test_trimindex_long2long(
    indexorframe: str,
    trimfreq: str,
    tr_start: str,
    tr_end: str,
    tz: str,
    offset_hours: int,
    offset_hours_ignore: int,
):
    """Test if indices with daily-or-longer frequencies are correctly trimmed."""
    # If input is daily-or-longer, and has offset, the offset parameter to the trim function should be ignored.
    start = f"2020-01-05 {offset_hours:02}:00:00"
    tr_start = f"{tr_start} {offset_hours:02}:00:00"
    end = f"2022-12-21 {offset_hours:02}:00:00"
    tr_end = f"{tr_end} {offset_hours:02}:00:00"
    test_general(
        indexorframe,
        start,
        end,
        "D",
        tz,
        trimfreq,
        offset_hours_ignore,
        tr_start,
        tr_end,
    )


@pytest.mark.parametrize("indexorframe", ["index", "s", "s_unit", "df", "df_unit"])
@pytest.mark.parametrize("offset_hours", [0, 6])
@pytest.mark.parametrize("offset_hours_ignore", [0, 2, 6])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("freq", "start", "end"),
    [
        ("MS", "2020-02-01", "2022-12-01"),
        ("QS", "2020-04-01", "2022-10-01"),
        ("AS", "2021-01-01", "2022-01-01"),
    ],
)
@pytest.mark.parametrize("trimfreq", ["H", "15T"])
def test_trimindex_long2short(
    indexorframe: str,
    start: str,
    end: str,
    freq: str,
    tz: str,
    trimfreq: str,
    offset_hours: int,
    offset_hours_ignore: int,
):
    """Test if indices with daily-or-longer frequency are correctly trimmed in complex cases."""
    # If trimfrequency is below-daily, the offset parameter to the trim function should be ignored.
    start = tr_start = f"{start} {offset_hours:02}:00:00"
    end = tr_end = f"{end} {offset_hours:02}:00:00"
    test_general(
        indexorframe,
        start,
        end,
        freq,
        tz,
        trimfreq,
        offset_hours_ignore,
        tr_start,
        tr_end,
    )


@pytest.mark.parametrize("indexorframe", ["index", "s", "s_unit", "df", "df_unit"])
@pytest.mark.parametrize("offset_hours", [0, 6])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("trimfreq", "tr_start", "tr_end"),
    [
        ("D", "2020-01-05", "2022-12-21"),
        ("MS", "2020-02-01", "2022-12-01"),
        ("QS", "2020-04-01", "2022-10-01"),
        ("AS", "2021-01-01", "2022-01-01"),
    ],
)
def test_trimindex_short2long(
    indexorframe: str, trimfreq: str, tr_start: str, tr_end: str, tz, offset_hours: int
):
    """Test if indices with below-daily frequency are correctly trimmed."""
    # Only in case a below-daily index is trimmed to a daily-or-longer frequency is the offset parameter used.
    start, tr_start = "2020-01-04 22:00:00", f"{tr_start} {offset_hours:02}:00:00"
    end, tr_end = "2022-12-21 22:00:00", f"{tr_end} {offset_hours:02}:00:00"
    test_general(
        indexorframe, start, end, "H", tz, trimfreq, offset_hours, tr_start, tr_end
    )


def test_general(
    indexorframe: str,
    start: str,
    end: str,
    freq: str,
    tz: str,
    trimfreq: str,
    offset_hours: int,
    tr_start: str,
    tr_end: str,
):
    i = pd.date_range(start, end, freq=freq, inclusive="left", tz=tz)
    if not tr_start:
        i_expected = pd.DatetimeIndex([], freq=freq, tz=tz)
    else:
        i_expected = pd.date_range(tr_start, tr_end, freq=freq, inclusive="left", tz=tz)

    if indexorframe == "index":
        test_trimindex(i, trimfreq, offset_hours, i_expected)
    elif indexorframe == "s":
        test_trimseries(i, trimfreq, offset_hours, i_expected, False)
    elif indexorframe == "s_unit":
        test_trimseries(i, trimfreq, offset_hours, i_expected, True)
    elif indexorframe == "df":
        test_trimdataframe(i, trimfreq, offset_hours, i_expected, False)
    elif indexorframe == "df_unit":
        test_trimdataframe(i, trimfreq, offset_hours, i_expected, True)


def test_trimindex(
    i: pd.DatetimeIndex, trimfreq: str, offset_hours: int, expected: pd.DatetimeIndex
):
    result = tools.trim.index(i, trimfreq, offset_hours)
    testing.assert_index_equal(result, expected)


def test_trimseries(
    i: pd.DatetimeIndex,
    trimfreq: str,
    offset_hours: int,
    i_result: pd.DatetimeIndex,
    with_units: bool,
):
    dtype = "pint[MW]" if with_units else float
    fr = pd.Series(range(len(i)), i, dtype=dtype)
    for num, ts in enumerate(i):
        if i == i_result[0]:
            break
    else:
        raise ValueError
    expected = pd.Series(range(num, num + len(i_result)), i_result, dtype=dtype)

    result = tools.trim.frame(fr, trimfreq, offset_hours)
    testing.assert_series_equal(result, expected)


def test_trimdataframe(
    i: pd.DatetimeIndex,
    trimfreq: str,
    offset_hours: int,
    i_result: pd.DatetimeIndex,
    with_units: bool,
):
    dtype = "pint[MW]" if with_units else float
    fr = pd.DataFrame({"a": range(len(i))}, i, dtypes={"a": dtype})
    for num, ts in enumerate(i):
        if i == i_result[0]:
            break
    else:
        raise ValueError
    expected = pd.DataFrame(
        {"a": range(num, num + len(i_result))}, i_result, dtypes={"a": dtype}
    )

    result = tools.trim.frame(fr, trimfreq, offset_hours)
    testing.assert_frames_equal(result, expected)
