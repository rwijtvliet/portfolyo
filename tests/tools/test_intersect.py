from typing import Iterable, List, Union

import pandas as pd
import pytest

from portfolyo import testing, tools

COMMON_END = "2022-02-02"

TESTCASES = [  # startdates, freq, expected_startdate
    # One starts at first day of year.
    (("2020-01-01", "2020-01-20"), "15T", "2020-01-20"),
    (("2020-01-01", "2020-01-20"), "15T", "2020-01-20"),
    (("2020-01-01", "2020-01-20"), "H", "2020-01-20"),
    (("2020-01-01", "2020-01-20"), "H", "2020-01-20"),
    (("2020-01-01", "2020-01-20"), "D", "2020-01-20"),
    (("2020-01-01", "2020-01-20"), "D", "2020-01-20"),
    (("2020-01-01", "2020-03-01"), "MS", "2020-03-01"),
    (("2020-01-01", "2020-03-01"), "MS", "2020-03-01"),
    (("2020-01-01", "2020-04-01"), "QS", "2020-04-01"),
    (("2020-01-01", "2020-04-01"), "QS", "2020-04-01"),
    (("2020-01-01", "2021-01-01"), "AS", "2021-01-01"),
    (("2020-01-01", "2021-01-01"), "AS", "2021-01-01"),
    # Both start in middle of year.
    (("2020-04-21", "2020-06-20"), "15T", "2020-06-20"),
    (("2020-04-21", "2020-06-20"), "15T", "2020-06-20"),
    (("2020-04-21", "2020-06-20"), "H", "2020-06-20"),
    (("2020-04-21", "2020-06-20"), "H", "2020-06-20"),
    (("2020-04-21", "2020-06-20"), "D", "2020-06-20"),
    (("2020-04-21", "2020-06-20"), "D", "2020-06-20"),
]


@pytest.mark.parametrize("types", ["series", "df"])
@pytest.mark.parametrize("ignore_tz", [True, False])
def test_frames_ignore_tz(types: str, ignore_tz: bool):
    idx_a = pd.date_range(
        "2020", "2022", freq="MS", inclusive="left", tz="Europe/Berlin"
    )
    a = pd.Series(range(0, 24), idx_a)
    idx_b = pd.date_range("2020-02", "2021-09", freq="MS", inclusive="left")
    b = pd.Series(range(0, 19), idx_b)

    exp_idx_a = pd.date_range(
        "2020-02", "2021-09", freq="MS", inclusive="left", tz="Europe/Berlin"
    )
    exp_idx_b = idx_b
    exp_a = pd.Series(range(1, 20), exp_idx_a)
    exp_b = pd.Series(range(0, 19), exp_idx_b)

    if types == "series":
        if not ignore_tz:
            with pytest.raises(ValueError):
                _ = tools.intersect.frames(a, b, ignore_tz=ignore_tz)
            return
        result_a, result_b = tools.intersect.frames(a, b, ignore_tz=ignore_tz)
        testing.assert_series_equal(result_a, exp_a)
        testing.assert_series_equal(result_b, exp_b)
    else:
        a, b = pd.DataFrame({"col_a": a}), pd.DataFrame({"col_b": b})
        if not ignore_tz:
            with pytest.raises(ValueError):
                _ = tools.intersect.frames(a, b, ignore_tz=ignore_tz)
            return
        exp_a, exp_b = pd.DataFrame({"col_a": exp_a}), pd.DataFrame({"col_b": exp_b})
        result_a, result_b = tools.intersect.frames(a, b, ignore_tz=ignore_tz)
        testing.assert_frame_equal(result_a, exp_a)
        testing.assert_frame_equal(result_b, exp_b)


def get_idx(
    startdate: str, starttime: str, tz: str, freq: str, enddate: str = COMMON_END
) -> pd.DatetimeIndex:
    # Empty index.
    if startdate is None:
        return pd.DatetimeIndex([], freq=freq, tz=tz)
    # Normal index.
    ts_start = pd.Timestamp(f"{startdate} {starttime}", tz=tz)
    ts_end = pd.Timestamp(f"{enddate} {starttime}", tz=tz)
    return pd.date_range(ts_start, ts_end, freq=freq, inclusive="left")


def get_frames(
    idxs: Iterable[pd.DatetimeIndex], ref_idx: pd.DatetimeIndex = None
) -> List[Union[pd.Series, pd.DataFrame]]:
    frames = []
    for i, idx in enumerate(idxs):
        # Get data.
        if ref_idx is None:
            startnum = 0
            index = idx
        else:
            startnum = idx.get_loc(ref_idx[0]) if len(ref_idx) > 0 else 0
            index = ref_idx
        # Create series.
        fr = pd.Series(range(startnum, startnum + len(index)), index, dtype=int)
        # Possibly turn into dataframe.
        if i % 2 == 0:
            fr = pd.DataFrame({"a": fr})
        frames.append(fr)
    return frames


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("indexorframe", ["idx", "fr"])
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
@pytest.mark.parametrize(("startdates", "freq", "expected_startdate"), TESTCASES)
def test_intersect(
    indexorframe: str,
    startdates: Iterable[str],
    starttime: str,
    tz: str,
    freq: str,
    expected_startdate: str,
):
    """Test if intersection of indices gives correct result."""
    idxs = [get_idx(startdate, starttime, tz, freq) for startdate in startdates]
    do_test_intersect(indexorframe, idxs, expected_startdate, starttime, tz, freq)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("indexorframe", ["idx", "fr"])
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
@pytest.mark.parametrize(("startdates", "freq", "expected_startdate"), TESTCASES)
def test_intersect_distinctfreq(
    indexorframe: str,
    startdates: Iterable[str],
    starttime: str,
    tz: str,
    freq: str,
    expected_startdate: str,
):
    """Test if intersection of indices with distinct frequencies gives correct result."""
    otherfreq = "H" if freq == "D" else "D"
    idxs = [
        get_idx(startdates[0], starttime, tz, freq),
        get_idx(startdates[1], starttime, tz, otherfreq),
    ]
    do_test_intersect(indexorframe, idxs, ValueError)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("indexorframe", ["idx", "fr"])
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
@pytest.mark.parametrize(("startdates", "freq", "expected_startdate"), TESTCASES)
def test_intersect_distincttz(
    indexorframe: str,
    startdates: Iterable[str],
    starttime: str,
    tz: str,
    freq: str,
    expected_startdate: str,
):
    """Test if intersection of indices with distinct timezones gives correct result."""
    othertz = None if tz == "Europe/Berlin" else "Europe/Berlin"
    idxs = [
        get_idx(startdates[0], starttime, tz, freq),
        get_idx(startdates[1], starttime, othertz, freq),
    ]
    do_test_intersect(indexorframe, idxs, ValueError)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("indexorframe", ["idx", "fr"])
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
@pytest.mark.parametrize(("startdates", "freq", "expected_startdate"), TESTCASES)
def test_intersect_distinctstartofday(
    indexorframe: str,
    startdates: Iterable[str],
    starttime: str,
    tz: str,
    freq: str,
    expected_startdate: str,
):
    """Test if intersection of indices with distinct frequencies gives correct result."""
    otherstarttime = "00:00" if starttime == "06:00" else "06:00"
    idxs = [
        get_idx(startdates[0], starttime, tz, freq),
        get_idx(startdates[1], otherstarttime, tz, freq),
    ]
    do_test_intersect(indexorframe, idxs, ValueError)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("indexorframe", ["idx", "fr"])
@pytest.mark.parametrize("freq", tools.freq.FREQUENCIES)
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
def test_intersect_nooverlap(indexorframe: str, tz: str, freq: str, starttime: str):
    """Test if intersection of non-overlapping indices gives correct result."""
    idxs = [
        get_idx("2020-01-01", starttime, tz, freq, "2022-01-01"),
        get_idx("2023-01-01", starttime, tz, freq, "2025-01-01"),
    ]
    do_test_intersect(indexorframe, idxs, None, "", tz, freq)


def do_test_intersect(
    indexorframe: str,
    idxs: Iterable[pd.DatetimeIndex],
    expected_startdate: Union[str, Exception],
    expected_starttime: str = None,
    expected_tz: str = None,
    expected_freq: str = None,
):
    if indexorframe == "idx":
        do_test_fn = do_test_intersect_index
    else:
        do_test_fn = do_test_intersect_frame
    do_test_fn(idxs, expected_startdate, expected_starttime, expected_tz, expected_freq)


def do_test_intersect_index(
    idxs: Iterable[pd.DatetimeIndex],
    expected_startdate: Union[str, Exception],
    expected_starttime: str = None,
    expected_tz: str = None,
    expected_freq: str = None,
):
    # Error case.
    if isinstance(expected_startdate, type) and issubclass(
        expected_startdate, Exception
    ):
        with pytest.raises(expected_startdate):
            tools.intersect.indices(*idxs)
        return
    # Normal case.
    result = tools.intersect.indices(*idxs)
    expected = get_idx(
        expected_startdate, expected_starttime, expected_tz, expected_freq
    )
    testing.assert_index_equal(result, expected)


def do_test_intersect_frame(
    idxs: Iterable[pd.DatetimeIndex],
    expected_startdate: Union[str, Exception],
    expected_starttime: str = None,
    expected_tz: str = None,
    expected_freq: str = None,
):
    frames = get_frames(idxs)

    # Error case.
    if type(expected_startdate) is type and issubclass(expected_startdate, Exception):
        with pytest.raises(expected_startdate):
            tools.intersect.frames(*frames)
        return

    # Normal case.
    result_frames = tools.intersect.frames(*frames)
    expected_index = get_idx(
        expected_startdate, expected_starttime, expected_tz, expected_freq
    )
    expected_frames = get_frames(idxs, expected_index)

    for result, expected in zip(result_frames, expected_frames):
        if isinstance(result, pd.Series):
            testing.assert_series_equal(result, expected)
        else:
            testing.assert_frame_equal(result, expected)
