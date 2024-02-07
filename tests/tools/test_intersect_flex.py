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

COMMON_END_2 = "2023-01-01"
TESTCASES_2 = [  # startdates, freq, expected_dates
    # One starts at first day of year.
    (("2020-01-01", "2020-01-20"), ("15T", "H"), "2020-01-20"),
    (("2020-01-01", "2020-01-20"), ("15T", "D"), "2020-01-20"),
    (("2022-04-01", "2021-02-01"), ("H", "MS"), "2022-04-01"),
    (("2020-01-01", "2020-04-01"), ("H", "QS"), "2020-04-01"),
    (("2020-01-01", "2021-01-01"), ("D", "AS"), "2021-01-01"),
    # Both start in middle of year.
    (("2020-04-21", "2020-06-20"), ("15T", "H"), "2020-06-20"),
    (("2020-04-21", "2020-06-20"), ("15T", "D"), "2020-06-20"),
    (("2020-04-21", "2020-07-01"), ("H", "MS"), "2020-07-01"),
    (("2020-04-21", "2020-07-01"), ("H", "QS"), "2020-07-01"),
    (("2020-04-21", "2021-01-01"), ("D", "AS"), "2021-01-01"),
]


def get_idx(
    startdate: str,
    starttime: str,
    tz: str,
    freq: str,
    enddate: str,
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
@pytest.mark.parametrize(("startdates", "freq", "expected_startdate"), TESTCASES)
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
# @pytest.mark.parametrize("indexorframe", ["idx", "fr"])
def test_intersect_flex_ignore_start_of_day(
    # indexorframe: str,
    startdates: Iterable[str],
    starttime: str,
    tz: str,
    freq: str,
    expected_startdate: str,
):
    otherstarttime = "00:00" if starttime == "06:00" else "06:00"
    idxs = [
        get_idx(
            startdates[0],
            starttime,
            tz,
            freq,
            COMMON_END,
        ),
        get_idx(
            startdates[1],
            otherstarttime,
            tz,
            freq,
            COMMON_END,
        ),
    ]
    do_test_intersect(
        "idx",
        idxs,
        expected_startdate,
        expected_tz=tz,
        expected_freq=freq,
        expected_starttime=starttime,
        expected_otherstarttime=otherstarttime,
        expected_othertz=tz,
        expected_otherfreq=freq,
        enddate=COMMON_END,
        ignore_start_of_day=True,
    )


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
@pytest.mark.parametrize(("startdates", "freq", "expected_startdate"), TESTCASES)
# @pytest.mark.parametrize("indexorframe", ["idx", "fr"])
def test_intersect_flex_ignore_tz(
    # indexorframe: str,
    startdates: Iterable[str],
    starttime: str,
    tz: str,
    freq: str,
    expected_startdate: str,
):
    othertz = None if tz == "Europe/Berlin" else "Europe/Berlin"
    idxs = [
        get_idx(startdates[0], starttime, tz, freq, COMMON_END),
        get_idx(startdates[1], starttime, othertz, freq, COMMON_END),
    ]
    do_test_intersect(
        "idx",
        idxs,
        expected_startdate,
        expected_tz=tz,
        expected_freq=freq,
        expected_starttime=starttime,
        expected_otherstarttime=starttime,
        expected_othertz=othertz,
        expected_otherfreq=freq,
        enddate=COMMON_END,
        ignore_tz=True,
    )


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(("startdates", "freq", "expected_startdate"), TESTCASES_2)
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
def test_intersect_flex_ignore_freq(
    # indexorframe: str,
    startdates: Iterable[str],
    starttime: str,
    tz: str,
    freq: Iterable[str],
    expected_startdate: str,
):
    """Test if intersection of indices with distinct frequencies gives correct result."""

    idxs = [
        get_idx(startdates[0], starttime, tz, freq[0], COMMON_END_2),
        get_idx(startdates[1], starttime, tz, freq[1], COMMON_END_2),
    ]
    do_test_intersect(
        "idx",
        idxs,
        expected_startdate,
        expected_tz=tz,
        expected_freq=freq[0],
        expected_starttime=starttime,
        expected_otherstarttime=starttime,
        expected_othertz=tz,
        expected_otherfreq=freq[1],
        enddate=COMMON_END_2,
        ignore_freq=True,
    )


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(("startdates", "freq", "expected_startdate"), TESTCASES_2)
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
def test_ignore_all(  # indexorframe: str,
    startdates: Iterable[str],
    starttime: str,
    tz: str,
    freq: Iterable[str],
    expected_startdate: str,
):
    otherstarttime = "00:00" if starttime == "06:00" else "06:00"
    othertz = None if tz == "Europe/Berlin" else "Europe/Berlin"
    idxs = [
        get_idx(startdates[0], starttime, tz, freq[0], COMMON_END_2),
        get_idx(startdates[1], otherstarttime, othertz, freq[1], COMMON_END_2),
    ]
    do_test_intersect(
        "idx",
        idxs,
        expected_startdate,
        expected_tz=tz,
        expected_freq=freq[0],
        expected_starttime=starttime,
        expected_otherstarttime=otherstarttime,
        expected_othertz=othertz,
        expected_otherfreq=freq[1],
        enddate=COMMON_END_2,
        ignore_freq=True,
        ignore_start_of_day=True,
        ignore_tz=True,
    )


def do_test_intersect(
    indexorframe: str,
    idxs: Iterable[pd.DatetimeIndex],
    expected_startdate: Union[str, Exception],
    expected_starttime: str = None,
    expected_tz: str = None,
    expected_freq: str = None,
    expected_otherstarttime: str = None,
    expected_othertz: str = None,
    expected_otherfreq: str = None,
    enddate: str = None,
    ignore_start_of_day: bool = False,
    ignore_tz: bool = False,
    ignore_freq: bool = False,
):
    if indexorframe == "idx":
        do_test_intersect_index(
            idxs,
            expected_startdate,
            expected_starttime,
            expected_tz,
            expected_freq,
            expected_otherstarttime,
            expected_othertz,
            expected_otherfreq,
            enddate,
            ignore_start_of_day,
            ignore_tz,
            ignore_freq,
        )
    else:
        do_test_intersect_frame(
            idxs,
            expected_startdate,
            expected_starttime,
            expected_tz,
            expected_freq,
            expected_otherstarttime,
            expected_othertz,
            expected_otherfreq,
            enddate,
            ignore_start_of_day,
            ignore_tz,
            ignore_freq,
        )


def do_test_intersect_index(
    idxs: Iterable[pd.DatetimeIndex],
    expected_startdate: Union[str, Exception],
    expected_starttime: str = None,
    expected_tz: str = None,
    expected_freq: str = None,
    expected_otherstarttime: str = None,
    expected_othertz: str = None,
    expected_otherfreq: str = None,
    enddate: str = None,
    ignore_start_of_day: bool = False,
    ignore_tz: bool = False,
    ignore_freq: bool = False,
):
    # Error case.
    if isinstance(expected_startdate, type) and issubclass(
        expected_startdate, Exception
    ):
        with pytest.raises(expected_startdate):
            tools.intersect.indices_flex(
                *idxs,
                ignore_start_of_day=False,
                ignore_tz=False,
                ignore_freq=ignore_freq,
            )
        return
    # Normal case.
    out_a, out_b = tools.intersect.indices_flex(
        *idxs,
        ignore_start_of_day=ignore_start_of_day,
        ignore_tz=ignore_tz,
        ignore_freq=ignore_freq,
    )
    expected_a = get_idx(
        expected_startdate,
        expected_starttime,
        expected_tz,
        expected_freq,
        enddate,
    )
    expected_b = get_idx(
        expected_startdate,
        expected_otherstarttime,
        expected_othertz,
        expected_otherfreq,
        enddate,
    )
    testing.assert_index_equal(out_a, expected_a)
    testing.assert_index_equal(out_b, expected_b)


def do_test_intersect_frame(
    idxs: Iterable[pd.DatetimeIndex],
    expected_startdate: Union[str, Exception],
    expected_starttime: str = None,
    expected_tz: str = None,
    expected_freq: str = None,
    expected_otherstarttime: str = None,
    expected_othertz: str = None,
    expected_otherfreq: str = None,
    enddate: str = None,
    ignore_start_of_day: bool = False,
    ignore_tz: bool = False,
    ignore_freq: bool = False,
):
    frames = get_frames(idxs)

    # Error case.
    if type(expected_startdate) is type and issubclass(expected_startdate, Exception):
        with pytest.raises(expected_startdate):
            tools.intersect.frames(*frames)
        return

    # Normal case.

    result_frames = tools.intersect.frames(
        *frames,
        ignore_freq=ignore_freq,
        ignore_start_of_day=ignore_start_of_day,
        ignore_tz=ignore_tz,
    )
    expected_index = get_idx(
        expected_startdate,
        expected_starttime,
        expected_tz,
        expected_freq,
        enddate,
    )
    expected_frames = get_frames(idxs, expected_index)
    for result, expected in zip(result_frames, expected_frames):
        if isinstance(result, pd.Series):
            testing.assert_series_equal(result, expected)
        else:
            testing.assert_frame_equal(result, expected)
