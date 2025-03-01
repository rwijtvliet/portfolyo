"""Test if slice attributes works properly with portfolio line."""

import pandas as pd
import pytest

from portfolyo import dev


def get_idx(
    startdate: str,
    starttime: str,
    enddate: str,
    tz: str,
    freq: str,
    inclusive: str,
) -> pd.DatetimeIndex:
    # Empty index.
    if startdate is None:
        return pd.DatetimeIndex([], freq=freq, tz=tz)
    # Normal index.
    ts_start = pd.Timestamp(f"{startdate} {starttime}", tz=tz)
    ts_end = pd.Timestamp(f"{enddate} {starttime}", tz=tz)
    return pd.date_range(ts_start, ts_end, freq=freq, inclusive=inclusive)


@pytest.mark.parametrize("freq", ["MS", "YS", "YS-APR", "QS", "QS-FEB", "D"])
@pytest.mark.parametrize("slice_start", ["2021", "2022", "2022-01-02", "2022-05-23 14:34"])
@pytest.mark.parametrize("sod", ["00:00", "06:00"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("inclusive", ["left", "both"])
def test_flat_slice_start(slice_start: str, freq: str, tz: str, sod: str, inclusive: str):
    index = get_idx("2020", starttime=sod, enddate="2024", freq=freq, inclusive=inclusive, tz=tz)
    pfl1 = dev.get_flatpfline(index)
    assert pfl1.slice[slice_start:] == pfl1.loc[slice_start:]


@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("freq", ["MS", "YS", "YS-APR", "QS", "QS-MAR", "D"])
@pytest.mark.parametrize(
    "slice_end",
    [
        # (<param for slice>, <param for loc>)
        ("2021", "2020"),
        ("2022", "2021"),
        ("2021-07", "2021-06"),
        ("2022-01-02", "2022-01-01"),
    ],
)
@pytest.mark.parametrize("sod", ["00:00", "06:00"])
@pytest.mark.parametrize("inclusive", ["left", "both"])
def test_flat_slice_end(slice_end: str, freq: str, tz: str, sod: str, inclusive: str):
    index = get_idx("2020", starttime=sod, enddate="2024", freq=freq, inclusive=inclusive, tz=tz)
    pfl1 = dev.get_flatpfline(index)
    assert pfl1.slice[: slice_end[0]] == pfl1.loc[: slice_end[1]]


@pytest.mark.parametrize("freq", ["MS", "YS", "YS-APR", "QS", "QS-MAR", "D"])
@pytest.mark.parametrize(
    "where",
    ["2022", "2022-03", "2022-04-21", "2022-05-23 14:34"],
)
@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("sod", ["00:00", "06:00"])
@pytest.mark.parametrize("inclusive", ["left", "both"])
def test_flat_slice_whole(where: str, freq: str, tz: str, sod: str, inclusive: str):
    """Test that slicing splits the pfl in 2 non-overlapping pieces without gap
    (i.e., ensure that each original timestamp is in exactly one of the resulting pieces.)
    """
    index = get_idx("2020", starttime=sod, enddate="2024", freq=freq, inclusive=inclusive, tz=tz)
    pfl1 = dev.get_flatpfline(index)
    left, right = pfl1.slice[:where], pfl1.slice[where:]
    # Test that each timestamp is present at least once.
    pd.testing.assert_index_equal(left.index.union(right.index), index)
    # Test that no timestamp is present twice.
    assert len(left.index.intersection(right.index)) == 0


@pytest.mark.parametrize("freq", ["MS", "YS", "YS-APR", "QS", "QS-MAR", "D"])
@pytest.mark.parametrize(
    "slice_start",
    [
        "2021",
        "2022",
        "2022-01-02",
        "2022-01-02 14:00",
    ],
)
@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("sod", ["00:00", "06:00"])
@pytest.mark.parametrize("inclusive", ["left", "both"])
def test_nested_slice_start(slice_start: str, freq: str, tz: str, sod: str, inclusive: str):
    index = get_idx("2020", starttime=sod, enddate="2024", freq=freq, inclusive=inclusive, tz=tz)
    pfl1 = dev.get_nestedpfline(index)
    assert pfl1.slice[slice_start:] == pfl1.loc[slice_start:]


@pytest.mark.parametrize("freq", ["MS", "YS", "YS-APR", "QS", "QS-MAR", "D"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize(
    "slice_end",
    [
        # (<param for slice>, <param for loc>)
        ("2021", "2020"),
        ("2022", "2021"),
        ("2021-07", "2021-06"),
        ("2022-01-02", "2022-01-01"),
    ],
)
@pytest.mark.parametrize("sod", ["00:00", "06:00"])
@pytest.mark.parametrize("inclusive", ["left", "both"])
def test_nested_slice_end(slice_end: str, freq: str, tz: str, sod: str, inclusive: str):
    index = get_idx("2020", starttime=sod, enddate="2024", freq=freq, inclusive=inclusive, tz=tz)
    pfl1 = dev.get_nestedpfline(index)
    assert pfl1.slice[: slice_end[0]] == pfl1.loc[: slice_end[1]]


@pytest.mark.parametrize("freq", ["MS", "YS", "YS-APR", "QS", "QS-MAR", "D"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize(
    "where",
    ["2022", "2022-03", "2022-04-21", "2022-05-23 14:34"],
)
@pytest.mark.parametrize("sod", ["00:00", "06:00"])
@pytest.mark.parametrize("inclusive", ["left", "both"])
def test_nested_slice_whole(where: str, freq: str, tz: str, sod: str, inclusive: str):
    """Test that slicing splits the pfl in 2 non-overlapping pieces without gap
    (i.e., ensure that each original timestamp is in exactly one of the resulting pieces.)
    """
    index = get_idx("2020", starttime=sod, enddate="2024", freq=freq, inclusive=inclusive, tz=tz)
    pfl1 = dev.get_nestedpfline(index)
    left, right = pfl1.slice[:where], pfl1.slice[where:]
    # Test that each timestamp is present at least once.
    pd.testing.assert_index_equal(left.index.union(right.index), index)
    # Test that no timestamp is present twice.
    assert len(left.index.intersection(right.index)) == 0


@pytest.mark.parametrize("freq", ["h", "15min"])
@pytest.mark.parametrize("sod", ["00:00", "06:00"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("startdate", ["2021", "2022", "2022-01-02"])
def test__start_less_than_daily(startdate: str, freq: str, tz: str, sod: str):
    index = get_idx("2020", starttime=sod, enddate="2024", freq=freq, inclusive="left", tz=tz)
    pfl1 = dev.get_flatpfline(index)
    slice_start = f"{startdate} {sod}"
    assert pfl1.slice[slice_start:] == pfl1.loc[slice_start:]


@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize(
    "enddate",
    [
        # (<param for slice>, <param for loc>)
        ("2021", "2020-12-31"),
        ("2022", "2021-12-31"),
        ("2021-07", "2021-06-30"),
        ("2022-01-02", "2022-01-01"),
    ],
)
def test__end_less_than_daily(enddate: str, tz: str):
    index = get_idx(
        "2020", starttime="00:00", enddate="2024", freq="15min", inclusive="left", tz=tz
    )
    pfl1 = dev.get_flatpfline(index)
    slice_end = f"{enddate[0]} 00:00"
    loc_end = f"{enddate[1]} 23:45"
    assert pfl1.slice[:slice_end] == pfl1.loc[:loc_end]
