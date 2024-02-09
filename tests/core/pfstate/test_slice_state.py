"""Test if slice attributes works properly with portfolio state."""

import pytest
import pandas as pd
from portfolyo import dev


@pytest.mark.parametrize("freq", ["MS", "AS", "QS", "D", "15T"])
@pytest.mark.parametrize("slice_start", ["2021", "2022", "2022-01-02"])
@pytest.mark.parametrize(
    "slice_end",
    [
        # (<param for slice>, <param for loc>)
        ("2021", "2020"),
        ("2022", "2021"),
        ("2022-01-02", "2022-01-01"),
    ],
)
def test_slice_state(slice_start, slice_end, freq):
    index = pd.date_range("2020", "2024", freq=freq)
    pfs = dev.get_pfstate(index)

    pfs_to_concat = [pfs.slice[: slice_end[0]], pfs.slice[slice_start:]]
    pfs_to_concat2 = [pfs.loc[: slice_end[1]], pfs.loc[slice_start:]]
    assert pfs_to_concat == pfs_to_concat2


@pytest.mark.parametrize("freq", ["MS", "AS", "QS", "D", "15T"])
@pytest.mark.parametrize(
    "slice_start",
    [
        "2021",
        "2022",
        "2022-01-02",
        "2022-01-02 14:00",
    ],
)
def test_state_slice_start(slice_start, freq):
    index = pd.date_range("2020", "2024", freq=freq, inclusive="left")
    pfs = dev.get_pfstate(index)
    assert pfs.slice[slice_start:] == pfs.loc[slice_start:]


@pytest.mark.parametrize("freq", ["MS", "AS", "QS", "D", "15T"])
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
def test_state_slice_end(slice_end, freq):
    index = pd.date_range("2020", "2024", freq=freq, inclusive="left")
    pfs = dev.get_pfstate(index)
    assert pfs.slice[: slice_end[0]] == pfs.loc[: slice_end[1]]


@pytest.mark.parametrize("freq", ["MS", "AS", "QS", "D", "15T"])
@pytest.mark.parametrize(
    "where",
    ["2022", "2022-03", "2022-04-21", "2022-05-23 14:34"],
)
def test_state_slice_whole(where: str, freq: str):
    """Test that slicing splits the pfl in 2 non-overlapping pieces without gap
    (i.e., ensure that each original timestamp is in exactly one of the resulting pieces.)
    """
    index = pd.date_range("2020", "2024", freq=freq, inclusive="left")
    pfs = dev.get_pfstate(index)
    left, right = pfs.slice[:where], pfs.slice[where:]
    # Test that each timestamp is present at least once.
    pd.testing.assert_index_equal(left.index.union(right.index), index)
    # Test that no timestamp is present twice.
    assert len(left.index.intersection(right.index)) == 0
