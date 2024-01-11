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
