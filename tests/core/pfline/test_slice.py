"""Test if slice attributes works properly with portfolio line."""

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
def test_flat_slice(slice_start, slice_end, freq):
    index = pd.date_range("2020", "2024", freq=freq)
    pfl1 = dev.get_flatpfline(index)

    pfls_to_concat = [pfl1.slice[: slice_end[0]], pfl1.slice[slice_start:]]
    pfls_to_concat2 = [pfl1.loc[: slice_end[1]], pfl1.loc[slice_start:]]
    assert pfls_to_concat == pfls_to_concat2


@pytest.mark.parametrize("freq", ["MS", "AS", "QS", "D", "15T"])
@pytest.mark.parametrize("slice_start", ["2021", "2022", "2022-01-02"])
@pytest.mark.parametrize("children", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "slice_end",
    [
        # (<param for slice>, <param for loc>)
        ("2021", "2020"),
        ("2022", "2021"),
        ("2022-01-02", "2022-01-01"),
    ],
)
def test_nested_slice(slice_start, slice_end, freq, children):
    index = pd.date_range("2020", "2024", freq=freq)
    pfl1 = dev.get_nestedpfline(index, childcount=children)

    pfls_to_concat = [pfl1.slice[: slice_end[0]], pfl1.slice[slice_start:]]
    pfls_to_concat2 = [pfl1.loc[: slice_end[1]], pfl1.loc[slice_start:]]
    assert pfls_to_concat == pfls_to_concat2
