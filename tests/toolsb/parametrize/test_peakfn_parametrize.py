import datetime as dt
from typing import Iterable

import pytest

from portfolyo import toolsb


@pytest.mark.parametrize(
    ("peak_left", "peak_right", "isoweekdays"),
    [
        (dt.time(hour=0), dt.time(hour=0), [1, 2, 3, 4, 5, 6]),
        (dt.time(hour=1), None, [1, 2, 3, 4, 5, 6, 7]),
        (None, None, None),
    ],
)
def test_functioncreation_ok(
    peak_left: dt.time | None, peak_right: dt.time | None, isoweekdays: Iterable[int]
):
    """Test if function creation in OK cases."""
    _ = toolsb.peakfn.factory(peak_left, peak_right, isoweekdays)


@pytest.mark.parametrize(
    ("peak_left", "peak_right", "isoweekdays"),
    [
        (None, None, [1, 2, 3, 4, 5, 6, 7]),
        (dt.time(hour=0), dt.time(hour=0), [1, 2, 3, 4, 5, 6, 7]),
    ],
)
def test_functioncreation_nok(
    peak_left: dt.time | None, peak_right: dt.time | None, isoweekdays: Iterable[int]
):
    """Test if an error is raised when creating impossible functions."""
    with pytest.raises(ValueError):
        _ = toolsb.peakfn.factory(peak_left, peak_right, isoweekdays)
