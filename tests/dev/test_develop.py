"""Simply test if functions don't raise an error"""

import pandas as pd
import pytest

from portfolyo import Kind, dev
from portfolyo.core.pfline import classes


@pytest.mark.parametrize("as_str", [True, False])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("freq", "start"),
    [
        (None, None),
        ("H", "2020"),
        ("D", "2020"),
        ("MS", "2020"),
        ("QS", "2020"),
        ("AS", "2020"),
        ("H", "2020-04-21 15:00"),
        ("D", "2020-04-21"),
        ("MS", "2020-04"),
        ("QS", "2020-04"),
    ],
)
def test_index(freq, tz, start, as_str):
    """Test index creation."""
    if not as_str:
        start = pd.Timestamp(start)
    _ = dev.get_index() if freq is None else dev.get_index(freq, tz, start)


@pytest.mark.parametrize("request_unit", [True, False])
@pytest.mark.parametrize(
    ("name", "name_has_unit"),
    [("q", True), ("w", True), ("r", True), ("p", True), ("x", False)],
)
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("freq", "start"),
    [
        (None, None),
        ("H", "2020-04-21 15:00"),
        ("D", "2020-04-21"),
        ("MS", "2020-04"),
        ("QS", "2020-04"),
        ("AS", "2020"),
    ],
)
def test_series(freq, tz, start, name, name_has_unit, request_unit):
    """Test series creation."""
    i = None if freq is None else dev.get_index(freq, tz, start)
    s = dev.get_series(i, name, request_unit)
    if request_unit and name_has_unit:
        _ = s.pint.magnitude
    else:
        with pytest.raises((ValueError, AttributeError)):
            _ = s.pint.magnitude


@pytest.mark.parametrize("request_unit", [True, False])
@pytest.mark.parametrize(
    ("cols", "first_col_has_unit"),
    [
        ("q", True),
        ("wq", True),
        ("rp", True),
        ("prwq", True),
        ("qx", True),
        ("xq", False),
        ("xy", False),
    ],
)
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("freq", "start"),
    [
        (None, None),
        ("H", "2020-04-21 15:00"),
        ("D", "2020-04-21"),
        ("MS", "2020-04"),
        ("QS", "2020-04"),
        ("AS", "2020"),
    ],
)
def test_dataframe(freq, tz, start, cols, first_col_has_unit, request_unit):
    """Test dataframe creation."""
    i = None if freq is None else dev.get_index(freq, tz, start)
    df = dev.get_dataframe(i, cols, request_unit)
    unit = df.pint.dequantify().columns.get_level_values("unit")[0]
    if request_unit and first_col_has_unit:
        assert unit != "No Unit"
    else:
        assert unit == "No Unit"


@pytest.mark.parametrize("kind", [Kind.COMPLETE, Kind.VOLUME, Kind.PRICE])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("freq", "start"),
    [
        (None, None),
        ("H", "2020-04-21 15:00"),
        ("D", "2020-04-21"),
        ("MS", "2020-04"),
        ("QS", "2020-04"),
        ("AS", "2020"),
    ],
)
def test_flatnestedpfline(freq, tz, start, kind):
    """Test flatpfline and nestedpfline creation."""
    i = None if freq is None else dev.get_index(freq, tz, start)
    _ = dev.get_flatpfline(i, kind)
    _ = dev.get_nestedpfline(i, kind)


@pytest.mark.parametrize("max_nlevels", [1, 2, 3])
@pytest.mark.parametrize("kind", [Kind.COMPLETE, Kind.VOLUME, Kind.PRICE])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("freq", "start"),
    [
        (None, None),
        ("H", "2020-04-21 15:00"),
        ("D", "2020-04-21"),
        ("MS", "2020-04"),
        ("QS", "2020-04"),
        ("AS", "2020"),
    ],
)
def test_pfline(freq, tz, start, kind, max_nlevels):
    """Test that pfline can be created."""
    i = None if freq is None else dev.get_index(freq, tz, start)
    pfl = dev.get_randompfline(i, kind, max_nlevels)
    if max_nlevels == 1:
        assert isinstance(pfl, classes.FlatPfLine)
