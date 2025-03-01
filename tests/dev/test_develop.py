"""Simply test if functions don't raise an error"""

import datetime as dt

import pytest

from portfolyo import Kind, dev
from portfolyo.core.pfline import classes


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("freq", "startdate", "start_of_day"),
    [
        (None, None, None),
        ("h", "2020", None),
        ("D", "2020", None),
        ("MS", "2020", None),
        ("QS", "2020", None),
        ("QS-MAR", "2020", None),
        ("YS", "2020", None),
        ("YS-APR", "2020", None),
        ("h", "2020-04-21", dt.time(hour=15)),
        ("D", "2020-04-21", None),
        ("MS", "2020-04", None),
        ("QS", "2020-04", None),
        ("QS-MAR", "2020-03", None),
        ("YS-APR", "2020-04", None),
    ],
)
def test_index(freq, tz, startdate, start_of_day):
    """Test index creation."""
    _ = dev.get_index() if freq is None else dev.get_index(freq, tz, startdate)


@pytest.mark.parametrize("request_unit", [True, False])
@pytest.mark.parametrize(
    ("name", "name_has_unit"),
    [("q", True), ("w", True), ("r", True), ("p", True), ("x", False)],
)
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("freq", "startdate", "start_of_day"),
    [
        (None, None, None),
        ("h", "2020-04-21", dt.time(hour=15)),
        ("D", "2020-04-21", None),
        ("MS", "2020-04", None),
        ("QS", "2020-04", None),
        ("QS-MAR", "2020", None),
        ("QS-MAR", "2020-03", None),
        ("YS", "2020", None),
        ("YS-APR", "2020", None),
    ],
)
def test_series(freq, tz, startdate, name, name_has_unit, request_unit, start_of_day):
    """Test series creation."""
    i = None if freq is None else dev.get_index(freq, tz, startdate, start_of_day=start_of_day)
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
    ("freq", "startdate", "start_of_day"),
    [
        (None, None, None),
        ("h", "2020-04-21", dt.time(hour=15)),
        ("D", "2020-04-21", None),
        ("MS", "2020-04", None),
        ("QS", "2020-04", None),
        ("QS-MAR", "2020", None),
        ("QS-MAR", "2020-03", None),
        ("YS", "2020", None),
        ("YS-APR", "2020", None),
        ("YS-APR", "2020-04", None),
    ],
)
def test_dataframe(freq, tz, startdate, cols, first_col_has_unit, request_unit, start_of_day):
    """Test dataframe creation."""
    i = None if freq is None else dev.get_index(freq, tz, startdate, start_of_day=start_of_day)
    df = dev.get_dataframe(i, cols, request_unit)
    unit = df.pint.dequantify().columns.get_level_values("unit")[0]
    if request_unit and first_col_has_unit:
        assert unit != "No Unit"
    else:
        assert unit == "No Unit"


@pytest.mark.parametrize("kind", [Kind.COMPLETE, Kind.VOLUME, Kind.PRICE])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("freq", "startdate", "start_of_day"),
    [
        (None, None, None),
        ("h", "2020-04-21", dt.time(hour=15)),
        ("D", "2020-04-21", None),
        ("MS", "2020-04", None),
        ("QS", "2020-04", None),
        ("QS-MAR", "2020", None),
        ("QS-MAR", "2020-03", None),
        ("YS", "2020", None),
        ("YS-APR", "2020", None),
        ("YS-APR", "2020-04", None),
    ],
)
def test_flatnestedpfline(freq, tz, startdate, kind, start_of_day):
    """Test flatpfline and nestedpfline creation."""
    i = None if freq is None else dev.get_index(freq, tz, startdate, start_of_day=start_of_day)
    _ = dev.get_flatpfline(i, kind)
    _ = dev.get_nestedpfline(i, kind)


@pytest.mark.parametrize("max_nlevels", [1, 2, 3])
@pytest.mark.parametrize("kind", [Kind.COMPLETE, Kind.VOLUME, Kind.PRICE])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("freq", "startdate", "start_of_day"),
    [
        (None, None, None),
        ("h", "2020-04-21", dt.time(hour=15)),
        ("D", "2020-04-21", None),
        ("MS", "2020-04", None),
        ("QS", "2020-04", None),
        ("QS-MAR", "2020-03", None),
        ("YS", "2020", None),
        ("YS-APR", "2020", None),
        ("YS-APR", "2020-04", None),
    ],
)
def test_pfline(freq, tz, startdate, kind, max_nlevels, start_of_day):
    """Test that pfline can be created."""
    i = None if freq is None else dev.get_index(freq, tz, startdate, start_of_day=start_of_day)
    pfl = dev.get_randompfline(i, kind, max_nlevels)
    if max_nlevels == 1:
        assert isinstance(pfl, classes.FlatPfLine)
