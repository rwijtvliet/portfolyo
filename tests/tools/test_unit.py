import numpy as np
import pytest
import pint

from portfolyo import tools
import pandas as pd
from portfolyo.tools.testing import assert_series_equal
from portfolyo.tools.unit import Q_, ureg

UNIT_IDENTITIES = [
    ("MWh", "MWH", "megawatthour"),
    ("kWh", "KWH", "kilowatthour"),
    ("J", "Joule"),
    ("W", "Watt"),
    ("euro", "EUR", "Eur"),
    ("cent", "eurocent", "cents", "cEur"),
    ("Eur/MWh", "Eur / MWH", "euro / megawatthour"),
]


@pytest.mark.parametrize("units", UNIT_IDENTITIES)
def test_units_consistent(units):
    unit = tools.unit.ureg.Unit(units[0])
    for u in units:
        assert tools.unit.ureg.Unit(u) == unit


@pytest.mark.parametrize("units", UNIT_IDENTITIES)
def test_quantities_consistent(units):
    quantity = tools.unit.Q_(1.4, units[0])
    for u in units:
        assert tools.unit.Q_(1.4, u) == quantity


QUANTITY_IDENTITIES = [
    (
        Q_(1.0, ureg.megawatthour),
        Q_(1.0, ureg.MWh),
        Q_(1.0, "MWh"),  # most common constructor
        ureg("1 MWH"),
        1 * ureg.megawatthour,
        1 * ureg.MWh,  # most common constructor
        1 * ureg("MWh"),
        Q_(1000.0, ureg.kilowatthour),
        Q_(1000.0, ureg.kWh),
        Q_(1000.0, "kWh"),
        ureg("1000 kWh"),
        1000 * ureg.kilowatthour,
        1000 * ureg.kWh,
        1000 * ureg("kWh"),
        Q_(1.0e6, "Wh"),
        1e6 * ureg.Wh,
        1e-3 * ureg.GWh,
        1e-6 * ureg.TWh,
        3.6 * ureg.GJ,
        3600 * ureg.MJ,
        3.6e6 * ureg.kJ,
        3.6e9 * ureg.J,
        1 * ureg.MW * ureg.h,
        1 * ureg.kW * 1000 * ureg.min * 60,
    ),
    (Q_(23.0, "MW"), Q_("46 MWh") / (2 * ureg.h), Q_(23.0, "MWh") / ureg.h),
    (
        30.0 * ureg.euro / (40.0 * ureg.kWh),
        750.0 * ureg.euro_per_MWh,
        75.0 * ureg.cent_per_kWh,
    ),
    (30.0 * ureg.MWh / (40.0 * ureg.kWh), 750),
    (30.0 * ureg.kWh / (40.0e6 * ureg.W * 3600 * ureg.s), 0.75e-3),
]


@pytest.mark.parametrize("quants", QUANTITY_IDENTITIES)
def test_extended_identities(quants):
    for q in quants:
        assert np.isclose(q, quants[0])


# Normalize.


@pytest.fixture(params=[10, 10.0, -15, -15.0])
def intfloat(request):
    return request.param


@pytest.fixture(params=[[10, 10, -15], [-15.0, 200, -18]])
def intfloats(request):
    return request.param


@pytest.fixture(params=["Eur/MWh", "MWh", "tce/h"])
def units(request):
    return request.param


@pytest.fixture(params=[["Eur/MWh", "MWh", "tce/h"]])
def units_mixeddim(request):
    return request.param


@pytest.fixture(params=[[("MWh", 1), ("GWh", 1000), ("TWh", 1e6)]])
def units_onedim(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(pd.date_range("2020", freq="h", periods=3), id="hourlyindex"),
        pytest.param(pd.date_range("2020", freq="MS", periods=3), id="monthlyindex"),
        pytest.param([0, 1, 2], id="integerindex"),
    ]
)
def index(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(tools.unit.normalize_value, id="normalize_value"),
        pytest.param(tools.unit.normalize, id="normalize"),
    ]
)
def fn_to_test_value(request):
    return request.param


@pytest.fixture(
    params=[pytest.param(True, id="strict"), pytest.param(False, id="notstrict")]
)
def strict(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(tools.unit.normalize_frame, id="normalize_frame"),
        pytest.param(tools.unit.normalize, id="normalize"),
    ]
)
def fn_to_test_frame_part1(request):
    return request.param


@pytest.fixture
def fn_to_test_frame(fn_to_test_frame_part1, strict):
    def partial(fr):
        return fn_to_test_frame_part1(fr, strict)

    return partial


@pytest.fixture
def floatt(intfloat):
    return float(intfloat)


@pytest.fixture
def floatts(intfloats):
    return [float(v) for v in intfloats]


@pytest.fixture
def intfloat_quantity(intfloat, units):
    return Q_(intfloat, units)


@pytest.fixture
def float_quantity(floatt, units):
    return Q_(floatt, units)


@pytest.fixture
def intfloat_series(intfloats, index):
    return pd.Series(intfloats, index)


@pytest.fixture
def float_series(floatts, index):
    return pd.Series(floatts, index)


@pytest.fixture
def quantity_series(intfloats, index, units):
    return pd.Series([Q_(v, units) for v in intfloats], index)


@pytest.fixture
def dimensionless_quantity_series(intfloats, index):
    return pd.Series([Q_(v, "") for v in intfloats], index)


@pytest.fixture
def pint_series(floatts, index, units):
    return pd.Series(floatts, index).astype(f"pint[{units}]")


@pytest.fixture
def onedim_quantity_series(intfloats, index, units_onedim):
    return pd.Series([Q_(v, u[0]) for v, u in zip(intfloats, units_onedim)], index)


@pytest.fixture
def onedim_pint_series(floatts, index, units_onedim):
    return pd.Series([v * u[1] for v, u in zip(floatts, units_onedim)], index).astype(
        f"pint[{units_onedim[0][0]}]"
    )


@pytest.fixture
def mixeddim_quantity_series(intfloats, index, units_mixeddim):
    return pd.Series([Q_(v, u) for v, u in zip(intfloats, units_mixeddim)], index)


def test_normalize_value_intfloat(intfloat, floatt, fn_to_test_value):
    result = fn_to_test_value(intfloat)
    expected = floatt

    assert result == expected


def test_normalize_value_quantity(intfloat_quantity, float_quantity, fn_to_test_value):
    result = fn_to_test_value(intfloat_quantity)
    expected = float_quantity

    assert result == expected
    assert result.magnitude == expected.magnitude
    assert result.units == expected.units


def test_normalize_series_intfloat(intfloat_series, float_series, fn_to_test_frame):
    result = fn_to_test_frame(intfloat_series)
    expected = float_series

    assert_series_equal(result, expected)


def test_normalize_series_quantities(quantity_series, pint_series, fn_to_test_frame):
    result = fn_to_test_frame(quantity_series)
    expected = pint_series

    assert_series_equal(result, expected)


def test_normalize_series_dimensionlessquantities(
    dimensionless_quantity_series, float_series, fn_to_test_frame
):
    result = fn_to_test_frame(dimensionless_quantity_series)
    expected = float_series

    assert_series_equal(result, expected)


def test_normalize_series_onedimquantities(
    onedim_quantity_series, onedim_pint_series, fn_to_test_frame
):
    result = fn_to_test_frame(onedim_quantity_series)
    expected = onedim_pint_series

    assert_series_equal(result, expected)


def test_normalize_series_mixeddimquantities(
    mixeddim_quantity_series, fn_to_test_frame, strict
):
    if strict:
        with pytest.raises(pint.DimensionalityError):
            _ = fn_to_test_frame(mixeddim_quantity_series)
        return

    result = fn_to_test_frame(mixeddim_quantity_series)
    expected = mixeddim_quantity_series

    assert_series_equal(result, expected)


# TODO: add dataframe tests.
# TODO: add nanvalus
