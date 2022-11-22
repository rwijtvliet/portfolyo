import numpy as np
import pytest
from portfolyo import tools
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
        Q_(1, ureg.megawatthour),
        Q_(1, ureg.MWh),
        Q_(1, "MWh"),  # most common constructor
        ureg("1 MWH"),
        1 * ureg.megawatthour,
        1 * ureg.MWh,  # most common constructor
        1 * ureg("MWh"),
        Q_(1000, ureg.kilowatthour),
        Q_(1000, ureg.kWh),
        Q_(1000, "kWh"),
        ureg("1000 kWh"),
        1000 * ureg.kilowatthour,
        1000 * ureg.kWh,
        1000 * ureg("kWh"),
        Q_(1e6, "Wh"),
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
    (Q_(23, "MW"), Q_("46 MWh") / (2 * ureg.h), Q_(23, "MWh") / ureg.h),
    (30 * ureg.euro / (40 * ureg.kWh), 750 * ureg.euro_per_MWh, 75 * ureg.cent_per_kWh),
    (30 * ureg.MWh / (40 * ureg.kWh), 750),
    (30 * ureg.kWh / (40e6 * ureg.W * 3600 * ureg.s), 0.75e-3),
]


@pytest.mark.parametrize("quants", QUANTITY_IDENTITIES)
def test_extended_identities(quants):
    for q in quants:
        assert np.isclose(q, quants[0])
