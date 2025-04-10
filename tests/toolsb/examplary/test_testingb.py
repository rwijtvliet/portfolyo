import numpy as np
import pytest

from portfolyo import toolsb
from portfolyo.toolsb.unit import Q_


@pytest.mark.parametrize(
    "equals",
    [
        (5, 5.0),
        (np.nan, np.nan),
        (Q_(5, "MW"), Q_(5.0, "MW"), Q_(0.005, "GW"), Q_(5000, "kW"), Q_(5, "MWh/h")),
        (Q_(np.nan, "MW"), Q_(np.nan, "MW")),
    ],
)
def test_assertscalarequal(equals):
    for e1 in equals:
        for e2 in equals:
            toolsb.testing.assert_scalar_equal(e1, e2)


@pytest.mark.parametrize(
    "unequals",
    [(5, 5.1), (Q_(5, "Eur"), Q_(5, "Usd")), (Q_(np.nan, "MW"), Q_(np.nan, "MWh"))],
)
def test_assertscalarequal_nok(unequals):
    for i1, u1 in enumerate(unequals):
        for i2, u2 in enumerate(unequals):
            if i1 == i2:
                continue
            with pytest.raises(AssertionError):
                toolsb.testing.assert_scalar_equal(u1, u2)
