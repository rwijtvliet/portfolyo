"""Units, applied to physical dimensions and portfolyo colums."""

from typing import Callable, Iterable, Literal

import numpy as np
import pandas as pd
import pint

from . import index as tools_index
from . import testing as tools_testing
from .types import Col, FloatSeries, IntSeries, PintSeries
from .unit import get_basedimty, ureg

# =========================
# Physical characterization
# =========================


class _Dimty:  # dimensionality
    @classmethod
    def dimensionless_dim(cls) -> pint.util.UnitsContainer:
        return ureg.get_dimensionality("")

    @classmethod
    def time_dim(cls) -> pint.util.UnitsContainer:
        return ureg.get_dimensionality("[time]")

    @classmethod
    def energy_dims(cls) -> set[pint.util.UnitsContainer]:
        return {ureg.get_dimensionality("[energy]")}

    @classmethod
    def emissions_dims(cls) -> set[pint.util.UnitsContainer]:
        return {ureg.get_dimensionality("[emissions]")}

    @classmethod
    def quantity_dims(cls) -> set[pint.util.UnitsContainer]:
        return cls.energy_dims() | cls.emissions_dims()

    @classmethod
    def energyrate_dims(cls) -> set[pint.util.UnitsContainer]:
        return {energy_dim / cls.time_dim() for energy_dim in cls.energy_dims()}

    @classmethod
    def emissionsrate_dims(cls) -> set[pint.util.UnitsContainer]:
        return {emissions_dim / cls.time_dim() for emissions_dim in cls.emissions_dims()}

    @classmethod
    def quantityrate_dims(cls) -> set[pint.util.UnitsContainer]:
        return cls.energyrate_dims() | cls.emissionsrate_dims()

    @classmethod
    def currency_dims(cls) -> set[pint.util.UnitsContainer]:
        return {
            ureg.get_dimensionality(dim)
            for dim, defn in ureg._dimensions.items()
            if defn.is_base and dim.startswith("[currency_")
        }

    @classmethod
    def energyprice_dims(cls) -> set[pint.util.UnitsContainer]:
        return {c / e for e in cls.energy_dims() for c in cls.currency_dims()}

    @classmethod
    def emissionsprice_dims(cls) -> set[pint.util.UnitsContainer]:
        return {c / e for e in cls.emissions_dims() for c in cls.currency_dims()}

    @classmethod
    def quantityprice_dims(cls) -> set[pint.util.UnitsContainer]:
        return cls.energyprice_dims() | cls.emissionsprice_dims()

    @classmethod
    def exchangerate_dims(cls) -> set[pint.util.UnitsContainer]:
        return {c1 / c2 for c1 in cls.currency_dims() for c2 in cls.currency_dims() if c1 != c2}


def _create_validation_fn(
    wanted_dimties: Iterable[pint.util.UnitsContainer], wanted_dimty_name: str
):
    def validate(
        obj: pint.util.UnitsContainer | pint.Unit | pint.Quantity | pd.Series | str | float | int,
    ) -> None:
        f"""Validate if object is a {wanted_dimty_name} value; raise ValueError if not."""
        if (dimty := get_basedimty(obj)) not in wanted_dimties:
            raise ValueError(f"This is not a {wanted_dimty_name} value: {obj} ({dimty}).")

    return validate


validate_is_dimless = _create_validation_fn(
    {_Dimty.dimensionless_dim()},
    "dimensionless",
)
validate_is_energy = _create_validation_fn(
    _Dimty.energy_dims(),
    "energy",
)
validate_is_emissions = _create_validation_fn(
    _Dimty.emissions_dims(),
    "emissions",
)
validate_is_quantity = _create_validation_fn(
    _Dimty.quantity_dims(),
    "quantity",
)
validate_is_energyrate = _create_validation_fn(
    _Dimty.energyrate_dims(),
    "energy rate (i.e., power = energy per unit of time)",
)
validate_is_emissionsrate = _create_validation_fn(
    _Dimty.emissionsrate_dims(),
    "emissions rate (i.e., emissions per unit of time)",
)
validate_is_quantityrate = _create_validation_fn(
    _Dimty.quantityrate_dims(),
    "quantity rate (i.e., energy per unit of time, or emissions per unit of time)",
)
validate_is_currency = _create_validation_fn(
    _Dimty.currency_dims(),
    "currency",
)
validate_is_energyprice = _create_validation_fn(
    _Dimty.energyprice_dims(),
    "energy price (i.e., energy per unit of currency)",
)
validate_is_emissionsprice = _create_validation_fn(
    _Dimty.emissionsprice_dims(),
    "emissions price (i.e., emissions per unit of currency)",
)
validate_is_quantityprice = _create_validation_fn(
    _Dimty.quantityprice_dims(),
    "quantity price (i.e., energy per unit of currency, or emissions per unit of currency)",
)
validate_is_exchangerate = _create_validation_fn(
    _Dimty.exchangerate_dims(),
    "exchange rate (i.e., price of a currency per unit of another currency)",
)


# ====
# wqpr
# ====


def validate_compatible(
    objs: Iterable[
        pint.util.UnitsContainer
        | pint.Unit
        | pint.Quantity
        | FloatSeries
        | IntSeries
        | PintSeries
        | str
        | float
        | int
    ],
):
    """Validate if objects are compatible. Incompatible means: mixing energy and emissions,
    or mixing distinct currencies."""
    # Turn combined dimensionalities (e.g. [energy]/[time]) into individual ones ([energy] and [time]).
    dims = {dim for obj in objs for dim in get_basedimty(obj)}
    # Check mixing of energy and emissions.
    have_energy = any(dim in _Dimty.energy_dims() for dim in dims)
    have_emissions = any(dim in _Dimty.emissions_dims() for dim in dims)
    if have_energy and have_emissions:
        raise ValueError("Found incompatible combination: energies and emissions.")
    # Check mixing of distinct currencies.
    distinct_currency_count = len(set(dim for dim in dims if dim in _Dimty.currency_dims()))
    if distinct_currency_count > 1:
        raise ValueError("Found incompatible combination: distinct currencies.")


def validate_complete(
    objs: Iterable[
        pint.util.UnitsContainer | pint.Unit | pint.Quantity | pd.Series | str | float | int
    ],
):
    """Validate if objects are complete. Complete means: contains energy or emissions, energy rate
    or emissions rate, energy price or emissions price, and revenue."""
    dims = {get_basedimty(obj) for obj in objs}
    if not any(dim in _Dimty.quantity_dims() for dim in dims):
        raise ValueError("Did not find a quantity (i.e., energy or emissions).")
    if not any(dim in _Dimty.quantityrate_dims() for dim in dims):
        raise ValueError("Did not find a quantity rate (i.e., energy rate or emissions rate).")
    if not any(dim in _Dimty.quantityprice_dims() for dim in dims):
        raise ValueError("Did not find a quantity price (i.e., energy price or emissions price).")
    if not any(dim in _Dimty.currency_dims() for dim in dims):
        raise ValueError("Did not find a currency.")


_COL_TO_DIMTIESFN: dict[Col, Callable[[], set[pint.util.UnitsContainer]]] = {
    "q": _Dimty.quantity_dims,
    "w": _Dimty.quantityrate_dims,
    "r": _Dimty.currency_dims,
    "p": _Dimty.quantityprice_dims,
}


def col_to_dimties(
    col: Col | Literal["nodim"], nodim_allowed: bool = False
) -> set[pint.util.UnitsContainer]:
    """Return dimensionalities (in base dimensions) allowed for a given column.

    Parameters
    ----------
    col
        The column for which to get the allowed dimensionality.
    nodim_allowed, optional (default: False)
        If False, ``col`` may be one of 'w', 'q', 'p', 'r'. If True, may also be 'nodim'.

    Returns
    -------
        Set of allowed dimensionalities.
    """
    if col == "nodim":
        if nodim_allowed:
            return {_Dimty.dimensionless_dim()}
        raise ValueError("Value 'nodim' not allowed for parameter `col`.")
    return _COL_TO_DIMTIESFN[col]()


def valid_col(
    obj: pint.util.UnitsContainer | pint.Unit | pint.Quantity | str | float | int | PintSeries,
    nodim_allowed: bool = False,
) -> Col | Literal["nodim"]:
    """Return column allowed for a given object.

    Parameters
    ----------
    obj
        Data from which to analyse the units (dimensionality) to find column it can be used for.
    nodim_allowed, optional (default: False)
        If False, check if ``col`` is one of 'w', 'q', 'p', 'r'. If True, check also if it is 'nodim'.

    Returns
    -------
        Column for which the data can be used.
    """
    dim = get_basedimty(obj)
    for col, dimtiesfn in _COL_TO_DIMTIESFN.items():
        if dim in dimtiesfn():
            return col
    if nodim_allowed and dim == _Dimty.dimensionless_dim():
        return "nodim"
    raise ValueError(
        f"Dimensionality of object {obj} ({dim}) not valid as a column for a portfolio line."
    )


def complete_and_verify_consistency(
    w: pd.Series | None, q: pd.Series | None, p: pd.Series | None, r: pd.Series | None
) -> tuple[pd.Series | None, pd.Series | None, pd.Series | None, pd.Series | None]:
    """Calculate as many of the series as possible. Series are assumed to have same index."""
    # Get duration.
    for s in [w, q, p, r]:
        if s is not None:
            duration = tools_index.duration(s.index)
            break
    else:
        return None, None, None, None

    # Volumes.
    if w is not None and q is not None:
        try:
            tools_testing.assert_series_equal(w, q / duration, check_names=False)
        except AssertionError as e:
            raise ValueError(f"Values for w and q are not consistent: {w:=}, {q:=}") from e
    elif w is not None and q is None:
        q = w * duration
    elif w is None and q is not None:
        w = q / duration
    elif w is None and q is None and p is not None and r is not None:
        q = r / p
        w = q / duration

    # If we are here, there are no more options to find w and q.
    # If they are not both None, they are consistent with each other but might be inconsistent with p and r.

    # Price.
    if p is None and q is not None and r is not None:
        p = r / q

    # If we are here, there are no more options to find p.
    # It may be inconsistent with w, q and r.

    # Revenue.
    if r is None and q is not None and p is not None:
        r = q * p
        # Make correction for edge case: p unknown (nan or inf) and q==0 --> assume r=0
        mask = np.isclose(q.pint.m, 0) & (p.isna() | np.isinf(p.pint.m))
        if mask.any():
            r[mask] = 0

    # If we land here, there are no more options to find r.
    # It may be inconsistent with w, q and p.

    # Consistency.
    if q is not None and p is not None and r is not None:
        # Check for consistency, but ignore edge cases:
        # - p unknown (nan or inf) and q==0 --> ignore
        # - q unknown (nan or inf) and p==0 --> ignore
        ign1 = np.isclose(q.pint.m, 0) & (p.isna() | np.isinf(p.pint.m))
        ign2 = np.isclose(p.pint.m, 0) & (q.isna() | np.isinf(q.pint.m))
        ignore = ign1 | ign2
        try:
            tools_testing.assert_series_equal(
                r[~ignore], p[~ignore] * q[~ignore], check_names=False
            )
        except AssertionError as e:
            raise ValueError("Values for r, p, and q are not consistent.") from e

    return w, q, p, r


# @dataclasses.dataclass(frozen=True, kw_only=True)
# class PreferredUnits:
#     """Units to use when converting or printing physical quantities.
#
#     Parameters
#     ----------
#     q
#         Unit for quantity, i.e., unit for energy (e.g. MWh) or for emissions (e.g. tCO2).
#     r
#         Unit for revenue, i.e., unit for currency (e.g. EUR or USD).
#     w, optional
#         Unit for 'quantity per time'.
#         If q is unit of energy: power, e.g. kW; if q is unit of emissions: emissions rate. e.g. kgCO2/h.
#         Default: unit specified for q per hour.
#     p, optional
#         Unit for price, i.e., for 'currency per quantity'.
#         If q is unit of energy: energy price, e.g., Eur/MWh; if q is unit of emissions: emissions
#         price, e.g. Eur/tCO2.
#         Default: unit specified for r divided by unit specified for q.
#
#     Notes
#     -----
#     Units specified for ``r`` and ``p`` must use same currency (dimension). E.g. if r='EUR', then
#     p='EUR/MWh' and p='ctEur/kWh' are both valid; but p='USD/MWh' is not.
#     """
#
#     r: pint.Unit | str
#     q: pint.Unit | str
#     w: pint.Unit | str | None = None
#     p: pint.Unit | str | None = None
#
#     def __post_init__(self):
#         # Verify units for q and r. Verify unit is (a) KNOWN and (b) of correct dimensionality.
#         # . q
#         # object.__setattr__(self, "q", convert_unit(self.q))
#         # validate_is_quantity(self.q)
#         # # . r
#         # object.__setattr__(self, "r", convert_unit(self.r))
#         # validate_is_currency(self.r)
#         # # Verify (if specified) or calculate (if not specified) units for w and p.
#         # # . w
#         # if self.w is None:
#         #     object.__setattr__(self, "w", self.q / Unit("h"))  # no additional checks needed
#         # else:
#         #     object.__setattr__(self, "w", convert_unit(self.w))
#         #     validate_is_quantityrate(self.w)
#         # # . p
#         # if self.p is None:
#         #     object.__setattr__(self, "p", self.r / self.q)
#         # else:
#         #     object.__setattr__(self, "p", convert_unit(self.p))
#         #     validate_is_quantityprice(self.p)
#         #
#         # # If we are here, each unit individually has the correct dimensionality.
#         #
#         # # Check if all units are compatible with eachother.
#         # validate_compatible([self.q, self.r, self.w, self.p])
#         #
#         # # Add mapping to quickly find preferred unit from dimension or unit.
#         # map = {(unit := getattr(self, col)).dimensionality: unit for col in COLS}
#         # object.__setattr__(self, "_map", map)
#         pass
#
#     # def __or__(self, other) -> Self:
#     #     if not isinstance(other, Self):
#     #         raise TypeError(f"Can only do union on PreferredUnit instances; got {type(other)}.")
#
#     @classmethod
#     def from_units(cls, units: Iterable[pint.Unit]) -> Self:
#         """Create preferred units from iterable. Match unit to correct column."""
#         found_units = {}
#         for unit in units:
#             col = valid_col(unit)
#             found_unit = found_units.get(col)
#             if found_unit is None:
#                 found_units[col] = unit
#             elif found_unit is not unit:
#                 raise ValueError(
#                     f"Found multiple units for same dimension ({col}): {unit} and {found_unit}."
#                 )
#         return cls(**found_units)
#
#     def get_units(
#         self,
#         obj: pint.util.UnitsContainer | pint.Unit | pint.Quantity | pd.Series | str | float | int,
#     ) -> pint.Unit | None:
#         """Return preferred unit for given dimension or unit (or None if no preference found)."""
#         dimty = get_basedimty(obj)  # ensure base dimensions
#         return self._map.get(dimty)
