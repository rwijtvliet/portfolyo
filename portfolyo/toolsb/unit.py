"""Working with pint units."""

import dataclasses
import functools
from pathlib import Path
from typing import overload, Iterable
from typing_extensions import Self

import pandas as pd
import pint
import pint_pandas

from .types import Col, COLS, PintSeries
from . import _decorator as tools_decorator

_FILEPATH = Path(__file__).parent / "unitdefinitions.txt"


# ==============
# Ureg and units
# ==============

ureg = pint.UnitRegistry(
    str(_FILEPATH),
    auto_reduce_dimensions=True,
    case_sensitive=False,
)
ureg.formatter.default_format = "~P"  # short by default
ureg.setup_matplotlib()
pint.set_application_registry(ureg)

# Set for export.
Q_ = ureg.Quantity
Unit = ureg.Unit


# Conversion and validation.
# --------------------------


@functools.lru_cache()
def convert(unit: pint.Unit | str | None) -> pint.Unit:
    """Convert argument to correct/expected type."""
    if unit is None:
        unit = Unit("")  # intepret as dimensionless
    elif not isinstance(unit, pint.Unit):
        try:
            unit = Unit(unit)
        except pint.UndefinedUnitError as e:
            raise ValueError(
                f"Unit '{unit}' not defined. Add to unit registry by relating to existing unit, e.g."
                " with 'ureg.define('BTU = 0.293071 Wh')'. To add a currency, specify dimension"
                " name instead, e.g. with 'ureg.define('THB = [currency_Thai]')'. The dimension"
                " name must start with 'currency_', and units must not collide with existing ones."
            ) from e
    return unit


coerce = tools_decorator.create_coercedecorator(
    conversion=convert, validation=None, default_param="unit"
)

# =====================================
# Units, dimensions, quantities, series
# =====================================


def get_basedimty(
    obj: pint.util.UnitsContainer | pint.Unit | pint.Quantity | pd.Series | str | float | int,
) -> pint.util.UnitsContainer:
    """Get base dimensionality of ``obj``."""
    if isinstance(obj, pd.DataFrame):
        raise TypeError("Can't get dimensionality of DataFrame; apply to individual Series.")

    if isinstance(obj, pd.Series):
        if pd.api.types.is_numeric_dtype(obj.dtype):
            return get_basedimty(1)  # dimensionless
        elif isinstance(obj.dtype, pint_pandas.PintType):
            return get_basedimty(obj.pint.units)
        else:
            raise ValueError("Series does not have a unit and is also not numeric.")

    # Defer to function pint provides.
    return ureg.get_dimensionality(obj)


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


def validate_compatible(
    objs: Iterable[
        pint.util.UnitsContainer | pint.Unit | pint.Quantity | pd.Series | str | float | int
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
        raise ValueError("Found incompatible combination of energies and emissions.")
    # Check mixing of distinct currencies.
    distinct_currency_count = len(set(dim for dim in dims if dim in _Dimty.currency_dims()))
    if distinct_currency_count > 1:
        raise ValueError("Found incompatible combination of distinct currencies.")


# ====
# wqpr
# ====


_COL_TO_DIMTIES = {
    "q": _Dimty.quantity_dims,
    "w": _Dimty.quantityrate_dims,
    "r": _Dimty.currency_dims,
    "p": _Dimty.quantityprice_dims,
}


def col_to_dimties(col: Col) -> set[pint.util.UnitsContainer]:
    """Return dimensionalities (in base dimensions) allowed for a given column."""
    return _COL_TO_DIMTIES[col]()


def valid_col(
    obj: pint.util.UnitsContainer | pint.Unit | pint.Quantity | pd.Series | str | float | int,
) -> str:
    """Return column allowed for a given object."""
    dim = get_basedimty(obj)
    for col, dimtyfn in _COL_TO_DIMTIES.items():
        if dim in dimtyfn():
            return col
    raise ValueError(
        f"Dimensionality of object {obj} ({dim}) not valid as a column for a portfolio line."
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class PreferredUnits:
    """Units to use when converting or printing physical quantities.

    Parameters
    ----------
    q
        Unit for quantity, i.e., unit for energy (e.g. MWh) or for emissions (e.g. tCO2).
    r
        Unit for revenue, i.e., unit for currency (e.g. EUR or USD).
    w, optional
        Unit for 'quantity per time'.
        If q is unit of energy: power, e.g. kW; if q is unit of emissions: emissions rate. e.g. kgCO2/h.
        Default: unit specified for q per hour.
    p, optional
        Unit for price, i.e., for 'currency per quantity'.
        If q is unit of energy: energy price, e.g., Eur/MWh; if q is unit of emissions: emissions
        price, e.g. Eur/tCO2.
        Default: unit specified for r divided by unit specified for q.

    Notes
    -----
    Units specified for ``r`` and ``p`` must use same currency (dimension). E.g. if r='EUR', then
    p='EUR/MWh' and p='ctEur/kWh' are both valid; but p='USD/MWh' is not.
    """

    r: pint.Unit | str
    q: pint.Unit | str
    w: pint.Unit | str | None = None
    p: pint.Unit | str | None = None

    def __post_init__(self):
        # Verify units for q and r. Verify unit is (a) KNOWN and (b) of correct dimensionality.
        # . q
        object.__setattr__(self, "q", convert(self.q))
        validate_is_quantity(self.q)
        # . r
        object.__setattr__(self, "r", convert(self.r))
        validate_is_currency(self.r)
        # Verify (if specified) or calculate (if not specified) units for w and p.
        # . w
        if self.w is None:
            object.__setattr__(self, "w", self.q / Unit("h"))  # no additional checks needed
        else:
            object.__setattr__(self, "w", convert(self.w))
            validate_is_quantityrate(self.w)
        # . p
        if self.p is None:
            object.__setattr__(self, "p", self.r / self.q)
        else:
            object.__setattr__(self, "p", convert(self.p))
            validate_is_quantityprice(self.p)

        # If we land here, each unit individually has the correct dimensionality.

        # Check if all units are compatible with eachother.
        validate_compatible([self.q, self.r, self.w, self.p])

        # Add mapping to quickly find preferred unit from dimension or unit.
        map = {(unit := getattr(self, col)).dimensionality: unit for col in COLS}
        object.__setattr__(self, "_map", map)

    @classmethod
    def from_units(cls, units: Iterable[pint.Unit]) -> Self:
        """Create preferred units from iterable. Match unit to correct column."""
        found_units = {}
        for unit in units:
            col = valid_col(unit)
            if found_units.get(col) in [None, unit]:
                found_units[col] = unit
            else:
                raise ValueError("Found multiple units for same dimension.")
        return cls(**found_units)

    def get_units(
        self,
        obj: pint.util.UnitsContainer | pint.Unit | pint.Quantity | pd.Series | str | float | int,
    ) -> pint.Unit | None:
        """Return preferred unit for given dimension or unit (or None if no preference found)."""
        dimty = get_basedimty(obj)  # ensure base dimensions
        return self._map.get(dimty)


def _coerce_skalar_to_preferred(
    val: int | float | pint.Quantity, preferred: PreferredUnits | None = None
) -> pint.Quantity:
    if isinstance(val, float | int):
        return Q_(float(val), "")  # dimensionless
    elif isinstance(val, pint.Quantity):
        if preferred and (new_units := preferred.get_units(val.units)):
            return val.to(new_units)
        return val


def _coerce_series_to_preferred(
    val: pd.Series, preferred: PreferredUnits | None = None
) -> PintSeries | pd.Series:
    if pd.api.types.is_numeric_dtype(val.dtype):
        return val.astype(float).astype("pint[]")  # dimensionless

    if isinstance(val.dtype, pint_pandas.PintType):
        if preferred and (new_units := preferred.get_units(val.pint.units)):
            return val.pint.to(new_units)

    elif pd.api.types.is_object_dtype(val.dtype) and isinstance(val.iloc[0], pint.Quantity):
        units = val.iloc[0].units
        if preferred and (new_units := preferred.get_units(units)):
            units = new_units
        try:
            return val.astype(f"pint[{units}]")
        except pint.DimensionalityError:  # not all have same dimension
            return val.apply(lambda v: _coerce_skalar_to_preferred(v, preferred))

    return val  # bools, timestamps, ...; unknown unit;


@overload
def coerce_to_preferred(
    val: int | float | pint.Quantity, preferred: PreferredUnits | None = None
) -> pint.Quantity:
    ...


@overload
def coerce_to_preferred(val: pd.Series, preferred: PreferredUnits | None = None) -> pd.Series:
    ...


@overload
def coerce_to_preferred(val: pd.DataFrame, preferred: PreferredUnits | None = None) -> pd.DataFrame:
    ...


def coerce_to_preferred(
    val: int | float | pint.Quantity | pd.Series | pd.DataFrame,
    preferred: PreferredUnits | None = None,
) -> float | pint.Quantity | pd.Series | pd.DataFrame:
    """Convert ``val`` to preferred units. Also turns floats and ints into dimensionless.

    Parameters
    ----------
    val
        The value to convert to preferred units.
    preferredunits


    Returns
    -------
        Same object, in other units.
    """
    if isinstance(val, int | float | pint.Quantity):
        return _coerce_skalar_to_preferred(val, preferred)
    elif isinstance(val, pd.Series):
        return _coerce_series_to_preferred(val, preferred)
    elif isinstance(val, pd.DataFrame):
        return pd.DataFrame({col: _coerce_series_to_preferred(s) for col, s in val.items()})
    raise TypeError("``val`` must be an int, float, Quantity, Series, or DataFrame.")
