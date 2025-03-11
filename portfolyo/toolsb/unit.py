"""Working with pint units."""

import dataclasses
from pathlib import Path
from typing import Tuple, overload

import pandas as pd
import pint
import pint_pandas

from .types import Series_or_DataFrame

_FILEPATH = Path(__file__).parent / "unitdefinitions.txt"
pint_pandas.DEFAULT_SUBDTYPE = float

# Prepare the unit registry.

ureg = pint.UnitRegistry(
    str(_FILEPATH),
    auto_reduce_dimensions=True,
    case_sensitive=False,
)
ureg.formatter.default_format = "~P"  # short by default
ureg.setup_matplotlib()
pint.set_application_registry(ureg)

# Set for export.
PA_ = pint_pandas.PintArray
Q_ = ureg.Quantity
Unit = ureg.Unit

_DEFAULT_NAMES_AND_UNITS = {"a": "b"}


def _energyunits() -> str:
    return (
        "Currently defined units for energy are "
        + ", ".join(str(u) for u in ureg.get_compatible_units("[energy]"))
        + "."
    )


def _emissionsunits() -> str:
    return (
        "Currently defined units for emissions are "
        + ", ".join(str(u) for u in ureg.get_compatible_units("[emissions]"))
        + "."
    )


def _currencyunits() -> str:
    currencydimensions = [dim for dim in ureg._dimensions.keys() if dim.startswith("[currency")]
    return (
        "Currently defined units for currency are "
        + ", ".join(str(u) for dim in currencydimensions for u in ureg.get_compatible_units(dim))
        + "."
    )


def _assert_valid_quantity_unit(unit: str) -> None:
    if not isinstance(unit, Unit):
        try:
            unit = Unit(unit)
        except pint.UndefinedUnitError as e:
            raise ValueError(
                f"Unit '{unit}' not defined. Add to unit registry as unit to 'energy' or"
                " 'emissions' dimension, by relating to existing unit. E.g. with"
                " 'ureg.define('BTU = 0.293071 Wh')' or 'ureg.define('gCO2 = 1e-6 tCO2')'."
                f" {_energyunits()} {_emissionsunits()}"
            ) from e

        if (dim := unit.dimensionality) not in ["[energy]", "[emissions]"]:
            raise ValueError(
                f"``q`` must be a unit with dimension [energy] or [emissions]; got {dim}. ({_energyunits()} {_emissionsunits()})"
            )


@dataclasses.dataclass(frozen=True)
class WqprUnits:
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
    Units specified for ``r`` and ``p`` must use same currency unit.
    """

    q: str
    r: str
    w: str | None = None
    p: str | None = None

    def __post_init__(self):
        # Verify units for q and r. Verify unit is (a) KNOWN and (b) of correct dimensionality.
        # . q
        try:
            unit = Unit(self.q)
        except pint.UndefinedUnitError as e:
            raise ValueError(
                f"Unit '{self.q}' not defined. Add to unit registry as unit to 'energy' or"
                " 'emissions' dimension, by relating to existing unit. E.g. with"
                " 'ureg.define('BTU = 0.293071 Wh')' or 'ureg.define('gCO2 = 1e-6 tCO2')'."
                f" {_energyunits()} {_emissionsunits()}"
            ) from e
        if (dim := unit.dimensionality) not in ["[energy]", "[emissions]"]:
            raise ValueError(
                f"``q`` must be a unit with dimension [energy] or [emissions]; got {dim}. ({_energyunits()} {_emissionsunits()})"
            )
        # . r
        try:
            unit = Unit(self.r)
        except pint.UndefinedUnitError as e:
            raise ValueError(
                f"Unit '{self.r}' not defined. Add to unit registry as a unit to existing currency"
                " by relating to an existing unit of that currency, e.g. with 'ureg.define('pence = 0.01 GBP')'."
                " Or add as new currency, e.g. with 'ureg.define('THB = [currency_Thai]')'. The dimension"
                " name must start with 'currency', and units must not collide with existing ones."
            ) from e
        if len(unit.dimensionality) == 0:
            raise ValueError(f"Unit '{self.r}' is dimensionless. Specify a currency unit.")
        if len(dim := unit.dimensionality) > 1:
            raise ValueError(f"Unit '{self.r}' is not a currency unit; got dimension {dim}.")
        dim, exp = next(iter(unit.dimensionality.items()))
        if not dim.startswith("[currency") or exp != 1:
            raise ValueError(
                f"Unit '{self.r}' is not a currency unit; got dimension {unit.dimensionality}. ({_currencyunits()})"
            )

        # Verify (if specified) or calculate (if not specified) units for w and p.
        pass


def to_name(unit: pint.Unit) -> str:
    """Find the standard column name belonging to unit `unit`. Checks on dimensionality,
    not exact unit."""
    for name, u in _DEFAULT_NAMES_AND_UNITS.items():
        if u.dimensionality == unit.dimensionality:
            return name
    raise pint.UndefinedUnitError(f"No standard name found for unit '{unit}'.")


def from_name(name: str) -> pint.Unit:
    """Find standard unit belonging to a column name."""
    if name in _DEFAULT_NAMES_AND_UNITS:
        return _DEFAULT_NAMES_AND_UNITS[name]
    raise ValueError(f"No standard unit found for name '{name}'.")


@overload
def defaultunit(val: int | float) -> float:
    ...


@overload
def defaultunit(val: pint.Quantity) -> pint.Quantity:
    ...


@overload
def defaultunit(val: pd.Series) -> pd.Series:
    ...


@overload
def defaultunit(val: pd.DataFrame) -> pd.DataFrame:
    ...


def defaultunit(
    val: int | float | pint.Quantity | pd.Series | pd.DataFrame,
) -> float | pint.Quantity | pd.Series | pd.DataFrame:
    """Convert ``val`` to base units. Also turns dimensionless values into floats.

    Parameters
    ----------
    val : int, float, pint.Quantity, pd.Series, or pd.DataFrame
        The value to convert to base units.

    Returns
    -------
    float, pint.Quantity, pd.Series, or pd.DataFrame
        In base units.
    """
    # Do the conversion.
    if isinstance(val, int):
        return float(val)
    elif isinstance(val, float):
        return val
    elif isinstance(val, pint.Quantity):
        if val.units == ureg.Unit("dimensionless"):
            return val.magnitude
        return val.to_base_units()
    elif isinstance(val, pd.Series):
        if isinstance(val.dtype, pint_pandas.PintType):
            if val.pint.units == ureg.Unit("dimensionless"):
                return val.astype(float)
            return val.pint.to_base_units()
        elif pd.api.types.is_object_dtype(val.dtype) and isinstance(val.iloc[0], Q_):
            try:
                unit = val.iloc[0].to_base_units().units
                pintseries = val.astype(f"pint[{unit}]")
                return defaultunit(pintseries)
            except pint.DimensionalityError:  # not all have same dimension
                # convert to base units instead.
                magn_unit_tupls = [split_magn_unit(v) for v in val.values]
                qq = [m if u is None else Q_(m, u) for m, u in magn_unit_tupls]
                return pd.Series(qq, val.index)
        elif pd.api.types.is_integer_dtype(val.dtype):
            return val.astype(float)
        else:  # series of floats, bools, timestamps, ...
            return val
    elif isinstance(val, pd.DataFrame):
        return pd.DataFrame({col: defaultunit(s) for col, s in val.items()})
    raise TypeError("``val`` must be an int, float, Quantity, Series, or DataFrame.")


@overload
def split_magn_unit(val: int | float) -> Tuple[float, None]:
    ...


@overload
def split_magn_unit(val: pint.Quantity) -> Tuple[float, None | pint.Unit]:
    ...


@overload
def split_magn_unit(
    val: pd.Series,
) -> Tuple[pd.Series, None | pint.Unit | pd.Series]:
    ...


def split_magn_unit(
    val: int | float | pint.Quantity | pd.Series,
) -> Tuple[float, None | pint.Unit] | Tuple[pd.Series, None | pint.Unit | pd.Series]:
    """Split ``val`` into magnitude and units. If ``val`` is a Series with uniform
    dimension, the unit is returned as a pint Unit. If not, it is returned as a Series.
    """
    val = defaultunit(val)
    if isinstance(val, pint.Quantity):
        return val.magnitude, val.units
    elif isinstance(val, pd.Series):
        if isinstance(val.dtype, pint_pandas.PintType):
            return val.pint.magnitude, val.pint.units
        elif pd.api.types.is_object_dtype(val.dtype) and isinstance(val.iloc[0], Q_):
            # series of quantities?
            m = [q.magnitude for q in val.values]
            u = [q.units for q in val.values]
            return pd.Series(m, val.index), pd.Series(u, val.index)
        return val, None
    elif isinstance(val, pd.DataFrame):
        raise TypeError("For dataframes, handle the series seperately.")
    else:  # int, float, bool, timestamp, ...
        return val, None


def avoid_frame_of_objects(fr: Series_or_DataFrame) -> Series_or_DataFrame:
    """Ensure a Series or Dataframe does not have objects as its values,
    if possible.

    Parameters:
    -----------
    fr : Series_or_DataFrame
        The input data structure, which can be either a pandas Series or DataFrame.
        Expected int-Series, float-Series, pint-Series, or Series of pint quantities (of equal dimensionality).

    Returns:
    --------
    Series_or_DataFrame
        The transformed data structure.


    """

    if isinstance(fr, pd.DataFrame):
        return pd.DataFrame({col: avoid_frame_of_objects(s) for col, s in fr.items()})

    # fr is now a Series.

    if pd.api.types.is_integer_dtype(fr):
        return fr.astype(float)
    if pd.api.types.is_float_dtype(fr):
        return fr
    if hasattr(fr, "pint"):
        if isinstance(fr.dtype, pint_pandas.PintType):
            return fr
        # We may have a series of pint quantities. Convert to pint-series, if possible.
        try:
            return fr.astype(f"pint[{fr.iloc[0].units}]")
        except pint.DimensionalityError as e:
            dimensions = {v.dimensionality for v in fr.values}
            raise pint.DimensionalityError(
                f"Expected a Series with quantities of the same dimension; got {dimensions}."
            ) from e
    raise TypeError(
        "Expected int-Series, float-Series, pint-Series, or Series of pint quantities (of equal dimensionality)."
    )
