"""Module with tools for dealing with units ("'nits" to keep "units" available in
name space.) """

from pathlib import Path
from typing import Union, Dict
import pint
import pint_pandas
import pandas as pd


path = Path(__file__).parent / "unitdefinitions.txt"

ureg = pint_pandas.PintType.ureg = pint.UnitRegistry(
    str(path),
    system="powerbase",
    auto_reduce_dimensions=True,
    case_sensitive=False,
)
ureg.default_format = "~P"  # short by default
ureg.setup_matplotlib()

# Set for export.
PA_ = pint_pandas.PintArray
Q_ = ureg.Quantity


def pintunit_remove(u: pint.Unit) -> str:
    units = f"{u}" or "dimensionless"
    return f"pint[{units}]"
    # TODO: replace with f'{units:P}'


# def to_pref_unit(self: pint.Quantity):
#     for unit in (ureg.MW, ureg.euro_per_MWh):
#         if self.dimensionality == unit.dimensionality:
#             return self.to(unit)
#     return self

# def cast2quant(val, unit:pint.Unit) -> pint.Quantity:
#     """Cast a value `val` to a quantity with the given unit."""
#     return Q_(val, unit)

NAMES_AND_UNITS = {
    "w": ureg.MW,
    "q": ureg.MWh,
    "p": ureg.euro_per_MWh,
    "r": ureg.euro,
    "duration": ureg.hour,
    "t": ureg.degC,
    "nodim": ureg.dimensionless,
}


def to_compact(value: Union[pint.Quantity, pd.Series, pd.DataFrame]):
    """Convert to more compact unit by moving absolute magnitude into readable range."""
    if isinstance(value, pint.Quantity):
        return value.to_compact()
    elif isinstance(value, pd.Series):
        newunits = value.abs().max().to_compact().units
        return value.pint.to(newunits)
    elif isinstance(value, pd.DataFrame):
        return pd.DataFrame({name: to_compact(s) for name, s in value.items()})
    else:
        raise TypeError(
            "`value` must be a Quantity, or Series or DataFrame of quantities."
        )


def unit2name(unit: pint.Unit) -> str:
    """Find the standard column name belonging to unit `unit`. Checks on dimensionality,
    not exact unit."""
    for name, u in NAMES_AND_UNITS.items():
        if u.dimensionality == unit.dimensionality:
            return name
    raise pint.UndefinedUnitError(f"No standard name found for unit '{unit}'.")


def name2unit(name: str) -> pint.Unit:
    """Find standard unit belonging to a column name."""
    if name in NAMES_AND_UNITS:
        return NAMES_AND_UNITS[name]
    raise ValueError(f"No standard unit found for name '{name}'.")


def set_unit(s: pd.Series, unit: Union[pint.Unit, str]) -> pd.Series:
    """Make series unit-aware. If series is already unit-aware, convert to specified unit.
    If not, assume values are in specified unit.

    Parameters
    ----------
    s : pd.Series
    unit : Union[pint.Unit, str]
        If None, remove the unit.

    Returns
    -------
    pd.Series
        Same as input series, but with specified unit.
    """
    if not isinstance(unit, pint.Unit):
        unit = ureg.Unit(unit)
    # sets unit if none set yet, otherwise converts if possible
    return s.astype(pintunit_remove(unit))


def set_units(
    df: pd.DataFrame, units: Dict[str, Union[pint.Unit, str]]
) -> pd.DataFrame:
    """Make dataframe unit-aware. If dataframe is already unit-aware, convert to specified
    units. If not, assume values are in specified unit.

    Parameters
    ----------
    df : pd.DataFrame
    units : Dict[str, Union[pint.Unit, str]]
        key = column name, value = unit to set to that column

    Returns
    -------
    pd.DataFrame
        Same as input dataframe, but with specified units.
    """
    df = df.copy()  # don't change incoming dataframe
    for name, unit in units.items():
        df[name] = set_unit(df[name], unit)
    return df


def drop_units(fr: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """
    Convert ``fr`` to base units and return only the magnitude.

    If ``fr`` is not unit-aware, return as-is.
    """
    if isinstance(fr, pd.Series):
        if hasattr(fr, "pint"):
            return fr.pint.to_base_units().pint.m
        return fr
    else:
        return pd.DataFrame({col: drop_units(s) for col, s in fr.items()})
