"""
Working with pint units.
"""

from pathlib import Path
from typing import Union

import pandas as pd
import pint
import pint_pandas

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
    # TODO: Unused. Remove?
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


def to_name(unit: pint.Unit) -> str:
    """Find the standard column name belonging to unit `unit`. Checks on dimensionality,
    not exact unit."""
    for name, u in NAMES_AND_UNITS.items():
        if u.dimensionality == unit.dimensionality:
            return name
    raise pint.UndefinedUnitError(f"No standard name found for unit '{unit}'.")


def from_name(name: str) -> pint.Unit:
    """Find standard unit belonging to a column name."""
    if name in NAMES_AND_UNITS:
        return NAMES_AND_UNITS[name]
    raise ValueError(f"No standard unit found for name '{name}'.")


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
