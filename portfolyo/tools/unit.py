"""
Working with pint units.
"""

from pathlib import Path
from typing import Tuple, Union

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
Unit = ureg.Unit

NAMES_AND_UNITS = {
    "w": ureg.MW,
    "q": ureg.MWh,
    "p": ureg.euro_per_MWh,
    "r": ureg.euro,
    "duration": ureg.hour,
    "t": ureg.degC,
    "nodim": ureg.dimensionless,
}


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


def defaultunit(
    val: Union[int, float, pint.Quantity, pd.Series, pd.DataFrame],
) -> Union[float, pint.Quantity, pd.Series, pd.DataFrame]:
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
        if pd.api.types.is_integer_dtype(val.dtype):
            return val.astype(float)
        elif pd.api.types.is_float_dtype(val.dtype):
            return val
        elif isinstance(val.dtype, pint_pandas.PintType):
            if val.pint.units == ureg.Unit("dimensionless"):
                return val.astype(float)
            return val.pint.to_base_units()
        elif pd.api.types.is_object_dtype(val.dtype):  # maybe series of quantities
            try:
                magn_unit_tupls = [split_magn_unit(v) for v in val.values]
                mm, uu = zip(*magn_unit_tupls)
                if len(set(uu)) == 1:  # if all same unit: return as pint-Series
                    if (u := next(iter(uu))) is not None:
                        return pd.Series(mm, val.index).astype(f"pint[{u}]")
                    return pd.Series(mm, val.index)  # floats
                # otherwise, return again as series of quantities, at least in baseunit
                qq = [m if u is None else Q_(m, u) for m, u in magn_unit_tupls]
                return pd.Series(qq, val.index)
            except TypeError:
                pass
        raise TypeError(f"``val`` is a Series with an unallowed dtype: '{val.dtype}'.")
    elif isinstance(val, pd.DataFrame):
        return pd.DataFrame({col: defaultunit(s) for col, s in val.items()})
    raise TypeError("``val`` must be an int, float, Quantity, Series, or DataFrame.")


def split_magn_unit(
    val: Union[int, float, pint.Quantity, pd.Series]
) -> Union[
    Tuple[float, None],
    Tuple[float, pint.Unit],
    Tuple[pd.Series, None],
    Tuple[pd.Series, pint.Unit],
    Tuple[pd.Series, pd.Series],
]:
    """Split ``val`` into magnitude and units. If ``val`` is a Series with uniform
    dimension, the unit is returned as a pint Unit. If not, it is returned as a Series.
    """
    val = defaultunit(val)
    if isinstance(val, float):
        return val, None
    elif isinstance(val, pint.Quantity):
        return val.magnitude, val.units
    elif isinstance(val, pd.Series):
        if pd.api.types.is_float_dtype(val.dtype):
            return val, None
        elif isinstance(val.dtype, pint_pandas.PintType):
            return val.pint.magnitude, val.pint.units
        else:  # must be series of quantities
            m = [q.magnitude for q in val.values]
            u = [q.units for q in val.values]
            return pd.Series(m, val.index), pd.Series(u, val.index)
    elif isinstance(val, pd.DataFrame):
        raise TypeError("For dataframes, handle the series seperately.")
    raise TypeError("``val`` must be an int, float, Quantity, or Series.")
