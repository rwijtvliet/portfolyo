"""
Working with pint units.
"""

from pathlib import Path
from typing import Tuple, overload
from .types import Series_or_DataFrame
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
ureg.formatter.default_format = "~P"  # short by default
ureg.setup_matplotlib()

# Set for export.
PA_ = pint_pandas.PintArray
Q_ = ureg.Quantity
Unit = ureg.Unit


NAMES_AND_DIMENSIONS = {
    "w": ureg.get_dimensionality({"[energy]": 1, "[time]": -1}),
    "q": ureg.get_dimensionality({"[energy]": 1}),
    "p": ureg.get_dimensionality({"[currency]": 1, "[energy]": -1}),
    "r": ureg.get_dimensionality({"[currency]": 1}),
    "duration": ureg.get_dimensionality({"[time]": 1}),
    "t": ureg.get_dimensionality({"[temperature]": 1}),
    "nodim": ureg.get_dimensionality({}),
}


def to_name(unit: pint.Unit) -> str:
    """Find the standard column name belonging to unit `unit`. Checks on dimensionality,
    not exact unit."""
    for name, dim in NAMES_AND_DIMENSIONS.items():
        if dim == unit.dimensionality:
            return name
    raise pint.UndefinedUnitError(f"No standard name found for unit '{unit}'.")


@overload
def defaultunit(val: int | float) -> float: ...


@overload
def defaultunit(val: pint.Quantity) -> pint.Quantity: ...


@overload
def defaultunit(val: pd.Series) -> pd.Series: ...


@overload
def defaultunit(val: pd.DataFrame) -> pd.DataFrame: ...


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
def split_magn_unit(val: int | float) -> Tuple[float, None]: ...


@overload
def split_magn_unit(val: pint.Quantity) -> Tuple[float, None | pint.Unit]: ...


@overload
def split_magn_unit(
    val: pd.Series,
) -> Tuple[pd.Series, None | pint.Unit | pd.Series]: ...


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


def avoid_frame_of_objects(
    fr: Series_or_DataFrame, strict: bool = True
) -> Series_or_DataFrame:
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
            units = float if (u := fr.pint.units) == "dimensionless" else f"pint[{u}]"
            return fr.pint.magnitude.astype(float).astype(units)
        # We MAY have a series of pint quantities. Convert to pint-series, if possible.
        units = {v.units for v in fr.values if isinstance(v, Q_)}
        dims = {u.dimensionality for u in units}
        if len(units):
            if len(dims) == 1:
                return fr.astype(f"pint[{units.pop()}]")
            if not strict:
                return fr
            raise pint.DimensionalityError(
                f"Expected a Series with quantities of the same dimension; got {dims}."
            )
    raise TypeError(
        "Expected int-Series, float-Series, pint-Series, or Series of pint quantities (of equal dimensionality)."
    )
