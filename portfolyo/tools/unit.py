"""
Working with pint units.
"""

from pathlib import Path
from typing import Tuple, overload
from .types import Series_or_DataFrame
import pandas as pd
import pint
import pint_pandas
import numpy as np

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
    if isinstance(val, int):
        return float(val), None
    elif isinstance(val, float):
        return val, None
    elif isinstance(val, pint.Quantity):
        return float(val.magnitude), val.units
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
    else:  # bool, timestamp, ...
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
            return _normalize_pintseries(fr)
        else:
            # We MAY have a series of pint quantities. Convert to pint-series, if possible.
            return _normalize_pintobjects(fr, strict)
    raise TypeError(
        "Expected int-Series, float-Series, pint-Series, or Series of pint quantities (of equal dimensionality)."
    )


def _normalize_pintseries(s: pd.Series) -> pd.Series:
    float_magnitudes = s.pint.magnitude.astype(float)
    if s.pint.dimensionless:
        return float_magnitudes
    return float_magnitudes.astype(f"pint[{s.pint.units}]")


def _normalize_pintobjects(s: pd.Series, strict: bool) -> pd.Series:
    # If we have a series of quantities (and nan-values), convert to pint-series if possible.
    is_q = s.apply(lambda v: isinstance(v, Q_))
    if not any(is_q):
        if not strict:
            return s
        raise ValueError("Expected at least one quantity.")

    is_q_or_nan = s.apply(lambda v: isinstance(v, Q_) or np.isnan(v))
    if not all(is_q_or_nan):
        if not strict:
            return s
        raise ValueError("Expected only quantities (and np.nan).")

    units = {v.units for v in s.loc[is_q].values}
    dims = {u.dimensionality for u in units}

    if len(dims) > 1:
        if not strict:
            return s
        raise ValueError(
            f"Expected a Series with quantities of the same dimension; got {dims}."
        )

    # All values are quantities of the same dimension.

    return s.astype(f"pint[{units.pop()}]")
