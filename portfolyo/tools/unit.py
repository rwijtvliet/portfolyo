"""Working with pint units."""

from pathlib import Path

from .types import Series_or_DataFrame
import pandas as pd
import pint
import pint_pandas

path = Path(__file__).parent / "unitdefinitions.txt"


ureg = pint_pandas.PintType.ureg = pint.UnitRegistry(
    str(path),
    auto_reduce_dimensions=True,
    autoconvert_to_preferred=True,
    case_sensitive=False,
)
ureg.formatter.default_format = "~P"  # short by default
ureg.setup_matplotlib()

# Set for export.
PA_ = pint_pandas.PintArray
Q_ = ureg.Quantity
Unit = ureg.Unit


NAMES_AND_DIMENSIONS = {
    "w": ureg.get_dimensionality("[energy]/[time]"),
    "q": ureg.get_dimensionality("[energy]"),
    "p": ureg.get_dimensionality("[currency]/[energy]"),
    "r": ureg.get_dimensionality("[currency]"),
    "duration": ureg.get_dimensionality("[time]"),
    "t": ureg.get_dimensionality("[temperature]"),
    "nodim": ureg.get_dimensionality("[]"),
}

# DEFAULT_PREFERRED = [ureg.Eur]
# ureg.default_preferred_units = DEFAULT_PREFERRED


def to_name(arg: pint.Unit | pint.Quantity) -> str:
    """Find the standard column name belonging to ``arg``, which can be a unit or
    a quantity. Checks on dimensionality, not exact unit."""
    for name, dim in NAMES_AND_DIMENSIONS.items():
        if arg.dimensionality == dim:
            return name
    raise ValueError(f"No standard name found for dimension '{arg.dimensionality}'.")


# @overload
# def split_magn_unit(val: int | float) -> Tuple[float, None]: ...
#
#
# @overload
# def split_magn_unit(val: pint.Quantity) -> Tuple[float, None | pint.Unit]: ...
#
#
# @overload
# def split_magn_unit(
#     val: pd.Series,
# ) -> Tuple[pd.Series, None | pint.Unit | pd.Series]: ...
#
#
# def split_magn_unit(
#     val: int | float | pint.Quantity | pd.Series,
# ) -> Tuple[float, None | pint.Unit] | Tuple[pd.Series, None | pint.Unit | pd.Series]:
#     """Split ``val`` into magnitude and units. If ``val`` is a Series with uniform
#     dimension, the unit is returned as a pint Unit. If not, it is returned as a Series.
#     """
#     if isinstance(val, int):
#         return float(val), None
#     elif isinstance(val, float):
#         return val, None
#     elif isinstance(val, pint.Quantity):
#         return float(val.magnitude), val.units
#     elif isinstance(val, pd.Series):
#         if isinstance(val.dtype, pint_pandas.PintType):
#             return val.pint.magnitude, val.pint.units
#         elif pd.api.types.is_object_dtype(val.dtype) and isinstance(val.iloc[0], Q_):
#             # series of quantities?
#             m = [q.magnitude for q in val.values]
#             u = [q.units for q in val.values]
#             return pd.Series(m, val.index), pd.Series(u, val.index)
#         return val, None
#     elif isinstance(val, pd.DataFrame):
#         raise TypeError("For dataframes, handle the series seperately.")
#     else:  # bool, timestamp, ...
#         return val, None
#


def normalize_value(v: float | int | pint.Quantity) -> float | pint.Quantity:
    """Ensure a value is a float or a (non-dimensionless) quantity.

    Parameters
    ----------
    v
        Input value to normalize.

    Returns
    -------
        Normalized value.
    """
    if isinstance(v, int):
        return float(v)

    elif isinstance(v, float):
        return v

    elif isinstance(v, pint.Quantity):
        float_magnitude = float(v.magnitude)
        if v.dimensionless:
            return float_magnitude
        return Q_(float_magnitude, v.units)
    raise TypeError(f"Expected float, int, or pint Quantity; got {type(v)}.")


def normalize_frame(
    fr: Series_or_DataFrame, strict: bool = True
) -> Series_or_DataFrame:
    """Ensure a Series (or each Series in a Dataframe) is a float-series or a (non-dimensionless) pint-Series, if possible.

    Parameters:
    -----------
    fr
        The input data structure, which can be either a pandas Series or DataFrame.
        Expect int-Series, float-Series, pint-Series, or Series of pint quantities (of equal dimensionality).
    strict, optional (default: True)
        If True, raises Error if ``fr`` cannot be converted into a frame without objects.

    Returns:
    --------
        The transformed data structure.
    """

    if isinstance(fr, pd.DataFrame):
        return fr.apply(normalize_frame, axis=1)

    # fr is now a Series.

    if pd.api.types.is_integer_dtype(fr):
        # Int-series --> convert to floats.
        return fr.astype(float)

    elif pd.api.types.is_float_dtype(fr):
        # Float-series --> return as-is.
        return fr

    elif isinstance(fr.dtype, pint_pandas.PintType):
        # Pint-series --> return as floats or pint-series (with float magnitude).
        float_magnitudes = fr.pint.magnitude.astype(float)
        if fr.pint.dimensionless:
            return float_magnitudes
        return float_magnitudes.astype(f"pint[{fr.pint.units}]")

    elif hasattr(fr, "pint"):
        # Series of pint quantities. MAY be uniform dimension.
        try:
            pintseries = fr.pint.convert_object_dtype()  # may have int magnitudes
        except pint.DimensionalityError as e:
            if not strict:
                return fr
            raise e

        # handle int-magnitudes and/or dimensionless data
        return normalize_frame(pintseries)

    raise TypeError(
        "Expected int-Series, float-Series, pint-Series, or Series of pint quantities (of equal dimensionality)."
    )


def normalize(
    v: float | int | pint.Quantity | pd.Series | pd.DataFrame, strict: bool = True
) -> float | pint.Quantity | pd.Series | pd.DataFrame:
    """Ensure dimensionless values are floats (or float-series), and that other values are quantities (or pint-Series) with float magnitudes.

    Parameters
    ----------
    v
        Input data to normalize
    strict, optional (default: True)
        If True, raises Error if Series cannot be converted into a series without pint quantities.

    Returns
    -------
        Normalized data.
    """
    if isinstance(v, float | int | pint.Quantity):
        return normalize_value(v)
    else:
        return normalize_frame(v, strict)
