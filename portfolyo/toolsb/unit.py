"""Working with pint units."""

import functools
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, overload

import pandas as pd
import pint
import pint_pandas
from typing_extensions import Self

from . import _decorator as tools_decorator
from .types import (
    FloatSeries,
    IntSeries,
    MultiDimQuantitySeries,
    NontimeDataframe,
    NontimeSeries,
    OtherScalar,
    OtherUniformSeries,
    PintSeries,
    Series_or_Dataframe,
    SingleDimQuantitySeries,
    TimeDataframe,
    TimeSeries,
)

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


def get_basedimty(
    obj: (
        pint.util.UnitsContainer
        | pint.Unit
        | pint.Quantity
        | IntSeries
        | FloatSeries
        | PintSeries
        | str
        | float
        | int
    ),
) -> pint.util.UnitsContainer:
    """Get base dimensionality of ``obj``. If ``obj`` is a Series, it must not be a series of pint
    Quantities."""
    if isinstance(obj, pd.DataFrame):
        raise TypeError(
            "Can't get dimensionality of DataFrame; call function for individual Series."
        )

    if isinstance(obj, pd.Series):
        if pd.api.types.is_integer_dtype(obj.dtype) or pd.api.types.is_float_dtype(obj.dtype):
            return ureg.get_dimensionality(1)  # dimensionless
        elif isinstance(obj.dtype, pint_pandas.PintType):
            return ureg.get_dimensionality(obj.pint.units)
        else:
            raise ValueError("Series does not have a unit and is also not numeric.")

    # Defer to function pint provides.
    return ureg.get_dimensionality(obj)


# Conversion and validation: Unit.
# --------------------------------


def validate_unit(unit: Any) -> None:
    """Check if ``unit`` is valid unit. If not, raise Error."""
    if not isinstance(unit, pint.Unit):
        raise ValueError(
            f"'{unit}' is not a (defined) unit. Add to unit registry by relating to existing unit,"
            " e.g. with 'ureg.define('BTU = 0.293071 Wh')'. To add a currency, specify dimension"
            " name instead, e.g. with 'ureg.define('THB = [currency_Thai]')'. The dimension"
            " name must start with 'currency_', and units must not collide with existing ones."
        )


@functools.lru_cache()
def coerce_unit(unit: pint.Unit | str | None) -> pint.Unit:
    """Convert ``unit`` into valid frequency; raise Error if unsuccessful."""
    if unit is None:
        unit = Unit("")  # intepret as dimensionless
    elif not isinstance(unit, pint.Unit):
        try:
            unit = Unit(unit)
        except pint.UndefinedUnitError:
            pass

    validate_unit(unit)
    return unit


# Conversion and validation: Quantity.
# ------------------------------------


def validate_quantity(sk: Any) -> None:
    """Check if ``sk`` is valid quantity. If not, raise Error."""
    if not isinstance(sk, pint.Quantity):
        raise ValueError(f"This is not a Quantity: {sk}.")


def coerce_quantity(sk: int | float | pint.Quantity) -> pint.Quantity:
    """Convert ``sk`` into quantity; raise Error if unsuccessful."""
    if isinstance(sk, int):
        return Q_(float(sk), "")
    elif isinstance(sk, float):
        return Q_(sk, "")

    validate_quantity(sk)
    return sk


# Conversion and validation: Series
# ---------------------------------


def _convert_to_pintseries_if_possible(
    s: (
        IntSeries
        | FloatSeries
        | PintSeries
        | OtherUniformSeries
        | SingleDimQuantitySeries
        | MultiDimQuantitySeries
    ),
) -> PintSeries | OtherUniformSeries | MultiDimQuantitySeries:
    """If possible, turn Series into PintSeries. Converts Series of Quantities with uniform
    dimensionality into pintseries. If not possible, return as-is."""

    # NOTE: Can't use `.is_numeric_dtype`, because also true for pint dtype.
    if pd.api.types.is_integer_dtype(s.dtype):
        return s.astype(float).astype("pint[]")
    elif pd.api.types.is_float_dtype(s.dtype):
        return s.astype("pint[]")

    elif pd.api.types.is_object_dtype(s.dtype) and isinstance(s.iloc[0], pint.Quantity):
        units = s.iloc[0].units
        try:
            return s.astype(f"pint[{units}]")  # works if all quantities have same dimension
        except pint.DimensionalityError:
            return s  # series of quantities with distinct dimension; keep as-is

    return s  # bools, timestamps, ..., quantities


def validate_pintseries(s: pd.Series) -> None:
    """Check if ``s`` is a pintseries (i.e., series with pint dtype); if not, raise Error."""
    if not isinstance(s.dtype, pint_pandas.PintType):
        raise ValueError(f"This is not a pintseries: {s}.")


def coerce_pintseries(s: pd.Series) -> PintSeries:
    """Convert ``s`` into pintseries; raise Error if unsuccessful."""
    s = _convert_to_pintseries_if_possible(s)
    validate_pintseries(s)
    return s


def validate_pintdataframe(df: pd.DataFrame) -> None:
    """Check if ``df`` is a pintdataframe (i.e., dataframe with pintseries); if not, raise Error."""
    for _, s in df.items():
        validate_pintseries(s)


def coerce_pintdataframe(df: pd.DataFrame) -> PintDataframe:
    """Convert ``df`` into pintdataframe; raise Error if unsuccessful."""
    df = pd.DataFrame({col: _convert_to_pintseries_if_possible(s) for col, s in df.items()})
    validate_pintdataframe(df)
    return df


def coerce_not_quantityseries(s: pd.Series) -> PintSeries | dict[Any, pint.Quantity]:
    """Convert ``s`` into pintseries or a dictionary. Do not allow quantities to remain as series
    element."""
    s = _convert_to_pintseries_if_possible(s)
    try:
        validate_pintseries(s)
        return s
    except Exception:
        pass

    # If we are here, ``s`` is not a pintseries, and we now expect: only objects.

    if not any(isinstance(v, pint.Quantity) for v in s.values):
        raise ValueError(
            f"Unexpected series: cannot convert to pintseries, but does not contain any quantities: {s}."
        )
    if not all(isinstance(v, pint.Quantity) for v in s.values):
        raise ValueError(
            f"Unexpected series: cannot convert to pintseries, but does not contain only quantities: {s}."
        )
    return s.to_dict()


# Conversion and validation: Unit preference.
# -------------------------------------------


class UnitPref(dict):
    """Mapping base dimensionality (pint.util.UnitsContainer) -> unit (pint.Unit)."""

    def __init__(self, /, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Convert and validate data.
        converted = {}
        for dimty, unit in self.items():
            converted[get_basedimty(dimty)] = coerce_unit(unit)
        self.clear()
        self.update(converted)

    def __getitem__(self, dimty: pint.util.UnitsContainer) -> pint.Unit:
        dimty = get_basedimty(dimty)
        return super().__getitem__(dimty)

    def __setitem__(self, dimty: pint.util.UnitsContainer, unit: pint.Unit) -> None:
        unit = coerce_unit(unit)  # ensure Unit instance
        dimty = get_basedimty(dimty)
        super().__setitem__(dimty, unit)

    def __delitem__(self, dimty: pint.util.UnitsContainer) -> None:
        dimty = get_basedimty(dimty)
        super().__delitem__(dimty)

    def __repr__(self) -> str:
        return f"Unit preferences: {super().__repr__()}"

    @classmethod
    def from_objs(
        cls,
        objs: Iterable[pint.Unit | float | int | pint.Quantity | pd.Series | pd.DataFrame],
        collision: Literal["update", "ignore", "raise"] = "ignore",
    ) -> Self:
        """Create instance from one or more objects that a unit can be derived from. If multiple
        units are found for a dimensionality, keep first (if ``collision`` == 'ignore', default),
        keep last (if ``collision`` == 'update') or raise exception (if ``collision == 'raise')."""
        self = cls()
        for obj in objs:
            self.add(obj, collision)
        return self

    def add(
        self,
        obj: (
            str
            | pint.Unit
            | int
            | float
            | pint.Quantity
            | IntSeries
            | FloatSeries
            | PintSeries
            | SingleDimQuantitySeries
            | MultiDimQuantitySeries
            | pd.DataFrame
        ),
        collision: Literal["update", "ignore", "raise"] = "ignore",
    ) -> None:
        """Add a new unit as the preferred unit for its dimensionality.

        Parameters
        ----------
        obj
            Any object that a unit can be derived from.
        collision
            What to do if a (different) preferred unit for that dimensionality already exists.
            - "raise" to raise ValueError.
            - "ignore" to keep existing.
            - "update" to replace existing with provided unit.

        Notes
        -----
        If applicable, `obj` is traversed left to right and top to bottom. If distinct units with
        the same dimensionality are found, `collision` determines, which is kept (or if Exception
        is raised).
        """
        if isinstance(obj, str | pint.Unit):
            self._add_fromunit(obj, collision)
        elif isinstance(obj, float | int | pint.Quantity):
            self._add_fromskalar(obj, collision)
        elif isinstance(obj, pd.Series):
            self._add_fromseries(obj, collision)
        else:  # Dataframe
            self._add_fromdataframe(obj, collision)

    def _add_fromunit(
        self, unit: pint.Unit, collision: Literal["update", "ignore", "raise"]
    ) -> None:
        """Add ``unit`` as preferred unit for its dimensionality."""
        unit = coerce_unit(unit)  # ensure Unit instance
        dimty = get_basedimty(unit)
        if dimty not in self:  # add
            self[dimty] = unit
            return

        existing_unit = self[dimty]
        if collision == "ignore" or existing_unit == unit:  # no issue
            return
        elif collision == "update":
            self[dimty] = unit
        elif collision == "raise":
            raise ValueError(
                f"An existing unit ({existing_unit}) was found and it differs from the provided unit"
                f" ({unit}); dimensionality {dimty}."
            )

    def _add_fromskalar(
        self,
        sk: int | float | pint.Quantity | Any,
        collision: Literal["update", "ignore", "raise"],
    ) -> None:
        """Add unit (if it has any) from skalar ``sk`` as preferred unit for its dimensionality."""
        sk = coerce_quantity(sk)  # turns float and int into quantities
        if isinstance(sk, pint.Quantity):
            self._add_fromunit(sk.units, collision)

    def _add_fromseries(
        self,
        s: (
            IntSeries
            | FloatSeries
            | PintSeries
            | NonuniformSeries
            | SingleDimQuantitySeries
            | MultiDimQuantitySeries
        ),
        collision: Literal["update", "ignore", "raise"],
    ) -> None:
        """Collect all units used in ``s``, and add each as preferred unit for its dimensionality."""
        if isinstance(s.dtype, pint_pandas.PintType):  # one unit for entire series
            self._add_fromunit(s.pint.units, collision)
        elif pd.api.types.is_object_dtype(s.dtype):  # may contain quantities
            for sk in s.values:
                self._add_fromskalar(sk, collision)

    def _add_fromdataframe(
        self,
        df: pd.DataFrame,
        collision: Literal["update", "ignore", "raise"],
    ) -> None:
        """Collect all units used in ``df``, and add each as preferred unit for its dimensionality."""
        for _, s in df.items():
            self._add_fromseries(s, collision)
        return

    # def add_or_lookup_unit(self, unit: pint.Unit) -> pint.Unit:
    #     """If a unit is stored for the dimensionality of `unit`, return it. If not, store
    #     `unit` as the preferred unit for that dimensionality, and return it."""
    #     dimty = get_basedimty(unit)
    #     if dimty not in self:
    #         self[dimty] = unit
    #     else:
    #         unit = self[dimty]  # update
    #     return unit

    def get_fromunit(self, unit: pint.Unit) -> pint.Unit | None:
        """Look-up the unit stored for the dimensionality of `unit`, and return it (or None if none
        present."""
        unit = coerce_unit(unit)  # ensure Unit instance
        return self.get(get_basedimty(unit))

    def merge(self, other: Self) -> Self:
        """Merge 2 instances. Raises ValueError if same dimensionality with distinct units found."""
        # Check if duplicate dimensionalities have same unit.
        for dimty in set(self.keys()).intersection(set(other.keys())):
            if (u1 := self[dimty]) != (u2 := other[dimty]):
                raise ValueError(
                    f"Found distinct units ({u1} and {u2}) for dimensionality {dimty}."
                )
        return UnitPref(self | other)


def _coerce_quantity_to_preferred(
    sk: int | float | pint.Quantity, pref: Mapping[pint.util.UnitsContainer, pint.Unit]
) -> pint.Quantity:
    sk = coerce_quantity(sk)
    if (new_units := pref.get(get_basedimty(sk))) is not None:
        return sk.to(new_units)
    return sk


def _coerce_series_to_preferred(
    s: IntSeries | FloatSeries | PintSeries | SingleDimQuantitySeries | MultiDimQuantitySeries,
    pref: Mapping[pint.util.UnitsContainer, pint.Unit],
) -> pd.Series | dict[Any, pint.Quantity]:
    s_or_dict = coerce_not_quantityseries(s)
    if isinstance(s_or_dict, pd.Series):
        if (new_units := pref.get(get_basedimty(s_or_dict))) is not None:
            return s_or_dict.pint.to(new_units)
    else:  # dict: do element-by-element
        return {key: _coerce_quantity_to_preferred(sk, pref) for key, sk in s_or_dict.items()}


@overload
def convert_to_preferred(
    obj: int | float | pint.Quantity, pref: Mapping[pint.util.UnitsContainer, pint.Unit]
) -> pint.Quantity: ...


@overload
def convert_to_preferred(
    obj: pd.Series, pref: Mapping[pint.util.UnitsContainer, pint.Unit]
) -> pd.Series: ...


@overload
def convert_to_preferred(
    obj: pd.DataFrame, pref: Mapping[pint.util.UnitsContainer, pint.Unit]
) -> pd.DataFrame: ...


def convert_to_preferred(
    obj: (
        int | float | pint.Quantity | IntSeries | FloatSeries | PintSeries | pd.DataFrame | Iterable
    ),
    pref: Mapping[pint.util.UnitsContainer, pint.Unit],
) -> pint.Quantity | PintSeries | Iterable:
    """Convert ``obj`` to preferred units. Also turns floats and ints into dimensionless.

    Parameters
    ----------
    obj
        Value to convert to preferred units.
    pref
        Unit preference for given dimensionalities.

    Returns
    -------
        Same object, in other units.
    """
    if isinstance(obj, pd.DataFrame):
        return pd.DataFrame({col: _coerce_series_to_preferred(s, pref) for col, s in obj.items()})
    elif isinstance(obj, pd.Series):
        return _coerce_series_to_preferred(obj, pref)
    else:  # assume skalar, so int, float, quantity, but also bool, timestamp, ...
        return _coerce_quantity_to_preferred(obj, pref)
