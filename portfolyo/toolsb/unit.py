"""Working with pint units."""

import functools
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, overload

import pandas as pd
import pint
import pint_pandas
from typing_extensions import Self

from . import _decorator as tools_decorator
from .types import PintSeries

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
    obj: pint.util.UnitsContainer | pint.Unit | pint.Quantity | pd.Series | str | float | int,
) -> pint.util.UnitsContainer:
    """Get base dimensionality of ``obj``."""
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


@functools.lru_cache()
def convert_unit(unit: pint.Unit | str | None) -> pint.Unit:
    """If possible, turn `unit` into Unit. If not possible, return as-is."""
    if unit is None:
        unit = Unit("")  # intepret as dimensionless
    elif not isinstance(unit, pint.Unit):
        try:
            unit = Unit(unit)
        except pint.UndefinedUnitError:
            pass
    return unit


def validate_unit(unit: Any) -> None:
    if not isinstance(unit, pint.Unit):
        raise ValueError(
            f"'{unit}' is not a (defined) unit. Add to unit registry by relating to existing unit,"
            " e.g. with 'ureg.define('BTU = 0.293071 Wh')'. To add a currency, specify dimension"
            " name instead, e.g. with 'ureg.define('THB = [currency_Thai]')'. The dimension"
            " name must start with 'currency_', and units must not collide with existing ones."
        )


coerce_unit = tools_decorator.coerce_fn(convert_unit, validate_unit)

apply_coercion_unit = tools_decorator.create_coerciondecorator(
    convert_unit, None, default_param="unit"
)


# Conversion and validation: Quantity.
# ------------------------------------


def convert_quantity(sk: int | float | pint.Quantity) -> pint.Quantity:
    """If possible, turn `sk` into Quantity. If not possible, return as-is."""
    if isinstance(sk, int):
        return Q_(float(sk), "")
    elif isinstance(sk, float):
        return Q_(sk, "")
    return sk


def validate_quantity(sk: Any) -> None:
    if not isinstance(sk, pint.Quantity):
        raise ValueError(f"This is not a Quantity: {sk}.")


coerce_quantity = tools_decorator.coerce_fn(convert_quantity, validate_quantity)


apply_coercion_quantity = tools_decorator.create_coerciondecorator(
    convert_quantity, validate_quantity
)


# Conversion and validation: Series and Dataframe.
# ------------------------------------------------


@overload
def convert_pintframe(fr: pd.Series) -> pd.Series: ...


@overload
def convert_pintframe(fr: pd.DataFrame) -> pd.DataFrame: ...


def convert_pintframe(fr: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """If possible, turn Series/DataFrame into (collection of) pintseries. Converts Series of
    Quantities with uniform dimensionality into pintseries. If not possible, return as-is.
    """
    if isinstance(fr, pd.DataFrame):
        return pd.DataFrame({c: convert_pintframe(s) for c, s in fr.items()})

    # If we are here, `fr` is a series.

    # NOTE: Can't use `.is_numeric_dtype`, because also true for pint dtype.
    if pd.api.types.is_integer_dtype(fr.dtype):
        return fr.astype(float).astype("pint[]")
    elif pd.api.types.is_float_dtype(fr.dtype):
        return fr.astype("pint[]")

    elif pd.api.types.is_object_dtype(fr.dtype) and isinstance(fr.iloc[0], pint.Quantity):
        units = fr.iloc[0].units
        try:
            return fr.astype(f"pint[{units}]")
        except pint.DimensionalityError:
            return fr  # series of quantities with distinct dimension; keep as-is

    return fr  # bools, timestamps, ...


def validate_pintframe(fr: pd.Series | pd.DataFrame) -> None:
    if isinstance(fr, pd.DataFrame):
        for _, s in fr.items():
            validate_pintframe(s)
        return

    # If we are here, `fr` is a series.

    if not isinstance(fr.dtype, pint_pandas.PintType):
        raise ValueError(f"This is not a pintseries: {fr}.")


coerce_pintframe = tools_decorator.coerce_fn(convert_pintframe, validate_pintframe)

apply_coercion_pintframe = tools_decorator.create_coerciondecorator(
    convert_pintframe, validate_pintframe, default_param="fr"
)

# additional, further-reaching conversions.


@overload
def convert_pintframe_reducedunits(fr: pd.Series) -> pd.Series: ...


@overload
def convert_pintframe_reducedunits(fr: pd.DataFrame) -> pd.DataFrame: ...


def convert_pintframe_reducedunits(
    fr: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """Like ``convert_pintframe``, but if possible also reduce number of units by converting like
    dimensionalities (e.g. MW and kW) to one unit. For series: relevant if series of quantities.
    For dataframes: additionally relevant if pintseries with same dimensionality. If not possible,
    return as-is."""
    fr = convert_pintframe(fr)  # Float and ints to quantities. Series of quantities to pintseries.
    pref = UnitPref.from_objs(fr)  # collect units
    return convert_to_preferred(fr, pref)  # apply units


def validate_pintframe_oneunitperdim(fr: pd.Series | pd.DataFrame) -> None:
    """Validate that pintframe has only one unit for each dimensionality."""
    UnitPref().add(fr, "raise")  # will raise error if multiple units found for same dimensionality


coerce_pintframe_oneunitperdim = tools_decorator.coerce_fn(
    convert_pintframe_reducedunits, validate_pintframe_oneunitperdim
)

apply_coercion_pintframe_oneunitperdim = tools_decorator.create_coerciondecorator(
    convert_pintframe_reducedunits, validate_pintframe_oneunitperdim
)


def validate_pintframe_oneunit(fr: pd.Series | pd.DataFrame) -> None:
    """Validate that pintframe has only one dimensionality with only one unit."""
    unitpref = UnitPref()
    unitpref.add(fr, "raise")  # will raise error if multiple units found for same dimensionality
    if len(unitpref) > 1:
        raise ValueError(f"Found multiple dimensionalities: {list(unitpref.keys())}.")


coerce_pintframe_oneunit = tools_decorator.coerce_fn(
    convert_pintframe_reducedunits, validate_pintframe_oneunit
)

apply_coercion_pintframe_oneunit = tools_decorator.create_coerciondecorator(
    convert_pintframe_reducedunits, validate_pintframe_oneunit
)


# Conversion and validation: Unit preference.
# -------------------------------------------


class UnitPref(dict):
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
        obj: pint.Unit | float | int | pint.Quantity | pd.Series | pd.DataFrame,
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
        else:  # Series or Dataframe
            self._add_fromframe(obj, collision)

    def _add_fromunit(
        self, unit: pint.Unit, collision: Literal["update", "ignore", "raise"]
    ) -> None:
        """Add `unit` as preferred unit for its dimensionality."""
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
                f"An existing unit ({existing_unit}) was found and it differs from the provided unit ({unit}); dimensionality {dimty}."
                " To avoid raising an error, see `.add_or_lookup`."
            )

    def _add_fromskalar(
        self,
        sk: float | int | pint.Quantity | Any,
        collision: Literal["update", "ignore", "raise"],
    ) -> None:
        """Add unit (if it has any) from skalar `sk` as preferred unit for its dimensionality."""
        sk = convert_quantity(sk)  # turns float and int into quantities
        if isinstance(sk, pint.Quantity):
            self._add_fromunit(sk.units, collision)

    def _add_fromframe(
        self,
        fr: pd.Series | pd.DataFrame,
        collision: Literal["update", "ignore", "raise"],
    ) -> None:
        """Collect all units used in `fr`, and add each as preferred unit for its dimensionality."""
        if isinstance(fr, pd.DataFrame):
            for _, s in fr.items():
                self._add_fromframe(s, collision)
            return

        # If we are here, `fr` is a series.

        if isinstance(fr.dtype, pint_pandas.PintType):  # one unit for entire series
            self._add_fromunit(fr.pint.units, collision)
        elif pd.api.types.is_object_dtype(fr.dtype):  # may contain quantities
            for sk in fr.values:
                self._add_fromskalar(sk, collision)

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
        """Look-up the unit stored for the dimenisionality of `unit`, and return it (or None if none
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


def _convert_skalar_to_preferred(
    sk: float | int | pint.Quantity | Any,
    pref: Mapping[pint.util.UnitsContainer, pint.Unit],
) -> pint.Quantity:
    sk = convert_quantity(sk)
    if isinstance(sk, pint.Quantity):
        if (new_units := pref.get(get_basedimty(sk))) is not None:
            return sk.to(new_units)
    return sk  # bools, timestamps, ...; unknown unit; skalar without unit preference


def _convert_series_to_preferred(
    s: pd.Series,
    pref: Mapping[pint.util.UnitsContainer, pint.Unit],
) -> PintSeries | pd.Series:
    s = convert_pintframe(s)
    if isinstance(s.dtype, pint_pandas.PintType):
        if (new_units := pref.get(get_basedimty(s))) is not None:
            return s.pint.to(new_units)
    elif pd.api.types.is_object_dtype(s.dtype):
        # Do element-by-element
        return pd.Series([_convert_skalar_to_preferred(sk, pref) for sk in s], s.index, name=s.name)
    return s  # bools, timestamps, ...; unknown unit; pintseries without unit preference


@overload
def convert_to_preferred(
    obj: int | float | pint.Quantity,
    pref: Mapping[pint.util.UnitsContainer, pint.Unit],
) -> pint.Quantity: ...


@overload
def convert_to_preferred(
    obj: pd.Series,
    pref: Mapping[pint.util.UnitsContainer, pint.Unit],
) -> pd.Series: ...


@overload
def convert_to_preferred(
    obj: pd.DataFrame,
    pref: Mapping[pint.util.UnitsContainer, pint.Unit],
) -> pd.DataFrame: ...


def convert_to_preferred(
    obj: int | float | pint.Quantity | pd.Series | pd.DataFrame,
    pref: Mapping[pint.util.UnitsContainer, pint.Unit],
) -> pint.Quantity | pd.Series | pd.DataFrame:
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
        return pd.DataFrame({col: _convert_series_to_preferred(s, pref) for col, s in obj.items()})
    elif isinstance(obj, pd.Series):
        return _convert_series_to_preferred(obj, pref)
    else:  # assume skalar, so int, float, quantity, but also bool, timestamp, ...
        return _convert_skalar_to_preferred(obj, pref)
