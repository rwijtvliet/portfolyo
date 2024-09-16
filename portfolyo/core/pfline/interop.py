"""Ensure interoperability by extracting power, energy, price, revenue, and
dimensionless values/timeseries from data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd
from pint import DimensionalityError

from ... import tools
from . import classes, create

if TYPE_CHECKING:  # needed to avoid circular imports
    from .classes import FlatPfLine

_ATTRIBUTES = ("w", "q", "p", "r", "nodim")


@dataclass(frozen=True)
class InOp:
    """Class to check increase interoperability. Tries to extract power (w), energy (q),
    price (p), revenue (r), adimensional (nodim) and dim-agnostic (agn) information from
    the provided data.

    Typical usage:

    . Initialisation:
        inop = Inop(w=..., q=..., ...)
    or
        inop = Inop.from_data(...)

    . Turn all into timeseries:
        inop = inop.to_timeseries()

    . Assign unit-agnostic to unit-specific:
        inop = inop.assign_agn('p')

    . Check for consistency and fill the missing attributes as much as possible:
        inop = inop.make_consistent()

    . And finally, return as dataframe with columns 'w', 'q', 'p', 'r', 'nodim'
        inop = inop.to_df()
    """

    w: tools.unit.Q_ | pd.Series = None
    q: tools.unit.Q_ | pd.Series = None
    p: tools.unit.Q_ | pd.Series = None
    r: tools.unit.Q_ | pd.Series = None
    nodim: tools.unit.Q_ | pd.Series = None  # explicitly dimensionless
    agn: float | pd.Series = None  # agnostic TODO factor out

    def __post_init__(self):
        # Add correct units and check type.
        object.__setattr__(self, "w", _check_unit(self.w, "w"))
        object.__setattr__(self, "q", _check_unit(self.q, "q"))
        object.__setattr__(self, "p", _check_unit(self.p, "p"))
        object.__setattr__(self, "r", _check_unit(self.r, "r"))
        object.__setattr__(self, "nodim", _check_unit(self.nodim, "nodim"))

    @classmethod
    def from_data(cls, data):
        return _from_data(data)

    def to_timeseries(self, ref_index=None) -> InOp:
        """Turn all values into timeseries or None. If none of the attributes is a
        timeseries, and no ``ref_index`` is provided, raise Error. If >1 is a timeseries,
        store only the timestamps where they overlap (i.e., intersection)."""
        # Get index.
        indices = [] if ref_index is None else [ref_index]
        for attr in _ATTRIBUTES:
            if isinstance(val := getattr(self, attr), pd.Series):
                indices.append(val.index)
        index = tools.intersect.indices(*indices)  # raises error if none passed
        if not len(index):
            raise ValueError("Data has no overlapping timestamps.")
        # Save all values as timeseries.
        kwargs = {}
        for attr in _ATTRIBUTES:
            val = getattr(self, attr)
            if val is None:
                continue
            elif isinstance(val, pd.Series):
                kwargs[attr] = val.loc[index]
            elif isinstance(val, tools.unit.Q_):
                kwargs[attr] = pd.Series(val.m, index, dtype=f"pint[{val.units:P}]")
            else:  # float
                kwargs[attr] = pd.Series(val, index)
        # Return as new InOp instance.
        return InOp(**kwargs)

    def drop(self, da: str) -> InOp:
        """Drop part of the information and return new InOp object."""
        return InOp(**{attr: getattr(self, attr) for attr in _ATTRIBUTES if attr != da})

    def make_consistent(self) -> InOp:
        """Fill as much of the data as possible. All data must be None or timeseries, and
        self.agn must have been assigned (or dropped)."""

        self._assert_all_timeseries()

        # If we land here, there is no self.agn, and all other attributes are timeseries (or None).

        w, q, p, r, nodim = self.w, self.q, self.p, self.r, self.nodim

        # Volumes.
        if w is not None and q is not None:
            try:
                tools.testing.assert_series_equal(
                    w, q / q.index.duration, check_names=False
                )
            except AssertionError as e:
                raise ValueError("Values for w and q are not consistent.") from e
        elif w is not None and q is None:
            q = w * w.index.duration
        elif w is None and q is not None:
            w = q / q.index.duration
        elif w is None and q is None and p is not None and r is not None:
            q = r / p
            w = q / q.index.duration

        # If we land here, there are no more options to find w and q.
        # They are consistent with each other but might be inconsistent with p and r.

        # Price.
        if p is None and q is not None and r is not None:
            p = r / q

        # If we land here, there are no more options to find p.
        # It may be inconsistent with w, q and r.

        # Revenue.
        if r is None and q is not None and p is not None:
            r = q * p
            # Make correction for edge case: p unknown (nan or inf) and q==0 --> assume r=0
            mask = np.isclose(q.pint.m, 0) & (p.isna() | np.isinf(p.pint.m))
            if mask.any():
                r[mask] = 0

        # If we land here, there are no more options to find r.
        # It may be inconsistent with w, q and p.

        # Consistency.
        if q is not None and p is not None and r is not None:
            # Check for consistency, but ignore edge cases:
            # - p unknown (nan or inf) and q==0 --> ignore
            # - q unknown (nan or inf) and p==0 --> ignore
            ign1 = np.isclose(q.pint.m, 0) & (p.isna() | np.isinf(p.pint.m))
            ign2 = np.isclose(p.pint.m, 0) & (q.isna() | np.isinf(q.pint.m))
            ignore = ign1 | ign2
            try:
                tools.testing.assert_series_equal(
                    r[~ignore], p[~ignore] * q[~ignore], check_names=False
                )
            except AssertionError as e:
                raise ValueError("Values for r, p, and q are not consistent.") from e

        return InOp(w=w, q=q, p=p, r=r, nodim=nodim)

    def to_df(self) -> pd.DataFrame:
        """Create dataframe with (at most) columns w, q, p, r, nodim. All data must be
        None or timeseries, and self.agn must have been assigned (or dropped). Also, you'll
        probably want to have run the `.make_consistent()` method."""

        self._assert_all_timeseries()

        # If we land here, all attributes are timeseries (or None).

        series = {}
        for attr in _ATTRIBUTES:
            val = getattr(self, attr)
            if val is None:
                continue
            series[attr] = val

        return pd.DataFrame(series)

    def _assert_all_timeseries(self):
        """Raise Error if object (still) has agnostic data or if not all data are timeseries."""

        # Guard clause.
        errors = {}
        for attr in _ATTRIBUTES:
            val = getattr(self, attr)
            if val is None:
                continue
            if isinstance(val, pd.Series):
                continue
            errors[attr] = type(val)
        if errors:
            raise ValueError(
                "Object contains non-timeseries data; first use `.to_timeseries()`."
            )

    def __bool__(self) -> bool:
        return not all(getattr(self, attr) is None for attr in _ATTRIBUTES)

    def __or__(self, other: InOp) -> InOp:
        return _union(self, other)

    __ror__ = __or__

    def __eq__(self, other) -> bool:
        return _equal(self, other)


def _check_unit(
    v: float | int | tools.unit.Q_ | pd.Series, attr: str
) -> float | tools.unit.Q | pd.Series:
    """Check the unit and dimensionality of a value.

    This function verifies if the given value `v` has the correct unit dimensionality
    corresponding to the attribute `attr`.

    """
    # Retrieve the expected dimensionality for the given attribute.
    dim = tools.unit.NAMES_AND_DIMENSIONS[attr]
    if v is None:
        return v

    # Check if the value is a float or int and ensure it is dimensionless.
    if isinstance(v, float | int):
        if dim is not tools.unit.NAMES_AND_DIMENSIONS["nodim"]:
            raise DimensionalityError(
                dim,
                tools.unit.NAMES_AND_DIMENSIONS["nodim"],
                extra_msg="Float or int only allowed for dimensionless value. To specify a physical quantity, add a unit.",
            )
        else:
            return float(v)

    elif isinstance(v, tools.unit.Q_):
        if not v.dimensionality == dim:
            raise DimensionalityError(
                dim,
                v.pint.dimensionality,
                extra_msg=f"Incorrect dimension for this attribute; expected {dim}, got {v.pint.dimensionality}",
            )
            # if the dim is nodim, we retun float
        elif v.dimensionality == tools.unit.NAMES_AND_DIMENSIONS["nodim"]:
            return float(v)
        # else
        else:
            return v

    elif isinstance(v, pd.Series) and isinstance(v.index, pd.DatetimeIndex):
        # Turn into floats-series or pint-series.
        v = tools.unit.avoid_frame_of_objects(v)
        if pd.api.types.is_float_dtype(v):
            if dim is not tools.unit.NAMES_AND_DIMENSIONS["nodim"]:
                raise DimensionalityError(
                    dim,
                    tools.unit.NAMES_AND_DIMENSIONS["nodim"],
                    extra_msg=f". Float or int only allowed for dimensionless value, got {dim}. To specify a physical quantity, add a unit.",
                )
            else:
                return v
        else:
            if not v.pint.dimensionality == dim:
                raise DimensionalityError(
                    dim,
                    v.pint.dimensionality,
                    extra_msg=f"Incorrect dimension for this attribute; expected {dim}, got {v.pint.dimensionality}",
                )
        # Check if series is standardised.
        try:
            tools.standardize.assert_frame_standardized(v)
        except AssertionError as e:
            raise ValueError(
                "Timeseries not in expected form. See ``portfolyo.standardize()`` for more information."
            ) from e

        return v

    raise TypeError(
        f"Value should be a number, Quantity, or timeseries; got {type(v)}."
    )


def _unit2attr(unit) -> str:
    attr = tools.unit.to_name(unit)  # Error if dimension unknown
    if attr not in _ATTRIBUTES:
        raise NotImplementedError(f"Cannot handle data with this unit ({unit}).")
    return attr


def _from_data(
    data: float | tools.unit.Q_ | pd.Series | Dict | pd.DataFrame | Iterable | Mapping,
) -> InOp:
    """Turn ``data`` into a InterOp object."""

    if data is None:
        return InOp()

    elif isinstance(data, int):
        return InOp(nodim=float(data))

    elif isinstance(data, float):
        return InOp(nodim=data)

    elif isinstance(data, tools.unit.Q_):
        return InOp(**{_unit2attr(data.units): data})

    elif isinstance(data, pd.Series) and isinstance(data.index, pd.DatetimeIndex):
        # timeseries
        data = tools.unit.avoid_frame_of_objects(data)
        if pd.api.types.is_float_dtype(data):
            return InOp(nodim=data)
        else:
            return InOp(**{_unit2attr(data.pint.units): data})

    elif (
        isinstance(data, pd.DataFrame)
        or isinstance(data, pd.Series)
        or isinstance(data, Mapping)
    ):

        def dimabbr(key):  # following keys return 'w': 'w', ('w', 'pf1'), ('pf1', 'w')
            if key in _ATTRIBUTES:
                return key
            elif not isinstance(key, str) and isinstance(key, Iterable):
                if (da := dimabbr(key[0])) is not None:
                    return da
                if (da := dimabbr(key[-1])) is not None:
                    return da
            return None

        inops = []
        for key, value in data.items():
            if da := dimabbr(key):
                inops.append(InOp(**{da: value}))
            else:
                raise KeyError(
                    f"Found item with unexpected key/name '{key}'. Should be one of {', '.join(_ATTRIBUTES)}."
                )
        return _multiple_union(inops)

    elif isinstance(data, Iterable):
        return _multiple_union((InOp.from_data(element) for element in data))

    raise TypeError(
        f"Expecting number, Quantity, timeseries, Mapping (e.g. dict or DataFrame), or Iterable; got {type(data)}."
    )


def _multiple_union(inops: Iterable[InOp]) -> InOp:
    inop_result = None
    for inop in inops:
        inop_result |= inop
    if inop_result is None:
        return InOp()  # raises Error
    return inop_result


def _union(inop1: InOp, inop2: InOp) -> InOp:
    """Combine 2 ``InOp`` objects, and raise error if same dimension is supplied twice."""
    if inop2 is None:
        return inop1
    if not isinstance(inop2, InOp):
        raise TypeError("Can only unite same object type.")
    kwargs = {}
    for attr in _ATTRIBUTES:
        val1, val2 = getattr(inop1, attr), getattr(inop2, attr)
        if val1 is not None and val2 is not None:
            raise ValueError(f"Got two values for attribute '{attr}'.")
        elif val1 is not None:
            kwargs[attr] = val1
        else:
            kwargs[attr] = val2
    return InOp(**kwargs)


def _equal(inop1: InOp, inop2: InOp) -> InOp:
    """2 ``InOp`` objects are equal if they have the same attributes."""
    if not isinstance(inop2, InOp):
        return False
    for attr in _ATTRIBUTES:
        val1, val2 = getattr(inop1, attr), getattr(inop2, attr)
        if type(val1) is not type(val2):
            return False
        if isinstance(val1, pd.Series):
            try:
                tools.testing.assert_series_equal(val1, val2, check_names=False)
            except AssertionError:
                return False
        elif val1 != val2:
            return False
    return True


def pfline_or_nodimseries(
    data: Any, ref_index: pd.DatetimeIndex
) -> None | pd.Series | FlatPfLine:
    """Turn ``data`` into PfLine if dimension-aware. If not, turn into Series."""

    # Already a PfLine.
    if isinstance(data, classes.PfLine):
        return data

    # Can be turned into a PfLine.
    try:
        return create.pfline(data)
    except ValueError:
        pass

    # Turn into InOp object with timeseries.
    inop = InOp.from_data(data).to_timeseries(ref_index)

    if inop.p is inop.q is inop.w is inop.r is inop.nodim is None:
        # Data was None
        return None

    elif inop.nodim is None:
        # Only dimension-aware data was supplied; must be able to turn into PfLine.
        return create.flatpfline(inop)

    elif inop.p is inop.q is inop.w is inop.r is None:
        # Only dimensionless data was supplied; is Series of factors.
        return inop.nodim

    else:
        raise NotImplementedError(
            "Found a mix of dimension-aware and dimensionless data."
        )
