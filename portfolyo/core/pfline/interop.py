"""Ensure interoperability by extracting power, energy, price, revenue, and
dimensionless values/timeseries from data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd

from ... import tools
from . import classes, create

if TYPE_CHECKING:  # needed to avoid circular imports
    from .classes import FlatPfLine

_ATTRIBUTES = ("w", "q", "p", "r", "nodim", "agn")


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
    agn: float | pd.Series = None  # agnostic

    def __post_init__(self):
        # Add correct units and check type.
        object.__setattr__(self, "w", _set_unit(self.w, "w"))
        object.__setattr__(self, "q", _set_unit(self.q, "q"))
        object.__setattr__(self, "p", _set_unit(self.p, "p"))
        object.__setattr__(self, "r", _set_unit(self.r, "r"))
        object.__setattr__(self, "nodim", _set_unit(self.nodim, "nodim"))
        object.__setattr__(self, "agn", _set_unit(self.agn, None))

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

    def assign_agn(self, da: str = None) -> InOp:
        """Set dimension-agnostic part as specific dimension (unless it's None)."""
        if self.agn is None or da is None:
            return self
        # keep = [a for a in _ATTRIBUTES if a != da]
        # return InOp(**{**{a: getattr(self, a) for a in keep}, da: self.agn})
        return self.drop("agn") | InOp(**{da: self.agn})

    def drop(self, da: str) -> InOp:
        """Drop part of the information and return new InOp object."""
        return InOp(**{attr: getattr(self, attr) for attr in _ATTRIBUTES if attr != da})

    def make_consistent(self) -> InOp:
        """Fill as much of the data as possible. All data must be None or timeseries, and
        self.agn must have been assigned (or dropped)."""

        self._assert_noagn_and_all_timeseries()

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

        self._assert_noagn_and_all_timeseries()

        # If we land here, there is no self.agn, and all other attributes are timeseries (or None).

        series = {}
        for attr in _ATTRIBUTES:
            val = getattr(self, attr)
            if val is None:
                continue
            series[attr] = val

        return pd.DataFrame(series)

    def _assert_noagn_and_all_timeseries(self):
        """Raise Error if object (still) has agnostic data or if not all data are timeseries."""
        if self.agn is not None:
            raise ValueError(
                "Object contains agnostic data; first use `.assign_agn()`."
            )

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


def _set_unit(
    v: float | int | tools.unit.Q_ | pd.Series, attr: str
) -> float | tools.unit.Q | pd.Series:
    """Set unit (if none set yet) or convert to unit."""
    if v is None:
        return None

    unit = tools.unit.from_name(attr) if attr else None

    if unit is None:  # should be unit-agnostic
        if isinstance(v, float):
            return v
        if isinstance(v, int):
            return float(v)
        if isinstance(v, pd.Series) and isinstance(v.index, pd.DatetimeIndex):
            v = _timeseries_of_floats_or_pint(v)  # float-series or pint-series
            if hasattr(v, "pint"):
                raise ValueError(
                    "Agnostic timeseries should not have a dimension and should not be "
                    f"dimensionless. Should be plain number values; found {v.pint.units}."
                )
            return v
        raise TypeError(
            f"Value should be a number or timeseries of numbers; got {type(v)}."
        )

    else:  # should be unit-aware
        if isinstance(v, float):
            return tools.unit.Q_(v, unit)  # add unit
        if isinstance(v, int):
            return tools.unit.Q_(float(v), unit)  # add unit
        if isinstance(v, tools.unit.Q_):
            return tools.unit.Q_(v, unit)  # convert to unit
        if isinstance(v, pd.Series) and isinstance(v.index, pd.DatetimeIndex):
            v = _timeseries_of_floats_or_pint(v)  # float-series or pint-series
            return v.astype(f"pint[{unit:P}]")
        raise TypeError(
            f"Value should be a number, Quantity, or timeseries; got {type(v)}."
        )


def _timeseries_of_floats_or_pint(s: pd.Series) -> pd.Series:
    """Check if a timeseries is a series of objects, and if so, see if these objects are
    actually Quantities."""

    # Turn into floats-series or pint-series.

    if s.dtype != object:
        if not hasattr(s, "pint"):
            s = s.astype(float)  # int to float
        else:
            magnitudes = s.pint.magnitude
            if pd.api.types.is_integer_dtype(magnitudes.dtype):
                # series of int to series of float
                s = pd.Series(magnitudes.astype(float), dtype=s.dtype)

    else:
        # object -> maybe series of Quantitis -> convert to pint-series.
        if not all(isinstance(val, tools.unit.Q_) for val in s.values):
            raise TypeError(f"Timeseries with unexpected data type: {s.dtype}.")
        quantities = [val.to_base_units() for val in s.values]
        magnitudes = [q.m for q in quantities]
        units = list(set([q.u for q in quantities]))
        if len(units) != 1:
            raise ValueError(f"Timeseries needs uniform unit; found {','.join(units)}.")
        s = pd.Series(magnitudes, s.index, dtype=f"pint[{units[0]:P}]")

    # Check if all OK.

    try:
        tools.standardize.assert_frame_standardized(s)
    except AssertionError as e:
        raise ValueError(
            "Timeseries not in expected form. See ``portfolyo.standardize()`` for more information."
        ) from e

    return s


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
        return InOp(agn=float(data))

    elif isinstance(data, float):
        return InOp(agn=data)

    elif isinstance(data, tools.unit.Q_):
        return InOp(**{_unit2attr(data.units): data})

    elif isinstance(data, pd.Series) and isinstance(data.index, pd.DatetimeIndex):
        # timeseries
        data = tools.unit.avoid_frame_of_objects(data)
        if data.dtype in [float, int]:
            return InOp(agn=data)
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
    data: Any, ref_index: pd.DatetimeIndex, agn_default: str = None
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
    inop = InOp.from_data(data).assign_agn(agn_default).to_timeseries(ref_index)

    if inop.p is inop.q is inop.w is inop.r is inop.nodim is None and (
        inop.agn is None or np.allclose(inop.agn, 0)
    ):
        # Data was None, or
        # Data was dimension-agnostic 0.0 and shouldn't be interpreted as something else.
        return None

    elif inop.agn is not None:
        raise ValueError(
            "Cannot do this operation. If you meant to specify data of a certain dimension"
            " or unit, make it more explicit, either by specifying 'w', 'p', etc. as e.g."
            " a dictionary key, or by setting a unit e.g. in a pint Quantity."
        )

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
