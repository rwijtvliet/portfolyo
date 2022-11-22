"""Ensure interoperability by extracting power, energy, price, revenue, and
dimensionless values/timeseries from data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Union

import numpy as np
import pandas as pd

from ... import testing, tools
from . import base, single

if TYPE_CHECKING:  # needed to avoid circular imports
    from .single import SinglePfLine

_ATTRIBUTES = ("w", "q", "p", "r", "nodim", "agn")


@dataclass
class InOp:
    """Class to check increase interoperability. Tries to extract power (w), energy (q),
    price (p), revenue (r), adimensional (nodim) and dim-agnostic (agn) information from
    the provided data; initially without checking for consistency."""

    w: Union[tools.unit.Q_, pd.Series] = None
    q: Union[tools.unit.Q_, pd.Series] = None
    p: Union[tools.unit.Q_, pd.Series] = None
    r: Union[tools.unit.Q_, pd.Series] = None
    nodim: Union[tools.unit.Q_, pd.Series] = None  # explicitly dimensionless
    agn: Union[float, pd.Series] = None  # agnostic

    def __post_init__(self):
        # Add correct units and check type.
        self.w = _set_unit(self.w, "w")
        self.q = _set_unit(self.q, "q")
        self.p = _set_unit(self.p, "p")
        self.r = _set_unit(self.r, "r")
        self.nodim = _set_unit(self.nodim, "nodim")
        self.agn = _set_unit(self.agn, None)

    @classmethod
    def from_data(cls, data):
        return _from_data(data)

    def to_timeseries(self, index=None):
        """Turn all values into timeseries or None. If none of the attributes is a
        timeseries, and no ``index`` is provided, raise Error. If >1 is a timeseries,
        store only the timestamps where they overlap (i.e., intersection)."""
        # Get index.
        indices = [] if index is None else [index]
        for attr in _ATTRIBUTES:
            if isinstance(val := getattr(self, attr), pd.Series):
                indices.append(val.index)
        index = tools.intersection.index(*indices)  # raises error if none passed
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
        return InOp(**kwargs)

    def assign_agn(self, da: str = None):
        """Set dimension-agnostic part as specific dimension (unless it's None)."""
        if self.agn is None or da is None:
            return self
        # keep = [a for a in _ATTRIBUTES if a != da]
        # return InOp(**{**{a: getattr(self, a) for a in keep}, da: self.agn})
        return self.drop("agn") | InOp(**{da: self.agn})

    def drop(self, da: str):
        """Drop part of the information and return new InOp object."""
        return InOp(**{attr: getattr(self, attr) for attr in _ATTRIBUTES if attr != da})

    def __bool__(self):
        return not all(getattr(self, attr) is None for attr in _ATTRIBUTES)

    def __or__(self, other):
        return _union(self, other)

    __ror__ = __or__

    def __eq__(self, other):
        return _equal(self, other)


def _set_unit(
    v: Union[float, int, tools.unit.Q_, pd.Series], attr: str
) -> Union[float, tools.unit.Q, pd.Series]:
    """Add unit (if no unit set yet) or convert to unit."""
    if v is None:
        return None

    unit = tools.unit.from_name(attr) if attr else None

    if unit is None:  # should be unit-agnostic
        if isinstance(v, int) or isinstance(v, float):
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
        if isinstance(v, float) or isinstance(v, int) or isinstance(v, tools.unit.Q_):
            return tools.unit.Q_(v, unit)  # add unit or convert to unit
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

    if s.dtype != object and not hasattr(s, "pint"):
        s = s.astype(float)  # int to float

    elif s.dtype == object:
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
    data: Union[float, tools.unit.Q_, pd.Series, Dict, pd.DataFrame, Iterable, Mapping]
) -> InOp:
    """Turn ``data`` into a InterOp object."""

    if data is None:
        return InOp()

    elif isinstance(data, int) or isinstance(data, float):
        return InOp(agn=data)

    elif isinstance(data, tools.unit.Q_):
        return InOp(**{_unit2attr(data.units): data})

    elif isinstance(data, pd.Series) and isinstance(data.index, pd.DatetimeIndex):
        # timeseries
        if hasattr(data, "pint"):  # pint timeseries
            return InOp(**{_unit2attr(data.pint.units): data})
        elif data.dtype == object:  # timeeries of objects -> maybe Quantities?
            if len(data) and isinstance(val := data.values[0], tools.unit.Q_):
                # use unit of first value to find dimension
                return InOp(**{_unit2attr(val.u): data})
        else:  # assume float or int
            return InOp(agn=data)

    elif (
        isinstance(data, pd.DataFrame)
        or isinstance(data, pd.Series)
        or isinstance(data, Mapping)
    ):

        def dimabbr(key):
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


def _multiple_union(inops) -> InOp:
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
                testing.assert_series_equal(val1, val2, check_names=False)
            except AssertionError:
                return False
        elif val1 != val2:
            return False
    return True


def pfline_or_nodimseries(
    data: Any, ref_index: pd.DatetimeIndex, agn_default: str = None
) -> Union[None, pd.Series, SinglePfLine]:
    """Turn ``data`` into PfLine if dimension-aware. If not, turn into Series."""

    # Already a PfLine.
    if isinstance(data, base.PfLine):
        return data

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
            "Cannot do this operation. If you meant to specify data of a certain dimension / unit, try making it more "
            "explicit by specifying 'w', 'p', etc. as e.g. a dictionary key, or by setting the unit."
        )

    elif inop.nodim is None:
        # Only dimension-aware data was supplied; must be able to turn into PfLine.
        return single.SinglePfLine(inop)

    elif inop.p is inop.q is inop.w is inop.r is None:
        # Only dimensionless data was supplied; is Series of factors.
        return inop.nodim

    else:
        raise NotImplementedError(
            "Cannot do arithmatic; found a mix of dimension-aware and dimensionless data."
        )
