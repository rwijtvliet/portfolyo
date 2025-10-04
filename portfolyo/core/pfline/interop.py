"""Ensure interoperability by extracting power, energy, price, revenue, and dimensionless values/timeseries from data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

import pandas as pd
import pint

from ... import toolsb
from ...toolsb.types import COLS, PintTimeDataframe, PintTimeSeries, TimeDataframe, TimeSeries


@dataclass
class InOp:
    """Class to check increase interoperability. Tries to extract quantity rate (w), quantity (q),
    price (p), revenue (r) and adimensional (nodim) information from the provided data.

    Typical usage:

    . Initialisation:
        inop = InOp(w=..., r=...)
      or
        inop = InOp.from_data(...)
      or from specific object types
        inop = InOp.from_timeseries(...)
        inop = InOp.from_mapping(...)

    . Then, turn all into dataframe with columns 'w', 'q', 'p', 'r', 'nodim' (possibly providing
      reference index, in case skalars were provided):
        df = inop.to_df(ref_index)

    Notes
    -----
    After initialisation, all provided data has be mapped to one of the 5 field values. No further
    checks are done to verify the data fits the fields (unless necessary for mapping), nor is the
    data processed (e.g. turned into timeseries) or made consistent (with same index and values/units
    that are consistent, i.e., with w = q/duration and [dimensionality w] = [dimensionality q] / [time].

    All these checks and transformations ARE done, however, when turning the data into a dataframe
    with .to_df().

    NO checks are done to verify that the units are valid in the context of the energy industry. So:
    the dimensionality of `q` is not necessarily [energy] or [emissions], but always
    [dimensionality q] == [dimensionality w] * [time] == [dimensionality r] / [dimensionality p],
    (if p and/or r specified).
    """

    fields: dict[Literal["w", "q", "p", "r", "nodim"], pint.Quantity | PintTimeSeries]

    def __post_init__(self):
        # Ensure each value is a Quantity or a PintTimeSeries. Also,
        # ensure dimensionality of each value fits with the field.
        self.fields = {
            field: _process_single_field(field, value)
            for field, value in self.fields.items()
            if value is not None
        }

    # Class methods to get data in. Only need to map field ('w', 'q', etc) to data. Do not need
    # to do any conversion of data; that is done post-init.

    @classmethod
    def from_data(cls, data: Any) -> InOp:
        if isinstance(data, int | float | pint.Quantity):
            return cls.from_skalar(data)
        elif isinstance(data, pd.Series):
            for fn in (cls.from_timeseries, cls.from_mapping):
                try:
                    return fn(data)
                except Exception as e:
                    last_error = e
            raise last_error()
        elif isinstance(data, pd.DataFrame):
            return cls.from_timedataframe(data)
        else:
            return cls.from_iterable(data)

    @classmethod
    def from_skalar(cls, data: float | int | pint.Quantity) -> InOp:
        data = toolsb.unit.coerce_quantity(data)
        field = toolsb.wqpr.valid_col(data, nodim_allowed=True)
        return cls({field: data})

    @classmethod
    def from_timeseries(cls, data: TimeSeries) -> InOp:
        data = toolsb.unit.coerce_pintseries(data)
        field = toolsb.wqpr.valid_col(data, nodim_allowed=True)
        return cls({field: data})

    @classmethod
    def from_timedataframe(cls, data: TimeDataframe) -> InOp:
        return cls(data.to_dict("series"))

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[
            Literal["w", "q", "p", "r", "nodim"], float | int | pint.Quantity | pd.Series
        ],
    ) -> InOp:
        return cls(data)

    @classmethod
    def from_iterable(cls, data: Iterable[float | int | pint.Quantity | pd.Series]) -> InOp:
        return _multiple_union(InOp.from_data(element) for element in data)

    # Further processing.

    def _to_timeseries(
        self, ref_index: pd.DatetimeIndex | None = None
    ) -> dict[Literal["w", "q", "p", "r", "nodim"], PintTimeSeries]:
        """Turn all fields into pinttimeseries (in-place). If none of the attributes is a
        timeseries, and no ``ref_index`` is provided, raise Error. If >1 is a timeseries, keep only
        the timestamps where they overlap (i.e., intersection)."""

        # Get index.
        indices = [value.index for value in self.fields.values() if isinstance(value, pd.Series)]
        if ref_index is not None:
            indices.append(ref_index)
        index = toolsb.index.intersect(indices)  # raises error if none passed or incompatible
        if index.empty:
            raise ValueError("Data has no overlapping timestamps.")

        # Turn all into timeseries.
        fields_as_pinttimeseries = {}
        for field, value in self.fields.items():
            if isinstance(value, pint.Quantity):
                pinttimeseries = pd.Series(value.magnitude, index).astype(f"pint[{value.units}]")
            else:  # pinttimeseries
                pinttimeseries = value.loc[index]
            fields_as_pinttimeseries[field] = pinttimeseries

        # For all but 'nodim': add data that is missing but can be calculated. Also check if redundant info is correct.
        w, q, p, r = toolsb.wqpr.complete_and_verify_consistency(
            **{field: fields_as_pinttimeseries.get(field) for field in COLS}
        )
        if w is not None:
            fields_as_pinttimeseries["w"] = w
        if q is not None:
            fields_as_pinttimeseries["q"] = q
        if p is not None:
            fields_as_pinttimeseries["p"] = p
        if r is not None:
            fields_as_pinttimeseries["r"] = r

        return fields_as_pinttimeseries

    # Get dataframe.

    def to_df(self, ref_index: pd.DatetimeIndex | None = None) -> PintTimeDataframe:
        """Create dataframe with one or more columns w, q, p, r, nodim."""
        return pd.DataFrame(self._to_timeseries(ref_index))

    def __bool__(self) -> bool:
        return bool(self.fields)

    def __or__(self, other: Any) -> InOp:
        if other is None:
            return self
        if not isinstance(other, InOp):
            raise TypeError("Can only unite same object type.")
        return _union(self, other)

    __ror__ = __or__

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, InOp):
            return False
        return _equal(self, other)


# def _guess_field(key):  # following keys return 'w': 'w', ('w', 'pf1'), ('pf1', 'w')
#     if key in _FIELDS:
#         return key
#     elif not isinstance(key, str) and isinstance(key, tuple):
#         if (field := guess_field(key[0])) is not None:
#             return field
#         if (field := guess_field(key[-1])) is not None:
#             return field
#     return None


def _process_single_field(
    field: Literal["w", "q", "p", "r", "nodim"],
    value: None | float | int | pint.Quantity | TimeSeries,
) -> None | pint.Quantity | PintTimeSeries:
    """For field ``field``, convert value ``value`` to Quantity or PintTimeSeries. Also, Check if
    dimensionality of fits the field ``field``."""

    # Ensure value is Quantity or PintTimeSeries.
    if isinstance(value, float | int | pint.Quantity):
        value = toolsb.unit.coerce_quantity(value)
    elif isinstance(value, pd.Series):
        value = toolsb.unit.coerce_pintseries(value)
        toolsb.index.validate(value.index)
    else:
        raise TypeError(f"Unexpected value type. Received: {value=} ({type(value)=}).")

    # Ensure value has expected dimensionality for the given field.
    expected_dims = toolsb.wqpr.col_to_dimties(field, nodim_allowed=True)
    received_dim = toolsb.unit.get_basedimty(value)
    if received_dim not in expected_dims:
        raise ValueError(
            f"For field {field}, expected one of following dimensions: {expected_dims}. Received: {received_dim} ({value=})."
        )

    return value


def _multiple_union(inops: Iterable[InOp]) -> InOp:
    inop_result = None
    for inop in inops:
        inop_result |= inop
    if inop_result is None:
        raise ValueError("Empty iterable provided (i.e., no InOp objects).")
    return inop_result


def _union(inop1: InOp, inop2: InOp) -> InOp:
    """Combine 2 ``InOp`` objects, and raise error if same field is supplied twice."""
    intersection = set(inop1.fields.keys()).intersection(set(inop2.fields.keys()))
    if intersection:
        raise ValueError(f"One or more fields were specified in both instances: {intersection}.")
    return InOp(inop1.fields | inop2.fields)


def _equal(inop1: InOp, inop2: InOp) -> bool:
    """``InOp`` objects are equal if they have the same fields and these fields have the same values."""
    if set(inop1.fields.keys()) != set(inop2.fields.keys()):
        return False
    for field in inop1.fields:
        value1, value2 = inop1.fields[field], inop2.fields[field]
        if type(value1) is not type(value2):
            return False
        if isinstance(value1, pd.Series):
            try:
                toolsb.testing.assert_series_equal(value1, value2, check_names=False)
            except AssertionError:
                return False
        elif value1 != value2:
            return False
    return True
