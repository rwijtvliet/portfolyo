"""Ensure interoperability by extracting power, energy, price, revenue, and
dimensionless values/timeseries from data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd
import pint

from ... import toolsb
from . import classes, create

if TYPE_CHECKING:  # needed to avoid circular imports
    from .classes import FlatPfLine

_ATTRIBUTES = ("w", "q", "p", "r", "nodim")


@dataclass(frozen=True)
class InOp:
    """Class to check increase interoperability. Tries to extract quartity rate (w), quantity (q),
    price (p), revenue (r) and adimensional (nodim) information from the provided data.

    Typical usage:

    . Initialisation:
        inop = Inop.from_data(...)

    . Turn all into timeseries:
        inop = inop.to_timeseries()

    . Check for consistency and fill the missing attributes as much as possible:
        inop = inop.make_consistent()

    . And finally, return as dataframe with columns 'w', 'q', 'p', 'r', 'nodim'
        inop = inop.to_df()
    """

    w: pint.Quantity | pd.Series | None = None
    q: pint.Quantity | pd.Series | None = None
    p: pint.Quantity | pd.Series | None = None
    r: pint.Quantity | pd.Series | None = None
    nodim: float | pd.Series | None = None  # explicitly dimensionless

    def __post_init__(self):
        object.__setattr__(self, "w", check_dimensionality(self.w, "w"))
        object.__setattr__(self, "q", check_dimensionality(self.q, "q"))
        object.__setattr__(self, "p", check_dimensionality(self.p, "p"))
        object.__setattr__(self, "r", check_dimensionality(self.r, "r"))
        object.__setattr__(self, "nodim", check_dimensionality(self.nodim, "nodim"))

    @classmethod
    def from_data(cls, data):
        return _from_data(data)

    def to_timeseries(self, ref_index=None) -> InOp:
        """Turn all values into timeseries or None. If none of the attributes is a
        timeseries, and no ``ref_index`` is provided, raise Error. If >1 is a timeseries,
        keep only the timestamps where they overlap (i.e., intersection)."""
        # Get index.
        indices = [] if ref_index is None else [ref_index]
        for attr in _ATTRIBUTES:
            if isinstance(val := getattr(self, attr), pd.Series):
                indices.append(val.index)
        index = toolsb.index.intersect(indices)  # raises error if none passed
        if not len(index):
            raise ValueError("Data has no overlapping timestamps.")
        # Save all values as timeseries.
        kwargs = {}
        for attr in _ATTRIBUTES:
            val = getattr(self, attr)
            if val is None:
                continue
            elif isinstance(val, float):
                kwargs[attr] = pd.Series(val, index)
            elif isinstance(val, pint.Quantity):
                kwargs[attr] = pd.Series(val.magnitude, index).astype(f"pint[{val.units}]")
            else:  # float- or pint-series
                kwargs[attr] = val.loc[index]
        # Return as new InOp instance.
        return InOp(**kwargs)

    # def drop(self, da: str) -> InOp:
    #     """Drop part of the information and return new InOp object."""
    #     return InOp(**{attr: getattr(self, attr) for attr in _ATTRIBUTES if attr != da})

    def make_consistent(self) -> InOp:
        """Fill as much of the data as possible. All data must be None or timeseries."""

        self._assert_all_timeseries()

        # If we land here, all attributes are timeseries (or None).

        w, q, p, r, nodim = self.w, self.q, self.p, self.r, self.nodim
        duration = toolsb.index.duration(frame.index)

        # Volumes.
        if w is not None and q is not None:
            try:
                toolsb.testing.assert_series_equal(w, q / duration(q), check_names=False)
            except AssertionError as e:
                raise ValueError("Values for w and q are not consistent.") from e
        elif w is not None and q is None:
            q = w * duration(w)
        elif w is None and q is not None:
            w = q / duration(q)
        elif w is None and q is None and p is not None and r is not None:
            q = r / p
            w = q / duration(q)

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
                toolsb.testing.assert_series_equal(
                    r[~ignore], p[~ignore] * q[~ignore], check_names=False
                )
            except AssertionError as e:
                raise ValueError("Values for r, p, and q are not consistent.") from e

        return InOp(w=w, q=q, p=p, r=r, nodim=nodim)

    def to_df(self) -> pd.DataFrame:
        """Create dataframe with (at most) columns w, q, p, r, nodim. All data must be
        None or timeseries. Also, you'll probably want to have run the `.make_consistent()` method.
        """

        self._assert_all_timeseries()

        # If we land here, all attributes are timeseries (or None).

        return pd.DataFrame(
            {attr: s for attr in _ATTRIBUTES if (s := getattr(self, attr)) is not None}
        )

    def _assert_all_timeseries(self):
        """Raise Error if not all data are timeseries."""
        for attr in _ATTRIBUTES:
            if isinstance(getattr(self, attr), None | pd.Series):
                continue
            raise ValueError(
                f"Attribute {attr} contains non-timeseries data; use `.to_timeseries()`."
            )

    def __bool__(self) -> bool:
        return not all(getattr(self, attr) is None for attr in _ATTRIBUTES)

    def __or__(self, other: InOp) -> InOp:
        return _union(self, other)

    __ror__ = __or__

    def __eq__(self, other) -> bool:
        return _equal(self, other)


def raisedimerror_receivedfloat(expected: pint.util.UnitsContainer) -> None:
    raise pint.DimensionalityError(
        expected,
        "[]",
        extra_msg="Float or int only allowed for dimensionless value. To specify a physical quantity, add a unit.",
    )


def raisedimerror_receivedincorrect(
    expected: pint.util.UnitsContainer, received: pint.util.UnitsContainer
) -> None:
    raise pint.DimensionalityError(
        expected,
        received,
        extra_msg=f"Incorrect dimension for this attribute; expected {expected}, got {received}.",
    )


def check_dimensionality(
    v: None | float | int | pint.Quantity | pd.Series, attr: str
) -> None | float | pint.Quantity | pd.Series:
    """Check the dimensionality of a value.

    This function verifies if the given value `v` has the correct unit dimensionality
    corresponding to the attribute `attr`.
    """
    # Retrieve the expected dimensionality for the given attribute.
    if v is None:
        return v

    expected_dims = toolsb.wqpr.col_to_dimties(attr)
    received_dim = toolsb.unit.get_basedimty(v)

    if received_dim not in expected_dims:
        raise ValueError(
            f"Expected one of following dimensions: {expected_dims}. Received: {received_dim}."
        )


def _unit2attr(unit) -> str:
    attr = toolsb.unit.to_name(unit)  # Error if dimension unknown
    if attr not in _ATTRIBUTES:
        raise NotImplementedError(f"Cannot handle data with this unit ({unit}).")
    return attr


def _from_data(
    data: float | pint.Quantity | pd.Series | Dict | pd.DataFrame | Iterable | Mapping | None,
) -> InOp:
    """Turn ``data`` into a InterOp object."""

    if data is None:
        return InOp()

    try:
        col = toolsb.wqpr.valid_col(data)
        return InOp(**{col: data})
    except ValueError:
        pass  # could be dataframe or iterable or series (!= timeseries)

    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series) or isinstance(data, Mapping):

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


def _union(inop1: InOp, inop2: InOp | None) -> InOp:
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


def _equal(inop1: InOp, inop2: InOp) -> bool:
    """2 ``InOp`` objects are equal if they have the same attributes."""
    if not isinstance(inop2, InOp):
        return False
    for attr in _ATTRIBUTES:
        val1, val2 = getattr(inop1, attr), getattr(inop2, attr)
        if type(val1) is not type(val2):
            return False
        if isinstance(val1, pd.Series):
            try:
                toolsb.testing.assert_series_equal(val1, val2, check_names=False)
            except AssertionError:
                return False
        elif val1 != val2:
            return False
    return True


def pfline_or_nodimseries(data: Any, ref_index: pd.DatetimeIndex) -> None | pd.Series | FlatPfLine:
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
        raise NotImplementedError("Found a mix of dimension-aware and dimensionless data.")
