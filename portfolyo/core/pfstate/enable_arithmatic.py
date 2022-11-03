"""Add arithmatic to PfState classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import pandas as pd

from ... import tools
from .. import pfline
from . import pfstate

if TYPE_CHECKING:  # needed to avoid circular imports
    from ..pfline import PfLine
    from ..pfstate import PfState


def _prep_data(value, refindex: pd.DatetimeIndex) -> Union[pd.Series, PfLine, PfState]:
    """Turn ``value`` into PfLine or PfState if possible. If not, turn into (normal or unit-aware) Series."""

    # Already a PfState.
    if isinstance(value, pfstate.PfState):
        return value

    # Already a PfLine.
    if isinstance(value, pfline.PfLine):
        return value

    # Series.
    if isinstance(value, pd.Series):
        if not hasattr(value, "pint"):  # has no unit
            return value

        try:
            name = tools.unit.to_name(value.pint.units)
        except ValueError:
            return value  # has unit, but unknown

        if name not in ["p", "q", "w"]:
            return value  # has know unit, but none from which PfLine can be made

        return pfline.SinglePfLine({name: value})

    # Just a single value.
    if isinstance(value, int) or isinstance(value, float):
        s = pd.Series(value, refindex)
        return _prep_data(s, refindex)
    elif isinstance(value, tools.unit.Q_):
        s = pd.Series(value.magnitude, refindex, dtype=f"pint[{value.units:P}]")
        return _prep_data(s, refindex)

    raise TypeError(f"Cannot handle inputs of this type; got {type(value)}.")


def _add_pfstates(pfs1: pfstate.PfState, pfs2: pfstate.PfState) -> pfstate.PfState:
    """Add two pfstates."""
    offtakevolume = pfs1.offtakevolume.volume + pfs2.offtakevolume.volume
    sourced = pfs1.sourced + pfs2.sourced

    # Unsourced price.
    # . The following line works... but not if volume == 0.
    # unsourcedprice = (pfs1.unsourced + pfs2.unsourced).price
    # . Therefore, use weighted average.
    values = pd.DataFrame({"s": pfs1.unsourcedprice.p, "o": pfs2.unsourcedprice.p})
    weights = pd.DataFrame({"s": pfs1.unsourced.q, "o": pfs2.unsourced.q})
    unsourcedprice = pfline.PfLine({"p": tools.wavg.dataframe(values, weights, axis=1)})

    return pfstate.PfState(offtakevolume, unsourcedprice, sourced)


def _multiply_pfstate_and_series(pfs: pfstate.PfState, s: pd.Series) -> pfstate.PfState:
    """Multiply pfstate and Series."""
    # Scale up volumes (and revenues), leave prices unchanged.
    return pfstate.PfState(pfs.offtakevolume * s, pfs.unsourcedprice, pfs.sourced * s)


def _divide_pfstates(pfs1: pfstate.PfState, pfs2: pfstate.PfState) -> pd.DataFrame:
    """Divide two pfstates."""
    series = {}
    for top, bottom in [
        ("offtake", "volume"),
        ("sourced", "volume"),
        ("sourced", "price"),
        ("unsourced", "volume"),
        ("unsourcedprice", "price"),
        ("pnl_cost", "price"),
    ]:
        pfl1 = getattr(getattr(pfs1, top), bottom)
        pfl2 = getattr(getattr(pfs2, top), bottom)
        top = top.replace("unsourcedprice", "unsourced")
        series[(top, bottom)] = pfl1 / pfl2
    return pd.DataFrame(series)


def _assert_index_compatibility(o1, o2):
    if o1.index.freq != o2.index.freq:
        raise NotImplementedError(
            "Cannot do arithmatic with timeseries of unequal frequency."
        )
    if o1.index.tz != o2.index.tz:
        raise NotImplementedError(
            "Cannot do arithmatic with timeseries in unequal timezones."
        )


class PfStateArithmatic:

    METHODS = ["neg", "add", "radd", "sub", "rsub", "mul", "rmul", "truediv"]

    def __neg__(self: PfState):
        # invert volumes and revenues, leave prices unchanged.
        return self.__class__(-self.offtakevolume, self.unsourcedprice, -self.sourced)

    def __add__(self: PfState, other):
        if not other:
            return self

        other = _prep_data(other, self.index)  # other is now PfState, PfLine, or Series
        _assert_index_compatibility(self, other)

        # Other is a PfState.
        if isinstance(other, pfstate.PfState):
            return _add_pfstates(self, other)

        raise NotImplementedError("This addition is not defined.")

    __radd__ = __add__

    def __sub__(self: PfState, other):
        return self + -other if other else self  # defer to mul and neg

    def __rsub__(self: PfState, other):
        return other + -self  # defer to mul and neg

    def __mul__(self: PfState, other):

        other = _prep_data(other, self.index)  # other is now PfState, PfLine, or Series
        _assert_index_compatibility(self, other)

        # Other is a Series (but not containing [power], [energy] or [price]).
        if isinstance(other, pd.Series):
            return _multiply_pfstate_and_series(self, other)

        raise NotImplementedError("This multiplication is not defined.")

    __rmul__ = __mul__

    def __truediv__(self: PfState, other):
        other = _prep_data(other, self.index)  # other is now PfState, PfLine, or Series
        _assert_index_compatibility(self, other)

        # Other is a PfState.
        if isinstance(other, pfstate.PfState):
            return _divide_pfstates(self, other)

        # Other is a Series (but not containing [power], [energy] or [price]).
        if isinstance(other, pd.Series):
            return self * (1 / other)  # defer to mul

        raise NotImplementedError("This division is not defined.")


def apply():
    for attr in PfStateArithmatic.METHODS:
        attrname = f"__{attr}__"
        setattr(pfstate.PfState, attrname, getattr(PfStateArithmatic, attrname))
