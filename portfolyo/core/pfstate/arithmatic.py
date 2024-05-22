"""Add arithmatic to PfState classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from ... import tools
from ..pfline import PfLine, create
from . import pfstate

if TYPE_CHECKING:  # needed to avoid circular imports
    from . import PfState


class Prep:
    def assert_objects_indexcompatibility(fn):
        """Indices must have same frequency and same start-of-day; if not, raise Error."""

        def wrapper(o1, o2, *args, **kwargs):
            try:
                tools.testing.assert_indices_compatible(o1.index, o2.index)
            except AssertionError as e:
                raise NotImplementedError from e
            return fn(o1, o2, *args, **kwargs)

        return wrapper

    def standardize_other(fn):
        """Turn other into None, PfState, PfLine, or Series."""

        def wrapper(pfs: PfState, other: Any):
            other = Prep._prep_data(other, pfs.index)
            return fn(pfs, other)

        return wrapper

    def _prep_data(value, refindex: pd.DatetimeIndex) -> pd.Series | PfLine | PfState:
        """Turn ``value`` into PfLine or PfState if possible. If not, turn into (normal or unit-aware) Series."""

        # None.
        if value is None:
            return None

        # Already a PfState.
        if isinstance(value, pfstate.PfState):
            return value

        # Already a PfLine.
        if isinstance(value, PfLine):
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

            return create.flatpfline({name: value})

        # Just a single value.
        if isinstance(value, int) or isinstance(value, float):
            s = pd.Series(value, refindex)
            return Prep._prep_data(s, refindex)
        elif isinstance(value, tools.unit.Q_):
            s = pd.Series(value.magnitude, refindex, dtype=f"pint[{value.units:P}]")
            return Prep._prep_data(s, refindex)

        raise TypeError(f"Cannot handle inputs of this type; got {type(value)}.")

        # TODO: refactor and use inop

    def returnself_if_otherNone(fn):
        def wrapper(pfs: PfState, other: Any):
            if other is None:
                return pfs
            return fn(pfs, other)

        return wrapper

    def raiseerror_if_otherNone(fn):
        def wrapper(pfs: PfState, other: Any):
            if other is None:
                raise NotImplementedError("Cannot do this operation with this operand.")
            return fn(pfs, other)

        return wrapper


class PfStateArithmatic:
    METHODS = ["neg", "add", "radd", "sub", "rsub", "mul", "rmul", "truediv"]

    def __neg__(self: PfState) -> PfState:
        # invert volumes and revenues, leave prices unchanged.
        return self.__class__(-self.offtakevolume, self.unsourcedprice, -self.sourced)

    @Prep.standardize_other  # other is None, PfState, Pfline, or Series
    @Prep.returnself_if_otherNone  # other is PfState, PfLine, or Series...
    @Prep.assert_objects_indexcompatibility  # ... with compatible index
    def __add__(self: PfState, other: Any) -> PfState:
        # Other is a PfState.
        if isinstance(other, pfstate.PfState):
            return Add.two_pfstates(self, other)

        raise NotImplementedError("This addition is not defined.")

    def __radd__(self: PfState, other: Any) -> PfState:
        return self + other  # defer to __add__

    @Prep.returnself_if_otherNone  # catch pfstate - None
    def __sub__(self: PfState, other: Any) -> PfState:
        return self + -other  # defer to __add__ and __neg__

    def __rsub__(self: PfState, other: Any) -> PfState:
        return -self + other  # defer to __add__ and __neg__

    @Prep.standardize_other  # other is None, PfState, Pfline, or Series
    @Prep.raiseerror_if_otherNone  # other is PfState, Pfline, or Series
    @Prep.assert_objects_indexcompatibility  # ... with a compatible index
    def __mul__(self: PfState, other: Any) -> PfState:
        # Other is a Series (but not containing [power], [energy] or [price]).
        if isinstance(other, pd.Series):
            return Multiply.pfstate_and_series(self, other)

        raise NotImplementedError("This Multiplication is not defined.")

    def __rmul__(self: PfState, other: Any) -> PfState:
        return self * other  # defer to __mul__

    @Prep.standardize_other  # other is None, PfState, Pfline, or Series
    @Prep.raiseerror_if_otherNone  # other is PfState, Pfline, or Series
    @Prep.assert_objects_indexcompatibility  # ... with a compatible index
    def __truediv__(self: PfState, other):
        # Other is a PfState.
        if isinstance(other, pfstate.PfState):
            return Divide.two_pfstates(self, other)

        # Other is a Series (but not containing [power], [energy] or [price]).
        if isinstance(other, pd.Series):
            return Multiply.pfstate_and_series(self, 1 / other)

        raise NotImplementedError("This division is not defined.")


class Add:
    def two_pfstates(pfs1: PfState, pfs2: PfState) -> PfState:
        offtakevolume = pfs1.offtakevolume + pfs2.offtakevolume
        sourced = pfs1.sourced + pfs2.sourced

        # Unsourced price.
        # . The following line works... but not if volume == 0.
        # unsourcedprice = (pfs1.unsourced + pfs2.unsourced).price
        # . Therefore, use weighted average.
        values = pd.DataFrame({"s": pfs1.unsourcedprice.p, "o": pfs2.unsourcedprice.p})
        weights = pd.DataFrame({"s": pfs1.unsourced.q, "o": pfs2.unsourced.q})
        unsourcedprice = create.flatpfline(
            {"p": tools.wavg.dataframe(values, weights, axis=1)}
        )

        return pfstate.PfState(offtakevolume, unsourcedprice, sourced)


class Multiply:
    def pfstate_and_series(pfs: PfState, s: pd.Series) -> PfState:
        # Scale up volumes (and revenues), leave prices unchanged.
        return pfstate.PfState(
            pfs.offtakevolume * s, pfs.unsourcedprice, pfs.sourced * s
        )


class Divide:
    def two_pfstates(pfs1: PfState, pfs2: PfState) -> pd.DataFrame:
        series = {}
        for top, bottom in [
            ("offtake", "volume"),
            ("sourced", "volume"),
            ("sourced", "price"),
            ("unsourced", "volume"),
            ("unsourcedprice", "price"),
            ("pnl_cost", "price"),
        ]:
            pfl1 = getattr(getattr(pfs1, top).flatten(), bottom)
            pfl2 = getattr(getattr(pfs2, top).flatten(), bottom)
            top = top.replace("unsourcedprice", "unsourced")
            series[(top, bottom)] = pfl1 / pfl2
        return pd.DataFrame(series)
