"""Enable arithmatic with PfLine classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ... import tools
from . import classes, create, interop
from .enums import Kind, Structure

if TYPE_CHECKING:  # needed to avoid circular imports
    from .classes import FlatPfLine, NestedPfLine, PfLine

STRICT = False  # TODO: make setting


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

    def assert_pflines_samekind(fn):
        """Both must be of same kind; if not, raise Error."""

        def wrapper(pfl1: PfLine, pfl2: PfLine):
            if pfl1.kind is not pfl2.kind:
                raise NotImplementedError(
                    "Cannot do this operation on portfolio lines of unequal kind."
                )
            return fn(pfl1, pfl2)

        return wrapper

    def assert_pflines_distinctkind(fn):
        """Both must be of distinct kind; if not, raise Error."""

        def wrapper(pfl1: PfLine, pfl2: PfLine):
            if pfl1.kind is pfl2.kind:
                raise NotImplementedError(
                    "Cannot do this operation on portfolio lines of equal kind."
                )
            return fn(pfl1, pfl2)

        return wrapper

    def ensure_pflines_samestructure(fn):
        """Both must be of same structure (flat or nested); if not, raise Error (if
        STRICT) or else flatten the nested one."""

        def wrapper(pfl1: PfLine, pfl2: PfLine):
            if pfl1.structure is not pfl2.structure:
                if STRICT:
                    raise NotImplementedError(
                        "Cannot do this operation on portfolio lines of unequal structure;"
                        " both must be flat, or both must be nested."
                    )
                pfl1, pfl2 = pfl1.flatten(), pfl2.flatten()  # flatten the nested one
            return fn(pfl1, pfl2)

        return wrapper

    def ensure_pflines_flat(fn):
        """Both must be flat; if not, raise Error (if STRICT) or else flatten the nested ones."""

        def wrapper(pfl1: PfLine, pfl2: PfLine):
            if not isinstance(pfl1, classes.FlatPfLine) or not isinstance(
                pfl2, classes.FlatPfLine
            ):
                if STRICT:
                    raise NotImplementedError(
                        "Cannot do this operation if one or more of the portfolio lines"
                        " are nested; first .flatten() both operands."
                    )
                pfl1, pfl2 = pfl1.flatten(), pfl2.flatten()  # flatten both
            return fn(pfl1, pfl2)

        return wrapper

    def ensure_otherpfline_flat(fn):
        def wrapper(pfl: PfLine, other: PfLine):
            if not isinstance(other, classes.classes.FlatPfLine):
                if STRICT:
                    raise NotImplementedError(
                        "Cannot do this operation if other operand is flat porfolio line;"
                        " .flatten() first."
                    )
                other = other.flatten()
            return fn(pfl, other)

        return wrapper

    def standardize_other(fn):
        """Turn other into None, PfLine or dimensionless Series."""

        def wrapper(pfl: PfLine, other: Any):
            other = interop.pfline_or_nodimseries(other, pfl.index, "nodim")
            return fn(pfl, other)

        return wrapper

    def returnself_if_otherNone(fn):
        def wrapper(pfl: PfLine, other: Any):
            if other is None:
                return pfl
            return fn(pfl, other)

        return wrapper

    def returnself_if_otherzerofloatseries(fn):
        def wrapper(pfl: PfLine, other: Any):
            if isinstance(other, pd.Series):
                if other.dtype == "pint[dimensionless]":
                    other = other.pint.m
                if other.dtype in [int, float] and np.allclose(other.values, 0.0):
                    return pfl

            return fn(pfl, other)

        return wrapper

    def raiseerror_if_otherNone(fn):
        def wrapper(pfl: PfLine, other: Any):
            if other is None:
                raise NotImplementedError("Cannot do this operation with this operand.")
            return fn(pfl, other)

        return wrapper

    def raiseerror_if_otherdimlessseries(fn):
        def wrapper(pfl: PfLine, other: Any):
            if isinstance(other, pd.Series):
                raise NotImplementedError("Cannot do this operation with this operand.")
            return fn(pfl, other)

        return wrapper


class PfLineArithmatic:
    def __neg__(self: PfLine) -> PfLine:
        return self * -1  # defer to __mul__

    @Prep.standardize_other  # other converted to None, a PfLine, or dimless Series
    @Prep.returnself_if_otherNone  # other is now a PfLine or dimless Series
    @Prep.returnself_if_otherzerofloatseries  # catch pfline + 0
    @Prep.raiseerror_if_otherdimlessseries  # other is now a PfLine...
    @Prep.assert_objects_indexcompatibility  # ...with a compatible index
    def __add__(self: PfLine, other: Any) -> PfLine:
        return Add.two_pflines(self, other)

    def __radd__(self: PfLine, other: Any) -> PfLine:
        return self + other  # defer to __add__

    @Prep.standardize_other  # other converted to None, a PfLine, or dimless Series
    @Prep.returnself_if_otherNone  # catch pfline - None
    @Prep.returnself_if_otherzerofloatseries  # cach pfline - 0
    def __sub__(self: PfLine, other: Any) -> PfLine:
        return self + -other  # defer to __add__ and __neg__

    def __rsub__(self: PfLine, other: Any) -> PfLine:
        return -self + other  # defer to __add__ and __neg__

    @Prep.standardize_other  # other converted to None, a PfLine, or dimless Series
    @Prep.raiseerror_if_otherNone  # other is now a PfLine or dimless Series...
    @Prep.assert_objects_indexcompatibility  # ... with a compatible index
    def __mul__(self: PfLine, other: Any) -> PfLine:
        if isinstance(other, pd.Series):
            return Multiply.pfline_and_series(self, other)
        else:
            return Multiply.two_pflines(self, other)

    def __rmul__(self: PfLine, other: Any) -> PfLine:
        return self * other  # defer to __mul__

    @Prep.standardize_other  # other converted to None, a PfLine, or dimless Series
    @Prep.raiseerror_if_otherNone  # other is now a PfLine or dimless Series...
    @Prep.assert_objects_indexcompatibility  # ... with a compatible index
    def __truediv__(self: PfLine, other: Any) -> PfLine | pd.Series:
        if isinstance(other, pd.Series):
            return Multiply.pfline_and_series(self, 1 / other)
        else:
            return Divide.two_pflines(self, other)

    @Prep.standardize_other  # other converted to None, a PfLine, or dimless Series
    @Prep.raiseerror_if_otherNone  # other is now a PfLine or dimless Series
    @Prep.raiseerror_if_otherdimlessseries  # other is now a PfLine...
    @Prep.assert_objects_indexcompatibility  # ... with a compatible index
    def __rtruediv__(self: PfLine, other: Any) -> PfLine | pd.Series:
        return Divide.two_pflines(other, self)  # NB order!

    @Prep.standardize_other  # other converted to None, a PfLine, or dimless Series
    @Prep.returnself_if_otherNone  # other is now a PfLine or dimless Series
    @Prep.raiseerror_if_otherdimlessseries  # other is now a PfLine...
    @Prep.assert_objects_indexcompatibility  # ... with a compatible index
    def __or__(self: PfLine, other: Any) -> PfLine:
        return Unite.two_pflines(self, other)

    def __ror__(self: PfLine, other: Any) -> PfLine:
        return self | other  # defer to __or__


class Add:
    @Prep.ensure_pflines_samestructure  # pfl1 and pfl2 are now both flat or both nested
    def two_pflines(pfl1: PfLine, pfl2: PfLine) -> PfLine:
        if isinstance(pfl1, classes.FlatPfLine):
            return Add.two_flatpflines(pfl1, pfl2)
        else:
            return Add.two_nestedpflines(pfl1, pfl2)

    @Prep.assert_pflines_samekind  # pfl1 and pfl2 now have same kind
    def two_flatpflines(pfl1: FlatPfLine, pfl2: FlatPfLine) -> FlatPfLine:
        # newdf = sum(tools.intersect.frames(pfl1.df, pfl2.df))  # keep common rows
        # if pfl1.kind is Kind.COMPLETE:
        #     newdf["p"] = newdf["r"] / newdf["q"]
        newdfs = tools.intersect.frames(pfl1.df, pfl2.df)  # keep only common rows
        newdf = sum(newdfs)
        if len(newdf.index) == 0:
            raise NotImplementedError(
                "Cannot perform operation on 2 portfolio lines without any overlapping timestamps."
            )
        if pfl1.kind is Kind.COMPLETE:
            # Calculate price from wavg instead of r/q, to handle edge case p1==p2, q==0.
            values = pd.DataFrame({"1": newdfs[0].p, "2": newdfs[1].p})
            weights = pd.DataFrame({"1": newdfs[0].q, "2": newdfs[1].q})
            newdf["p"] = tools.wavg.dataframe(values, weights, axis=1)
        return pfl1.__class__(newdf)

    @Prep.assert_pflines_samekind  # pfl1 and pfl2 now have same kind
    def two_nestedpflines(pfl1: NestedPfLine, pfl2: NestedPfLine) -> NestedPfLine:
        newchildren = {}  # collect children and add those with same name
        for name in set([*pfl1, *pfl2]):
            if name in pfl1 and name in pfl2:
                newchildren[name] = pfl1[name] + pfl2[name]
            elif name in pfl1:
                newchildren[name] = pfl1[name]
            else:
                newchildren[name] = pfl2[name]
        return pfl1.__class__(newchildren)


class Multiply:
    @Prep.ensure_pflines_flat  # pfl1 and pfl2 are now both flat
    def two_pflines(pfl1: PfLine, pfl2: PfLine) -> PfLine:
        return Multiply.two_flatpflines(pfl1, pfl2)

    def two_flatpflines(pfl1: FlatPfLine, pfl2: FlatPfLine) -> FlatPfLine:
        # Only relevant case is volume * price = revenue
        if set([pfl1.kind, pfl2.kind]) != {Kind.PRICE, Kind.VOLUME}:
            raise NotImplementedError(
                "Can only multiply volume with price information."
            )
        q, p = (pfl2.q, pfl1.p) if pfl1.kind is Kind.PRICE else (pfl1.q, pfl2.p)
        q, p = tools.intersect.frames(q, p)
        r = (q * p).pint.to_base_units()
        constructor = classes.constructor(Structure.FLAT, Kind.REVENUE)
        return constructor(pd.DataFrame({"r": r}))

    def pfline_and_series(pfl: PfLine, s: pd.Series) -> PfLine:
        if isinstance(pfl, classes.FlatPfLine):
            return Multiply.flatpfline_and_series(pfl, s)
        else:
            return Multiply.nestedpfline_and_series(pfl, s)

    def flatpfline_and_series(pfl: NestedPfLine, s: pd.Series) -> NestedPfLine:
        df, s = tools.intersect.frames(pfl.df, s)
        newdf = pd.DataFrame({col: series * s for col, series in df.items()})
        if pfl.kind is Kind.COMPLETE:  # correction: in this case, keep original prices
            newdf["p"] = df["p"]
        return pfl.__class__(newdf)

    def nestedpfline_and_series(pfl: NestedPfLine, s: pd.Series) -> NestedPfLine:
        newchildren = {name: child * s for name, child in pfl.items()}
        return pfl.__class__(newchildren)


class Divide:
    @Prep.ensure_pflines_flat  # pfl1 and pfl2 are now both flat
    def two_pflines(pfl1: PfLine, pfl2: PfLine) -> pd.Series | PfLine:
        return Divide.two_flatpflines(pfl1, pfl2)

    def two_flatpflines(pfl1: FlatPfLine, pfl2: FlatPfLine) -> pd.Series | FlatPfLine:
        if pfl1.kind is pfl2.kind:
            if pfl1.kind is Kind.COMPLETE:
                raise NotImplementedError(
                    "Cannot divide complete PfLines. First select e.g. .volume or .price."
                )

            if pfl1.kind is Kind.PRICE:
                series = pfl1.p, pfl2.p
            elif pfl1.kind is Kind.VOLUME:
                series = pfl1.q, pfl2.q
            elif pfl1.kind is Kind.REVENUE:
                series = pfl1.r, pfl2.r
            series = tools.intersect.frames(*series)
            s = series[0] / series[1]
            if not len(s):
                raise ValueError("Data has no overlapping timestamps.")
            return s.rename("fraction")  # pint[dimensionless]

        # Unequal kind.

        if (pfl1.kind, pfl2.kind) == (Kind.REVENUE, Kind.PRICE):
            r, p = tools.intersect.frames(pfl1.r, pfl2.p)
            data = {"q": r / p}
        elif (pfl1.kind, pfl2.kind) == (Kind.REVENUE, Kind.VOLUME):
            r, q = tools.intersect.frames(pfl1.r, pfl2.q)
            data = {"p": r / q}
        else:
            raise NotImplementedError(
                "To divide PfLines of unequal kind, the numerator must have revenues,"
                " and denominator must have volumes or prices."
            )
        return create.flatpfline(data)


class Unite:
    @Prep.ensure_pflines_flat  # pfl1 and pfl2 are now both flat
    def two_pflines(pfl1: PfLine, pfl2: PfLine) -> PfLine:
        return Unite.two_flatpflines(pfl1, pfl2)

    @Prep.assert_pflines_distinctkind  # pfl1 and pfl2 are now of distinct kind
    def two_flatpflines(pfl1: FlatPfLine, pfl2: FlatPfLine) -> FlatPfLine:
        if Kind.COMPLETE in {pfl1.kind, pfl2.kind}:
            raise NotImplementedError(
                "Cannot do union when one of the operands is a complete PfLines. First"
                " select e.g. .volume or .price."
            )

        # Collect the complete dataframe.
        data = pd.concat(tools.intersect.frames(pfl1.df, pfl2.df), axis=1)
        return create.flatpfline(data)
