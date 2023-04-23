"""Enable arithmatic with PfLine classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import pandas as pd

from ... import tools
from . import base, flat, interop, nested
from .base import Kind

if TYPE_CHECKING:  # needed to avoid circular imports
    from .base import PfLine


def _assert_index_compatibility(fn):
    """Check if indices are compatible before calling the wrapped function."""

    def wrapper(o1, o2):
        if o1.index.freq != o2.index.freq:
            raise NotImplementedError(
                "Cannot do arithmatic with timeseries of unequal frequency."
            )
        if o1.index[0].time() != o2.index[0].time():
            raise NotImplementedError(
                "Cannot do arithmatic with timeseries that have unequal start-of-day."
            )
        return fn(o1, o2)

    return wrapper


@_assert_index_compatibility
def _add_pflines(pfl1: PfLine, pfl2: PfLine):
    """Add two pflines."""
    if pfl1.kind is not pfl2.kind:
        raise NotImplementedError(
            "Cannot add or subtract portfolio lines of unequal kind."
        )
    if type(pfl1) is not type(pfl2):  # ONE is NestedPfLine, ONE is FlatPfLine.
        raise TypeError(
            "Addition and subtraction only possible for 2 flat, or 2 nested, PfLines."
        )

    if isinstance(pfl1, nested.NestedPfLine):  # NestedPfLines.
        # Collect children and add those with same name.
        new_children = {}
        for name in set([*pfl1, *pfl2]):
            if name in pfl1 and name in pfl2:
                new_children[name] = pfl1[name] + pfl2[name]
            elif name in pfl1:
                new_children[name] = pfl1[name]
            else:
                new_children[name] = pfl2[name]
        return nested.NestedPfLine(new_children)

    # FlatPfLines.

    new_df = sum(tools.intersect.frames(pfl1._df, pfl2._df))  # keep only common rows
    if pfl1.kind is Kind.COMPLETE:
        new_df["p"] = new_df["r"] / new_df["q"]

    # new_dfs = tools.intersect.frames(pfl1._df, pfl2._df)  # keep only common rows
    # new_df = sum(new_dfs)
    # if pfl1.kind is Kind.COMPLETE:
    #     # Calculate price from wavg instead of r/q, to handle edge case p1==p2, q==0.
    #     values = pd.DataFrame({"1": new_dfs[0].p, "2": new_dfs[1].p})
    #     weights = pd.DataFrame({"1": new_dfs[0].q, "2": new_dfs[1].q})
    #     new_df["p"] = tools.wavg.dataframe(values, weights, axis=1)
    return flat.FlatPfLine(new_df, _internal=True)


@_assert_index_compatibility
def _multiply_pflines(pfl1: PfLine, pfl2: PfLine):
    """Multiply two pflines."""
    if isinstance(pfl1, nested.NestedPfLine) or isinstance(pfl2, nested.NestedPfLine):
        raise NotImplementedError(
            "Multiplication only possible between 2 flat PfLines."
        )

    if set([pfl1.kind, pfl2.kind]) != {Kind.PRICE, Kind.VOLUME}:
        raise NotImplementedError("Can only multiply volume with price information.")

    if pfl1.kind is Kind.PRICE:
        series = pfl2.q, pfl1.p
    else:
        series = pfl1.q, pfl2.p
    series = tools.intersect.frames(*series)
    return flat.FlatPfLine(series[0] * series[1])


@_assert_index_compatibility
def _multiply_pfline_and_dimensionlessseries(pfl: PfLine, s: pd.Series):
    """Multiply pfline and dimensionless series."""

    if isinstance(pfl, nested.NestedPfLine):  # NestedPfLine
        return nested.NestedPfLine({name: child * s for name, child in pfl.items()})

    # FlatPfLine.

    s, df = tools.intersect.frames(s, pfl._df)
    new_df = pd.DataFrame({col: series * s for col, series in df.items()})
    if pfl.kind is Kind.COMPLETE:  # correction
        new_df["p"] = new_df["r"] / new_df["q"]
    return flat.FlatPfLine(new_df, _internal=True)


@_assert_index_compatibility
def _divide_pflines(pfl1: PfLine, pfl2: PfLine) -> Union[pd.Series, PfLine]:
    """Divide two pflines."""

    if isinstance(pfl1, nested.NestedPfLine) or isinstance(pfl2, nested.NestedPfLine):
        raise NotImplementedError("Division only possible between 2 flat PfLines.")

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

    if pfl1.kind is not Kind.REVENUE:
        raise NotImplementedError(
            "To divide PfLines of unequal kind, the numerator must have revenues."
        )
    if pfl2.kind is Kind.PRICE:
        series = pfl1.r, pfl2.p
    elif pfl2.kind is Kind.VOLUME:
        series = pfl1.r, pfl2.q
    else:
        raise NotImplementedError(
            "To divide PfLines of unequal kind, the denominator must have volumes or prices."
        )
    series = tools.intersect.frames(*series)
    return flat.FlatPfLine(series[0] / series[1])


@_assert_index_compatibility
def _unite_pflines(pfl1: PfLine, pfl2: PfLine) -> PfLine:
    """Unite two pflines."""

    if isinstance(pfl1, nested.NestedPfLine) or isinstance(pfl2, nested.NestedPfLine):
        raise NotImplementedError("Can only do union if both PfLines are flat.")

    kinds = {pfl1.kind: pfl1, pfl2.kind: pfl2}
    if pfl1.kind is pfl2.kind:
        raise NotImplementedError("Cannot do union between PfLines of the same kind.")
    if Kind.COMPLETE in kinds.keys():
        raise NotImplementedError(
            "Cannot do union when one of the operands is a complete PfLines. First"
            " select e.g. .volume or .price."
        )

    data = {}
    if (pfl := kinds.get(Kind.VOLUME)) is not None:
        data["q"] = pfl.q
    if (pfl := kinds.get(Kind.PRICE)) is not None:
        data["p"] = pfl.p
    if (pfl := kinds.get(Kind.REVENUE)) is not None:
        data["r"] = pfl.r
    return flat.FlatPfLine(data)


class PfLineArithmatic:
    # Write without initial __ to facilitate identification

    def neg__(self: PfLine):
        return self * -1  # defer to mul

    def add__(self: PfLine, other) -> PfLine:
        other = interop.pfline_or_nodimseries(other, self.index)
        # other is now None, a PfLine, or dimless Series.
        if other is None:
            return self
        elif isinstance(other, base.PfLine):
            return _add_pflines(self, other)
        else:
            raise NotImplementedError("This addition is not defined.")

    radd__ = add__

    def sub__(self: PfLine, other):
        other = interop.pfline_or_nodimseries(other, self.index)
        # other is now None, a PfLine, or dimless Series.
        if other is None:
            return self
        elif isinstance(other, base.PfLine):
            return _add_pflines(self, -other)  # defer to add and neg
        else:
            raise NotImplementedError("This subtraction is not defined.")

    def rsub__(self: PfLine, other):
        return -self + other  # defer to add and neg

    def mul__(self: PfLine, other) -> PfLine:
        other = interop.pfline_or_nodimseries(other, self.index, "nodim")
        # other is now None, a PfLine, or dimless Series.
        if other is None:
            raise NotImplementedError("This multiplication is not defined.")
        elif isinstance(other, base.PfLine):
            return _multiply_pflines(self, other)
        else:
            return _multiply_pfline_and_dimensionlessseries(self, other)

    rmul__ = mul__

    def truediv__(self: PfLine, other):
        other = interop.pfline_or_nodimseries(other, self.index, "nodim")
        # other is now None, a PfLine, or dimless Series.
        if other is None:
            raise NotImplementedError("This division is not defined.")
        elif isinstance(other, base.PfLine):
            return _divide_pflines(self, other)
        else:
            return self * (1 / other)  # defer to mul

    def rtruediv__(self: PfLine, other):
        other = interop.pfline_or_nodimseries(other, self.index, "nodim")
        # other is now None, a PfLine, or dimless Series.
        if other is None:
            raise NotImplementedError("This division is not defined.")
        elif isinstance(other, base.PfLine):
            return _divide_pflines(other, self)
        else:
            raise NotImplementedError("This division is not defined.")

    def or__(self: PfLine, other):
        other = interop.pfline_or_nodimseries(other, self.index)
        # other is now None, a PfLine, or dimless Series.
        if other is None:
            return self
        elif isinstance(other, base.PfLine):
            return _unite_pflines(self, other)
        else:
            raise NotImplementedError("This union is not defined.")

    ror__ = or__


def apply():
    for attr in dir(PfLineArithmatic):
        if attr.endswith("__") and not attr.startswith("__"):
            setattr(base.PfLine, f"__{attr}", getattr(PfLineArithmatic, attr))
