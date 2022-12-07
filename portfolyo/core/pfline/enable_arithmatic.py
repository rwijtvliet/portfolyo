"""Enable arithmatic with PfLine classes."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pandas as pd

from ... import tools
from . import base, interop, multi, single
from .base import Kind

if TYPE_CHECKING:  # needed to avoid circular imports
    from .base import PfLine

# Compatibility:
#
# General
#
# Physically true:
# unitA + unitA = unitA
# unitA * dimensionless = unitA
# unitA / dimensionless = unitA
# dimensionless / unitA = 1/unitA
#
# In addition, accepted as true:
# unitA + dimension-agnostic = unitA
# Eur/MWh * MWh -> all-PfLine
# Eur/MWh * MW -> all-PfLine
#
#
# Implementation
#
# Before anything else: turn 'other' into p-PfLine or q-PfLine if possible, or else into
# a pd.Series. So, if other is single quantity, or pd.Series, in Eur/MWh, MW, or MWh,
# this is turned into p-PfLine or q-PfLine.
#                 other
#                 0 or 0.0 or None                                 => return self
#                 Eur/MWh                                          => p-PfLine
#                 MW, MWh                                          => q-PfLine
#                 other unit or dimensionless                      => pd.Series


class PfLineFlattenedWarning(Warning):
    pass


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
        raise NotImplementedError("Cannot add portfolio lines of unequal kind.")

    if isinstance(pfl1, multi.MultiPfLine) and isinstance(pfl2, multi.MultiPfLine):
        # If BOTH are MultiPfLines, collect children and add those with same name.
        names = set([*[name for name in pfl1], *[name for name in pfl2]])
        children = {}
        for name in names:
            if name in pfl1 and name in pfl2:
                children[name] = pfl1[name] + pfl2[name]
            elif name in pfl1:
                children[name] = pfl1[name]
            else:
                children[name] = pfl2[name]
        return multi.MultiPfLine(children)

    elif isinstance(pfl1, multi.MultiPfLine) or isinstance(pfl2, multi.MultiPfLine):
        # ONE is MultiPfLine, ONE is SinglePfLine.
        warnings.warn(
            "When adding a SinglePfLine and MultiPfLine, the MultiPfLine is flattened.",
            PfLineFlattenedWarning,
        )

    # At least one of them is a SinglePfLine.

    dfs = [pfl.df(pfl.summable, flatten=True) for pfl in [pfl1, pfl2]]
    dfs = tools.intersect.frames(*dfs)  # keep only common rows
    return single.SinglePfLine(sum(dfs))


# TODO: Decide if this should return a Single or Multi PfLine
@_assert_index_compatibility
def _multiply_pflines(pfl1: PfLine, pfl2: PfLine):
    """Multiply two pflines."""
    if set([pfl1.kind, pfl2.kind]) != {Kind.PRICE_ONLY, Kind.VOLUME_ONLY}:
        raise NotImplementedError("Can only multiply volume with price information.")

    if isinstance(pfl1, multi.MultiPfLine) or isinstance(pfl2, multi.MultiPfLine):
        warnings.warn(
            "PfLine instances are flattened before multiplication.",
            PfLineFlattenedWarning,
        )

    if pfl1.kind is Kind.PRICE_ONLY:
        data = {"q": pfl2.q, "p": pfl1.p}
    else:  # pfl1.kind is Kind.VOLUME_ONLY:
        data = {"q": pfl1.q, "p": pfl2.p}
    return single.SinglePfLine(data)


@_assert_index_compatibility
def _multiply_pfline_and_dimensionlessseries(pfl: PfLine, s: pd.Series):
    """Multiply pfline and dimensionless series."""
    # Scale the price p (kind == 'p') or the volume q (kind == 'q'), returning PfLine of same kind.
    if isinstance(pfl, multi.MultiPfLine):
        return multi.MultiPfLine({name: child * s for name, child in pfl.items()})
    df = pfl.df(pfl.summable)
    df, s = tools.intersect.frames(df, s)  # keep only common rows
    return single.SinglePfLine(df.mul(s, axis=0))


@_assert_index_compatibility
def _divide_pflines(pfl1: PfLine, pfl2: PfLine) -> pd.Series:
    """Divide two pflines."""
    if pfl1.kind is not pfl2.kind or pfl1.kind is Kind.ALL:
        raise NotImplementedError(
            "Can only divide portfolio lines if both contain price-only or both contain volume-only information."
        )
    if isinstance(pfl1, multi.MultiPfLine) or isinstance(pfl2, multi.MultiPfLine):
        warnings.warn(
            "PfLine instances are flattened before division.",
            PfLineFlattenedWarning,
        )

    if pfl1.kind is Kind.PRICE_ONLY:
        series = pfl1.p, pfl2.p
    else:  # self.kind is Kind.VOlUME_ONLY
        series = pfl1.q, pfl2.q
    series = tools.intersect.frames(*series)
    s = series[0] / series[1]
    return s.rename("fraction")  # pint[dimensionless]


class PfLineArithmatic:
    METHODS = ["neg", "add", "radd", "sub", "rsub", "mul", "rmul", "truediv"]

    def __neg__(self: PfLine):
        if isinstance(self, multi.MultiPfLine):
            return multi.MultiPfLine({name: -child for name, child in self.items()})

        # multiply price (kind == 'p'), volume (kind == 'q') or volume and revenue (kind == 'all') with -1
        # Workaround to make negation work for pint quantity
        df = (-self.df(self.summable).pint.dequantify()).pint.quantify()
        return single.SinglePfLine(df)

    def __add__(self: PfLine, other) -> PfLine:
        # interpret dim-agnostic 'other' as price
        default = "p" if self.kind is Kind.PRICE_ONLY else None
        other = interop.pfline_or_nodimseries(other, self.index, default)

        # other is now None, a PfLine, or dimless Series.

        if other is None:
            return self
        elif isinstance(other, base.PfLine):
            return _add_pflines(self, other)
        else:
            raise NotImplementedError("This addition is not defined.")

    __radd__ = __add__

    def __sub__(self: PfLine, other):
        # interpret non-zero, dim-agnostic 'other' as price
        default = "p" if self.kind is Kind.PRICE_ONLY else None
        other = interop.pfline_or_nodimseries(other, self.index, default)

        # other is now None, a PfLine, or dimless Series.

        if other is None:
            return self
        elif isinstance(other, base.PfLine):
            return _add_pflines(self, -other)
        else:
            raise NotImplementedError("This subtraction is not defined.")

    def __rsub__(self: PfLine, other):
        return -self + other  # defer to add and neg

    def __mul__(self: PfLine, other) -> PfLine:
        default = "nodim"  # interpret non-zero, dim-agnostic as dimless (i.e., factor)
        other = interop.pfline_or_nodimseries(other, self.index, default)

        # other is now None, a PfLine, or dimless Series.

        if other is None:
            raise NotImplementedError("This multiplication is not defined.")
        elif isinstance(other, base.PfLine):
            return _multiply_pflines(self, other)
        else:
            return _multiply_pfline_and_dimensionlessseries(self, other)

    __rmul__ = __mul__

    def __truediv__(self: PfLine, other):
        default = "nodim"  # interpret dim-agnostic as dimless (i.e., factor)
        other = interop.pfline_or_nodimseries(other, self.index, default)

        # other is now None, a PfLine, or dimless Series.

        if other is None:
            raise NotImplementedError("This division is not defined.")
        elif isinstance(other, base.PfLine):
            return _divide_pflines(self, other)
        else:
            return self * (1 / other)  # defer to mul


def apply():
    for attr in PfLineArithmatic.METHODS:
        attrname = f"__{attr}__"
        setattr(base.PfLine, attrname, getattr(PfLineArithmatic, attrname))
