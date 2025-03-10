from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from ... import tools
from . import classes
from .enums import Structure

if TYPE_CHECKING:
    from .classes import FlatPfLine, NestedPfLine, PricePfLine


def flatten(self: NestedPfLine) -> FlatPfLine:
    constructor = classes.constructor(Structure.FLAT, self.kind)
    return constructor(self.df)  # use flattened toplevel dataframe for initialisation


def po(
    self: NestedPfLine, peak_fn: tools.peakfn.PeakFunction, freq: str = "MS"
) -> pd.DataFrame:
    return self.flatten().po(peak_fn, freq)


def hedge_with(
    self: NestedPfLine,
    p: PricePfLine,
    how: str = "val",
    peak_fn: tools.peakfn.PeakFunction = None,
    freq: str = "MS",
) -> FlatPfLine:
    return self.flatten().hedge_with(p, how, peak_fn, freq)


def __bool__(self: NestedPfLine) -> bool:
    # True if a) has children of which b) any are true
    return any(self.children.keys())


def __eq__(self: NestedPfLine, other: Any) -> bool:
    if not isinstance(other, self.__class__):
        return False
    return self.children == other.children


@property
def loc(self: NestedPfLine) -> LocIndexer:
    return LocIndexer(self)


@property
def slice(self: NestedPfLine) -> SliceIndexer:
    return SliceIndexer(self)


def reindex(self: NestedPfLine, index: pd.DatetimeIndex) -> NestedPfLine:
    newchildren = {name: child.reindex(index) for name, child in self.pfl.items()}
    return self.pfl.__class__(newchildren)


class LocIndexer:
    """Helper class to obtain NestedPfLine instance, whose index is subset of original index."""

    def __init__(self, pfl: NestedPfLine):
        self.pfl = pfl

    def __getitem__(self, arg) -> NestedPfLine:
        newchildren = {name: child.loc[arg] for name, child in self.pfl.items()}
        return self.pfl.__class__(newchildren)


class SliceIndexer:
    """Helper class to obtain NestedPfLine instance, whose index is subset of original index.
    Exclude end point from the slice."""

    def __init__(self, pfl: NestedPfLine):
        self.pfl = pfl

    def __getitem__(self, arg) -> NestedPfLine:
        newchildren = {name: child.slice[arg] for name, child in self.pfl.items()}
        return self.pfl.__class__(newchildren)
