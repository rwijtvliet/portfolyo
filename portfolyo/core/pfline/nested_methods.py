from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .classes import NestedPfLine, FlatPfLine

from . import classes
from .enums import Structure


def flatten(self: NestedPfLine) -> FlatPfLine:
    constructor = classes.constructor(Structure.FLAT, self.kind)
    return constructor(self.df)  # use flattened toplevel dataframe for initialisation


def map_to_year(self: NestedPfLine, year: int, holiday_country: str) -> NestedPfLine:
    newchildren = {
        name: child.map_to_year(year, holiday_country) for name, child in self.items()
    }
    return self.__class__(newchildren)


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


class LocIndexer:
    """Helper class to obtain NestedPfLine instance, whose index is subset of original index."""

    def __init__(self, pfl: NestedPfLine):
        self.pfl = pfl

    def __getitem__(self, arg) -> NestedPfLine:
        newchildren = {name: child.loc[arg] for name, child in self.pfl.items()}
        return self.pfl.__class__(newchildren)
