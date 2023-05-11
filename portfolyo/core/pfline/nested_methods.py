from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from .classes import NestedPfLine, FlatPfLine, PfLine

from . import classes, create
from .enums import Structure


def flatten(self: NestedPfLine) -> FlatPfLine:
    constructor = classes.constructor(structure=Structure.FLAT, kind=self.kind)
    return constructor(self.df)  # use flattened toplevel dataframe for initialisation


def set_child(self: NestedPfLine, name: str, child: Union[PfLine, Any]) -> NestedPfLine:
    """Set/add/update child; returns new pfline instance without changing current instance."""
    try:
        child = create.pfline(child)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Parameter ``child`` cannot be turned into a PfLine; got {child}."
        ) from e
    if child.kind is not self.kind:
        raise ValueError(
            f"Incompatible kinds; the portfolio line has {self.kind} but the child has {child.kind}."
        )
    newchildren = {**self, name: child}
    return self.__class__(newchildren)


def drop_child(self: NestedPfLine, name: str) -> NestedPfLine:
    """Drop child; returns new pfline instance without changing current instance."""
    if name not in self.children:
        raise KeyError(f"Portfolio line does not have child with name '{name}'.")
    if len(self.children) == 1:
        raise RuntimeError("Cannot remove the last child of a portfolio line.")
    newchildren = {n: child for n, child in self.items() if n != name}
    return self.__class__(newchildren)


def __getitem__(self: NestedPfLine, name: str):
    if name not in self.children:
        raise KeyError(
            f"Portfolio line does not have child with name '{name}'."
            f" Names of available children: {', '.join(self.children.keys())}."
        )
    return self.children[name]


def items(self: NestedPfLine):
    """Iterate over children in (name, child)-tuples."""
    return self.children.items()


def __iter__(self: NestedPfLine):
    return iter(self.children.keys())


def __len__(self: NestedPfLine):
    return len(self.children)


def __getattr__(self: NestedPfLine, name: str):  # allow access to children by attribute
    if name not in self.children:
        raise AttributeError(f"No such attribute '{name}'.")
    return self.children[name]


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
