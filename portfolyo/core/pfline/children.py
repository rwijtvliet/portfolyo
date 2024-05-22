from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Mapping

from ... import tools
from . import create

if TYPE_CHECKING:
    from .classes import NestedPfLine, PfLine


class ChildFunctionality(Mapping):
    def set_child(self: NestedPfLine, name: str, child: PfLine | Any) -> NestedPfLine:
        """Set/add/update child; returns new pfline instance without changing current instance."""
        if name in ["w", "q", "p", "r"]:
            raise ValueError("Name cannot be one of 'w', 'q', 'p', 'r'.")
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
        try:
            tools.testing.assert_indices_compatible(self.index, child.index)
        except AssertionError as e:
            raise ValueError(
                "Index of new child is not compatible with the existing data."
            ) from e
        idx = tools.intersect.indices(self.index, child.index)
        if len(idx) == 0:
            raise ValueError(
                "Delivery period of the new child does not have any overlap with the existing data."
            )
        if len(idx) < len(self.index):
            warnings.warn(
                "Delivery period of the new child covers only part of the delivery period"
                " of the existing children. Only the common time period is kept."
            )
        newchildren = {**self, name: child}
        newchildren = {name: child.loc[idx] for name, child in newchildren.items()}
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

    def __getattr__(
        self: NestedPfLine, name: str
    ):  # allow access to children by attribute
        if name not in self.children:
            raise AttributeError(f"No such attribute '{name}'.")
        return self.children[name]
