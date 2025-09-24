from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Mapping

from ... import toolsb
from . import create, pflinee

if TYPE_CHECKING:
    from .pflinee import NestedPfLine, PfLine


class ChildMethods(Mapping):
    def set_child(self, name: str, child: PfLine | Any) -> NestedPfLine:
        """Set/add/update child; returns new pfline instance without changing current instance."""
        if name in ["w", "q", "p", "r"]:
            raise ValueError("Name cannot be one of 'w', 'q', 'p', 'r'.")
        try:
            child = create.pfline(child)  # TODO: provide self.commodity to use as default value
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Parameter ``child`` cannot be turned into a PfLine; got {child}."
            ) from e
        if child.kind is not self.kind:
            raise ValueError(
                f"Incompatible kinds; portfolio line has {self.kind} but child has {child.kind}."
            )
        if child.commodity is not self.commodity:
            raise ValueError(
                f"Incompatible commodities; portfolio line has {self.commodity} but child has {child.commodity}."
            )
        try:
            toolsb.testing.assert_index_compatible(self.index, child.index)
        except AssertionError as e:
            raise ValueError("Index of new child is not compatible with existing data.") from e
        idx = toolsb.index.intersect([self.index, child.index])
        if len(idx) == 0:
            raise ValueError(
                "Delivery period of new child does not have any overlap with existing data."
            )
        if len(idx) < len(self.index):
            warnings.warn(
                "Delivery period of new child covers only part of delivery period of existing"
                " children. Only common time period (i.e., the overlap) is kept."
            )
        newchildren = {**self, name: child}
        newchildren = {name: child.loc[idx] for name, child in newchildren.items()}
        return pflinee.NestedPfLine(newchildren, self.kind, self.commodity)

    def drop_child(self, name: str) -> NestedPfLine:
        """Drop child; returns new pfline instance without changing current instance."""
        if name not in self.children:
            raise KeyError(f"Portfolio line does not have child with name '{name}'.")
        if len(self.children) == 1:
            raise RuntimeError("Cannot remove the last child of a portfolio line.")
        newchildren = {n: child for n, child in self.items() if n != name}
        return pflinee.NestedPfLine(newchildren, self.kind, self.commodity)

    def __getitem__(self, name: str):
        if name not in self.children:
            raise KeyError(
                f"Portfolio line does not have child with name '{name}'."
                f" Names of available children: {', '.join(self.children.keys())}."
            )
        return self.children[name]

    def items(self):
        """Iterate over children in (name, child)-tuples."""
        return self.children.items()

    def __iter__(self):
        return iter(self.children.keys())

    def __len__(self):
        return len(self.children)

    def __getattr__(self, name: str):  # allow access to children by attribute
        if name not in self.children:
            raise AttributeError(f"No such attribute '{name}'.")
        return self.children[name]
