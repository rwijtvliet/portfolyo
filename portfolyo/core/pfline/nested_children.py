from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from ... import tools
from . import nested, pfline

if TYPE_CHECKING:
    from .nested import NestedPfLine
    from .pfline import PfLine


class ChildMethods(Mapping):
    def set_child(self: NestedPfLine, name: str, child: PfLine | Any) -> NestedPfLine:
        """Set/add/update child; returns new pfline instance without changing current instance."""
        if name in ["w", "q", "p", "r"]:
            raise ValueError("Name cannot be one of 'w', 'q', 'p', 'r'.")
        try:
            child = pfline.create(child, self.commodity)
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
            tools.testing.assert_index_compatible(self.index, child.index)
        except AssertionError as e:
            raise ValueError("Index of new child is not compatible with existing data.") from e
        idx = tools.index.intersect([self.index, child.index])
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
        return nested.NestedPfLine(newchildren, self.kind, self.commodity)

    def drop_child(self: NestedPfLine, name: str) -> NestedPfLine:
        """Drop child; returns new pfline instance without changing current instance."""
        if name not in self.children:
            raise KeyError(f"Portfolio line does not have child with name '{name}'.")
        if len(self.children) == 1:
            raise RuntimeError("Cannot remove the last child of a portfolio line.")
        newchildren = {n: child for n, child in self.items() if n != name}
        return nested.NestedPfLine(newchildren, self.kind, self.commodity)

    def __getitem__(self: NestedPfLine, name: str) -> PfLine:
        if name not in self.children:
            raise KeyError(
                f"Portfolio line does not have child with name '{name}'."
                f" Names of available children: {', '.join(self.children.keys())}."
            )
        return self.children[name]

    def items(self: NestedPfLine) -> Iterable[tuple[str, PfLine]]:
        """Iterate over children in (name, child)-tuples."""
        return self.children.items()

    def __iter__(self: NestedPfLine):
        return iter(self.children.keys())

    def __len__(self: NestedPfLine) -> int:
        return len(self.children)

    def __getattr__(self: NestedPfLine, name: str):  # allow access to children by attribute
        if name not in self.children:
            raise AttributeError(f"No such attribute '{name}'.")
        return self.children[name]
