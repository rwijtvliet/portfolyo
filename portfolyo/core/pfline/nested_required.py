"""Implementation of methods, required by abc, for nested pflines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from ... import tools
from ...tools.types import Frequencylike
from . import flat, indexer, nested, nested_helper
from .enums import Kind, Structure

if TYPE_CHECKING:
    from .flat import FlatPfLine
    from .nested import NestedPfLine


class NestedRequiredMethods:

    @property
    def volume(self: NestedPfLine) -> NestedPfLine:
        if self.kind is Kind.VOLUME:
            return self
        elif self.kind is Kind.COMPLETE:
            newchildren = {name: child.volume for name, child in self.items()}
            return nested.NestedPfLine(newchildren, Kind.VOLUME, self.commodity)
        raise TypeError("This portfolio line does not contain volume information.")

    @property
    def price(self: NestedPfLine) -> NestedPfLine | FlatPfLine:
        if self.kind is Kind.PRICE:
            return self
        elif self.kind is Kind.COMPLETE:
            # Price of (nested) complete PfLine is not sum of prices of its children, so return flat.
            return flat.FlatPfLine(self.df[["p"]], Kind.PRICE, self.commodity)
        raise TypeError("This portfolio line does not contain price information.")

    @property
    def revenue(self: NestedPfLine) -> NestedPfLine:
        if self.kind is Kind.REVENUE:
            return self
        elif self.kind is Kind.COMPLETE:
            newchildren = {name: child.revenue for name, child in self.items()}
            return nested.NestedPfLine(newchildren, self.REVENUE, self.commodity)
        raise TypeError("This portfolio line does not contain revenue information.")

    def set_commodity(self, commodity: Commodity | None) -> NestedPfLine:
        """Return new instance with the specified commodity. Use ``None`` to remove commodity."""
        if self.commodity is commodity:  # already correct
            return self

        # Need to change commodity.
        new_children = nested_helper.apply_commodity(self.children, commodity)
        return nested.NestedPfLine(new_children, self.kind, commodity)

    def asfreq(self: NestedPfLine, freq: Frequencylike = "MS") -> NestedPfLine:
        freq = tools.freq.coerce(freq)
        newchildren = {name: child.asfreq(freq) for name, child in self.items()}
        return nested.NestedPfLine(newchildren, self.kind, self.commodity)

    def flatten(self: NestedPfLine) -> FlatPfLine:
        return flat.FlatPfLine(self.df, self.kind, self.commodity)  # use toplevel df for init

    def reindex(self: NestedPfLine, index: pd.DatetimeIndex) -> NestedPfLine:
        newchildren = {name: child.reindex(index) for name, child in self.items()}
        return nested.NestedPfLine(newchildren, self.kind, self.commodity)

    def agg(self: NestedPfLine) -> pd.DataFrame:
        dfs = [self.flatten().agg().to_frame("").T]
        for name, child in self.items():
            if child.structure is Structure.FLAT:
                dfs.append(child.agg().to_frame(name).T)
            else:
                dfs.append(tools.frame.add_header(child.agg(), name, 0))
        return tools.frame.concat(dfs)

    def __bool__(self: NestedPfLine) -> bool:
        return any(self.children.keys())  # True if a) has children of which b) any are true

    def __eq__(self: NestedPfLine, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.children == other.children

    def po(self: NestedPfLine, freq: Frequencylike) -> pd.DataFrame:
        dfs = [self.flatten().po(freq)]
        for name, child in self.items():
            dfs.append(tools.frame.add_header(child.po(freq), name, 1))
        return tools.frame.concat(dfs)

    # ---

    @property
    def loc(self: NestedPfLine) -> indexer.NestedLoc:
        return indexer.NestedLoc(self)

    @property
    def iloc(self: NestedPfLine) -> indexer.NestedIloc:
        return indexer.NestedIloc(self)

    @property
    def slice(self: NestedPfLine) -> indexer.NestedSlice:
        return indexer.NestedSlice(self)
