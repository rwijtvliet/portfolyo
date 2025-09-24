from typing import TYPE_CHECKING

from ...toolsb.types import Frequencylike
from . import indexer

if TYPE_CHECKING:
    from .pflinee import FLatPfLine, NestedPfLine, PfLine


class NestedMethods:
    @property
    def loc(self) -> indexer.NestedLoc:
        return indexer.NestedLoc(self)

    @property
    def slice(self) -> indexer.NestedSlice:
        return indexer.NestedSlice(self)

    def asfreq(self, freq: Frequencylike = "MS") -> NestedPfLine:
        freq = toolsb.freq.coerce(freq)
        newchildren = {name: child.asfreq(freq) for name, child in self.items()}
        return NestedPfLine(newchildren, self.kind, self.commodity)

    def flatten(self) -> FlatPfLine:
        return FlatPfLine(self.df, self.commodity)  # use toplevel df for initialisation

    def reindex(self, index: pd.DatetimeIndex) -> NestedPfLine:
        newchildren = {name: child.reindex(index) for name, child in self.pfl.items()}
        return NestedPfLine(newchildren, self.kind, self.commodity)

    def agg(self) -> pd.DataFrame:
        dfs = [self.flatten().agg().to_frame("").T]
        for name, child in self.items():
            if child.structure is Structure.FLAT:
                dfs.append(child.agg().to_frame(name).T)
            else:
                dfs.append(toolsb.frame.add_header(child.agg(), name, 0))
        return toolsb.frame.concat(dfs)

    def __bool__(self) -> bool:
        return any(self.children.keys())  # True if a) has children of which b) any are true

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.children == other.children

    def po(self: PfLine, freq: Frequencylike) -> pd.DataFrame:
        dfs = [self.flatten().po(freq)]
        for name, child in self.items():
            dfs.append(toolsb.frame.add_header(child.po(freq), name, 1))
        return toolsb.frame.concat(dfs)
