"""Class that is created when instantiating a nested pfline. Brings together all functionality."""

import dataclasses
from typing import ClassVar, Mapping

import pandas as pd

from ..commodity import Commodity
from .enums import Kind, Structure
from .nested_children import ChildMethods
from .nested_required import NestedRequiredMethods
from .pfline import PfLine


@dataclasses.dataclass(frozen=True, repr=False, eq=False)
class NestedPfLine(PfLine, NestedRequiredMethods, ChildMethods):
    # Normal instance fields.
    # . Class is only called internally, so expect children to be in correct format.
    #   Meaning: all have same `kind` and are in correct units for `commodity`.
    children: Mapping[str, PfLine]
    kind: Kind
    commodity: Commodity
    # Class variables.
    structure: ClassVar[Structure] = Structure.NESTED
    # Calculated instance fields.
    df: pd.DataFrame = dataclasses.field(init=False)

    def __post_init__(self):
        df = sum(child.df for child in self.children.values())
        if self.kind is Kind.COMPLETE:
            df["p"] = df["r"] / df["q"]  # TODO: convert to correct unit
        object.__setattr__(self, "df", df)

    # dataframe = dataframeexport.Nested.dataframe
