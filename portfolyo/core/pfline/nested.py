"""Class that is created when instantiating a nested pfline. Brings together all functionality."""

import dataclasses
from collections import defaultdict
from typing import Any, ClassVar, Mapping

import pandas as pd

from ..commodity import Commodity
from . import flat, nested_helper
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
    commodity: Commodity
    # Class variables.
    structure: ClassVar[Structure] = Structure.NESTED
    # Calculated instance fields.
    kind: Kind = dataclasses.field(init=False)
    df: pd.DataFrame = dataclasses.field(init=False)

    def __post_init__(self):
        # Calculate kind.
        kindset = set([child.kind for child in self.children.values()])  # always >= 1
        if len(kindset) > 1:  # error
            kinds1 = defaultdict(list)
            for name, child in self.children.items():
                kinds1[child.kind].append(name)
            kinds2 = " and ".join([f"{kind} ({','.join(names)})" for kind, names in kinds1.items()])
            raise ValueError(f"All children must be of the same kind; found {kinds2}.")
        object.__setattr__(self, "kind", kindset.pop())

        # Calculate dataframe.
        df = sum(child.df for child in self.children.values())
        if self.kind is Kind.COMPLETE:
            df["p"] = df["r"] / df["q"]  # TODO: convert to correct unit
        object.__setattr__(self, "df", df)

    # dataframe = dataframeexport.Nested.dataframe


def create(data: Mapping[str, Any], commodity: Commodity | None) -> NestedPfLine:
    """Create a NestedPfLine instance from the provided data, if possible.

    Parameters
    ----------
    data
        Generally: mapping, between strings (as keys) and portfolio lines, or objects
        that can be converted into portfolio lines (as values).
    commodity, optional (default: none)
        Commodity the data applies to. Used to set units etc.

    Returns
    -------
        Nested portfolio line
    """
    # Catch easy cases.
    if isinstance(data, NestedPfLine):  # data already correct instancE; quick-return
        return data.set_commodity(commodity)
    elif isinstance(data, flat.FlatPfLine):
        raise TypeError("Cannot create nested portfolio line from a flat portfolio line.")

    # Data must be processed to find children and kind.
    children = nested_helper.create_children(data)
    children = nested_helper.apply_commodity(children, commodity)
    kind = nested_helper.get_kind(children)
    return nested.NestedPfLine(children, kind, commodity)
