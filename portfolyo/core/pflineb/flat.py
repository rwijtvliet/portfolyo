"""Class that is created when instantiating a flat pfline. Brings together all functionality."""

import dataclasses
from typing import ClassVar

import pandas as pd

from ..commodity import Commodity
from .enums import Kind, Structure
from .flat_required import FlatRequiredMethods
from .pfline import PfLine


@dataclasses.dataclass(frozen=True, repr=False, eq=False)
class FlatPfLine(PfLine, FlatRequiredMethods):
    # Normal instance fields.
    # . Class is only called internally, so expect df to be in correct format.
    #   Meaning: correct columns for `kind`, and correct units for `commodity`.
    df: pd.DataFrame
    kind: Kind
    commodity: Commodity
    # Class variables.
    structure: ClassVar[Structure] = Structure.FLAT

    # dataframe = dataframeexport.Flat.dataframe
