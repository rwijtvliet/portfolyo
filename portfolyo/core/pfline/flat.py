"""Class that is created when instantiating a flat pfline. Brings together all functionality."""

import dataclasses
from typing import TYPE_CHECKING, ClassVar, Iterable, Mapping

import pandas as pd
import pint

from ...toolsb.types import Col, PintTimeDataframe, PintTimeSeries
from ..commodity import Commodity
from . import flat_helper, nested
from .enums import Kind, Structure
from .flat_required import FlatRequiredMethods
from .pfline import PfLine

if TYPE_CHECKING:
    from .flat import FlatPfLine
    from .pfline import PfLine


@dataclasses.dataclass(frozen=True, repr=False, eq=False)
class FlatPfLine(PfLine, FlatRequiredMethods):
    # Normal instance fields.
    # . Class is only called internally, so expect df to be in correct format.
    #   Meaning: correct columns for `kind`, and correct units for `commodity`.
    df: PintTimeDataframe
    kind: Kind
    commodity: Commodity | None
    # Class variables.
    structure: ClassVar[Structure] = Structure.FLAT

    # dataframe = dataframeexport.Flat.dataframe


def create(
    data: (
        Mapping[Col, PintTimeSeries | pint.Quantity]
        | PintTimeDataframe
        | PintTimeSeries
        | Iterable[PintTimeSeries | pint.Quantity]
    ),
    commodity: Commodity | None,
) -> FlatPfLine:
    """Create a FlatPfLine instance from the provided data, if possible.

    Parameters
    ----------
    data
        Generally: mapping with one or more attributes or items ``w``, ``q``, ``r``, ``p``;
        all timeseries. Most commonly a ``pandas.DataFrame`` or a dictionary of
        ``pandas.Series``, but may also be e.g. another PfLine object.
        If they contain (distinct) ``pint`` data types, ``data`` may also be a single
        ``pandas.Series`` or a collection of ``pandas.Series``.
    commodity, optional (default: none)
        Commodity the data applies to. Used to set units etc.

    Returns
    -------
        Flat portfolio line
    """
    # Catch easy cases.
    if isinstance(data, FlatPfLine):  # data already correct instance; quick-return
        return data.set_commodity(commodity)
    elif isinstance(data, nested.NestedPfLine):  # data is a PfLine, but not a flat one: flatten
        return data.flatten().set_commodity(commodity)

    # Data must be processed to find dataframe and kind.
    df = flat_helper.create_df(data)
    df = flat_helper.apply_commodity(df, commodity)
    kind = flat_helper.get_kind(df)
    return FlatPfLine(df, kind, commodity)
