"""Verify input data and turn into object needed in FlatPfLine instantiation."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Tuple

import pandas as pd
import pint

from ... import toolsb
from ...toolsb.types import Col, PintTimeDataframe, PintTimeSeries
from ..commodity import Commodity
from .enums import Kind
from .interop import InOp


def create_df(
    data: (
        Mapping[Col, PintTimeSeries | pint.Quantity]
        | PintTimeDataframe
        | PintTimeSeries
        | Iterable[PintTimeSeries | pint.Quantity]
    ),
) -> PintTimeDataframe:
    """From data, create a DataFrame with datetimeindex, and columns `w` and `q`, or column `p`, or
    column `r`, or all (`w`, `q`, `p`, `r`); all with pint dtype. (i.e., with a unit). Also, do some
    data verification."""

    inop = InOp.from_data(data)

    # Check data types.
    if "nodim" in inop.fields:
        raise ValueError(
            f"Found explicitly dimensionless data: {inop.fields['nodim']}  Add a ``pint`` unit to"
            " indicate dimensionality."
        )

    # Make actual dataframe.
    inop.to_timeseries()
    inop.make_consistent()
    return inop.to_df()


def get_kind(df: pd.DataFrame) -> Kind:
    """Kind of data, based on columns in dataframe."""
    found = set(df.columns)
    for kind in Kind:
        if set(kind.available) == found:
            return kind

    raise ValueError(f"Unexpected columns for ``df``: {df.columns}.")


def apply_commodity(df: PintTimeDataframe, commodity: Commodity | None) -> PintTimeDataframe:
    """Apply ``commodity`` to ``df``, i.e., convert to correct units and do few checks."""
    if commodity is None:
        return df

    # Check dimensionality of columns is correct.
