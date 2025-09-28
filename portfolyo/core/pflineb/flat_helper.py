"""Verify input data and turn into object needed in FlatPfLine instantiation."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Tuple

import pandas as pd
import pint

from portfolyo.toolsb.types import TimeDataframe

from ... import toolsb
from ...toolsb.types import TimeDataframe, TimeSeries
from .enums import Kind
from .interop import InOp


def dataframe_and_kind(data: Any) -> Tuple[pd.DataFrame, Kind]:
    """From data, create a DataFrame with columns `w` and `q`, or column `p`, or column
    `r`, or all (`w`, `q`, `p`, `r`); with relevant units set to them. Also, do some data
    verification, and find the kind of the data from the columns in the dataframe."""
    df = _dataframe(data)
    kind = _kind(df)
    return df, kind


def _dataframe(
    data: (
        | TimeSeries
        | TimeDataframe
|            Mapping[str, TimeSeries | pint.Quantity]
        | Iterable[TimeSeries | pint.Quantity]
    )
) -> pd.DataFrame:
    """From data, create a DataFrame with columns `w` and `q`, or column `p`, or column
    `r`, or all (`w`, `q`, `p`, `r`); with relevant units set to them. Also, do some data
    verification."""

    inop = InOp.from_data(data)

    # Check data types.
    if inop.fields.get("nodim") is not None:
        raise ValueError(
            f"Found explicitly dimensionless ({inop.fields['nodim']}) data. Add a ``pint`` unit to"
            " indicate dimensionality."
        )

    # Make actual dataframe.
    inop.to_timeseries()
    inop.make_consistent()
    return inop.to_df()


def _kind(df: pd.DataFrame) -> Kind:
    """Kind of data, based on columns in dataframe."""
    found = set(df.columns)
    for kind in Kind:
        if set(kind.available) == found:
            return kind

    raise ValueError(f"Unexpected columns for ``df``: {df.columns}.")
