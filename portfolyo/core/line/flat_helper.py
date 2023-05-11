"""Verify input data and turn into object needed in FlatPfLine instantiation."""

from __future__ import annotations

from typing import Any, Tuple

import pandas as pd

from . import interop
from .enums import Kind


def dataframe_and_kind(data: Any) -> Tuple[pd.DataFrame, Kind]:
    """From data, create a DataFrame with columns `w` and `q`, or column `p`, or column
    `r`, or all (`w`, `q`, `p`, `r`); with relevant units set to them. Also, do some data
    verification, and find the kind of the data from the columns in the dataframe."""
    df = _dataframe(data)
    kind = _kind(df)
    return df, kind


def _dataframe(data: Any) -> pd.DataFrame:
    """From data, create a DataFrame with columns `w` and `q`, or column `p`, or column
    `r`, or all (`w`, `q`, `p`, `r`); with relevant units set to them. Also, do some data
    verification."""
    # Turn into interoperability object.
    if isinstance(data, interop.InOp):
        inop = data
    else:
        inop = interop.InOp.from_data(data)

    # Check data types.
    if inop.agn is not None or inop.nodim is not None:
        err = []
        if inop.nodim is not None:
            err.append(f"explicityly dimensionless ({inop.nodim})")
        if inop.agn is not None:
            err.append(f"dimension-agnostic ({inop.agn})")
        raise ValueError(
            f"Found {' and '.join(err)} data. Use 'w', 'q', 'p', 'r' (e.g. as dictionary"
            " keys), or explicitly pass values with a ``pint`` unit, to indicate dimensionality."
        )

    # Make actual dataframe.
    return inop.to_timeseries().make_consistent().to_df()


def _kind(df: pd.DataFrame) -> Kind:
    """Kind of data, based on columns in dataframe."""
    found = set(df.columns)
    for kind in Kind:
        if set(kind.available) == found:
            return kind

    raise ValueError(f"Unexpected columns for ``df``: {df.columns}.")
