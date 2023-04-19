"""Verify input data and turn into object needed in FlatPfLine instantiation."""

from __future__ import annotations

from typing import Any

import pandas as pd

# from . import flat, nested  #<-- moved to end of file
from . import flat, interop, nested
from .base import Kind


def dataframe(data: Any, __internal: bool = False) -> pd.DataFrame:
    """From data, create a DataFrame with columns `w` and `q`, or column `p`, or column
    `r`, or all (`w`, `q`, `p`, `r`); with relevant units set to them. Also, do some data
    verification."""
    # Shortcut if PfLine is passed.
    if isinstance(data, flat.FlatPfLine):
        return data._df
    if isinstance(data, nested.NestedPfLine):
        # return data.flatten()._df  <-- don't use, causes infinite recursion due to __init__ calls.
        return pd.DataFrame({col: getattr(data, col) for col in data.kind.available})
    # Shortcut if called internally.
    if __internal and isinstance(data, pd.DataFrame):
        return data

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


def kind(df: pd.DataFrame) -> Kind:
    """Kind of data, based on columns in (consistent) dataframe."""
    has_w, has_q, has_p, has_r = (col in df for col in "wqpr")
    if has_w and has_q and not has_p and not has_r:
        return Kind.VOLUME
    if not has_w and not has_q and has_p and not has_r:
        return Kind.PRICE
    if not has_w and not has_q and not has_p and has_r:
        return Kind.REVENUE
    if has_w and has_q and has_p and has_r:
        return Kind.COMPLETE
    raise ValueError(f"Unexpected columns for ``df``: {df.columns}.")
