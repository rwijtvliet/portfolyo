"""Verify input data and turn into object needed in FlatPfLine instantiation."""

from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd
import pint

from ... import tools
from ...tools.types import Col, PintTimeDataframe, PintTimeSeries
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
    return inop.to_df()


def apply_commodity(df: PintTimeDataframe, commodity: Commodity | None) -> PintTimeDataframe:
    """Apply ``commodity`` to ``df``, i.e., convert to correct units and do few checks."""
    # Check correct index and column dtype.
    tools.frame.coerce_timeframe(df)  # allowed frequency, full number of days
    tools.frame.coerce_pintframe(df)  # units set on each column

    # No commodity: rudimentary checks.
    if commodity is None:
        for col, s in df.items():
            if tools.wqpr.valid_col(s) != col:
                raise ValueError(
                    f"Data does not have correct unit for this column ({col}): got {s.pint.unit}."
                )
        return df

    # If we are here, the commodity is specified.

    # Check correct timezone.
    if not tools.tz.is_equal(expected := commodity.tz, found := df.index.tz):
        raise ValueError(
            f"Data does not have correct timezone. Expected (by commodity): {expected}. Data has {found}."
        )
    # Check correct start-of-day.
    if (expected := commodity.startofday) != (found := df.index[0].time()):
        raise ValueError(
            f"Data does not have correct start-of-day. Expected (by commodity): {expected}. Data has {found}."
        )
    # Check long enough frequency.
    if tools.freq.up_or_down(found := df.index.freq, shortest := commodity.freq) > 0:
        raise ValueError(
            f"Data has frequency ({found}) that is shorter than minimum ({shortest}) for this commodity."
        )
    # Convert to correct units (raises error if impossible).
    df = pd.DataFrame({col: s.pint.to(commodity.col_to_units[col]) for col, s in df.items()})
    return df
