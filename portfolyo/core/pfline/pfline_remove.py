from __future__ import annotations

import abc
import dataclasses
import functools
import pathlib
from typing import Any, Literal, Mapping

import pandas as pd

from portfolyo.core import pflineb

from ... import toolsb
from ..commodity import Commodity
from . import text as pfl_text
from .enums import Kind, Structure

# Constructors:

# PfLine({'w': ...}, commodity = gas)
# PfLine(list_of_values, columns = ['w'], commodity = gas)
# PfLine(series_with_unit, commodity = gas)
# PfLine(pfline, commodity = gas) --> verify if pfline has same commodity
# PfLine(pfline, commodity = None) --> get commodity from pfline


# Data coercion steps:
# a) ensure all data is a series
# b) link to column name and turn into dataframe
# c) ensure columns have units (use commodity to add them)
# d) ensure existing columns are consistent
# e) add missing columns
# f)


def _apply_correct_timezone(pfl: PfLineb, target_tz) -> None:
    if not isinstance(pfl.index, pd.DatetimeIndex):
        raise ValueError("PfLine data must have datetime index.")
    source_tz = pfl.index.tz
    if source_tz == target_tz:
        return
    elif source_tz is None:  # agnostic -> aware
        fn = lambda s: s.tz_localize(target_tz, ambiguous="infer")
    elif target_tz is None:  # aware -> agnostic
        fn = lambda s: s.tz_localize(None)
    else:  # aware -> aware
        fn = lambda s: s.tz_convert(target_tz)

    for col, s in tuple(pfl.items()):
        pfl[col] = fn(s)


def _verify_no_excess_columns_in_flat_pfl(pfl: PfLineb) -> None:
    """Ensure no expected columns are present."""
    for col in pfl.columns:
        if col not in toolsb.types.COLS:
            raise ValueError(
                f"Found unexpected column name {col}; expected all column names to be one of {toolsb.types.COLS}."
            )


def _add_units_to_existing_nonpint_columns_in_flat_pfl(pfl: PfLineb, no_units: str) -> None:
    """If a column does not have units, raise ValueError (if ``no_units``=='raise') or assume unit
    defined in ``pfl.commodity``."""
    for col, s in tuple(pfl.items()):
        if pd.api.types.is_integer_dtype(s.dtype) or pd.api.types.is_float_dtype(s.dtype):
            if no_units == "raise":
                raise ValueError(
                    f"Found column {col} without units. Initialise with values that have units, or"
                    " set ``implicit == 'units' to assume the default units of the commodity."
                )
            pfl[col] = s.astype(float).astype(f"pint[{pfl.commodity.col_to_units[col]}]")


def _coerce_pintseries_in_flat_pfl(pfl: PfLineb) -> None:
    """Ensure each column is a pintseries."""
    for col, s in tuple(pfl.items()):
        pfl[col] = toolsb.unit.coerce_pintframe_oneunit(s)


def _add_missing_columns_to_flat_pfl_and_check_consistency(pfl: PfLineb) -> None:
    """Add missing columns and check existing columns are consistent."""
    w, q, p, r = (pfl.get(col) for col in "wqpr")
    duration = toolsb.index.duration(pfl.index)

    # Actual logic.
    w, q, p, r = toolsb.wqpr.complete_and_verify_consistency(duration, w, q, p, r)

    # Store.
    if w is not None:
        pfl["w"] = w
    if q is not None:
        pfl["q"] = q
    if p is not None:
        pfl["p"] = p
    if r is not None:
        pfl["r"] = r

    # Sort.
    column_rank = {col: i for i, col in enumerate(toolsb.types.COLS)}
    pfl.sort_index(axis=1, key=lambda columns: columns.map(column_rank), inplace=True)


class PfLineb(abc.ABC):
    """Class to hold a energy- or emissions-related timeseries data.

    Depending on the type of information, will contain one or more of the following:
    - Attridute ``q`` with energy or emissions timeseries (e.g. in GWh or tCO2);
    - Attribute ``w`` with energy rate or emissions rate timeseries (e.g. in kW or tCO2/min);
    - Attribute ``p`` with price timeseries (e.g. in Eur/MWh or Usd/tCO2);
    - Attribute ``r`` with revenue timeseries (e.g. in Eur or Usd).

    In addition, may contain nested PfLines, i.e., children that add up to the PfLine's data.

    Parameters
    ----------
    data
        Data to create portfolio line from.
        For flat PfLine: mapping with one or more attributes or items ``w``, ``q``, ``r``, ``p``;
        all timeseries. Most commonly a ``pandas.DataFrame`` or a dictionary of ``pandas.Series``,
        but may also be e.g. another PfLine object. If they contain a (distinct) ``pint`` data type,
        may also be a single ``pandas.Series`` or a collection of ``pandas.Series``.
        For nested PfLine: mapping, from strings (as the child names) to portfolio lines, or to
        objects that can be converted into portfolio lines.
    commodity, optional
        Commodity describing characteristics of the commodity and the market it is traded on.
    """

    def __new__(
        cls, data: Any, /, *, commodity: Commodity | None = None, _skip_verification: bool = False
    ):
        if cls is not PfLineb:  # User actually called one of its descendents. Just moving along.
            return super().__new__(cls)

        # Construction if data contains pfline instances.
        elif isinstance(data, Mapping) and any(
            isinstance(value, PfLineb) for value in data.values()
        ):
            pass

        # --- actual object initialisation ---
        # Let pandas construct the DataFrame normally.
        super().__init__(data, index, columns, dtype, copy)
        # ------------------------------------

        self.commodity = commodity
        assert self.commodity is not None

        # Data valication and coercion.

        # . Index.
        self.index = toolsb.index.coerce(self.index)
        tz1 = None if self.index.tz is None else self.index.tz.zone  # datetimeindex tz: .zone
        tz2 = None if self.commodity.tz is None else self.commodity.tz.key  # ZoneInfo object: .key
        if tz1 != tz2:
            raise ValueError(
                f"Timezone mismatch: timezone in the data ({tz1}) is distinct from timezone required"
                f" by the commodity ({tz2}). First convert or localize your data."
            )

        # . Columns.
        if not isinstance(self.columns, pd.MultiIndex):  # Flat
            if commodity is None:
                raise ValueError("No commodity provided.")
            self.structure: Structure = Structure.FLAT
            _verify_no_excess_columns_in_flat_pfl(self)
            _add_units_to_existing_nonpint_columns_in_flat_pfl(self, no_units)
            _coerce_pintseries_in_flat_pfl(self)
            _add_missing_columns_to_flat_pfl_and_check_consistency(self)
            _ensure_correct_units_in_flat_pfl(self)
            self.kind: Kind = Kind.from_cols(self.columns)

        self._freeze()  # ensure immutable

    def _freeze(self):
        object.__setattr__(self, "_frozen", True)

    @property
    def _constructor(self):
        return lambda *args, **kwargs: PfLineb(
            *args, **kwargs, commodity=self.commodity, _skip_verification=True
        )

    # def __eq__(self, other: Any) -> bool:
    #     return (
    #         type(other) is PfLineb
    #         and self.commodity == other.commodity
    #         and self.kind is other.kind
    #         and self.structure is other.structure
    #         and self.children == other.children
    #     )

    # Ensure immutability.

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False) and name not in ("_frozen", "_item_cache"):
            raise AttributeError(f"{self.__class__.__name__} is immutable")
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        if getattr(self, "_frozen", False):
            raise TypeError(f"{self.__class__.__name__} is immutable")
        super().__setitem__(key, value)

    @property
    def iloc(self):
        return _IlocIndexer(self)

    @property
    def loc(self):
        return _LocIndexer(self)

    @property
    def slice(self):
        return _SliceIndexer(self)

    def __finalize__(self, other, method=None, **kwargs):
        # Ensures immutability persists after operations like `.copy()` or `.loc[]`
        result = super().__finalize__(other, method=method, **kwargs)
        if getattr(other, "_frozen", False):
            result._freeze()
        return result

    # Printing.

    def __repr__(self) -> str:
        dftext = pd.DataFrame(self).pint.dequantify().__repr__()
        return "\n".join(pfl_text.pflheader(self)) + "\n\n" + dftext

    def print(self: PfLineb, num_of_ts: int = 5, color: bool = True) -> None:
        """Treeview of the portfolio line.

        Parameters
        ----------
        num_of_ts
            How many timestamps to show for each PfLine.
        color
            Make tree structure clearer by including colors. May not work on all output devices.

        Returns
        -------
        None
        """
        print(pfl_text.pfl_as_string(self, num_of_ts, color))

    # Export.

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self)

    @functools.wraps(pd.DataFrame.to_excel)
    def to_excel(self, *args, **kwargs) -> None:
        pd.DataFrame(self).pint.dequantify().tz_localize(None).to_excel(*args, **kwargs)

    @functools.wraps(pd.DataFrame.to_clipboard)
    def to_clipboard(self, *args, **kwargs) -> None:
        pd.DataFrame(self).pint.dequantify().tz_localize(None).to_clipboard(*args, **kwargs)
