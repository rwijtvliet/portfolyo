from __future__ import annotations

from typing import Literal, Mapping

import pandas as pd

from ... import toolsb
from ..commodity import Commodity
from .enums import Structure

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


def _verify_no_excess_columns_in_flat_pfl(pfl: PfLineb) -> None:
    """Ensure no expected columns are present."""
    for col, s in pfl.items():
        if col not in toolsb.types.COLS:
            raise ValueError(
                f"Found unexpected column name {col}; expected all column names to be one of {toolsb.types.COLS}."
            )


def _add_units_to_existing_nonpint_columns_in_flat_pfl(pfl: PfLineb, no_units: str) -> None:
    """If a column does not have units, raise ValueError (if ``no_units``=='raise') or assume unit
    defined in ``pfl.commodity``."""
    for col, s in pfl.items():
        if pd.api.types.is_integer_dtype(s.dtype) or pd.api.types.is_float_dtype(s.dtype):
            if no_units == "raise":
                raise ValueError(
                    f"Found column {col} without units. Initialise with values that have units, or"
                    " set ``implicit == 'units' to assume the default units of the commodity."
                )
            pfl[col] = s.astype(float).astype(f"pint[{pfl.commodity.col_to_units[col]}]")


def _coerce_pintseries_in_flat_pfl(pfl: PfLineb) -> None:
    """Ensure each column is a pintseries."""
    for col, s in pfl.items():
        pfl[col] = toolsb.unit.coerce_pintframe_oneunit(s)


def _add_missing_columns_to_flat_pfl_and_check_consistency(pfl: PfLineb) -> None:
    """Add missing columns and check existing columns are consistent."""
    w, q, p, r = (pfl.get(col) for col in "wqpr")
    duration = toolsb.index.duration(pfl.index)

    # Actual logic.
    w, q, p, r = toolsb.wqpr.complete_and_verify(duration, w, q, p, r)

    # Store.
    if w is not None:
        pfl["w"] = w
    if q is not None:
        pfl["q"] = q
    if p is not None:
        pfl["p"] = p
    if r is not None:
        pfl["r"] = r


def _ensure_correct_units_in_flat_pfl(pfl: PfLineb) -> None:
    """Convert each series to the correct unit."""
    for col, s in pfl.items():
        units = pfl.commodity.col_to_units[col]
        pfl[col] = s.pint.to(units)


class PfLineb(pd.DataFrame):
    """Subclass of pandas.DataFrame to hold a energy- or emissions-related timeseries data.

    Depending on the type of information, will contain one or more of the following:
    - Column q with energy or emissions timeseries (e.g. in GWh or tCO2);
    - Column w with energy rate or emissions rate timeseries (e.g. in kW or tCO2/min);
    - Column p with price timeseries (e.g. in Eur/MWh or Usd/tCO2);
    - Column r with revenue timeseries (e.g. in Eur or Usd).

    In addition, may contain nested PfLines, i.e., children that add up to the PfLine's data.

    Parameters
    ----------
    data
        Mapping (column 'w', 'q', 'p' and/or 'r' -> pd.Series)
    commodity
        Commodity describing characteristics of the commodity and the market it is traded on.
    implicit
        Iterable containing 0 or more of the following values:
        - 'units': if units are missing, the unit preferences in ``commodity`` are assumed.

    See also
    --------
    pandas.DataFrame
    """

    def __init__(
        self,
        data=None,
        index=None,
        columns=None,
        dtype=None,
        copy=None,
        *,
        commodity: Commodity | None = None,
        no_units: Literal["raise", "imply"] = "raise",
        _skip_verification: bool = False,
    ):

        # Cases to check:
        # a) is a PfLine object passed? -> Verify commodity matches. Create copy.
        # b) is a Dataframe without any of the columns w q p r? --> ValueError
        # c) is a Dataframe with ONLY columns w q p r? --> flat pfline. Convert data to commodity.unitpref
        # d) is a DataFrame with additional columns --> nested pfline. Verify each subdataframe valid, convert to PfLine with commodity, and verify sum to top-level
        # e) is a Series or an iterable of series, with units -> use commodity to find out the columns and convert to correct unit

        # Construction if data IS a pfline instance.
        if isinstance(data, PfLineb):
            # Guard clauses.
            if commodity and data.commodity is not commodity:
                raise ValueError(
                    "Commodity mismatch: commodity of ``data`` is distinct from ``commodity`` "
                    f"parameter ({data.commodity} vs {commodity})."
                )
            if index is not None or columns is not None or dtype is not None or copy is not None:
                raise ValueError(
                    "Expect ``index``, ``columns``, ``dtype`` and ``copy`` to be None."
                )
            self = data.copy()
            return

        # Construction if data contains pfline instances.
        elif isinstance(data, Mapping) and any(
            isinstance(value, PfLineb) for value in data.values()
        ):
            pass

        # Let pandas construct the DataFrame normally.
        super().__init__(data, index, columns, dtype, copy)

        self.commodity = commodity

        # Data valication and coercion.

        # . Index.
        self.index = toolsb.index.coerce(self.index)

        # . Columns.
        if not isinstance(self.columns, pd.MultiIndex):  # Flat
            self.structure: Structure = Structure.FLAT

            if commodity is None:
                raise ValueError("No commodity provided.")
            _verify_no_excess_columns_in_flat_pfl(self)
            _add_units_to_existing_nonpint_columns_in_flat_pfl(self, no_units)
            _coerce_pintseries_in_flat_pfl(self)
            _add_missing_columns_to_flat_pfl_and_check_consistency(self)
            _ensure_correct_units_in_flat_pfl(self)

        self._freeze()  # ensure immutable

    def _freeze(self):
        object.__setattr__(self, "_frozen", True)

    @property
    def _constructor(self):
        return PfLineb

    # Ensure immutability.

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False) and name not in ("_frozen", "_item_cache"):
            raise AttributeError(f"{self.__class__.__name__} is immutable")
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        if getattr(self, "_frozen", False):
            raise TypeError(f"{self.__class__.__name__} is immutable")
        super().__setitem__(key, value)

    def __finalize__(self, other, method=None, **kwargs):
        # Ensures immutability persists after operations like `.copy()` or `.loc[]`
        result = super().__finalize__(other, method=method, **kwargs)
        if getattr(other, "_frozen", False):
            result._freeze()
        return result

    # Printing.

    def __repr__(self) -> str:
        return repr(super())
