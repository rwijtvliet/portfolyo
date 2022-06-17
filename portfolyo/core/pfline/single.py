"""
Dataframe-like class to hold general energy-related timeseries; either volume ([MW] or
[MWh]), price ([Eur/MWh]) or both; in all cases there is a single timeseries for each.
"""

from __future__ import annotations

from . import single_helper
from .base import PfLine
from .. import changefreq

from typing import Dict, Iterable, Union
import pandas as pd
import numpy as np


class SinglePfLine(PfLine):
    """Portfolio line without children. Has a single dataframe; .children is the empty
    dictionary.

    Parameters
    ----------
    data: Any
        Generally: object with one or more attributes or items ``w``, ``q``, ``r``, ``p``;
        all timeseries. Most commonly a ``pandas.DataFrame`` or a dictionary of
        ``pandas.Series``, but may also be e.g. another PfLine object.


    Returns
    -------
    SinglePfLine

    Notes
    -----
    * If the timeseries or values in ``data`` do not have a ``pint`` data type, the
    standard units are assumed (MW, MWh, Eur, Eur/MWh).
    * If the timeseries or values in ``data`` do have a ``pint`` data type, they are
    converted into the standard units.
    """

    def __new__(cls, data):
        # Catch case where data is already a valid class instance.
        if isinstance(data, SinglePfLine):
            return data  # TODO: return copy
        # Otherwise, do normal thing.
        return super().__new__(cls, data)

    def __init__(self, data: Union[PfLine, Dict, pd.DataFrame, pd.Series]):
        self._df = single_helper.make_dataframe(data)

    # Implementation of ABC methods.

    @property
    def children(self) -> Dict:
        return {}

    @property
    def index(self) -> pd.DatetimeIndex:
        return self._df.index

    @property
    def w(self) -> pd.Series:
        if self.kind == "p":
            return pd.Series(np.nan, self.index, name="w", dtype="pint[MW]")
        else:
            return pd.Series(self.q / self.index.duration, name="w").pint.to("MW")

    @property
    def q(self) -> pd.Series:
        if self.kind == "p":
            return pd.Series(np.nan, self.index, name="q", dtype="pint[MWh]")
        else:
            return self._df["q"]

    @property
    def p(self) -> pd.Series:
        if self.kind == "q":
            return pd.Series(np.nan, self.index, name="p", dtype="pint[Eur/MWh]")
        elif self.kind == "all":
            return pd.Series(self.r / self.q, name="p").pint.to("Eur/MWh")
        else:  # self.kind == 'p'
            return self._df["p"]

    @property
    def r(self) -> pd.Series:
        if self.kind != "all":
            return pd.Series(np.nan, self.index, name="r", dtype="pint[Eur]")
        return self._df["r"]

    @property
    def kind(self) -> str:
        if "q" in self._df:
            return "all" if "r" in self._df else "q"
        if "p" in self._df:
            return "p"
        raise ValueError(f"Unexpected value for ._df: {self._df}.")

    def df(
        self, cols: Iterable[str] = None, *args, has_units: bool = True, **kwargs
    ) -> pd.DataFrame:
        # *args, **kwargs needed because base class has this signature.
        if cols is None:
            cols = self.available
        series = {col: self[col] for col in cols}
        if not has_units:
            series = {col: s.pint.m for col, s in series.items()}
        return pd.DataFrame(series)

    def asfreq(self, freq: str = "MS") -> SinglePfLine:
        if self.kind == "p":
            df = changefreq.averagable(self.df("p"), freq)
        elif self.kind == "q":
            df = changefreq.summable(self.df("q"), freq)
        else:  # self.kind == 'all'
            df = changefreq.summable(self.df("qr"), freq)
        return SinglePfLine(df)

    @property
    def loc(self) -> _LocIndexer:
        return _LocIndexer(self)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self._df.pint.to_base_units().equals(other._df.pint.to_base_units())

    def __bool__(self) -> bool:
        # False if all relevant timeseries are 0.
        if self.kind == "p":
            return not np.allclose(self.p.pint.magnitude, 0)
        elif self.kind == "q":
            return not np.allclose(self.w.pint.magnitude, 0)
        else:  # kind == 'all'
            return not (
                np.allclose(self.w.pint.magnitude, 0)
                and np.allclose(self.r.pint.magnitude, 0)
            )

    # Additional methods, unique to this class.

    # (no additional methods)


class _LocIndexer:
    """Helper class to obtain SinglePfLine instance, whose index is subset of original index."""

    def __init__(self, spfl):
        self.spfl = spfl

    def __getitem__(self, arg) -> SinglePfLine:
        new_df = self.spfl.df().loc[arg]
        return SinglePfLine(new_df)
