"""
Dataframe-like class to hold general energy-related timeseries; either volume ([MW] or
[MWh]), price ([Eur/MWh]) or both; in all cases there is a single timeseries for each.
"""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, Union

import numpy as np
import pandas as pd

from ... import tools
from ...testing import testing
from . import single_helper
from .base import Kind, PfLine


class SinglePfLine(PfLine):
    """Flat portfolio line, i.e., without children. Has a single dataframe.

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
            return data
        # Otherwise, do normal thing.
        return super().__new__(cls, data)

    def __init__(self, data: Union[PfLine, Dict, pd.DataFrame, pd.Series]):
        if self is data:
            return  # don't continue initialisation, it's already the correct object
        self._df = single_helper.make_dataframe(data)

    # Implementation of ABC methods.

    @property
    def index(self) -> pd.DatetimeIndex:
        return self._df.index

    @property
    def w(self) -> pd.Series:
        if self.kind is Kind.PRICE_ONLY:
            return pd.Series(np.nan, self.index, name="w", dtype="pint[MW]")
        else:
            return pd.Series(self.q / self.index.duration, name="w").pint.to("MW")

    @property
    def q(self) -> pd.Series:
        if self.kind is Kind.PRICE_ONLY:
            return pd.Series(np.nan, self.index, name="q", dtype="pint[MWh]")
        else:
            return self._df["q"]

    @property
    def p(self) -> pd.Series:
        if self.kind is Kind.VOLUME_ONLY:
            return pd.Series(np.nan, self.index, name="p", dtype="pint[Eur/MWh]")
        elif self.kind is Kind.ALL:
            return pd.Series(self.r / self.q, name="p").pint.to("Eur/MWh")
        else:  # self.kind is Kind.PRICE_ONLY
            return self._df["p"]

    @property
    def r(self) -> pd.Series:
        if self.kind is not Kind.ALL:
            return pd.Series(np.nan, self.index, name="r", dtype="pint[Eur]")
        return self._df["r"]

    @property
    def kind(self) -> Kind:
        if "q" in self._df:
            return Kind.ALL if "r" in self._df else Kind.VOLUME_ONLY
        if "p" in self._df:
            return Kind.PRICE_ONLY
        raise ValueError(f"Unexpected value for ._df: {self._df}.")

    def df(
        self, cols: Iterable[str] = None, *args, has_units: bool = True, **kwargs
    ) -> pd.DataFrame:
        # *args, **kwargs needed because base class has this signature.
        if cols is None:
            cols = self.available
        series = {col: getattr(self, col) for col in cols}
        if not has_units:
            series = {col: s.pint.m for col, s in series.items()}
        return pd.DataFrame(series)

    def asfreq(self, freq: str = "MS") -> SinglePfLine:
        if self.kind is Kind.PRICE_ONLY:
            df = tools.changefreq.averagable(self.df("p"), freq)
        elif self.kind is Kind.VOLUME_ONLY:
            df = tools.changefreq.summable(self.df("q"), freq)
        else:  # self.kind is Kind.ALL
            df = tools.changefreq.summable(self.df("qr"), freq)
        return SinglePfLine(df)

    def map_to_year(self, year: int, holiday_country: str = None) -> SinglePfLine:
        if tools.freq.shortest(self.index.freq, "MS") == "MS":
            warnings.warn(
                "This PfLine has a monthly frequency or longer; changing the year is inaccurate, as"
                " details (number of holidays, weekends, offpeak hours, etc) cannot be taken into account."
            )
        # Use averageble data, which is necessary for mapping months (or longer) of unequal length (leap years).
        if self.kind is Kind.PRICE_ONLY:
            df = self.df("p")
        elif self.kind is Kind.VOLUME_ONLY:
            df = self.df("w")
        else:  # self.kind is Kind.ALL
            df = self.df("wp")
        return SinglePfLine(
            tools.changeyear.map_frame_to_year(df, year, holiday_country)
        )

    @property
    def loc(self) -> _LocIndexer:
        return _LocIndexer(self)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        try:
            testing.assert_frame_equal(self._df, other._df, rtol=1e-7)
            return True
        except AssertionError:
            return False
        # return self._df.pint.to_base_units().equals(other._df.pint.to_base_units())

    def __bool__(self) -> bool:
        # False if all relevant timeseries are 0.
        if self.kind is Kind.PRICE_ONLY:
            return not np.allclose(self.p.pint.magnitude, 0)
        elif self.kind is Kind.VOLUME_ONLY:
            return not np.allclose(self.w.pint.magnitude, 0)
        else:  # kind is Kind.ALL
            return not (
                np.allclose(self.w.pint.magnitude, 0)
                and np.allclose(self.r.pint.magnitude, 0)
            )

    def __setitem__(self, *args, **kwargs):
        raise TypeError("Flat portfolio line; cannot add (or change) children.")

    def __getitem__(self, *args, **kwargs):
        raise TypeError("Flat portfolio line is not subscriptable (has no children).")

    def __delitem__(self, name: str):
        raise TypeError("Flat portfolio line; cannot remove children.")

    # Additional methods, unique to this class.

    # (no additional methods)


class _LocIndexer:
    """Helper class to obtain SinglePfLine instance, whose index is subset of original index."""

    def __init__(self, spfl):
        self.spfl = spfl

    def __getitem__(self, arg) -> SinglePfLine:
        new_df = self.spfl.df().loc[arg]
        return SinglePfLine(new_df)
