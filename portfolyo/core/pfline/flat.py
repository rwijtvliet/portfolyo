"""
Dataframe-like class to hold general energy-related timeseries; either volume ([MW] or
[MWh]), price ([Eur/MWh]) or both; in all cases there is a single timeseries for each.
"""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, Mapping, Union

import numpy as np
import pandas as pd

from ... import tools
from ...testing import testing
from . import flat_helper
from .base import Kind, PfLine


class FlatPfLine(PfLine):
    """Flat portfolio line, i.e., without children. Has a single dataframe.

    Parameters
    ----------
    data: Any
        Generally: mapping with one or more attributes or items ``w``, ``q``, ``r``, ``p``;
        all timeseries. Most commonly a ``pandas.DataFrame`` or a dictionary of
        ``pandas.Series``, but may also be e.g. another PfLine object.
        If they contain a (distinct) ``pint`` data type, ``data`` may also be a single
        ``pandas.Series`` or a collection of ``pandas.Series``.

    Returns
    -------
    FlatPfLine

    Notes
    -----
    * If the timeseries or values in ``data`` do not have a ``pint`` data type, the
    standard units are assumed (MW, MWh, Eur, Eur/MWh).
    * If the timeseries or values in ``data`` do have a ``pint`` data type, they are
    converted into the standard units.
    """

    def __new__(cls, data, *args, **kwargs):
        # Catch case where data is already a valid class instance.
        if isinstance(data, FlatPfLine):
            return data
        # Otherwise, do normal thing.
        return super().__new__(cls)

    def __init__(
        self,
        data: Union[Mapping, PfLine, Dict, pd.DataFrame, pd.Series],
        _internal: bool = False,
    ):
        if self is data:
            return  # don't continue initialisation, it's already the correct object
        self._df = flat_helper.dataframe(data, _internal)
        self._kind = flat_helper.kind(self._df)

    # Implementation of ABC methods.

    kind: Kind = property(lambda self: self._kind)
    index: pd.DatetimeIndex = property(lambda self: self._df.index)

    def _get_series(self, col, unit) -> pd.Series:
        if col not in self._df.columns:
            return pd.Series(np.nan, self.index, name=col, dtype=f"pint[{unit}]")
        return self._df[col]

    w: pd.Series = property(lambda self: self._get_series("w", "MW"))
    q: pd.Series = property(lambda self: self._get_series("q", "MWh"))
    p: pd.Series = property(lambda self: self._get_series("p", "Eur/MWh"))
    r: pd.Series = property(lambda self: self._get_series("r", "Eur"))

    def df(
        self, cols: Iterable[str] = None, *args, has_units: bool = True, **kwargs
    ) -> pd.DataFrame:
        # *args, **kwargs needed because base class has this signature.
        if cols is None:
            cols = self._df.columns
        series = {col: getattr(self, col) for col in cols if col in "wqpr"}
        if not has_units:
            series = {col: s.pint.m for col, s in series.items()}
        return pd.DataFrame(series)

    @property
    def volume(self) -> FlatPfLine:
        if self.kind is Kind.VOLUME:
            return self
        if self.kind is Kind.COMPLETE:
            return FlatPfLine(self._df[["w", "q"]], _internal=True)
        raise ValueError("This portfolio line doesn't contain volumes.")

    @property
    def price(self) -> FlatPfLine:
        if self.kind is Kind.PRICE:
            return self
        if self.kind is Kind.COMPLETE:
            return FlatPfLine(self._df[["p"]], _internal=True)
        raise ValueError("This portfolio line doesn't contain prices.")

    @property
    def revenue(self) -> FlatPfLine:
        if self.kind is Kind.REVENUE:
            return self
        if self.kind is Kind.COMPLETE:
            return FlatPfLine(self._df[["r"]], _internal=True)
        raise ValueError("This portfolio line doesn't contain revenues.")

    def flatten(self) -> FlatPfLine:
        return self

    def asfreq(self, freq: str = "MS") -> FlatPfLine:
        if self.kind is Kind.VOLUME:
            df = self._df[["q"]]
            fn = tools.changefreq.summable
        elif self.kind is Kind.PRICE:
            df = self._df[["p"]]
            fn = tools.changefreq.averagable
        elif self.kind is Kind.REVENUE:
            df = self._df[["r"]]
            fn = tools.changefreq.summable
        else:  # self.kind is Kind.COMPLETE
            df = self._df[["q", "r"]]
            fn = tools.changefreq.summable
        return FlatPfLine(fn(df, freq))

    def map_to_year(self, year: int, holiday_country: str = None) -> FlatPfLine:
        if tools.freq.shortest(self.index.freq, "MS") == "MS":
            warnings.warn(
                "This PfLine has a monthly frequency or longer; changing the year is inaccurate, as"
                " details (number of holidays, weekends, offpeak hours, etc) cannot be taken into account."
            )
        # Use averageble data, which is necessary for mapping months (or longer) of unequal length (leap years).
        if self.kind is Kind.VOLUME:
            df = self._df[["w"]]
        elif self.kind is Kind.PRICE:
            df = self._df[["p"]]
        elif self.kind is Kind.REVENUE:
            df = self._df[["r"]]  # TODO: correct?
        else:  # self.kind is Kind.COMPLETE
            df = self._df[["w", "p"]]
        return FlatPfLine(tools.changeyear.map_frame_to_year(df, year, holiday_country))

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

    def __bool__(self) -> bool:
        # False if all relevant timeseries are 0.
        if self.kind is Kind.VOLUME:
            return not np.allclose(self._df["w"].pint.magnitude, 0)
        elif self.kind is Kind.PRICE:
            return not np.allclose(self._df["p"].pint.magnitude, 0)
        elif self.kind is Kind.REVENUE:
            return not np.allclose(self._df["r"].pint.magnitude, 0)
        else:  # kind is Kind.COMPLETE
            return not (
                np.allclose(self._df["w"].pint.magnitude, 0)
                and np.allclose(self._df["r"].pint.magnitude, 0)
            )

    def __getitem__(self, *args, **kwargs):
        raise TypeError("Flat portfolio line is not subscriptable (has no children).")

    # # Class should be immutable; remove __setitem__ and __delitem__
    # def __setitem__(self, *args, **kwargs):
    #     raise TypeError("Flat portfolio line; cannot add (or change) children.")
    # def __delitem__(self, name: str):
    #     raise TypeError("Flat portfolio line; cannot remove children.")

    # Additional methods, unique to this class.

    # (no additional methods)


class _LocIndexer:
    """Helper class to obtain FlatPfLine instance, whose index is subset of original index."""

    def __init__(self, pfl: FlatPfLine):
        self.pfl = pfl

    def __getitem__(self, arg) -> FlatPfLine:
        new_df = self.pfl._df.loc[arg]
        return FlatPfLine(new_df, _internal=True)
