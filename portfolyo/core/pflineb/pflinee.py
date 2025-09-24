from __future__ import annotations

import abc
import dataclasses
from typing import Any, Callable, ClassVar, Mapping  # noqa

import numpy as np
import pandas as pd

from portfolyo.toolsb.types import Frequencylike

from ... import toolsb
from ..shared.excelclipboard import ExcelClipboardOutput
from ..shared.ndframelike import NDFrameLike
from . import children, create, indexer
from .arithmatic import PfLineArithmatic
from .children import ChildMethods
from .enums import Kind, Structure
from .flat import FlatMethods
from .nested import NestedMethods
from .plot import PfLinePlot
from .text import TextMethods


# When creating a PfLine (or FlatPfLine or NestedPfLine) instance directly (i.e., with
# PfLine(data)), the code actually creates, initialises, and returns one of its
# subclasses. Normally, this would causes python to finally call the subclass' __init__
# method once more, using the original data as input, undoing the previous
# initialisation. To prevent this, use decorate the subclass with this decorator.
def dont_init_twice(Class):
    """Decorator for PfLine descendents, to allow PfLine to return a child instance."""
    original_init = Class.__init__
    Class._initialized = False

    def wrapped_init(self, *args, **kwargs):
        if not self._initialized:
            object.__setattr__(self, "_initialized", True)
            original_init(self, *args, **kwargs)

    Class.__init__ = wrapped_init

    return Class


class PfLine(NDFrameLike, TextMethods, PfLinePlot, ExcelClipboardOutput, PfLineArithmatic):
    """Class to hold a related energy timeseries. This can be volume data (with q
    [MWh] and w [MW]), price data (with p [Eur/MWh]), revenue data (with r [Eur]), or
    a combination of all.
    """

    def __new__(cls, data=None, *args, **kwargs):
        if cls is not PfLine:
            # User actually called one of its descendents. Just move along
            return super().__new__(cls)

        # User did indeed call PfLine and data must be processed by a descendent's __init__
        return create.pfline(data)

    def __post_init__(self):
        if set(self.df.columns) != set(self.kind.available):
            raise ValueError(f"Expected columns {self.kind.available}, received {self.df.columns}.")

    # Abstact methods to be implemented by FlatPfLine and NestedPfLine.

    @property
    @abc.abstractmethod
    def kind(self) -> Kind: ...

    @property
    @abc.abstractmethod
    def structure(self) -> Structure: ...

    @property
    @abc.abstractmethod
    def df(self) -> pd.DataFrame: ...

    @abc.abstractmethod
    def asfreq(self, freq: Frequencylike = "MS") -> PfLine:
        """Resample the instance to a new frequency.

        Parameters
        ----------
        freq, optional
            The frequency at which to resample. 'YS' for year, 'QS' for quarter, 'MS'
            (default) for month, 'D' for day, 'h' for hour, '15min' for quarterhour.

        Returns
        -------
            PfLine, resampled at wanted frequency.
        """
        ...

    @abc.abstractmethod
    def flatten(self) -> PfLine:
        """Return flattened instance, i.e., without children."""
        ...

    @abc.abstractmethod
    def reindex(self, index: pd.DatetimeIndex) -> PfLine:
        """Reindex; fill any new values with zero (where applicable)."""
        ...

    @abc.abstractmethod
    def agg(self) -> pd.Series | pd.DataFrame:
        """Aggregate values over entire delivery period."""
        ...

    @abc.abstractmethod
    def __bool__(self) -> bool:
        """Return True if object (or, its children) contains any non-zero data."""
        ...

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        """Return True if objects (or, their children) contain identical data."""
        ...

    @abc.abstractmethod
    def po(self: PfLine, freq: Frequencylike) -> pd.DataFrame:
        """Decompose the portfolio line into peak and offpeak values. Takes simple (duration-
        weighted) averages of volume [MW] and price [Eur/MWh] - does not hedge!

        Parameters
        ----------
        freq
            Frequency at which to calculate the peok and offpeak values, e.g. 'MS'
            for monthly.

        Returns
        -------
        pd.DataFrame
            The dataframe shows a composition into peak and offpeak values.

        Notes
        -----
        Only defined if commodity specifies a peakfunction.
        """
        ...

    # @abc.abstractmethod
    # def hedge_with(
    #     self: PfLine,
    #     p: PricePfLine,
    #     how: str = "val",
    #     peak_fn: toolsb.peakfn.PeakFunction = None,
    #     freq: str = "MS",
    # ) -> PfLine:
    #     """Hedge the volume in the portfolio line with a price curve.
    #
    #     Parameters
    #     ----------
    #     p : PricePfLine
    #         Portfolio line with prices to be used in the hedge.
    #     how : str, optional (Default: 'val')
    #         Hedge-constraint. 'vol' for volumetric hedge, 'val' for value hedge.
    #     peak_fn : PeakFunction, optional (default: None)
    #         To hedge with peak and offpeak products: function that returns boolean
    #         Series indicating if timestamps in index lie in peak period.
    #         If None, hedge with base products.
    #     freq : {'D' (days), 'MS' (months, default), 'QS' (quarters), 'YS' (years)}
    #         Frequency of hedging products. E.g. 'QS' to hedge with quarter products.
    #
    #     See also
    #     --------
    #     portfolyo.create_peakfn
    #     portfolyo.germanpower_peakfn
    #
    #     Returns
    #     -------
    #     PfLine
    #         Hedged volume and prices. Index with same frequency as original, but every
    #         timestamp within a given hedging frequency has the same volume [MW] and price.
    #         (or, one volume-price pair for peak, and another volume-price pair for offpeak.)
    #
    #     Notes
    #     -----
    #     If the PfLine contains prices, these are ignored.
    #     """
    #     ...

    # Methods implemented here.

    @property
    def index(self) -> pd.DatetimeIndex:
        """Index of the data, containing the left-bound timestamps of the delivery periods."""
        return self.df.index

    @property
    def start(self) -> pd.Timestamp:
        """Start (incl) of the portfolio line."""
        return self.df.index[0]

    @property
    def end(self) -> pd.Timestamp:
        """End (excl) of the portfolio line."""
        return toolsb.index.to_right(self.df.index)[-1]

    @property
    def w(self) -> pd.Series:
        """(Flat) quantity rate timeseries."""
        if self.kind in [Kind.VOLUME, Kind.COMPLETE]:
            return self.df["w"]
        raise TypeError("This portfolio line does not contain volume information.")

    @property
    def q(self) -> pd.Series:
        """(Flat) quantity timeseries."""
        if self.kind in [Kind.VOLUME, Kind.COMPLETE]:
            return self.df["q"]
        raise TypeError("This portfolio line does not contain volume information.")

    @property
    def p(self) -> pd.Series:
        """(Flat) price timeseries."""
        if self.kind in [Kind.PRICE, Kind.COMPLETE]:
            return self.df["p"]
        raise TypeError("This portfolio line does not contain price information.")

    @property
    def r(self) -> pd.Series:
        """(Flat) revenue timeseries."""
        if self.kind in [Kind.REVENUE, Kind.COMPLETE]:
            return self.df["r"]
        raise TypeError("This portfolio line does not contain revenue information.")

    @property
    def volume(self) -> PfLine:
        """Volume-only PfLine."""
        if self.kind is Kind.VOLUME:
            return self
        elif self.kind is Kind.COMPLETE:
            if self.structure is Structure.FLAT:
                return FlatPfLine(self.df[["w", "q"]], Kind.VOLUME, self.commodity)
            else:
                newchildren = {name: child.volume for name, child in self.items()}
                return NestedPfLine(newchildren, Kind.VOLUME, self.commodity)
        raise TypeError("This portfolio line does not contain volume information.")

    @property
    def price(self) -> PfLine:
        """Price-only PfLine."""
        if self.Kind is Kind.PRICE:
            return self
        elif self.Kind is Kind.COMPLETE:
            # Price of (nested) complete PfLine is not sum of prices of its children, so return flat.
            return FlatPfLine(self.df[["p"]], Kind.PRICE, self.commodity)
        raise TypeError("This portfolio line does not contain price information.")

    @property
    def revenue(self) -> PfLine:
        """Revenue-only PfLine"""
        if self.Kind is Kind.REVENUE:
            return self
        elif self.Kind is Kind.COMPLETE:
            if self.structure is Structure.FLAT:
                return FlatPfLine(self.df[["r"]], self.REVENUE, self.commodity)
            else:
                newchildren = {name: child.revenue for name, child in self.items()}
                return NestedPfLine(newchildren, self.REVENUE, self.commodity)
        raise TypeError("This portfolio line does not contain revenue information.")


@dataclasses.dataclass(frozen=True, repr=False, eq=False)
class FlatPfLine(PfLine, FlatMethods):
    # Normal instance fields.
    # . Class is only called internally, so expect df to be in correct format.
    #   Meaning: correct columns for `kind`, and correct units for `commodity`.
    df: pd.DataFrame
    kind: Kind
    commodity: Commodity
    # Class variables.
    structure: ClassVar[Structure] = Structure.FLAT

    # dataframe = dataframeexport.Flat.dataframe


@dataclasses.dataclass(frozen=True, repr=False, eq=False)
class NestedPfLine(PfLine, NestedMethods, ChildMethods):
    # Normal instance fields.
    # . Class is only called internally, so expect children to be in correct format.
    #   Meaning: all have same `kind` and are in correct units for `commodity`.
    children: Mapping[str, PfLine]
    kind: Kind
    commodity: Commodity
    # Class variables.
    structure: ClassVar[Structure] = Structure.NESTED
    # Calculated instance fields.
    df: pd.DataFrame = dataclasses.field(init=False)

    def __post_init__(self):
        df = sum(child.df for child in self.children.values())
        if self.kind is Kind.COMPLETE:
            df["p"] = df["r"] / df["q"]  # TODO: convert to correct unit
        object.__setattr__(self, "df", df)

    # dataframe = dataframeexport.Nested.dataframe
