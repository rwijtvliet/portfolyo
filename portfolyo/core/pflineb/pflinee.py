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
from .enums import Kind, Structure
from .plot import PfLinePlot
from .text import PfLineText


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


class PfLine(NDFrameLike, PfLineText, PfLinePlot, ExcelClipboardOutput, PfLineArithmatic):
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

    @abc.abstractmethod
    def hedge_with(
        self: PfLine,
        p: PricePfLine,
        how: str = "val",
        peak_fn: toolsb.peakfn.PeakFunction = None,
        freq: str = "MS",
    ) -> PfLine:
        """Hedge the volume in the portfolio line with a price curve.

        Parameters
        ----------
        p : PricePfLine
            Portfolio line with prices to be used in the hedge.
        how : str, optional (Default: 'val')
            Hedge-constraint. 'vol' for volumetric hedge, 'val' for value hedge.
        peak_fn : PeakFunction, optional (default: None)
            To hedge with peak and offpeak products: function that returns boolean
            Series indicating if timestamps in index lie in peak period.
            If None, hedge with base products.
        freq : {'D' (days), 'MS' (months, default), 'QS' (quarters), 'YS' (years)}
            Frequency of hedging products. E.g. 'QS' to hedge with quarter products.

        See also
        --------
        portfolyo.create_peakfn
        portfolyo.germanpower_peakfn

        Returns
        -------
        PfLine
            Hedged volume and prices. Index with same frequency as original, but every
            timestamp within a given hedging frequency has the same volume [MW] and price.
            (or, one volume-price pair for peak, and another volume-price pair for offpeak.)

        Notes
        -----
        If the PfLine contains prices, these are ignored.
        """
        ...

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
class FlatPfLine(PfLine):
    # Normal instance fields.
    # . Class is only called internally, so expect df to be in correct format.
    #   Meaning: correct columns for `kind`, and correct units for `commodity`.
    df: pd.DataFrame
    kind: Kind
    commodity: Commodity
    # Class variables.
    structure: ClassVar[Structure] = Structure.FLAT

    # Methods / properties.
    # dataframe = dataframeexport.Flat.dataframe
    # hedge_with = flat_methods.hedge_with

    # Methods required by parent.

    @property
    def loc(self) -> indexer.FlatLoc:
        return indexer.FlatLoc(self)

    @property
    def slice(self) -> indexer.FlatSlice:
        return indexer.FlatSlice(self)

    def asfreq(self, freq: Frequencylike = "MS") -> FlatPfLine:
        freq = toolsb.freq.coerce(freq)

        if self.kind is Kind.VOLUME:
            newdf = toolsb.changefreq.summable(self.df[["q"]], freq)
            newdf["w"] = newdf["q"] / toolsb.index.duration(newdf.index)  # TODO: check unit
        elif self.kind is Kind.PRICE:
            newdf = toolsb.changefreq.averagable(self.df[["p"]], freq)
        elif self.kind is Kind.REVENUE:
            newdf = toolsb.changefreq.summable(self.df[["r"]], freq)
        else:  # self.kind is Kind.COMPLETE:
            newdf = toolsb.changefreq.summable(self.df[["q", "r"]], freq)
            newdf["w"] = newdf["q"] / toolsb.index.duration(newdf.index)
            newdf["p"] = newdf["r"] / newdf["q"]

        if not len(newdf):
            raise ValueError(f"There are no full periods when changing to frequency {freq}.")
        return FlatPfLine(newdf, self.kind, self.commodity)

    def flatten(self) -> FlatPfLine:
        return self  # already flat

    def reindex(self, index: pd.DatetimeIndex) -> FlatPfLine:
        toolsb.testing.assert_index_compatible(self.index, index)

        if self.kind is Kind.COMPLETE:
            newdf = self.df[["w", "q", "r"]].reindex(index, fill_value=0)
            newdf["p"] = newdf["r"] / newdf["q"]  # TODO: convert to correct units
        else:
            newdf = self.df.reindex(index, fill_value=0)

        return FlatPfLine(newdf, self.kind, self.commodity)

    def agg(self) -> pd.Series:
        if self.kind is Kind.VOLUME:
            q = self.df["q"].sum()
            duration = toolsb.index.duration(self.index).sum()
            w = q / duration
            return pd.Series({"w": w, "q": q})
        elif self.kind is Kind.PRICE:
            duration = toolsb.index.duration(self.index)
            p = toolsb.wavg.series(self.df["p"], duration)
            return pd.Series({"p": p})
        elif self.kind is Kind.REVENUE:
            r = self.df["r"].sum()
            return pd.Series({"r": r})
        else:  # self.kind is Kind.COMPLETE:
            q = self.df["q"].sum()
            r = self.df["r"].sum()
            duration = toolsb.index.duration(self.index).sum()
            w = q / duration
            p = r / q
            return pd.Series({"w": w, "q": q, "p": p, "r": r})

    def __bool__(self) -> bool:
        return not all(np.allclose(self.df[col].pint.m, 0) for col in self.kind.summable)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        try:
            toolsb.testing.assert_frame_equal(self.df, other.df, rtol=1e-7)
            return True
        except AssertionError:
            return False

    def __getitem__(self, *args, **kwargs):
        raise TypeError("Flat portfolio line is not subscriptable (has no children).")

    def po(self: PfLine, freq: Frequencylike) -> pd.DataFrame:
        peak_fn = self.commodity.peak_fn
        df_dict = {}

        # Always include duration.
        duration = toolsb.index.duration(self.df.index)
        df_dict["duration"] = toolsb.peakconvert.tseries2poframe(duration, peak_fn, freq, True)

        # Add volume.
        if self.kind in [Kind.VOLUME, Kind.COMPLETE]:
            df_dict["q"] = toolsb.peakconvert.tseries2poframe(self.q, peak_fn, freq, True)
            df_dict["w"] = df_dict["q"] / df_dict["duration"]

        # Add revenue.
        if self.kind in [Kind.REVENUE, Kind.COMPLETE]:
            df_dict["r"] = tseries2poframe(self.r, peak_fn, freq, True)

        # Add price.
        if self.kind is Kind.PRICE:
            df_dict["p"] = tseries2poframe(self.p, peak_fn, freq, False)
        elif self.kind is Kind.COMPLETE:
            df_dict["p"] = df_dict["r"] / df_dict["q"]

        # Turn into dataframe.
        return pd.DataFrame({k: df.stack() for k, df in df_dict.items()})


@dataclasses.dataclass(frozen=True, repr=False, eq=False)
class NestedPfLine(children.ChildFunctionality, PfLine):
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
    # po = nested_methods.po
    # hedge_with = nested_methods.hedge_with

    # Methods required by parent.

    @property
    def loc(self) -> indexer.NestedLoc:
        return indexer.NestedLoc(self)

    @property
    def slice(self) -> indexer.NestedSlice:
        return indexer.NestedSlice(self)

    def asfreq(self, freq: Frequencylike = "MS") -> NestedPfLine:
        freq = toolsb.freq.coerce(freq)
        newchildren = {name: child.asfreq(freq) for name, child in self.items()}
        return NestedPfLine(newchildren, self.kind, self.commodity)

    def flatten(self) -> FlatPfLine:
        return FlatPfLine(self.df, self.commodity)  # use toplevel df for initialisation

    def reindex(self, index: pd.DatetimeIndex) -> NestedPfLine:
        newchildren = {name: child.reindex(index) for name, child in self.pfl.items()}
        return NestedPfLine(newchildren, self.kind, self.commodity)

    def agg(self) -> pd.DataFrame:
        dfs = [self.flatten().agg().to_frame("").T]
        for name, child in self.items():
            if child.structure is Structure.FLAT:
                dfs.append(child.agg().to_frame(name).T)
            else:
                dfs.append(toolsb.frame.add_header(child.agg(), name, 0))
        return toolsb.frame.concat(dfs)

    def __bool__(self) -> bool:
        return any(self.children.keys())  # True if a) has children of which b) any are true

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.children == other.children

    def po(self: PfLine, freq: Frequencylike) -> pd.DataFrame:
        dfs = [self.flatten().po(freq)]
        for name, child in self.items():
            dfs.append(toolsb.frame.add_header(child.po(freq), name, 1))
        return toolsb.frame.concat(dfs)
