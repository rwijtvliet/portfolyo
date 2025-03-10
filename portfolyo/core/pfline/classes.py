from __future__ import annotations

import abc
import dataclasses
from typing import Callable, Dict  # noqa

import numpy as np
import pandas as pd

from ... import tools
from ..shared.excelclipboard import ExcelClipboardOutput
from ..shared.ndframelike import NDFrameLike
from . import children, create, dataframeexport, flat_methods, nested_methods
from .arithmatic import PfLineArithmatic
from .enums import Kind, Structure
from .plot import PfLinePlot
from .text import PfLineText


def constructor(structure: Structure, kind: Kind) -> type:
    """Return reference to class constructor."""
    classmatrix = {
        (Structure.FLAT, Kind.VOLUME): FlatVolumePfLine,
        (Structure.FLAT, Kind.PRICE): FlatPricePfLine,
        (Structure.FLAT, Kind.REVENUE): FlatRevenuePfLine,
        (Structure.FLAT, Kind.COMPLETE): FlatCompletePfLine,
        (Structure.NESTED, Kind.VOLUME): NestedVolumePfLine,
        (Structure.NESTED, Kind.PRICE): NestedPricePfLine,
        (Structure.NESTED, Kind.REVENUE): NestedRevenuePfLine,
        (Structure.NESTED, Kind.COMPLETE): NestedCompletePfLine,
    }
    return classmatrix.get((structure, kind))


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


def series_property_raising_typeerror(what: str) -> Callable[[PfLine], pd.Series]:
    @property
    def get_series(self) -> pd.Series:
        raise TypeError(f"This portfolio line does not contain {what} information.")

    return get_series


class PfLine(
    NDFrameLike, PfLineText, PfLinePlot, ExcelClipboardOutput, PfLineArithmatic
):
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
        err = f"Expected columns {self.kind.available}, received {self.df.columns}."
        assert set(self.df.columns) == set(self.kind.available), err

    @property
    @abc.abstractmethod
    def kind(self) -> Kind:
        ...

    @property
    @abc.abstractmethod
    def structure(self) -> Structure:
        ...

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
        return tools.right.index(self.df.index)[-1]

    @property
    @abc.abstractmethod
    def w(self) -> pd.Series:
        """Return (flat) volume timeseries in [MW]."""
        ...

    @property
    @abc.abstractmethod
    def q(self) -> pd.Series:
        """Return (flat) volume timeseries in [MWh]."""
        ...

    @property
    @abc.abstractmethod
    def p(self) -> pd.Series:
        """Return (flat) price timeseries in [Eur/MWh]."""
        ...

    @property
    @abc.abstractmethod
    def r(self) -> pd.Series:
        """Return (flat) revenue timeseries in [Eur]."""
        ...

    @property
    @abc.abstractmethod
    def volume(self) -> PfLine:
        """Return volume-only PfLine."""
        ...

    @property
    @abc.abstractmethod
    def price(self) -> PfLine:
        """Return price-only PfLine."""
        ...

    @property
    @abc.abstractmethod
    def revenue(self) -> PfLine:
        """Return revenue-only PfLine"""
        ...

    @abc.abstractmethod
    def flatten(self) -> PfLine:
        """Return flattened instance, i.e., without children."""
        ...

    @abc.abstractmethod
    def reindex(self, index: pd.DatetimeIndex):
        """Reindex and fill any new values with zero (where applicable)."""
        ...

    @abc.abstractmethod
    def po(
        self: PfLine, peak_fn: tools.peakfn.PeakFunction, freq: str = "MS"
    ) -> pd.DataFrame:
        """Decompose the portfolio line into peak and offpeak values. Takes simple (duration-
        weighted) averages of volume [MW] and price [Eur/MWh] - does not hedge!

        Parameters
        ----------
        peak_fn : PeakFunction
            Function that returns boolean Series indicating if timestamps in index lie in peak period.
        freq : {'MS' (months, default), 'QS' (quarters), 'YS' (years)}
            Frequency of resulting dataframe.

        Returns
        -------
        pd.DataFrame
            The dataframe shows a composition into peak and offpeak values.

        Notes
        -----
        Only relevant for hourly (and shorter) data.
        """
        ...

    @abc.abstractmethod
    def hedge_with(
        self: PfLine,
        p: PricePfLine,
        how: str = "val",
        peak_fn: tools.peakfn.PeakFunction = None,
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

    @abc.abstractmethod
    def __bool__(self) -> bool:
        """Return True if object (i.e., its children) contains any non-zero data."""
        ...

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        """Return True if objects (i.e., their children) contain identical data."""
        ...


class VolumePfLine:
    kind = Kind.VOLUME
    w = property(lambda self: self.df["w"])
    q = property(lambda self: self.df["q"])
    p = series_property_raising_typeerror("price")
    r = series_property_raising_typeerror("revenue")
    volume = property(lambda self: self)
    price = series_property_raising_typeerror("price")
    revenue = series_property_raising_typeerror("revenue")


class PricePfLine:
    kind = Kind.PRICE
    w = series_property_raising_typeerror("volume")
    q = series_property_raising_typeerror("volume")
    p = property(lambda self: self.df["p"])
    r = series_property_raising_typeerror("revenue")
    volume = series_property_raising_typeerror("volume")
    price = property(lambda self: self)
    revenue = series_property_raising_typeerror("revenue")


class RevenuePfLine:
    kind = Kind.REVENUE
    w = series_property_raising_typeerror("volume")
    q = series_property_raising_typeerror("volume")
    p = series_property_raising_typeerror("price")
    r = property(lambda self: self.df["r"])
    volume = series_property_raising_typeerror("volume")
    price = series_property_raising_typeerror("price")
    revenue = property(lambda self: self)


class CompletePfLine:
    kind = Kind.COMPLETE
    w = property(lambda self: self.df["w"])
    q = property(lambda self: self.df["q"])
    p = property(lambda self: self.df["p"])
    r = property(lambda self: self.df["r"])
    # volume => on child clasess
    # price => on child clasess
    # revenue => on child classes


class FlatPfLine:
    structure = Structure.FLAT

    dataframe = dataframeexport.Flat.dataframe
    flatten = flat_methods.flatten
    po = flat_methods.po
    hedge_with = flat_methods.hedge_with
    loc = flat_methods.loc
    slice = flat_methods.slice
    reindex = flat_methods.reindex
    __getitem__ = flat_methods.__getitem__
    # __bool__ => on child classes
    __eq__ = flat_methods.__eq__


class NestedPfLine(children.ChildFunctionality):
    structure = Structure.NESTED

    dataframe = dataframeexport.Nested.dataframe
    flatten = nested_methods.flatten
    po = nested_methods.po
    hedge_with = nested_methods.hedge_with
    loc = nested_methods.loc
    slice = nested_methods.slice
    reindex = nested_methods.reindex
    __bool__ = nested_methods.__bool__
    __eq__ = nested_methods.__eq__


@dont_init_twice
@dataclasses.dataclass(frozen=True, repr=False, eq=False)
class FlatVolumePfLine(FlatPfLine, VolumePfLine, PfLine):
    # Class is only called internally, so expect df to be in correct format. Here: with columns 'w', 'q'.
    df: pd.DataFrame

    def asfreq(self, freq: str = "MS") -> FlatVolumePfLine:
        newdf = tools.changefreq.summable(self.df[["q"]], freq)
        if not len(newdf):
            raise ValueError(
                f"There are no full periods available when changing to the frequency {freq}."
            )
        newdf["w"] = newdf["q"] / tools.duration.index(newdf.index)  # TODO: check unit
        return FlatVolumePfLine(newdf)

    def __bool__(self) -> bool:
        return not np.allclose(self.df["w"].pint.magnitude, 0.0)


@dont_init_twice
@dataclasses.dataclass(frozen=True, repr=False, eq=False)
class NestedVolumePfLine(NestedPfLine, VolumePfLine, PfLine):
    # Class is only called internally, so expect children to be in correct format. Here: all are volume-pflines.
    children: Dict[str, VolumePfLine]
    df: pd.DataFrame = dataclasses.field(init=False)

    def __post_init__(self):
        df = sum(child.df for child in self.children.values())
        object.__setattr__(self, "df", df)

    def asfreq(self, freq: str = "MS") -> NestedVolumePfLine:
        newchildren = {name: child.asfreq(freq) for name, child in self.items()}
        return NestedVolumePfLine(newchildren)


@dont_init_twice
@dataclasses.dataclass(frozen=True, repr=False, eq=False)
class FlatPricePfLine(FlatPfLine, PricePfLine, PfLine):
    # Class is only called internally, so expect df to be in correct format. Here: with column 'p'.
    df: pd.DataFrame

    def asfreq(self, freq: str = "MS") -> FlatPricePfLine:
        newdf = tools.changefreq.averagable(self.df[["p"]], freq)
        if not len(newdf):
            raise ValueError(
                f"There are no full periods available when changing to the frequency {freq}."
            )
        return FlatPricePfLine(newdf)

    def __bool__(self) -> bool:
        return not np.allclose(self.df["p"].pint.magnitude, 0.0)


@dont_init_twice
@dataclasses.dataclass(frozen=True, repr=False, eq=False)
class NestedPricePfLine(NestedPfLine, PricePfLine, PfLine):
    # Class is only called internally, so expect children to be in correct format. Here: all are price-pflines.
    children: Dict[str, PricePfLine]
    df: pd.DataFrame = dataclasses.field(init=False)

    def __post_init__(self):
        df = sum(child.df for child in self.children.values())
        object.__setattr__(self, "df", df)

    def asfreq(self, freq: str = "MS") -> NestedPricePfLine:
        newchildren = {name: child.asfreq(freq) for name, child in self.items()}
        return NestedPricePfLine(newchildren)


@dont_init_twice
@dataclasses.dataclass(frozen=True, repr=False, eq=False)
class FlatRevenuePfLine(FlatPfLine, RevenuePfLine, PfLine):
    # Class is only called internally, so expect df to be in correct format. Here: with column 'r'.
    df: pd.DataFrame

    def asfreq(self, freq: str = "MS") -> FlatRevenuePfLine:
        newdf = tools.changefreq.summable(self.df[["r"]], freq)
        if not len(newdf):
            raise ValueError(
                f"There are no full periods available when changing to the frequency {freq}."
            )
        return FlatRevenuePfLine(newdf)

    def __bool__(self) -> bool:
        return not np.allclose(self.df["r"].pint.magnitude, 0.0)


@dont_init_twice
@dataclasses.dataclass(frozen=True, repr=False, eq=False)
class NestedRevenuePfLine(NestedPfLine, RevenuePfLine, PfLine):
    # Class is only called internally, so expect children to be in correct format. Here: all are revenue-pflines.
    children: Dict[str, RevenuePfLine]
    df: pd.DataFrame = dataclasses.field(init=False)

    def __post_init__(self):
        df = sum(child.df for child in self.children.values())
        object.__setattr__(self, "df", df)

    def asfreq(self, freq: str = "MS") -> NestedRevenuePfLine:
        newchildren = {name: child.asfreq(freq) for name, child in self.items()}
        return NestedRevenuePfLine(newchildren)


@dont_init_twice
@dataclasses.dataclass(frozen=True, repr=False, eq=False)
class FlatCompletePfLine(FlatPfLine, CompletePfLine, PfLine):
    # Class is only called internally, so expect df to be in correct format. Here: with columns 'w', 'q', 'p', 'r'.
    df: pd.DataFrame

    @property
    def volume(self) -> FlatVolumePfLine:
        return FlatVolumePfLine(self.df[["w", "q"]])

    @property
    def price(self) -> FlatPricePfLine:
        return FlatPricePfLine(self.df[["p"]])

    @property
    def revenue(self) -> FlatRevenuePfLine:
        return FlatRevenuePfLine(self.df[["r"]])

    def asfreq(self, freq: str = "MS") -> FlatCompletePfLine:
        newdf = tools.changefreq.summable(self.df[["q", "r"]], freq)
        if not len(newdf):
            raise ValueError(
                f"There are no full periods available when changing to the frequency {freq}."
            )
        newdf["w"] = newdf["q"] / tools.duration.index(newdf.index)
        newdf["p"] = newdf["r"] / newdf["q"]
        return FlatCompletePfLine(newdf)

    def reindex(self, index: pd.DatetimeIndex) -> FlatCompletePfLine:
        tools.testing.assert_indices_compatible(self.index, index)
        newdf = self.df[["w", "q", "r"]].reindex(index, fill_value=0)
        newdf["p"] = newdf["r"] / newdf["q"]
        return FlatCompletePfLine(newdf)

    def __bool__(self) -> bool:
        return not (
            np.allclose(self.df["w"].pint.magnitude, 0.0)
            and np.allclose(self.df["r"].pint.magnitude, 0.0)
        )


@dont_init_twice
@dataclasses.dataclass(frozen=True, repr=False, eq=False)
class NestedCompletePfLine(NestedPfLine, CompletePfLine, PfLine):
    # Class is only called internally, so expect children to be in correct format. Here: all are complete-pflines.
    children: Dict[str, CompletePfLine]
    df: pd.DataFrame = dataclasses.field(init=False)

    def __post_init__(self):
        df = sum(child.df[["w", "q", "r"]] for child in self.children.values())
        df["p"] = df["r"] / df["q"]
        object.__setattr__(self, "df", df)

    @property
    def volume(self) -> NestedVolumePfLine:
        newchildren = {name: child.volume for name, child in self.items()}
        return NestedVolumePfLine(newchildren)

    @property
    def price(self) -> FlatPricePfLine:
        # price of NestedCompletePfLine is not sum of prices of its children, so flatten first.
        newdf = self.df[["p"]]
        return FlatPricePfLine(newdf)

    @property
    def revenue(self) -> NestedRevenuePfLine:
        newchildren = {name: child.revenue for name, child in self.items()}
        return NestedRevenuePfLine(newchildren)

    def asfreq(self, freq: str = "MS") -> NestedCompletePfLine:
        newchildren = {name: child.asfreq(freq) for name, child in self.items()}
        return NestedCompletePfLine(newchildren)
