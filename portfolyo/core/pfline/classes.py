from __future__ import annotations

import abc
import dataclasses
from typing import Any, Callable, Dict, Iterable, Mapping, Union  # noqa

import numpy as np
import pandas as pd

from ... import tools
from ..mixins import ExcelClipboardOutput, PfLinePlot, PfLineText
from ..ndframelike import NDFrameLike
from . import (
    create,
    dataframeexport,
    decorators,
    flat_methods,
    nested_methods,
    prices,
    children,
)
from .arithmatic import PfLineArithmatic
from .enums import Kind, Structure


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
    def index(self) -> pd.DatetimeIndex:
        """Index of the data, containing the left-bound timestamps of the delivery periods."""
        return self.df.index

    @abc.abstractproperty
    def w(self) -> pd.Series:  # override if applicable
        """Return (flat) volume timeseries in [MW]."""
        ...

    @abc.abstractproperty
    def q(self) -> pd.Series:  # override if applicable
        """Return (flat) volume timeseries in [MWh]."""
        ...

    @abc.abstractproperty
    def p(self) -> pd.Series:  # override if applicable
        """Return (flat) price timeseries in [Eur/MWh]."""
        ...

    @abc.abstractproperty
    def r(self) -> pd.Series:  # override if applicable
        """Return (flat) revenue timeseries in [Eur]."""
        ...

    @abc.abstractproperty
    def volume(self) -> PfLine:  # override if applicable
        """Return volume-only PfLine."""
        ...

    @abc.abstractproperty
    def price(self) -> PfLine:  # override if applicable
        """Return price-only PfLine."""
        ...

    @abc.abstractproperty
    def revenue(self) -> PfLine:  # override if applicable
        """Return revenue-only PfLine"""
        ...

    @abc.abstractmethod
    def flatten(self) -> PfLine:
        """Return flattened instance, i.e., without children."""
        ...

    @abc.abstractmethod
    def map_to_year(self, year: int, holiday_country: str = None) -> PfLine:
        """Transfer the data to a hypothetical other year.

        Parameters
        ----------
        year : int
            Year to transfer the data to.
        holiday_country : str, optional (default: None)
            Country or region for which to assume the holidays. E.g. 'DE' (Germany), 'NL'
            (Netherlands), or 'USA'. See ``holidays.list_supported_countries()`` for
            allowed values.

        Returns
        -------
        PfLine

        Notes
        -----
        Useful for daily (and shorter) data. Copies over the data but takes weekdays (and
        holidays) of target year into consideration. See ``portfolyo.map_frame_as_year()``
        for more information.
        Inaccurate for monthly data and longer, because we only have one value per month,
        and can therefore not take different number of holidays/weekends (i.e., offpeak
        hours) into consideration.
        """
        ...

    @abc.abstractmethod
    def po(self: PfLine, freq: str = "MS") -> pd.DataFrame:
        """Decompose the portfolio line into peak and offpeak values. Takes simple averages
        of volume [MW] and price [Eur/MWh] - does not hedge!

        Parameters
        ----------
        freq : {'MS' (months, default), 'QS' (quarters), 'AS' (years)}
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
        freq: str = "MS",
        po: bool = None,
    ) -> PfLine:
        """Hedge the volume in the portfolio line with a price curve.

        Parameters
        ----------
        p : PricePfLine
            Portfolio line with prices to be used in the hedge.
        how : str, optional (Default: 'val')
            Hedge-constraint. 'vol' for volumetric hedge, 'val' for value hedge.
        freq : {'D' (days), 'MS' (months, default), 'QS' (quarters), 'AS' (years)}
            Frequency of hedging products. E.g. 'QS' to hedge with quarter products.
        po : bool, optional
            Type of hedging products. Set to True to split hedge into peak and offpeak.
            (Default: split if volume timeseries has hourly values or shorter.)

        Returns
        -------
        PfLine
            Hedged volume and prices. Index with same frequency as original, but every
            timestamp within a given hedging frequency has the same volume [MW] and price.
            (or, one volume-price pair for peak, and another volume-price pair for offpeak.)

        Notes
        -----
        - If the PfLine contains prices, these are ignored.
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
    w: pd.Series = property(lambda self: self.df["w"])
    q: pd.Series = property(lambda self: self.df["q"])
    p: pd.Series = series_property_raising_typeerror("price")
    r: pd.Series = series_property_raising_typeerror("revenue")
    volume: VolumePfLine = property(lambda self: self)
    price: PricePfLine = series_property_raising_typeerror("price")
    revenue: RevenuePfLine = series_property_raising_typeerror("revenue")


class PricePfLine:
    kind = Kind.PRICE
    w: pd.Series = series_property_raising_typeerror("volume")
    q: pd.Series = series_property_raising_typeerror("volume")
    p: pd.Series = property(lambda self: self.df["p"])
    r: pd.Series = series_property_raising_typeerror("revenue")
    volume: VolumePfLine = series_property_raising_typeerror("volume")
    price: PricePfLine = property(lambda self: self)
    revenue: RevenuePfLine = series_property_raising_typeerror("revenue")


class RevenuePfLine:
    kind = Kind.REVENUE
    w: pd.Series = series_property_raising_typeerror("volume")
    q: pd.Series = series_property_raising_typeerror("volume")
    p: pd.Series = series_property_raising_typeerror("price")
    r: pd.Series = property(lambda self: self.df["r"])
    volume: VolumePfLine = series_property_raising_typeerror("volume")
    price: PricePfLine = series_property_raising_typeerror("price")
    revenue: RevenuePfLine = property(lambda self: self)


class CompletePfLine:
    kind = Kind.COMPLETE
    w: pd.Series = property(lambda self: self.df["w"])
    q: pd.Series = property(lambda self: self.df["q"])
    p: pd.Series = property(lambda self: self.df["p"])
    r: pd.Series = property(lambda self: self.df["r"])


class FlatPfLine(PfLine):
    """Flat portfolio line, i.e., without children. Only has a single dataframe.

    Notes
    -----
    * If the timeseries or values in ``data`` do not have a ``pint`` data type, the
    standard units are assumed (MW, MWh, Eur, Eur/MWh).
    * If the timeseries or values in ``data`` do have a ``pint`` data type, they are
    converted into the standard units.
    """

    structure = Structure.FLAT

    dataframe = dataframeexport.Flat.dataframe
    flatten = flat_methods.flatten
    po = prices.Flat.po
    hedge_with = prices.Flat.hedge_with
    # map_to_year => on child classes
    loc = flat_methods.loc
    __getitem__ = flat_methods.__getitem__
    # __bool__ => on child classes
    __eq__ = flat_methods.__eq__


class NestedPfLine(PfLine, children.ChildFunctionality):
    structure = Structure.NESTED

    dataframe = dataframeexport.Nested.dataframe
    flatten = nested_methods.flatten
    po = prices.Nested.po
    hedge_with = prices.Nested.hedge_with
    map_to_year = nested_methods.map_to_year
    loc = nested_methods.loc
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

    @decorators.map_to_year_warning
    def map_to_year(self, year: int, holiday_country: str = None) -> FlatVolumePfLine:
        df = self.df[["w"]]  # Averageble data to allow mapping unequal-length periods
        newdf = tools.changeyear.map_frame_to_year(df, year, holiday_country)
        newdf["q"] = newdf["w"] * tools.duration.index(newdf.index)
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

    @decorators.map_to_year_warning
    def map_to_year(self, year: int, holiday_country: str = None) -> FlatVolumePfLine:
        df = self.df[["p"]]  # Averageble data to allow mapping unequal-length periods
        newdf = tools.changeyear.map_frame_to_year(df, year, holiday_country)
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

    @decorators.map_to_year_warning
    def map_to_year(self, year: int, holiday_country: str = None) -> FlatVolumePfLine:
        # Assume that revenue is scales proportionately with duration of period.
        # E.g. 290 Eur in leapyear Feb --> 280 Eur in non-leapyear Feb.
        df = self.df[["r"]] * tools.duration.index(self.df.index)  # Make averageble
        newdf = tools.changeyear.map_frame_to_year(df, year, holiday_country)
        newdf = newdf / tools.duration.index(newdf.index)  # Make summable again
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

    @decorators.map_to_year_warning
    def map_to_year(self, year: int, holiday_country: str = None) -> FlatVolumePfLine:
        df = self.df[["w", "p"]]  # Averagable
        newdf = tools.changeyear.map_frame_to_year(df, year, holiday_country)
        newdf["q"] = newdf["w"] * tools.duration.index(newdf.index)
        newdf["r"] = newdf["q"] * newdf["p"]
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
