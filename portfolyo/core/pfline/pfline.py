from __future__ import annotations

import abc
from typing import Any

import pandas as pd

from ... import tools
from ...tools.types import Frequencylike
from ..commodity import Commodity
from ..shared.ndframelike import NDFrameLike
from . import flat, nested
from .enums import Kind, Structure
from .text import TextMethods

# from .arithmatic import PfLineArithmatic
# from ..shared.excelclipboard import ExcelClipboardOutput
# from .plot import PfLinePlot


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


class PfLine(NDFrameLike, TextMethods):  # , PfLinePlot, ExcelClipboardOutput, PfLineArithmatic):
    """Class to hold a energy- or emissions-related timeseries data.

    Depending on the type of information, will contain one or more of the following:
    - Attridute ``q`` with energy or emissions timeseries (e.g. in GWh or tCO2);
    - Attribute ``w`` with energy rate or emissions rate timeseries (e.g. in kW or tCO2/min);
    - Attribute ``p`` with price timeseries (e.g. in Eur/MWh or Usd/tCO2);
    - Attribute ``r`` with revenue timeseries (e.g. in Eur or Usd).

    In addition, may contain nested PfLines, i.e., children that add up to the PfLine's data.
    """

    def __new__(cls, data=None, commodity: Commodity | None = None):
        if cls is not PfLine:
            # User actually called one of its descendents. Just move along
            return super().__new__(cls)

        # User did indeed call PfLine and data must be processed by a descendent's __init__
        return create(data, commodity)

    def __post_init__(self):
        if set(self.df.columns) != set(self.kind.available):
            raise ValueError(f"Expected columns {self.kind.available}, received {self.df.columns}.")

    # Abstact methods to be implemented by FlatPfLine and NestedPfLine.

    @property
    @abc.abstractmethod
    def kind(self) -> Kind:
        """The kind of the PfLine: volume, price, ..."""
        ...

    @property
    @abc.abstractmethod
    def structure(self) -> Structure:
        """The structure of the PfLine: flat or nested."""
        ...

    @property
    @abc.abstractmethod
    def commodity(self) -> Commodity:
        """The commodity of the PfLine, specifying e.g. the units."""
        ...

    @property
    @abc.abstractmethod
    def df(self) -> pd.DataFrame:
        """The underlying dataframe containing the timeseries."""
        ...

    @property
    @abc.abstractmethod
    def volume(self) -> PfLine:
        """Volume-only PfLine."""
        ...

    @property
    @abc.abstractmethod
    def price(self) -> PfLine:
        """Price-only PfLine."""
        ...

    @property
    @abc.abstractmethod
    def revenue(self) -> PfLine:
        """Revenue-only PfLine"""
        ...

    @abc.abstractmethod
    def set_commodity(self, commodity: Commodity | None) -> PfLine:
        """Return new instance with the specified commodity. Use ``None`` to remove commodity."""
        ...

    @abc.abstractmethod
    def asfreq(self, freq: Frequencylike = "MS") -> PfLine:
        """Resample the instance to a new frequency.

        Parameters
        ----------
        freq, optional
            Frequency at which to resample. e.g. 'YS' for year, 'QS' for quarter, 'MS' (default) for
            month, 'D' for day, 'h' for hour, '15min' for quarterhour.

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
            Frequency at which to calculate the peak and offpeak values, e.g. 'MS' for monthly.

        Returns
        -------
            Dataframe showing a decomposition into peak and offpeak values.

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
    #     peak_fn: tools.peakfn.PeakFunction = None,
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
        return tools.stamp.to_right(self.df.index[-1], self.df.index.freq)

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


def create(data: Any, commodity: Commodity | None = None) -> flat.FlatPfLine | nested.NestedPfLine:
    """Create a PfLine instance from the provided data, if possible."""
    # Catch easy cases.
    if isinstance(data, PfLine):
        return data.set_commodity(commodity)

    # Data must be processed to see, which descendent class we need to return.
    errors = {}
    for name, fn in {"flat": flat.create, "nested": nested.create}.items():
        # Try passing data to other creation functions.
        try:
            return fn(data, commodity)
        except (ValueError, TypeError, KeyError) as e:
            errors[name] = e
            pass
    errormsg = "\n".join(f"- {name}: {e.args[0]}" for name, e in errors.items())
    raise ValueError(
        f"Cannot create flat or nested PfLine from the provided data, with the following reasons:\n{errormsg}"
    )
