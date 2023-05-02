"""
Abstract Base Class for PfLine.
"""

from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Iterable, Union

import pandas as pd

from ... import tools
from ...prices import convert, hedge
from ...prices.utils import duration_bpo
from ..mixins import OtherOutput, PfLinePlot, PfLineText

# from . import flat, multi, interop  #<-- moved to end of file
from ..ndframelike import NDFrameLike

# Developer notes: we would like to be able to handle 2 cases with volume AND financial
# information. We would like to...
# ... handle the situation where, for some timestamp, the volume q == 0 but the revenue
#   r != 0, because this occasionally arises for the sourced volume, e.g. after buying
#   and selling the same volume at unequal price. So: we want to be able to store q and r.
# ... keep price information even if the volume q == 0, because at a later time this price
#   might still be needed, e.g. if a perfect hedge becomes unperfect. So: we want to be
#   able to store q and p.
# Both cases can be catered to. The first as a 'FlatPfLine', where the timeseries for
# q and r are used in the instance creation. The price is not defined at the timestamp in
# the example, but can be calculated for other timestamps, and downsampling is also still
# possble.
# The second is a bit more complex. It is possible as a 'NestedPfLine'. This has then 2
# 'FlatPfLine' instances as its children: one made from each of the timeseries for q
# and p.


if TYPE_CHECKING:
    from .flat import FlatPfLine  # noqa
    from .nested import NestedPfLine  # noqa


class Kind(Enum):
    """Enumerate what kind of information (which dimensions) is present in a PfLine."""

    # abbreviation, available columns, summable (pfl1 + pfl2) columns
    VOLUME = "vol", "wq", "q"
    PRICE = "pri", "p", "p"
    REVENUE = "rev", "r", "r"
    COMPLETE = "all", "wqpr", "qr"

    @classmethod
    def _missing_(cls, val):
        for member in cls:
            if member.value[0] == val:
                return member

    @property
    def available(self):
        return tuple(self.value[1])

    @property
    def summable(self):
        return tuple(self.value[2])

    def __repr__(self):
        return f"<{self.value[0]}>"

    def __str__(self):
        return self.value[0]


class PfLine(NDFrameLike, PfLineText, PfLinePlot, OtherOutput):
    """Class to hold a related energy timeseries. This can be volume data (with q
    [MWh] and w [MW]), price data (with p [Eur/MWh]), revenue data (with r [Eur]), or
    a combination of all.
    """

    def __new__(cls, data=None):
        if cls is not PfLine:
            # User actually called a descendent class.
            return super().__new__(cls)

        # User actually called PfLine().

        elif isinstance(data, PfLine):
            # Data is already a valid instance and can directly be used.
            return data

        # User called PfLine and data must be processed by a descendent's __init__

        errors = {}
        for subcls in [flat.FlatPfLine, nested.NestedPfLine]:
            # Try passing data to subclasses to see if they can handle it.
            try:
                return subcls(data)
            except (ValueError, TypeError, KeyError) as e:
                errors[subcls] = e
                pass
        errormsg = "\n".join(f"- {c.__name__}: {e.args[0]}" for c, e in errors.items())
        raise ValueError(
            f"Cannot create flat or nested PfLine, with the following reasons:\n{errormsg}"
        )

    # Additional abstract methods to be implemented by descendents.

    @property
    @abstractmethod
    def kind(self) -> Kind:
        """Kind of data that is stored in the instance."""
        ...

    @property
    @abstractmethod
    def w(self) -> pd.Series:
        """(Flattened) power timeseries in [MW]."""
        ...

    @property
    @abstractmethod
    def q(self) -> pd.Series:
        """(Flattened) energy timeseries in [MWh]."""
        ...

    @property
    @abstractmethod
    def p(self) -> pd.Series:
        """(Flattened) price timeseries in [Eur/MWh]."""
        ...

    @property
    @abstractmethod
    def r(self) -> pd.Series:
        """(Flattened) revenue timeseries in [Eur]."""
        ...

    @abstractmethod
    def df(
        self, cols: Iterable[str] = None, flatten: bool = True, has_units: bool = False
    ) -> pd.DataFrame:
        """DataFrame for portfolio line in default units.

        Parameters
        ----------
        cols : str, optional (default: all that are available)
            The columns (w, q, p, r) to include in the dataframe.
        flatten : bool, optional (default: True)
            - If True, include only aggregated timeseries (4 or less; 1 per dimension).
            - If False, include all children and their (intermediate and final)
              aggregations.
        has_units : bool, optional (default: True)
            - If True, return dataframe with ``pint`` units. (The unit can be extracted
              as a column level with ``.pint.dequantify()``).
            - If False, return dataframe with float values.

        Returns
        -------
        pd.DataFrame
        """
        ...

    @property
    @abstractmethod
    def volume(self) -> FlatPfLine:
        """Return (flattened) volume-only PfLine."""
        ...

    @property
    @abstractmethod
    def price(self) -> FlatPfLine:
        """Return (flattened) price-only PfLine."""
        ...

    @property
    @abstractmethod
    def revenue(self) -> FlatPfLine:
        """Return (flattened) revenue-only PfLine."""
        ...

    @abstractmethod
    def flatten(self) -> FlatPfLine:
        """Return flat instance, i.e., without children."""
        ...

    @abstractmethod
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
        Useful for daily data and shorter. Copies over the data but takes weekdays (and
        holidays) target year into consideration. See ``portfolyo.map_frame_as_year()``
        for more information.
        Inaccurate for monthly data and longer, because we only have one value per month,
        and can therefore not take different number of holidays/weekends (i.e., offpeak
        hours) into consideration.
        """
        ...

    @abstractmethod
    def __bool__(self) -> bool:
        """Return True if object (i.e., its children) contains any non-zero data."""
        ...

    # Dunder methods.

    @abstractmethod
    def __getitem__(self, *args, **kwargs):  # Get child
        ...

    # Implemented directly here.

    # Class should be immutable; remove __setitem__ and __delitem__
    def __setitem__(self, *args, **kwargs):
        raise TypeError("PfLine instances are immutable.")

    def __delitem__(self, *args, **kwargs):
        raise TypeError("PfLine instances are immutable.")

    def _set_col_val(
        self, col: str, val: Union[pd.Series, float, int, tools.unit.Q_]
    ) -> FlatPfLine:
        """Set or update a timeseries and return the modified instance."""

        if self.kind is Kind.COMPLETE:
            raise ValueError(
                "Cannot set column value when ``.kind`` is Kind.COMPLETE. First select "
                "the data you wish to keep, e.g. with ``.price``, ``.volume`` or ``.revenue``."
            )

        data = {col: s for col, s in self.df(flatten=True).items()}
        # Ensure volume can be overwritten, by removing conflicting volume information.
        if col == "q" and "w" in data:
            del data["w"]
        elif col == "w" and "q" in data:
            del data["q"]
        data[col] = val
        return flat.FlatPfLine(data)

    def set_w(self, w: Union[pd.Series, float, int, tools.unit.Q_]) -> FlatPfLine:
        """Set or update power timeseries [MW]; returns modified (and flattened) instance."""
        return self._set_col_val("w", w)

    def set_q(self, q: Union[pd.Series, float, int, tools.unit.Q_]) -> FlatPfLine:
        """Set or update energy timeseries [MWh]; returns modified (and flattened) instance."""
        return self._set_col_val("q", q)

    def set_p(self, p: Union[pd.Series, float, int, tools.unit.Q_]) -> FlatPfLine:
        """Set or update price timeseries [Eur/MWh]; returns modified (and flattened) instance."""
        return self._set_col_val("p", p)

    def set_r(self, r: Union[pd.Series, float, int, tools.unit.Q_]) -> FlatPfLine:
        """Set or update revenue timeseries [MW]; returns modified (and flattened) instance."""
        return self._set_col_val("r", r)

    def set_volume(self, other: PfLine) -> FlatPfLine:
        """Set or update volume information; returns modified (and flattened) instance."""
        if not isinstance(other, PfLine) or other.kind is not Kind.VOLUME:
            raise ValueError(
                "Can only set volume equal to a volume-only PfLine. Use .volume to obtain such a PfLine."
            )
        return self.set_q(other.q)

    def set_price(self, other: PfLine) -> FlatPfLine:
        """Set or update price information; returns modified (and flattened) instance."""
        if not isinstance(other, PfLine) or other.kind is not Kind.PRICE:
            raise ValueError(
                "Can only set price equal to a price-only PfLine. Use .price to obtain such a PfLine."
            )
        return self.set_p(other.p)

    def set_revenue(self, other: PfLine) -> FlatPfLine:
        """Set or update revenue information; returns modified (and flattened) instance."""
        if not isinstance(other, PfLine) or other.kind is not Kind.REVENUE:
            raise ValueError(
                "Can only set revenue equal to a revenue-only PfLine. Use .revenue to obtain such a PfLine."
            )
        return self.set_r(other.r)

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
        """
        if self.index.freq not in ["15T", "H"]:
            raise ValueError(
                "Only PfLines with (quarter)hourly values can be turned into peak and offpeak values."
            )
        if freq not in ["MS", "QS", "AS"]:
            raise ValueError(
                f"Value of paramater ``freq`` must be one of {'MS', 'QS', 'AS'} (got: {freq})."
            )

        prods = ("peak", "offpeak")

        # Get values.
        dfs = []
        if "w" in self.kind.available:
            vals = convert.tseries2bpoframe(self.w, freq)
            vals.columns = pd.MultiIndex.from_product([vals.columns, ["w"]])
            dfs.append(vals)
        if "p" in self.kind.available:
            vals = convert.tseries2bpoframe(self.p, freq)
            vals.columns = pd.MultiIndex.from_product([vals.columns, ["p"]])
            dfs.append(vals)
        df = pd.concat(dfs, axis=1)

        # Add duration.
        durs = duration_bpo(df.index)
        durs.columns = pd.MultiIndex.from_product([durs.columns, ["duration"]])
        df = pd.concat([df, durs], axis=1)

        # Add additional values and sort.
        if "q" in self.kind.available:
            for prod in prods:
                df[(prod, "q")] = df[(prod, "w")] * df[(prod, "duration")]
        if "r" in self.kind.available:
            for prod in prods:
                df[(prod, "r")] = (
                    df[(prod, "q")] * df[(prod, "p")]
                ).pint.to_base_units()
        i = pd.MultiIndex.from_product([prods, ("duration", *self.kind.available)])
        return df[i]

    def hedge_with(
        self: PfLine, p: PfLine, how: str = "val", freq: str = "MS", po: bool = None
    ) -> PfLine:
        """Hedge the volume in the portfolio line with a price curve.

        Parameters
        ----------
        p : PfLine
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
        - If ``p`` contains volumes, these are ignored.
        """
        if self.kind is Kind.PRICE:
            raise ValueError(
                "Cannot hedge a PfLine that does not contain volume information."
            )
        if self.index.freq not in ["15T", "H", "D"]:
            raise ValueError(
                "Can only hedge a PfLine with daily or (quarter)hourly information."
            )
        if not isinstance(p, PfLine):
            raise TypeError(
                f"Parameter ``p`` must be a PfLine instance; got {type(p)}."
            )
        if po is None:
            po = self.index.freq in ["15T", "H"]  # default: peak/offpeak if possible
        if po and self.index.freq not in ["15T", "H"]:
            raise ValueError(
                "Can only hedge with peak and offpeak products if PfLine has (quarter)hourly information."
            )

        wout, pout = hedge.hedge(self.w, p.p, how, freq, po)
        return flat.FlatPfLine({"w": wout, "p": pout})


# Must be at end, because they depend on PfLine existing.
from . import enable_arithmatic, flat, interop, nested  # noqa

enable_arithmatic.apply()
