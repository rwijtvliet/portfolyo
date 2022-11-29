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

# from . import single, multi, interop  #<-- moved to end of file
from ..ndframelike import NDFrameLike

# Developer notes: we would like to be able to handle 2 cases with volume AND financial
# information. We would like to...
# ... handle the situation where, for some timestamp, the volume q == 0 but the revenue
#   r != 0, because this occasionally arises for the sourced volume, e.g. after buying
#   and selling the same volume at unequal price. So: we want to be able to store q and r.
# ... keep price information even if the volume q == 0, because at a later time this price
#   might still be needed, e.g. if a perfect hedge becomes unperfect. So: we want to be
#   able to store q and p.
# Both cases can be catered to. The first as a 'SinglePfLine', where the timeseries for
# q and r are used in the instance creation. The price is not defined at the timestamp in
# the example, but can be calculated for other timestamps, and downsampling is also still
# possble.
# The second is a bit more complex. It is possible as a 'MultiPfLine'. This has then 2
# 'SinglePfLine' instances as its children: one made from each of the timeseries for q
# and p.


if TYPE_CHECKING:
    from .multi import MultiPfLine  # noqa
    from .single import SinglePfLine  # noqa


class Kind(Enum):
    """Enumerate what kind of information (which dimensions) is present in a PfLine."""

    VOLUME_ONLY = "vol"
    PRICE_ONLY = "pri"
    ALL = "all"

    def __repr__(self):
        return f"<{self.value}>"

    def __str__(self):
        return self.value


class PfLine(NDFrameLike, PfLineText, PfLinePlot, OtherOutput):
    """Class to hold a related energy timeseries. This can be volume timeseries with q
    [MWh] and w [MW], a price timeseries with p [Eur/MWh] or both.
    """

    def __new__(cls, data):
        if cls is not PfLine:
            # User actually called a descendent class.
            return super().__new__(cls)

        # User actually called PfLine().

        elif isinstance(data, PfLine):
            # Data is already a valid instance and can directly be used.
            return data

        # User called PfLine and data must be processed by a descendent's __init__

        subclasses = [single.SinglePfLine, multi.MultiPfLine]
        errors = {}
        for subcls in subclasses:
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

    @property
    @abstractmethod
    def kind(self) -> Kind:
        """Kind of data that is stored in the instance."""
        ...

    @abstractmethod
    def df(
        self, cols: Iterable[str], flatten: bool = True, has_units: bool = False
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
    def __setitem__(self, *args, **kwargs):  # Add or overwrite child
        ...

    @abstractmethod
    def __getitem__(self, *args, **kwargs):  # Get child
        ...

    @abstractmethod
    def __delitem__(self, *args, **kwargs):  # Remove child
        ...

    # Implemented directly here.

    @property
    def summable(self) -> str:
        """Which attributes/colums of this PfLine can be added to those of other PfLines
        to get consistent/correct new PfLine."""
        return {Kind.PRICE_ONLY: "p", Kind.VOLUME_ONLY: "q", Kind.ALL: "qr"}[self.kind]

    @property
    def available(self) -> str:  # which time series have values
        """Attributes/columns that are available. One of {'wq', 'p', 'wqpr'}."""
        return {Kind.PRICE_ONLY: "p", Kind.VOLUME_ONLY: "wq", Kind.ALL: "wqpr"}[
            self.kind
        ]

    def flatten(self) -> SinglePfLine:
        """Return flat instance, i.e., without children."""
        return single.SinglePfLine(self)

    @property
    def volume(self) -> SinglePfLine:
        """Return (flattened) volume-only PfLine."""
        # Design decision: could also be non-flattened.
        # if isinstance(self, multi.MultiPfLine):
        #     return multi.MultiPfLine({name: child.volume for name, child in self.items){}})
        return single.SinglePfLine({"q": self.q})

    @property
    def price(self) -> SinglePfLine:
        """Return (flattened) price-only PfLine."""
        # Design decision: could also be non-flattened if self.kind is not ALL
        return single.SinglePfLine({"p": self.p})

    def _set_col_val(
        self, col: str, val: Union[pd.Series, float, int, tools.unit.Q_]
    ) -> SinglePfLine:
        """Set or update a timeseries and return the modified instance."""

        if col == "r" and self.kind is Kind.ALL:
            raise NotImplementedError(
                "Cannot set `r` on a price-and-volume portfolio line; first select `.volume`"
                " or `.price` before applying `.set_r()`."
            )

        inop = interop.InOp(**{col: val}).to_timeseries(self.index)

        # Create input data for new (flat) instance.
        data = {col: getattr(inop, col)}
        if col in ["w", "q", "r"] and "p" in self.available:
            data["p"] = self.p
        elif col in ["p", "r"] and "q" in self.available:
            data["q"] = self.q
        return single.SinglePfLine(data)

    def set_w(self, w: Union[pd.Series, float, int, tools.unit.Q_]) -> SinglePfLine:
        """Set or update power timeseries [MW]; returns modified (and flattened) instance."""
        return self._set_col_val("w", w)

    def set_q(self, q: Union[pd.Series, float, int, tools.unit.Q_]) -> SinglePfLine:
        """Set or update energy timeseries [MWh]; returns modified (and flattened) instance."""
        return self._set_col_val("q", q)

    def set_p(self, p: Union[pd.Series, float, int, tools.unit.Q_]) -> SinglePfLine:
        """Set or update price timeseries [Eur/MWh]; returns modified (and flattened) instance."""
        return self._set_col_val("p", p)

    def set_r(self, r: Union[pd.Series, float, int, tools.unit.Q_]) -> SinglePfLine:
        """Set or update revenue timeseries [MW]; returns modified (and flattened) instance."""
        return self._set_col_val("r", r)

    def set_volume(self, other: PfLine) -> SinglePfLine:
        """Set or update volume information; returns modified (and flattened) instance."""
        if not isinstance(other, PfLine) or other.kind is not Kind.VOLUME_ONLY:
            raise ValueError(
                "Can only set volume equal to a volume-only PfLine. Use .volume to obtain such a PfLine."
            )
        return self.set_q(other.q)

    def set_price(self, other: PfLine) -> SinglePfLine:
        """Set or update price information; returns modified (and flattened) instance."""
        if not isinstance(other, PfLine) or other.kind is not Kind.PRICE_ONLY:
            raise ValueError(
                "Can only set price equal to a price-only PfLine. Use .price to obtain such a PfLine."
            )
        return self.set_p(other.p)

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
        if "w" in self.available:
            vals = convert.tseries2bpoframe(self.w, freq)
            vals.columns = pd.MultiIndex.from_product([vals.columns, ["w"]])
            dfs.append(vals)
        if "p" in self.available:
            vals = convert.tseries2bpoframe(self.p, freq)
            vals.columns = pd.MultiIndex.from_product([vals.columns, ["p"]])
            dfs.append(vals)
        df = pd.concat(dfs, axis=1)

        # Add duration.
        durs = duration_bpo(df.index)
        durs.columns = pd.MultiIndex.from_product([durs.columns, ["duration"]])
        df = pd.concat([df, durs], axis=1)

        # Add additional values and sort.
        if "q" in self.available:
            for prod in prods:
                df[(prod, "q")] = df[(prod, "w")] * df[(prod, "duration")]
        if "r" in self.available:
            for prod in prods:
                df[(prod, "r")] = (
                    df[(prod, "q")] * df[(prod, "p")]
                ).pint.to_base_units()
        i = pd.MultiIndex.from_product([prods, ("duration", *self.available)])
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
        if self.kind is Kind.PRICE_ONLY:
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
        return single.SinglePfLine({"w": wout, "p": pout})


# Must be at end, because they depend on PfLine existing.
from . import enable_arithmatic, interop, multi, single  # noqa

enable_arithmatic.apply()
