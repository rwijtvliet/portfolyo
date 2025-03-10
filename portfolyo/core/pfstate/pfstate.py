"""
Dataframe-like class to hold energy-related timeseries, specific to portfolios, at a
certain moment in time (e.g., at the current moment, without any historic data).
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Iterable, Optional

import pandas as pd

from ... import tools
from ..pfline import PfLine, create
from ..shared.excelclipboard import ExcelClipboardOutput
from ..shared.ndframelike import NDFrameLike
from . import pfstate_helper
from .arithmatic import PfStateArithmatic
from .plot import PfStatePlot
from .text import PfStateText


@dataclasses.dataclass(frozen=True, repr=False)
class PfState(
    NDFrameLike, PfStateText, PfStatePlot, ExcelClipboardOutput, PfStateArithmatic
):
    """Class to hold timeseries information of an energy portfolio, at a specific moment.

    Parameters
    ----------
    offtakevolume: PfLine (volume-only)
    unsourcedprice: PfLine (price-only)
    sourced : PfLine (price-and-volume), optional
        If not specified, assume no sourcing has taken place.

    All PfLines are trimmed to the length of the offtakevolume.

    Notes
    -----
    Sign conventions:
    - Volumes (`q`, `w`): >0 if volume flows into the portfolio.
    - Revenues (`r`): >0 if money flows out of the portfolio. Consequently, income is negative.
    - Prices (`p`): normally positive.

    Attributes
    ----------
    offtakevolume : volume-only PfLine
        Offtake. Volumes are <=0 for all timestamps (see 'Notes' above).
    sourced : price-and-volume PfLine
        Procurement. Volumes (and normally, revenues) are >=0 for all timestamps (see
        'Notes' above).
    unsourced : price-and-volume PfLine
        Procurement/trade that is still necessary until delivery. Volumes (and normally,
        revenues) are >0 if more volume must be bought, <0 if volume must be sold for a
        given timestamp (see 'Notes' above). NB: if volume for a timestamp is 0, its
        price is undefined (NaN) - to get the market prices in this portfolio, use the
        property `.unsourcedprice` instead.
    unsourcedprice : price-only PfLine
        Prices of the unsourced volume.
    netposition : price-and-volume PfLine
        Net portfolio positions. Convenience property for users with a "traders' view".
        Does not follow sign conventions (see 'Notes' above); volumes are <0 if
        portfolio is short and >0 if long. Identical to `.unsourced`, but with sign
        change for volumes and revenues (but not prices).
    pnl_cost : price-and-volume PfLine
        The expected costs needed to source the offtake volume; the sum of the sourced
        and unsourced positions.
    """

    offtakevolume: PfLine
    unsourcedprice: PfLine
    sourced: Optional[PfLine] = None

    @classmethod
    def from_series(
        cls,
        *,
        pu: pd.Series,
        qo: Optional[pd.Series] = None,
        qs: Optional[pd.Series] = None,
        rs: Optional[pd.Series] = None,
        wo: Optional[pd.Series] = None,
        ws: Optional[pd.Series] = None,
        ps: Optional[pd.Series] = None,
    ):
        """Create Portfolio instance from timeseries.

        Parameters
        ----------
        unsourced prices:
            `pu` [Eur/MWh]
        offtake volume: (at least) one of
            `qo` [MWh]
            `wo` [MW]
        sourced volume and revenue: (at least) two of
            (`qs` [MWh] or `ws` [MW])
            `rs` [Eur]
            `ps` [Eur/MWh]
            If no volume has been sourced, all 4 sourced timeseries may be None.

        Returns
        -------
        PfState
        """
        offtakevolume = create.flatpfline({"q": qo, "w": wo})
        unsourcedprice = create.flatpfline({"p": pu})
        if not (ws is qs is rs is ps is None):
            sourced = create.flatpfline({"w": ws, "q": qs, "r": rs, "p": ps})
        else:
            sourced = None
        return cls(offtakevolume, unsourcedprice, sourced)

    def __post_init__(self):
        offtakevolume, unsourcedprice, sourced = pfstate_helper.make_pflines(
            self.offtakevolume, self.unsourcedprice, self.sourced
        )
        object.__setattr__(self, "offtakevolume", offtakevolume)
        object.__setattr__(self, "unsourcedprice", unsourcedprice)
        object.__setattr__(self, "sourced", sourced)

    @property
    def index(self) -> pd.DatetimeIndex:  # from ABC
        return self.offtakevolume.index

    @property
    def offtake(self) -> PfLine:
        # Future development: return not volume-only but price-and-volume. (by including offtake prices)
        return self.offtakevolume

    @property
    def unsourced(self) -> PfLine:
        return -(self.offtake.volume + self.sourced.volume) | self.unsourcedprice

    @property
    def netposition(self) -> PfLine:
        return -self.unsourced

    @property
    def pnl_cost(self):
        return create.nestedpfline(
            {"sourced": self.sourced, "unsourced": self.unsourced}
        )

    @property
    def sourcedfraction(self) -> pd.Series:
        return self.sourced.volume / -self.offtake.volume

    @property
    def unsourcedfraction(self) -> pd.Series:
        return 1 - self.sourcedfraction

    def dataframe(
        self,
        cols: Iterable[str] = None,
        *args,
        has_units: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """DataFrame for portfolio state in default units.

        Parameters
        ----------
        cols : str, optional (default: all that are available)
            The columns (w, q, p, r) to include in the dataframe.
        has_units : bool, optional (default: True)
            - If True, return dataframe with ``pint`` units. (The unit can be extracted
              as a column level with ``.pint.dequantify()``).
            - If False, return dataframe with float values.

        Returns
        -------
        pd.DataFrame
        """
        dfs = []
        for part in ("offtakevolume", "pnl_cost", "sourced", "unsourced"):
            childlevels = 0 if part == "pnl_cost" else -1  # always flatten pnl_cost
            dfin = self[part].dataframe(
                cols, has_units=has_units, childlevels=childlevels
            )
            dfs.append(tools.frame.add_header(dfin, part))
        return tools.frame.concat(dfs, axis=1)

    # Methods that return new class instance.

    def set_offtakevolume(self, offtakevolume: PfLine) -> PfState:
        warnings.warn(
            "This operation changes the unsourced volume. This causes inaccuracies in its price,"
            " if the portfolio state has a frequency that is longer than the spot market."
        )
        return PfState(offtakevolume, self.unsourcedprice, self.sourced)

    def set_unsourcedprice(self, unsourcedprice: PfLine) -> PfState:
        return PfState(self.offtake.volume, unsourcedprice, self.sourced)

    def set_sourced(self, sourced: PfLine) -> PfState:
        warnings.warn(
            "This operation changes the unsourced volume. This causes inaccuracies in its price"
            " if the portfolio state has a frequency that is longer than the spot market."
        )
        return PfState(self.offtakevolume, self.unsourcedprice, sourced)

    def add_sourced(self, add_sourced: PfLine) -> PfState:
        return self.set_sourced(self.sourced + add_sourced)  # warns

    def asfreq(self, freq: str = "MS") -> PfState:  # from ABC
        """Resample the Portfolio to a new frequency.

        Parameters
        ----------
        freq : str, optional
            The frequency at which to resample. 'YS' for year, 'QS' for quarter, 'MS'
            (default) for month, 'D for day', 'h' for hour, '15min' for quarterhour.

        Returns
        -------
        PfState
            Resampled at wanted frequency.
        """
        # pu resampling is most important, so that prices are correctly weighted.
        offtakevolume = self.offtakevolume.asfreq(freq)
        unsourcedprice = self.unsourced.asfreq(freq).price  # ensures weighted avg
        sourced = self.sourced.asfreq(freq)
        return PfState(offtakevolume, unsourcedprice, sourced)

    def hedge_of_unsourced(
        self: PfState,
        how: str = "val",
        peak_fn: tools.peakfn.PeakFunction = None,
        freq: str = "MS",
    ) -> PfLine:
        """Hedge the unsourced volume, at unsourced prices in the portfolio.

        Parameters
        ----------
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
        PfLine.hedge_with
        portfolyo.create_peakfn
        portfolyo.germanpower_peakfn

        Returns
        -------
        PfLine
            Hedge (volumes and prices) of unsourced volume.
        """
        return self.unsourced.volume.hedge_with(self.unsourcedprice, how, peak_fn, freq)

    def source_unsourced(
        self: PfState,
        how: str = "val",
        peak_fn: tools.peakfn.PeakFunction = None,
        freq: str = "MS",
    ) -> PfState:
        """Simulate PfState if unsourced volume is hedged and sourced at market prices.

        Parameters
        ----------
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
        PfState.hedge_of_unsourced

        Returns
        -------
        PfState
            which is fully hedged at time scales of ``freq`` or longer.
        """
        tosource = self.hedge_of_unsourced(how, peak_fn, freq)
        return self.__class__(
            self.offtakevolume, self.unsourcedprice, self.sourced + tosource
        )

    def mtm_of_sourced(self) -> PfLine:
        """Mark-to-Market value of sourced volume."""
        return self.sourced.volume * (self.unsourcedprice - self.sourced.price)

    # Dunder methods.

    def __getitem__(self, name):
        if hasattr(self, name):
            return getattr(self, name)

    def __eq__(self, other):  # from ABC
        if not isinstance(other, PfState):
            return False
        return all(
            [self[part] == other[part] for part in ["offtake", "unsourced", "sourced"]]
        )

    def __bool__(self):
        return True

    @property
    def loc(self) -> _LocIndexer:  # from ABC
        return _LocIndexer(self)

    @property
    def slice(self) -> _SliceIndexer:  # from ABC
        return _SliceIndexer(self)


class _LocIndexer:
    """Helper class to obtain PfState instance, whose index is subset of original index."""

    def __init__(self, pfs):
        self.pfs = pfs

    def __getitem__(self, arg) -> PfState:
        offtakevolume = self.pfs.offtake.volume.loc[arg]
        unsourcedprice = self.pfs.unsourcedprice.loc[arg]
        sourced = self.pfs.sourced.loc[arg]
        return PfState(offtakevolume, unsourcedprice, sourced)


class _SliceIndexer:
    """Helper class to obtain PfState instance, whose index is subset of original index.
    Exclude end index from the slice"""

    def __init__(self, pfs):
        self.pfs = pfs

    def __getitem__(self, arg) -> PfState:
        offtakevolume = self.pfs.offtake.volume.slice[arg]
        unsourcedprice = self.pfs.unsourcedprice.slice[arg]
        sourced = self.pfs.sourced.slice[arg]
        return PfState(offtakevolume, unsourcedprice, sourced)
