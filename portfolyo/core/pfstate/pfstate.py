"""
Dataframe-like class to hold energy-related timeseries, specific to portfolios, at a
certain moment in time (e.g., at the current moment, without any historic data).
"""

from __future__ import annotations


from .pfstate_helper import make_pflines
from ..ndframelike import NDFrameLike
from ..pfline import PfLine, MultiPfLine
from ..mixins import PfStateText, PfStatePlot, OtherOutput
from ...tools import frames

from typing import Iterable, Optional, Union
import pandas as pd
import warnings


class PfState(NDFrameLike, PfStateText, PfStatePlot, OtherOutput):
    """Class to hold timeseries information of an energy portfolio, at a specific moment.

    Parameters
    ----------
    offtakevolume, unsourcedprice, sourced : PfLine
        `offtakevolume` may also be passed as pd.Series with name `q` or `w`.
        `unsourcedprice` may also be passed as pd.Series.
        `sourced` is optional; if non is specified, assume no sourcing has taken place.

    Attributes
    ----------
    offtakevolume : volume-only PfLine
        Offtake. Volumes are <0 for all timestamps (see 'Notes' below).
    sourced : price-and-volume PfLine
        Procurement. Volumes (and normally, revenues) are >0 for all timestamps (see
        'Notes' below).
    unsourced : price-and-volume PfLine
        Procurement/trade that is still necessary until delivery. Volumes (and normally,
        revenues) are >0 if more volume must be bought, <0 if volume must be sold for a
        given timestamp (see 'Notes' below). NB: if volume for a timestamp is 0, its
        price is undefined (NaN) - to get the market prices in this portfolio, use the
        property `.unsourcedprice` instead.
    unsourcedprice : price-only PfLine
        Prices of the unsourced volume.
    netposition : price-and-volume PfLine
        Net portfolio positions. Convenience property for users with a "traders' view".
        Does not follow sign conventions (see 'Notes' below); volumes are <0 if
        portfolio is short and >0 if long. Identical to `.unsourced`, but with sign
        change for volumes and revenues (but not prices).
    procurement : price-and-volume PfLine
        The expected costs needed to source the offtake volume; the sum of the sourced
        and unsourced positions.
    index : pandas.DateTimeIndex
        Left timestamp of row.

    Notes
    -----
    Sign conventions:
    . Volumes (`q`, `w`): >0 if volume flows into the portfolio.
    . Revenues (`r`): >0 if money flows out of the portfolio (i.e., costs).
    . Prices (`p`): normally positive.
    """

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
        if not (ws is qs is rs is ps is None):
            sourced = PfLine({"w": ws, "q": qs, "r": rs, "p": ps})
        else:
            sourced = None
        return cls(PfLine({"q": qo, "w": wo}), PfLine({"p": pu}), sourced)

    def __init__(
        self,
        offtakevolume: Union[PfLine, pd.Series],
        unsourcedprice: Union[PfLine, pd.Series],
        sourced: Optional[PfLine] = None,
    ):
        # The only internal data of this class is stored as PfLines.
        self._offtakevolume, self._unsourcedprice, self._sourced = make_pflines(
            offtakevolume, unsourcedprice, sourced
        )

    @property
    def index(self) -> pd.DatetimeIndex:  # from ABC
        return self._offtakevolume.index

    @property
    def offtake(self) -> PfLine:
        # Future development: return not volume-only but price-and-volume. (by including offtake prices)
        return self._offtakevolume

    @property
    def offtakevolume(self) -> PfLine:
        return self._offtakevolume

    @property
    def sourced(self) -> PfLine:
        return self._sourced

    @property
    def unsourced(self) -> PfLine:
        return -(self._offtakevolume + self._sourced.volume) * self._unsourcedprice

    @property
    def unsourcedprice(self) -> PfLine:
        return self._unsourcedprice

    @property
    def netposition(self) -> PfLine:
        return -self.unsourced

    @property
    def pnl_cost(self):
        return MultiPfLine({"sourced": self.sourced, "unsourced": self.unsourced})

    @property
    def sourcedfraction(self) -> pd.Series:
        return -self._sourced.volume / self._offtakevolume

    @property
    def unsourcedfraction(self) -> pd.Series:
        return 1 - self.sourcedfraction

    def df(
        self,
        cols: Iterable[str] = None,
        flatten: bool = False,
        *args,
        has_units: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """DataFrame for portfolio state in default units.

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

        dfs = []
        for part in ("offtakevolume", "pnl_cost", "sourced", "unsourced"):
            fl = True if part == "pnl_cost" else flatten  # always flatten pnl_cost
            dfin = self[part].df(cols, fl, has_units=has_units)
            dfs.append(frames.add_header(dfin, part))
        return frames.concat(dfs, axis=1)

    # Methods that return new class instance.

    def set_offtakevolume(self, offtakevolume: PfLine) -> PfState:
        warnings.warn(
            "This operation changes the unsourced volume. This causes inaccuracies in its price"
            " if the portfolio has a frequency that is longer than the spot market."
        )
        return PfState(offtakevolume, self._unsourcedprice, self._sourced)

    def set_unsourcedprice(self, unsourcedprice: PfLine) -> PfState:
        return PfState(self._offtakevolume, unsourcedprice, self._sourced)

    def set_sourced(self, sourced: PfLine) -> PfState:
        warnings.warn(
            "This operation changes the unsourced volume. This causes inaccuracies in its price"
            " if the portfolio has a frequency that is longer than the spot market."
        )
        return PfState(self._offtakevolume, self._unsourcedprice, sourced)

    def add_sourced(self, add_sourced: PfLine) -> PfState:
        return self.set_sourced(self.sourced + add_sourced)

    def asfreq(self, freq: str = "MS") -> PfState:  # from ABC
        """Resample the Portfolio to a new frequency.

        Parameters
        ----------
        freq : str, optional
            The frequency at which to resample. 'AS' for year, 'QS' for quarter, 'MS'
            (default) for month, 'D for day', 'H' for hour, '15T' for quarterhour.

        Returns
        -------
        PfState
            Resampled at wanted frequency.
        """
        # pu resampling is most important, so that prices are correctly weighted.
        offtakevolume = self.offtakevolume.asfreq(freq).volume
        unsourcedprice = self.unsourced.asfreq(freq).price  # ensures weighted avg
        sourced = self.sourced.asfreq(freq)
        return PfState(offtakevolume, unsourcedprice, sourced)

    def hedge_of_unsourced(
        self: PfState, how: str = "val", freq: str = "MS", po: bool = None
    ) -> PfLine:
        """Hedge the unsourced volume, at unsourced prices in the portfolio.

        See also
        --------
        PfLine.hedge

        Returns
        -------
        PfLine
            Hedge (volumes and prices) of unsourced volume.
        """
        return self.unsourced.volume.hedge_with(self.unsourcedprice, how, freq, po)

    def source_unsourced(
        self: PfState, how: str = "val", freq: str = "MS", po: bool = None
    ) -> PfState:
        """Simulate PfState if unsourced volume is hedged and sourced at market prices.

        See also
        --------
        .hedge_of_unsourced()

        Returns
        -------
        PfState
            which is fully hedged at time scales of `freq` or longer.
        """
        tosource = self.hedge_of_unsourced(how, freq, po)
        return self.__class__(
            self._offtakevolume, self._unsourcedprice, self.sourced + tosource
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


class _LocIndexer:
    """Helper class to obtain PfState instance, whose index is subset of original index."""

    def __init__(self, pfs):
        self.pfs = pfs

    def __getitem__(self, arg) -> PfState:
        offtakevolume = self.pfs.offtake.volume.loc[arg]
        unsourcedprice = self.pfs.unsourcedprice.loc[arg]
        sourced = self.pfs.sourced.loc[arg]
        return PfState(offtakevolume, unsourcedprice, sourced)


from . import enable_arithmatic  # noqa

enable_arithmatic.apply()
