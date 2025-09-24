from typing import TYPE_CHECKING

from ...toolsb.types import Frequencylike
from . import indexer

if TYPE_CHECKING:
    from .pflinee import FLatPfLine, NestedPfLine, PfLine


class FlatMethods:

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
            df_dict["r"] = toolsb.peakconvert.tseries2poframe(self.r, peak_fn, freq, True)

        # Add price.
        if self.kind is Kind.PRICE:
            df_dict["p"] = toolsb.peakconvert.tseries2poframe(self.p, peak_fn, freq, False)
        elif self.kind is Kind.COMPLETE:
            df_dict["p"] = df_dict["r"] / df_dict["q"]

        # Turn into dataframe. Index: (timestamp, peak/offpeak), columns: {'duration', 'q', ...}
        return pd.DataFrame({k: df.stack() for k, df in df_dict.items()})
