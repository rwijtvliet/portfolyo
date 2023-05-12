from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ... import tools
from ...prices import convert, hedge
from ...prices.utils import duration_bpo
from . import classes
from .enums import Kind, Structure

if TYPE_CHECKING:
    from .classes import FlatPfLine, NestedPfLine, PfLine, PricePfLine


class Flat:
    def po(self: FlatPfLine, freq: str = "MS") -> pd.DataFrame:
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
        for col in ("w", "p"):
            if col in self.kind.available:
                vals = convert.tseries2bpoframe(self.df[col], freq)
                vals.columns = pd.MultiIndex.from_product([vals.columns, [col]])
                dfs.append(vals)
        df = pd.concat(dfs, axis=1)

        # Add duration.
        durs = duration_bpo(df.index)
        durs.columns = pd.MultiIndex.from_product([durs.columns, ["duration"]])
        df = pd.concat([df, durs], axis=1)

        # Add additional values and sort.
        for prod in prods:
            if "q" in self.kind.available:
                df[(prod, "q")] = df[(prod, "w")] * df[(prod, "duration")]
            if "r" in self.kind.available:
                df[(prod, "r")] = (
                    df[(prod, "q")] * df[(prod, "p")]
                ).pint.to_base_units()
        colidx = pd.MultiIndex.from_product([prods, ("duration", *self.kind.available)])
        return df[colidx]

    def hedge_with(
        self: FlatPfLine,
        p: PricePfLine,
        how: str = "val",
        freq: str = "MS",
        po: bool = None,
    ) -> FlatPfLine:
        if self.kind is Kind.PRICE:
            raise ValueError(
                "Cannot hedge a PfLine that does not contain volume information."
            )
        if self.index.freq not in ["15T", "H", "D"]:
            raise ValueError(
                "Can only hedge a PfLine with daily or (quarter)hourly information."
            )
        if po is None:
            po = self.index.freq in ["15T", "H"]  # default: peak/offpeak if possible
        if po and self.index.freq not in ["15T", "H"]:
            raise ValueError(
                "Can only hedge with peak and offpeak products if PfLine has (quarter)hourly information."
            )

        wout, pout = hedge.hedge(self.w, p.p, how, freq, po)
        df = pd.DataFrame({"w": wout, "p": pout})
        df["q"] = df["w"] * tools.duration.index(df.index)
        df["r"] = df["p"] * df["q"]
        constructor = classes.constructor(Structure.FLAT, Kind.COMPLETE)
        return constructor(df)
        # TODO: move to tools or elsewhere, and reference from there.


class Nested:
    def po(self: NestedPfLine, freq: str = "MS") -> pd.DataFrame:
        return self.flatten().po(freq)

    def hedge_with(
        self: NestedPfLine,
        p: PfLine,
        how: str = "val",
        freq: str = "MS",
        po: bool = None,
    ) -> FlatPfLine:
        return self.flatten().hedge_with(p, how, freq, po)
