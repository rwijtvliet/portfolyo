from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from ... import tools
from ...tools.peakconvert import tseries2poframe
from . import classes
from .enums import Kind, Structure

if TYPE_CHECKING:
    from .classes import FlatPfLine, PfLine, PricePfLine


def flatten(self: FlatPfLine) -> FlatPfLine:
    return self


def po(
    self: PfLine, peak_fn: tools.peakfn.PeakFunction, freq: str = "MS"
) -> pd.DataFrame:
    df_dict = {}

    # Always include duration.
    duration = tools.duration.index(self.df.index)
    df_dict["duration"] = tseries2poframe(duration, peak_fn, freq, True)

    # Add volume.
    if self.kind in [Kind.VOLUME, Kind.COMPLETE]:
        df_dict["q"] = tseries2poframe(self.q, peak_fn, freq, True)
        df_dict["w"] = df_dict["q"] / df_dict["duration"]

    # Add revenue.
    if self.kind in [Kind.REVENUE, Kind.COMPLETE]:
        df_dict["r"] = tseries2poframe(self.r, peak_fn, freq, True)

    # Add price.
    if self.kind is Kind.PRICE:
        df_dict["p"] = tseries2poframe(self.p, peak_fn, freq, False)
    elif self.kind is Kind.COMPLETE:
        df_dict["p"] = df_dict["r"] / df_dict["q"]

    # Turn into dataframe and put 'peak' and 'offpeak' on top.
    return pd.DataFrame(df_dict).swaplevel(axis=1).sort_index(axis=1)


def hedge_with(
    self: PfLine,
    p: PricePfLine,
    how: str = "val",
    peak_fn: tools.peakfn.PeakFunction = None,
    freq: str = "MS",
) -> FlatPfLine:
    if self.kind not in [Kind.VOLUME, Kind.COMPLETE]:
        raise ValueError(
            "Cannot hedge a PfLine that does not contain volume information."
        )
    if self.index.freq not in ["15T", "H", "D"]:
        raise ValueError(
            "Can only hedge a PfLine with daily or (quarter)hourly information."
        )

    wout, pout = tools.hedge.hedge(self.w, p.p, how, peak_fn, freq)
    constructor = classes.constructor(Structure.FLAT, Kind.COMPLETE)
    return constructor({"w": wout, "p": pout})


def __eq__(self: FlatPfLine, other: Any) -> bool:
    if not isinstance(other, self.__class__):
        return False
    try:
        tools.testing.assert_frame_equal(self.df, other.df, rtol=1e-7)
        return True
    except AssertionError:
        return False


def __getitem__(self: FlatPfLine, *args, **kwargs):
    raise TypeError("Flat portfolio line is not subscriptable (has no children).")


@property
def loc(self: FlatPfLine) -> LocIndexer:
    return LocIndexer(self)


@property
def slice(self: FlatPfLine) -> SliceIndexer:
    return SliceIndexer(self)


class LocIndexer:
    """Helper class to obtain FlatPfLine instance, whose index is subset of original index."""

    def __init__(self, pfl: FlatPfLine):
        self.pfl = pfl

    def __getitem__(self, arg) -> FlatPfLine:
        newdf = self.pfl.df.loc[arg]
        try:
            tools.standardize.assert_frame_standardized(newdf)
        except AssertionError as e:
            raise ValueError(
                "Timeseries not in expected form. See ``portfolyo.standardize()`` for more information."
            ) from e

        return self.pfl.__class__(newdf)  # use same (leaf) class


class SliceIndexer:
    """Helper class to obtain FlatPfLine instance, whose index is subset of original index.
    Exclude end point from the slice."""

    def __init__(self, pfl: FlatPfLine):
        self.pfl = pfl

    def __getitem__(self, arg) -> FlatPfLine:
        mask = pd.Index([True] * len(self.pfl.df))
        if arg.start is not None:
            mask &= self.pfl.index >= arg.start
        if arg.stop is not None:
            mask &= self.pfl.index < arg.stop

        newdf = self.pfl.df.loc[mask]
        try:
            tools.standardize.assert_frame_standardized(newdf)
        except AssertionError as e:
            raise ValueError(
                "Timeseries not in expected form. See ``portfolyo.standardize()`` for more information."
            ) from e
        return self.pfl.__class__(newdf)  # use same (leaf) class
