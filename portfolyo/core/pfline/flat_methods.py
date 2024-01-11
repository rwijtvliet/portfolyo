from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ... import testing
import pandas as pd
from datetime import timedelta

if TYPE_CHECKING:
    from .classes import FlatPfLine


def flatten(self: FlatPfLine) -> FlatPfLine:
    return self


def __eq__(self: FlatPfLine, other: Any) -> bool:
    if not isinstance(other, self.__class__):
        return False
    try:
        testing.assert_frame_equal(self.df, other.df, rtol=1e-7)
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
        return self.pfl.__class__(newdf)  # use same (leaf) class


class SliceIndexer:
    """Helper class to obtain FlatPfLine instance, whose index is subset of original index.
    Exclude end index from the slice"""

    def __init__(self, pfl: FlatPfLine):
        self.pfl = pfl

    def __getitem__(self, arg) -> FlatPfLine:
        date_start = pd.to_datetime(arg.start)
        date_end = pd.to_datetime(arg.stop)

        if arg.stop is not None:
            date_end = date_end - timedelta(seconds=1)

        newdf = self.pfl.df.loc[date_start:date_end]
        # newdf = self.pfl.df.loc[arg]
        # if arg.stop is not None:
        #     newdf = newdf.iloc[:-1]
        return self.pfl.__class__(newdf)  # use same (leaf) class
