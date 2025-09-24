import pandas as pd

from ... import toolsb
from .pflinee import FlatPfLine, NestedPfLine


class FlatLoc:
    """Helper class to obtain FlatPfLine instance, whose index is subset of original index."""

    def __init__(self, pfl: FlatPfLine):
        self.pfl = pfl

    def __getitem__(self, arg) -> FlatPfLine:
        newdf = self.pfl.df.loc[arg]
        try:
            toolsb.standardize.assert_frame_standardized(newdf)
        except AssertionError as e:
            raise ValueError(
                "Timeseries not in expected form. See ``portfolyo.standardize()`` for more information."
            ) from e

        # TODO: .loc might have selected only one or 2 columns, and therefore changed the .kind of the PfLine.
        return FlatPfLine(newdf, self.pfl.kind, self.pfl.commodity)


class FlatSlice:
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
            toolsb.standardize.assert_frame_standardized(newdf)
        except AssertionError as e:
            raise ValueError(
                "Timeseries not in expected form. See ``portfolyo.standardize()`` for more information."
            ) from e
        return FlatPfLine(newdf, self.pfl.kind, self.pfl.commodity)


class NestedLoc:
    """Helper class to obtain NestedPfLine instance, whose index is subset of original index."""

    def __init__(self, pfl: NestedPfLine):
        self.pfl = pfl

    def __getitem__(self, arg) -> NestedPfLine:
        newchildren = {name: child.loc[arg] for name, child in self.pfl.items()}
        return NestedPfLine(newchildren, self.pfl.kind, self.pfl.commodity)


class NestedSlice:
    """Helper class to obtain NestedPfLine instance, whose index is subset of original index.
    Exclude end point from the slice."""

    def __init__(self, pfl: NestedPfLine):
        self.pfl = pfl

    def __getitem__(self, arg) -> NestedPfLine:
        newchildren = {name: child.slice[arg] for name, child in self.pfl.items()}
        return NestedPfLine(newchildren, self.pfl.kind, self.pfl.commodity)
