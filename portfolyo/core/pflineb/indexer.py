import pandas as pd

from ... import toolsb
from . import flat, nested


def _assert_flat_data_ok(newdf: pd.DataFrame, olddf: pd.DataFrame) -> None:
    if set(newdf.columns) != set(olddf.columns):
        raise ValueError(
            "This method can only be used to select a subset of the rows. To change the kind of"
            " the PfLine, use e.g. .volume or .price. Or, for full flexibility, use .df to work"
            " directly with the pandas dataframe."
        )
    try:
        toolsb.standardize.assert_frame_standardized(newdf)
    except AssertionError as e:
        raise ValueError(
            "Timeseries not in expected form. See ``portfolyo.standardize()`` for more information."
        ) from e


class FlatLoc:
    """Helper class to obtain FlatPfLine instance, whose index is subset of original index."""

    def __init__(self, pfl: flat.FlatPfLine):
        self.pfl = pfl

    def __getitem__(self, arg) -> flat.FlatPfLine:
        newdf = self.pfl.df.loc[arg]
        _assert_flat_data_ok(newdf, self.pfl.df)
        # TODO: maybe just use the user-input route instead of flat.FlatPfLine(), to ensure data is checked?
        return flat.FlatPfLine(newdf, self.pfl.kind, self.pfl.commodity)


class FlatIloc:
    """Helper class to obtain FlatPfLine instance, whose index is subset of original index."""

    def __init__(self, pfl: flat.FlatPfLine):
        self.pfl = pfl

    def __getitem__(self, arg) -> flat.FlatPfLine:
        newdf = self.pfl.df.iloc[arg]
        _assert_flat_data_ok(newdf, self.pfl.df)
        # TODO: .iloc might have selected only one or 2 columns, and therefore changed the .kind of the PfLine.
        return flat.FlatPfLine(newdf, self.pfl.kind, self.pfl.commodity)


class FlatSlice:
    """Helper class to obtain FlatPfLine instance, whose index is subset of original index.
    Exclude end point from the slice."""

    def __init__(self, pfl: flat.FlatPfLine):
        self.pfl = pfl

    def __getitem__(self, arg) -> flat.FlatPfLine:
        mask = pd.Index([True] * len(self.pfl.df))
        if arg.start is not None:
            mask &= self.pfl.index >= arg.start
        if arg.stop is not None:
            mask &= self.pfl.index < arg.stop

        newdf = self.pfl.df.loc[mask]
        _assert_flat_data_ok(newdf, self.pfl.df)
        return flat.FlatPfLine(newdf, self.pfl.kind, self.pfl.commodity)


class NestedLoc:
    """Helper class to obtain NestedPfLine instance, whose index is subset of original index."""

    def __init__(self, pfl: nested.NestedPfLine):
        self.pfl = pfl

    def __getitem__(self, arg) -> nested.NestedPfLine:
        newchildren = {name: child.loc[arg] for name, child in self.pfl.items()}
        return nested.NestedPfLine(newchildren, self.pfl.kind, self.pfl.commodity)


class NestedIloc:
    """Helper class to obtain NestedPfLine instance, whose index is subset of original index."""

    def __init__(self, pfl: nested.NestedPfLine):
        self.pfl = pfl

    def __getitem__(self, arg) -> nested.NestedPfLine:
        newchildren = {name: child.iloc[arg] for name, child in self.pfl.items()}
        return nested.NestedPfLine(newchildren, self.pfl.kind, self.pfl.commodity)


class NestedSlice:
    """Helper class to obtain NestedPfLine instance, whose index is subset of original index.
    Exclude end point from the slice."""

    def __init__(self, pfl: nested.NestedPfLine):
        self.pfl = pfl

    def __getitem__(self, arg) -> nested.NestedPfLine:
        newchildren = {name: child.slice[arg] for name, child in self.pfl.items()}
        return nested.NestedPfLine(newchildren, self.pfl.kind, self.pfl.commodity)
