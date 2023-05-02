"""
Module to create nested pflines. These behave exactly as flat pflines, with all the
same methods. But which include several Pflines as children that can be accessed by
their name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Mapping, Union

import numpy as np
import pandas as pd

from ... import tools
from . import flat, nested_helper
from .base import Kind, PfLine

if TYPE_CHECKING:
    from .flat import FlatPfLine


class NestedPfLine(PfLine, Mapping):
    """Portfolio line with children.

    Children's names can be found by looping over the object and accessed by indexing.
    (name, child)-tuples are available from the .items() method.

    Parameters
    ----------
    data: Any
        Generally: object with a mapping from strings to PfLine instances; most commonly a
        dictionary.
    """

    def __new__(cls, data, *args, **kwargs):
        # Catch case where data is already a valid class instance.
        if isinstance(data, NestedPfLine):
            return data
        # Otherwise, do normal thing.
        return super().__new__(cls)

    def __init__(
        self,
        data: Union[NestedPfLine, Mapping[str, PfLine], pd.DataFrame],
        _internal: bool = False,
    ):
        if self is data:
            return  # don't continue initialisation, it's already the correct object

        if _internal:
            children = data
        else:
            children = {
                name: self._prepare_child(name, child)
                for name, child in nested_helper.make_mapping(data).items()
            }
        self._children, self._kind = nested_helper.children_and_kind(children)
        self._df = nested_helper.dataframe(self._children, self._kind)

    # Implementation of ABC methods.

    kind: Kind = property(lambda self: self._kind)
    index: pd.DatetimeIndex = property(lambda self: self._df.index)

    def _get_series(self, col, unit) -> pd.Series:
        if col not in self._df.columns:
            return pd.Series(np.nan, self.index, name=col, dtype=f"pint[{unit}]")
        return self._df[col]

    w: pd.Series = property(lambda self: self._get_series("w", "MW"))
    q: pd.Series = property(lambda self: self._get_series("q", "MWh"))
    p: pd.Series = property(lambda self: self._get_series("p", "Eur/MWh"))
    r: pd.Series = property(lambda self: self._get_series("r", "Eur"))

    def df(
        self,
        cols: Iterable[str] = None,
        flatten: bool = True,
        *arg,
        has_units: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        if flatten:
            if cols is None:
                cols = self._df.columns
            series = {col: getattr(self, col) for col in cols if col in "wqpr"}
            if not has_units:
                series = {col: s.pint.m for col, s in series.items()}
            return pd.DataFrame(series)

        # One big dataframe.
        dfs = [self.df(cols, flatten=True)]
        for name, child in self.items():
            dfs.append(tools.frame.add_header(child.df(cols, flatten=False), name))
        return tools.frame.concat(dfs, 1)

    @property
    def volume(self) -> NestedPfLine:
        if self.kind is Kind.VOLUME:
            return self
        if self.kind is Kind.COMPLETE:
            return NestedPfLine(
                {name: child.volume for name, child in self.items()}, _internal=True
            )
        raise ValueError("This portfolio line doesn't contain volumes.")

    @property
    def price(self) -> NestedPfLine:
        if self.kind is Kind.PRICE:
            return self
        if self.kind is Kind.COMPLETE:
            raise ValueError(
                "This is a complete portfolio line, i.e., that also contains volume "
                "information. To get its prices, first .flatten() it."
            )
        raise ValueError("This portfolio line doesn't contain prices.")

    @property
    def revenue(self) -> NestedPfLine:
        if self.kind is Kind.REVENUE:
            return self
        if self.kind is Kind.COMPLETE:
            return NestedPfLine(
                {name: child.revenue for name, child in self.items()}, _internal=True
            )
        raise ValueError("This portfolio line doesn't contain revenues.")

    def flatten(self) -> FlatPfLine:
        return flat.FlatPfLine(self._df, _internal=True)

    def asfreq(self, freq: str) -> NestedPfLine:
        return NestedPfLine(
            {name: child.asfreq(freq) for name, child in self.items()}, _internal=True
        )

    def map_to_year(self, year: int, holiday_country: str) -> NestedPfLine:
        return NestedPfLine(
            {
                name: child.map_to_year(year, holiday_country)
                for name, child in self.items()
            },
            _internal=True,
        )

    @property
    def loc(self) -> _LocIndexer:
        return _LocIndexer(self)

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._children == other._children

    def __bool__(self) -> bool:
        # True if a) has children of which b) any are true
        return any(self._children.keys())

    def _prepare_child(self, name: str, child: Union[PfLine, Any]) -> PfLine:
        """Assure that name can be used for a child, and that child is PfLine."""
        if name in dir(self):  # don't use hasattr(): it runs the code in the properties
            raise ValueError(
                f"Cannot name child '{name}', this is a reserved attribute name."
            )
        if not isinstance(name, str):
            raise TypeError(
                f"Parameter ``name`` must be a string; got {name} ({type(name)})."
            )
        if not isinstance(child, PfLine):
            child = PfLine(child)
        return child

    def __getitem__(self, name: str):
        if name not in self._children:
            raise KeyError(f"Portfolio line does not have child with name '{name}'.")
        return self._children[name]

    # Class should be immutable; remove __setitem__ and __delitem__
    # def __setitem__(self, name: str, child: Union[PfLine, Any]):
    #     children = {**self._children, name: self._prepare_child(name, child)}
    #     self._children, _ = nested_helper.children_and_kind(children)
    #     self._df = nested_helper.dataframe(self._children, self._kind)
    # def __delitem__(self, name: str):
    #     if name not in self._children:
    #         raise KeyError(f"Portfolio line does not have child with name '{name}'.")
    #     if len(self._children) == 1:
    #         raise RuntimeError("Cannot remove the last child of a portfolio line.")
    #     del self._children[name]
    #     self._df = nested_helper.dataframe(self._children, self._kind)

    # Additional methods, unique to this class.

    # . Iterate over children.

    def items(self):
        """Iterate over children in (name, child)-tuples."""
        return self._children.items()

    def __iter__(self):
        return iter(self._children.keys())

    def __len__(self):
        return len(self._children)

    # . Allow access to children by attribute.

    def __getattr__(self, name):
        if name not in self._children:
            raise AttributeError(f"No such attribute '{name}'.")
        return self._children[name]

    # . Add and remove not-inplace.

    def set_child(self, name: str, child: Union[PfLine, Any]) -> PfLine:
        """Sets/adds/updates child; returns new pfline instance without changing current instance."""
        return NestedPfLine(
            {**self, name: self._prepare_child(name, child)}, _internal=True
        )

    def drop_child(self, name: str) -> PfLine:
        """Drop child; returns new pfline instance without changing current instance."""
        if name not in self._children:
            raise KeyError(f"Portfolio line does not have child with name '{name}'.")
        if len(self._children) == 1:
            raise RuntimeError("Cannot remove the last child of a portfolio line.")
        return NestedPfLine(
            {n: child for n, child in self.items() if n != name}, _internal=True
        )


class _LocIndexer:
    """Helper class to obtain NestedPfLine instance, whose index is subset of original index."""

    def __init__(self, pfl: NestedPfLine):
        self.pfl = pfl

    def __getitem__(self, arg) -> NestedPfLine:
        new_children = {name: child.loc[arg] for name, child in self.pfl.items()}
        return NestedPfLine(new_children, _internal=True)
