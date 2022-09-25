"""
Module to create multi-pflines. These behave exactly as single-pflines, with all the
same methods. But which include several Pflines as children that can be accessed by
their name.
"""

from __future__ import annotations

from . import multi_helper
from .base import PfLine, Kind

from typing import Dict, Iterable, Mapping, Optional, Union, Any
import pandas as pd
import numpy as np
import warnings


class MultiPfLine(PfLine, Mapping):
    """Portfolio line with children.

    Children's names can be found by looping over the object and accessed by indexing.
    (name, child)-tuples are available from the .items() method.

    Parameters
    ----------
    data: Any
        Generally: object with a mapping from strings to PfLine instances; most commonly a
        dictionary.
    """

    def __new__(cls, data):
        # Catch case where data is already a valid class instance.
        if isinstance(data, MultiPfLine):
            return data
        # Otherwise, do normal thing.
        return super().__new__(cls, data)

    def __init__(self, data: Union[MultiPfLine, Mapping[str, PfLine], pd.DataFrame]):
        if self is data:
            return  # don't continue initialisation, it's already the correct object
        self._children = {}
        for name, child in multi_helper.make_mapping(data).items():
            self[name] = child

    # Implementation of ABC methods.

    @property
    def index(self) -> pd.DatetimeIndex:
        return next(iter(self._children.values())).index

    @property
    def w(self) -> pd.Series:
        if self.kind is Kind.PRICE_ONLY:
            return pd.Series(np.nan, self.index, name="w", dtype="pint[MW]")
        else:
            return pd.Series(self.q / self.index.duration, name="w").pint.to("MW")

    @property
    def q(self) -> pd.Series:
        # TODO: simply flatten and then return volume-part?
        if self.kind is Kind.PRICE_ONLY:
            return pd.Series(np.nan, self.index, name="q", dtype="pint[MWh]")
        elif (qp_children := self._qp_children) is not None:
            return qp_children[Kind.VOLUME_ONLY].q
        else:  # all children have a sensible timeseries for .q
            return sum(child.q for child in self._children.values()).rename("q")

    @property
    def p(self) -> pd.Series:
        # TODO: simply flatten and then return price-part?
        if self.kind is Kind.VOLUME_ONLY:
            return pd.Series(np.nan, self.index, name="p", dtype="pint[Eur/MWh]")
        elif (qp_children := self._qp_children) is not None:
            return qp_children[Kind.PRICE_ONLY].p
        elif self.kind is Kind.ALL:  # all children have .kind == 'all'
            return pd.Series(self.r / self.q, name="p").pint.to("Eur/MWh")
        else:  # self.kind == 'p', all children have a sensible timeseries for .p
            return sum(child.p for child in self._children.values()).rename("p")

    @property
    def r(self) -> pd.Series:
        if self.kind is not Kind.ALL:
            return pd.Series(np.nan, self.index, name="r", dtype="pint[Eur]")
        elif (qp_children := self._qp_children) is not None:
            q, p = qp_children[Kind.VOLUME_ONLY].q, qp_children[Kind.PRICE_ONLY].p
            return pd.Series(q * p, name="r").pint.to("Eur")
        else:  # all children have .kind == 'all'
            return sum(child.r for child in self._children.values()).rename("r")

    @property
    def kind(self) -> Kind:
        if self._heterogeneous_children:
            return Kind.ALL
        return next(iter(self._children.values())).kind

    def df(
        self,
        cols: Iterable[str] = None,
        flatten: bool = True,
        *arg,
        has_units: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        if flatten:
            # TODO: just do self.flatten().df()?
            cols = self.available if cols is None else cols
            series = {col: getattr(self, col) for col in cols}
            if not has_units:
                series = {key: s.pint.m for key, s in series.items()}
            return pd.DataFrame(series)

        # One big dataframe. First: collect constituent dataframes.
        dfs = [self.df(cols, True)]
        dfdicts = [{n: c.df(cols, False)} for n, c in self._children.items()]
        dfs.extend([pd.concat(dfdict, axis=1) for dfdict in dfdicts])
        # Then: make all have same number of levels.
        n_target = max([df.columns.nlevels for df in dfs])
        for df in dfs:
            n_current = df.columns.nlevels
            keys = [""] * (n_target - n_current)
            oldcol = df.columns if n_current > 1 else [[item] for item in df.columns]
            df.columns = pd.MultiIndex.from_tuples(((*item, *keys) for item in oldcol))
        # Finally: put all together in big new dataframe.
        return pd.concat(dfs, axis=1)

    def asfreq(self, freq: str = "MS") -> MultiPfLine:
        if self._heterogeneous_children:
            warnings.warn(
                "This portfolio has its price and volume information stored in distinct child porfolios."
                " The portfolio is flattened before changing its frequency."
            )
            return self.flatten().asfreq(freq)
        return MultiPfLine(
            {name: child.asfreq(freq) for name, child in self._children.items()}
        )

    def map_to_year(self, year: int, holiday_country: str) -> MultiPfLine:
        return MultiPfLine(
            {
                name: child.map_to_year(year, holiday_country)
                for name, child in self._children.items()
            }
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

    def __setitem__(self, name: str, pfl: Union[PfLine, Any]):
        if name in dir(self):  # cannot use hasattr(): runs the code in the properties
            raise ValueError(
                f"Cannot name child '{name}', this is a reserved attribute name."
            )
        if not isinstance(name, str):
            raise TypeError(
                f"Parameter ``name`` must be a string; got {name} ({type(name)})."
            )
        self._children = multi_helper.verify_and_trim_dict({**self, name: PfLine(pfl)})

    def __getitem__(self, name: str):
        if name not in self._children:
            raise KeyError(f"Portfolio line does not have child with name '{name}'.")
        return self._children[name]

    def __delitem__(self, name: str):
        if name not in self._children:
            raise KeyError(f"Portfolio line does not have child with name '{name}'.")
        if len(self._children) == 1:
            raise RuntimeError("Cannot remove the last child of a portfolio line.")
        del self._children[name]

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

    def set_child(self, name: str, pfl: Union[PfLine, Any]) -> PfLine:
        """Sets/adds/updates child; returns new pfline instance without changing current instance."""
        return MultiPfLine({**self, name: pfl})

    def drop_child(self, name: str) -> PfLine:
        """Drop child; returns new pfline instance without changing current instance."""
        if name not in self._children:
            raise KeyError(f"Portfolio line does not have child with name '{name}'.")
        if len(self._children) == 1:
            raise RuntimeError("Cannot remove the last child of a portfolio line.")
        return MultiPfLine({n: child for n, child in self.items() if n != name})

    # . Other.

    @property
    def _heterogeneous_children(self) -> bool:
        """Return True if children are not all of same kind."""
        return bool(self._qp_children)

    @property
    def _qp_children(self) -> Optional[Dict[Kind, PfLine]]:
        """Helper method that returns the child providing the volume and the one providing the price."""
        qp_children = {child.kind: child for child in self._children.values()}
        if Kind.VOLUME_ONLY in qp_children and Kind.PRICE_ONLY in qp_children:
            return qp_children
        else:
            return None


class _LocIndexer:
    """Helper class to obtain MultiPfLine instance, whose index is subset of original index."""

    def __init__(self, mpfl):
        self.mpfl = mpfl

    def __getitem__(self, arg) -> MultiPfLine:
        new_dict = {name: child.loc[arg] for name, child in self.mpfl.items()}
        return MultiPfLine(new_dict)
