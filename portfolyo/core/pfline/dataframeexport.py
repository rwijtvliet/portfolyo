from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import pandas as pd

from ... import tools

if TYPE_CHECKING:
    from .classes import FlatPfLine, NestedPfLine

from . import classes


class Flat:
    def dataframe(
        self: FlatPfLine, cols: Iterable[str] = None, *, has_units: bool = True
    ) -> pd.DataFrame:
        """DataFrame for portfolio line in default units.

        Parameters
        ----------
        cols : str, optional (default: all that are available)
            The columns (w, q, p, r) to include in the dataframe.
            Columns that are not available are silently excluded.
        has_units : bool, optional (default: True)
            - If True, return dataframe with ``pint`` units. (The unit can be extracted
                as a column level with ``.pint.dequantify()``).
            - If False, return dataframe with float values.

        Returns
        -------
        pd.DataFrame
        """
        cols = cols or "wqpr"  # in case nothing was specified.
        cols = [col for col in cols if col in "wqpr" and col in self.kind.available]
        df = pd.DataFrame({col: self.df[col] for col in cols})
        return df if has_units else df.pint.m


class Nested:
    def dataframe(
        self: NestedPfLine,
        cols: Iterable[str] = None,
        childlevels: int = -1,
        *,
        has_units: bool = True,
    ) -> pd.DataFrame:
        """DataFrame for portfolio line in default units.

        Parameters
        ----------
        cols : str, optional (default: all that are available)
            The columns (w, q, p, r) to include in the dataframe.
            Columns that are not available are silently excluded.
        childlevels : int, optional (default: -1)
            Number of child levels to include. 0 to only show current level, without
            children. -1 to show all.
        has_units : bool, optional (default: True)
            - If True, return dataframe with ``pint`` units. (The unit can be extracted
                as a column level with ``.pint.dequantify()``).
            - If False, return dataframe with float values.

        Returns
        -------
        pd.DataFrame
        """
        flatdf = self.flatten().dataframe(cols, has_units=has_units)

        if childlevels == 0:
            return flatdf

        # One big dataframe.
        dfs = [flatdf]
        for name, child in self.items():
            if isinstance(child, classes.FlatPfLine):
                childdf = child.dataframe(cols, has_units=has_units)
            else:
                childdf = child.dataframe(cols, childlevels - 1, has_units=has_units)
            dfs.append(tools.frame.add_header(childdf, name))
        return tools.frame.concat(dfs, 1)
