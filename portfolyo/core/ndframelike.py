"""Base class for PfLine and PfState classes."""

from __future__ import annotations

import abc
from typing import Iterable

import pandas as pd


class NDFrameLike(abc.ABC):
    """Class that specifies which attributes from pandas Series and DataFrames must be
    implemented by descendents of this class (=PfLine and PfState)."""

    # Abstract methods to be implemented by descendents.

    @abc.abstractproperty
    def index(self) -> pd.DatetimeIndex:
        """Left timestamp of time period corresponding to each data row."""
        ...

    @abc.abstractmethod
    def asfreq(self, freq: str = "MS") -> NDFrameLike:
        """Resample the instance to a new frequency.

        Parameters
        ----------
        freq : str, optional
            The frequency at which to resample. 'AS' for year, 'QS' for quarter, 'MS'
            (default) for month, 'D for day', 'H' for hour, '15T' for quarterhour.

        Returns
        -------
        Instance
            Resampled at wanted frequency.
        """
        ...

    @abc.abstractproperty
    def loc(self):
        """Create a new instance with a subset of the rows (selection by row label(s) or
        a boolean array.)"""
        ...

    @abc.abstractmethod
    def dataframe(
        self, cols: Iterable[str] = None, has_units: bool = True, *args, **kwargs
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
        ...
