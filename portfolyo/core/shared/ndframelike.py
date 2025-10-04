"""Base class for PfLine and PfState classes."""

from __future__ import annotations

import abc

import pandas as pd


class NDFrameLike(abc.ABC):
    """Class that specifies which attributes from pandas Series and DataFrames must be
    implemented by descendents of this class (=PfLine and PfState)."""

    # Abstract methods to be implemented by descendents.

    @property
    @abc.abstractmethod
    def index(self) -> pd.DatetimeIndex:
        """Left timestamp of time period corresponding to each data row."""
        ...

    @abc.abstractmethod
    def asfreq(self, freq: str = "MS") -> NDFrameLike:
        """Resample the instance to a new frequency."""
        ...

    @property
    @abc.abstractmethod
    def loc(self):
        """Create a new instance with a subset of the rows (selection by row label(s) or
        a boolean array.)"""
        ...

    @property
    @abc.abstractmethod
    def iloc(self):
        """Create a new instance with a subset of the rows (selection by row index position)."""
        ...

    @property
    @abc.abstractmethod
    def slice(self):
        """Create a new instance with a subset of the rows.
        Different from loc since performs slicing with right-open interval."""
        ...
