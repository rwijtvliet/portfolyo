"""
Module with mixins, to add other output-methods to PfLine and PfState classes.
"""

from __future__ import annotations

import functools

import pandas as pd

from .ndframelike import NDFrameLike


def prepare_df(ndframelike: NDFrameLike) -> pd.DataFrame:
    """Prepare dataframe so it can easily be saved to excel or copied to clipboard."""
    return ndframelike.dataframe().pint.dequantify().tz_localize(None)


class ExcelClipboardOutput:  # for both PfLine and PfState
    @functools.wraps(pd.DataFrame.to_clipboard)
    def to_clipboard(self: NDFrameLike, *args, **kwargs) -> None:
        prepare_df(self).to_clipboard(*args, **kwargs)

    @functools.wraps(pd.DataFrame.to_excel)
    def to_excel(self: NDFrameLike, *args, **kwargs) -> None:
        prepare_df(self).to_excel(*args, *kwargs)
