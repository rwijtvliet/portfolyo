"""
Module with mixins, to add other output-methods to PfLine and PfState classes.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Union

import pandas as pd

if TYPE_CHECKING:  # needed to avoid circular imports
    from ..pfline import PfLine
    from ..pfstate import PfState


def prepare_df(pfl_or_pfs: Union[PfLine, PfState]) -> pd.DataFrame:
    """Prepare dataframe so it can easily be saved to excel or copied to clipboard."""
    return pfl_or_pfs.dataframe().pint.dequantify().tz_localize(None)


class ExcelClipboardOutput:  # for both PfLine and PfState
    @functools.wraps(pd.DataFrame.to_clipboard)
    def to_clipboard(self: Union[PfLine, PfState], *args, **kwargs) -> None:
        prepare_df(self).to_clipboard(*args, **kwargs)

    @functools.wraps(pd.DataFrame.to_excel)
    def to_excel(self: Union[PfLine, PfState], *args, **kwargs) -> None:
        prepare_df(self).to_excel(*args, *kwargs)
