"""Help the type checker."""

from typing import TypeVar

import pandas as pd

from ..core.pfline import PfLine
from ..core.pfstate import PfState

Indexable = TypeVar(
    "Series_DataFrame_PfLine_PfState", pd.Series, pd.DataFrame, PfLine, PfState
)
