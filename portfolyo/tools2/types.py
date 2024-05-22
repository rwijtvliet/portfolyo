"""Help the type checker."""

from typing import TypeVar

import pandas as pd

from ..core.pfline import PfLine
from ..core.pfstate import PfState

NDFrameLike = TypeVar("NDFrameLike", pd.Series, pd.DataFrame, PfLine, PfState)
