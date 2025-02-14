"""Help the type checker."""

from typing import TypeVar

import pandas as pd

Frequencylike = str | pd.offsets.BaseOffset
Series_or_DataFrame = TypeVar("Series_or_DataFrame", pd.Series, pd.DataFrame)
