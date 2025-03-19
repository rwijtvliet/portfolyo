"""Help the type checker."""

from typing import TypeVar, Literal

import pandas as pd

PintSeries = pd.Series  # aLias to make it more clear, what is being returned
Frequencylike = str | pd.DateOffset | pd.tseries.offsets.BaseOffset
Series_or_DataFrame = TypeVar("Series_or_DataFrame", pd.Series, pd.DataFrame)
Col = Literal["w", "q", "p", "r"]
COLS = Col.__args__
