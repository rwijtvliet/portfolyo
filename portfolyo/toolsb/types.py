"""Help the type checker."""

from typing import TypeVar

import pandas as pd

PintSeries = pd.Series  # aLias to make it more clear, what is being returned
Frequencylike = str | pd.DateOffset
Series_or_DataFrame = TypeVar("Series_or_DataFrame", pd.Series, pd.DataFrame)
