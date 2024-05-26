"""Help the type checker."""

from typing import TypeVar

import pandas as pd

Series_or_DataFrame = TypeVar("Series_or_DataFrame", pd.Series, pd.DataFrame)
