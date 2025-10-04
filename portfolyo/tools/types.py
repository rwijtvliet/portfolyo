"""Help the type checker."""

from typing import Any, Literal, TypeVar

import pandas as pd

Frequencylike = str | pd.DateOffset | pd.tseries.offsets.BaseOffset

# Aliases to make it more clear, what is being returned or expected.


# Scalars.

# 1) Convertability.
# a: Scalars that can be turned into a pint.Quantity
# int
# float
# pint.Quantity
# b: Scalars that cannot be turned into a pint.Quantity
OtherScalar = Any  # e.g. bool, Timestamp


# Series.

# 1) Index type.
NontimeSeries = pd.Series  # without DatetimeIndex
TimeSeries = pd.Series  # with DatetimeIndex

# 2) Convertability.
# a: Series that can be turned into a pintSeries
IntSeries = pd.Series  # with int dtype
FloatSeries = pd.Series  # with float dtype
PintSeries = pd.Series  # with pint dtype
SingleDimQuantitySeries = pd.Series  # object dtype, each element is Quantity with same dimension
# b: Uniform series that cannot be turned into a pintSeries
OtherUniformSeries = pd.Series  # with e.g. bool dtype
# c: Series with quantities that cannot be turned into a pintSeries
MultiDimQuantitySeries = pd.Series  # object dtype, each element is Quantity, not all have same dim

# 3) Combination of 1 and 2.
FloatTimeSeries = pd.Series  # Series with DatetimeIndex and float dtype
PintTimeSeries = pd.Series  # Series with DatetimeIndex and pint dtype
BoolTimeSeries = pd.Series  # Series with DatetimeIndex and bool dtype


# DataFrames.

# 1) Index type.
NontimeDataframe = pd.DataFrame  # without DatetimeIndex
TimeDataframe = pd.DataFrame  # with DatetimeIndex

# 2) Datatype.
PintDataframe = pd.DataFrame  # each column has pint dtype

# 3) Combination of 1 and 2.
PintTimeDataframe = pd.DataFrame  # with DatetimeIndex and each column has a pint dtype


# Series or Dataframe.

Series_or_Dataframe = TypeVar("Series_or_Dataframe", pd.Series, pd.DataFrame)
TimeSeries_or_TimeDataframe = TypeVar("TimeSeries_or_TimeDataframe", TimeSeries, TimeDataframe)


# Indicating data dimensions.

Col = Literal["w", "q", "p", "r"]
COLS = Col.__args__
