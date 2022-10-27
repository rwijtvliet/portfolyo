import pandas as pd

i = pd.date_range("2020-03-28", freq="D", periods=3, tz="Europe/Berlin")
shifted_own = i.shift()

i.freq = None
shifted_param = i.shift(freq="D")


print(shifted_own)
# DatetimeIndex(['2020-03-29 00:00:00+01:00', '2020-03-30 00:00:00+02:00',
#                '2020-03-31 00:00:00+02:00'],
#               dtype='datetime64[ns, Europe/Berlin]', freq='D')

print(shifted_param)
# DatetimeIndex(['2020-03-29 00:00:00+01:00', '2020-03-30 01:00:00+02:00',
#                '2020-03-31 00:00:00+02:00'],
#               dtype='datetime64[ns, Europe/Berlin]', freq=None)

"""
From the docstring:

"
freq : pandas.DateOffset, pandas.Timedelta or string, optional
    Frequency increment to shift by.
    If None, the index is shifted by its own `freq` attribute.
"

I would expect the shifting to be the same in both cases, as the freq attribute in the first
and the provided parameter in the second are the same.
"""
