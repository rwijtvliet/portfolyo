import pandas as pd

# Not working: using string directly as start of date_range:
i_error = pd.date_range(
    "2020-10-25 02:00+0200", freq="H", periods=3, tz="Europe/Berlin"
)
# AssertionError: Inferred time zone not equal to passed time zone

# Working: first turning into timestamp...and then using that as start of date_range:
i_works = pd.date_range(
    pd.Timestamp("2020-10-25 02:00+0200", tz="Europe/Berlin"), freq="H", periods=3
)
# DatetimeIndex(['2020-10-25 02:00:00+02:00', '2020-10-25 02:00:00+01:00', '2020-10-25 03:00:00+01:00'],
#               dtype='datetime64[ns, Europe/Berlin]', freq='H')
