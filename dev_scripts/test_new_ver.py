import pandas as pd
from portfolyo import Kind, dev

i = pd.date_range(
    "2020-04-06", "2020-04-16", freq="MS", inclusive="left", tz="Europe/Berlin"
)
pfl = dev.get_flatpfline(i, Kind.COMPLETE)
