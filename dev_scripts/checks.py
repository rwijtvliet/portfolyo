import pandas as pd
import portfolyo as pf
from portfolyo.core.shared import concat


def get_idx(
    startdate: str, starttime: str, tz: str, freq: str, enddate: str
) -> pd.DatetimeIndex:
    # Empty index.
    if startdate is None:
        return pd.DatetimeIndex([], freq=freq, tz=tz)
    # Normal index.
    ts_start = pd.Timestamp(f"{startdate} {starttime}", tz=tz)
    ts_end = pd.Timestamp(f"{enddate} {starttime}", tz=tz)
    return pd.date_range(ts_start, ts_end, freq=freq, inclusive="left")


index = pd.date_range("2020", "2024", freq="QS", inclusive="left")
# index2 = pd.date_range("2023", "2025", freq="QS", inclusive="left")
# pfl = pf.dev.get_flatpfline(index)
# pfl2 = pf.dev.get_flatpfline(index2)
# print(pfl)
# print(pfl2)

# pfs = pf.dev.get_pfstate(index)

# pfs2 = pf.dev.get_pfstate(index2)
# pfl3 = concat.general(pfl, pfl2)
# print(pfl3)

# print(index)
# print(index2)

whole_pfl = pf.dev.get_nestedpfline(index)
pfl_a = whole_pfl.slice[:"2021"]

pfl_b = whole_pfl.slice["2021":"2022"]
pfl_c = whole_pfl.slice["2022":]
result = concat.concat_pflines(pfl_a, pfl_b, pfl_c)
result2 = concat.concat_pflines(pfl_b, pfl_c, pfl_a)
print(result)
print(result2)
