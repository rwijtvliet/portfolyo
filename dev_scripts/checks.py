import pandas as pd
from portfolyo.core.pfline.classes import PfLine


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


def get_pfl(
    idx: pd.DatetimeIndex = None,
    # kind: Kind = Kind.COMPLETE,
) -> PfLine:
    pfl = PfLine(pd.Series(range(0, len(idx)), idx, dtype="pint[MW]"))
    return pfl


idxs = [
    get_idx("2020", "00:00", None, "AS", "2022"),
    get_idx("2022", "00:00", None, "AS", "2024"),
]

print(len(idxs[0]))
pfl = get_pfl(idxs[0])
pfl2 = get_pfl(idxs[1])
index = get_idx("2020", "00:00", None, "AS", "2024")
pfl3 = get_pfl(index)
print(pfl)
print(pfl2)
print(pfl3)
# index = pd.date_range("2020", "2024", freq="QS", inclusive="left")
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
