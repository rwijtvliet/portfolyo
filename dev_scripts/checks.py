import pandas as pd
import portfolyo as pf
from portfolyo.core.shared import concat

index = pd.date_range("2020", "2024", freq="QS", inclusive="left")
index2 = pd.date_range("2023", "2025", freq="QS", inclusive="left")
# pfl = pf.dev.get_nestedpfline(index)
# pfl2 = pf.dev.get_nestedpfline(index2)
# print(pfl)
# print(pfl2)

pfs = pf.dev.get_pfstate(index)
pfs2 = pf.dev.get_pfstate(index2)
pfl3 = concat.general(pfs, pfs2)
print(pfl3)

# print(index)
# print(index2)
