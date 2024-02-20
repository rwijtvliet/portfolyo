import pandas as pd
import portfolyo as pf
from portfolyo.core.shared import concat

index = pd.date_range("2020", "2024", freq="QS", inclusive="left")
index2 = pd.date_range("2024", "2025", freq="QS", inclusive="left")
pfl = pf.dev.get_nestedpfline(index)
pfl2 = pf.dev.get_nestedpfline(index2)

concat.general(pfl, pfl2)

# print(index)
# print(index2)
# print(pfl)
