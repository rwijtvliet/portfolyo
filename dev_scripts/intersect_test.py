import pandas as pd

import portfolyo as pf

index1 = pd.date_range("2020-01-01", "2021-04-01", freq="QS", inclusive="left")
index2 = pd.date_range("2020-04-01", "2021-03-01", freq="YS-APR", inclusive="left")

print(index1)
print(index2)
print()

intersection = pf.tools.intersect.indices_flex(
    index1, index2, ignore_freq=True, ignore_tz=False, ignore_start_of_day=False
)
print(intersection)

inter_1 = pd.date_range("2020-04-01", "2021-02-01", freq="QS", inclusive="left")
print(inter_1)
