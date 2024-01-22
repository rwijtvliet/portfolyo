import pandas as pd

from portfolyo import tools

# create a and b with  different frequencies
# create c with the same freq as a
# look how a intersect c behaves


a = pd.date_range("2020-01-01", "2022-02-02", freq="15T")
b = pd.date_range("2020-01-20", "2022-02-02", freq="D")
c = pd.date_range("2021-01-01", "2024-01-01", freq="QS")
d = pd.date_range("2022-04-01", "2024-07-01", freq="AS")
intersect = tools.intersect.indices(a, b, ignore_freq=True)
# intersect2 = tools.intersect.indices(a, c)
# intersect3 = tools.intersect.indices(b, d)
print(intersect)
# print("\n")
# print(intersect2)
# print(intersect == intersect3)
