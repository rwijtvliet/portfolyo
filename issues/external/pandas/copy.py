import pandas as pd

i1 = pd.date_range("2022", freq="D", periods=10)
i2 = i1.copy()  # COPY!!!

print(i1 is i2)
print(id(i1), id(i2))
print(i1.freq)
print(i2.freq)

i1.freq = None

print(i1 is i2)
print(id(i1), id(i2))
print(i1.freq)
print(i2.freq)
