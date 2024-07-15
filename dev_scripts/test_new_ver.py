from matplotlib import pyplot as plt
import pandas as pd
from portfolyo import Kind, dev

index = pd.date_range("2020-01-01", "2021-01-01", freq="MS", tz=None)
pfl = dev.get_pfline(index, nlevels=2, childcount=1, kind=Kind.VOLUME)
pfl.plot(children=True)
plt.show()
