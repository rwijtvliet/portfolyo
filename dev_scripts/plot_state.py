import portfolyo as pf
import pandas as pd
from portfolyo.core.pfline.enums import Kind
from portfolyo.core.pfstate.pfstate import PfState
import matplotlib.pyplot as plt

index = pd.date_range(
    "2022-06-01", "2024-02-01", freq="MS", tz="Europe/Berlin", inclusive="left"
)
offtakevolume = pf.dev.get_nestedpfline(index, kind=Kind.VOLUME, childcount=2)
sourced = pf.dev.get_nestedpfline(index, kind=Kind.COMPLETE, childcount=2)
unsourcedprice = pf.dev.get_nestedpfline(index, kind=Kind.PRICE, childcount=2)
pfs = PfState(-1 * offtakevolume, unsourcedprice, sourced)
pfs.plot(children=True)
plt.show()
