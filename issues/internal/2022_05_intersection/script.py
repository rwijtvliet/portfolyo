import pandas as pd
from portfolyo.core.pfline import multi_helper
from portfolyo.tools import stamps
from portfolyo import dev
from portfolyo import testing

freq = "D"
i1 = pd.date_range(
    start="2020-01-01",
    end="2020-06-01",
    freq=freq,
    inclusive="left",
    tz="Europe/Berlin",
)
start = "2020-03-01"
i2 = pd.date_range(
    start=start,
    end="2020-09-01",
    freq=freq,
    inclusive="left",
    tz="Europe/Berlin",
)

spfl1 = dev.get_singlepfline(i1, "all")
spfl2 = dev.get_singlepfline(i2, "all")
dic = {"PartA": spfl1, "PartB": spfl2}

intersection = stamps.intersection(spfl1.index, spfl2.index)


# %%
result = multi_helper.make_childrendict(dic)
for name, child in result.items():
    testing.assert_series_equal(child.q, dic[name].loc[intersection].q)
    testing.assert_series_equal(child.r, dic[name].loc[intersection].r)
    testing.assert_index_equal(child.index, intersection)

# %%
