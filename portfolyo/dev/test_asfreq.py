import pandas as pd
import portfolyo as pf

index = pd.date_range("2024-04-01", freq="MS", periods=12)
input_dict = {
    "w": pd.Series(
        [200, 220, 300.0, 150, 175, 150, 200, 220, 300.0, 150, 175, 150], index
    )
}
pfl = pf.PfLine(input_dict)

pfl2 = pfl.asfreq("YS-APR")
print(pfl2)
