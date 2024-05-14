import pandas as pd
import portfolyo as pf

pfl = pf.PfLine(
    {
        "w": pd.Series(
            [1, 2], pd.date_range("2024", freq="AS", periods=2), dtype="pint[MW]"
        )
    }
)
# pfl2 = pf.PfLine(
#     {
#         "w": pd.Series(
#             [1, 2], pd.date_range("2024", freq="AS", periods=2), dtype="pint[MWh]"
#         )
#     }
# )
pfl3 = pf.PfLine(
    {
        "w": pd.Series(
            [1, 2], pd.date_range("2024", freq="AS", periods=2), dtype="pint[blablu]"
        )
    }
)
print(pfl)
