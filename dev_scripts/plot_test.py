"""Create several plots to see if plotting still works."""

import pandas as pd
import matplotlib.pyplot as plt
import portfolyo as pf


def plots(children: int = 1):
    """Create 16 plots: (w, q, p, r) x (hourly, monthly) x (children, no children)."""
    index = pd.date_range("2024", freq="AS", periods=3)
    pfl1 = pf.PfLine(pd.Series([0.2, 0.22, 0.3], index, dtype="pint[GW]"))
    dict_of_children = {}
    for i in range(children):
        key = str(i)
        dict_of_children[key] = pfl1

    pfl = pf.PfLine(dict_of_children)

    pfl.print()
    pfl.plot(children=True)
    plt.show()

    # pf_nested_pr.print()
    # pf_nested_pr.plot(children=True)
    # plt.show()
    print("The number of children is:", children)


plots(int(input()))
