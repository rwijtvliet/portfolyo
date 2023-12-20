"""Create several plots to see if plotting still works."""

import matplotlib.pyplot as plt
import portfolyo as pf


def plot_bars(children: int = 1):
    """Create bar chart with 4 subplots: (q) x (daily, monthly) x (children, no children)."""
    index = pf.dev.get_index(freq="D")
    pfl1 = pf.dev.get_pfline(index, nlevels=2, childcount=children)
    # pfl2 = pfl1.asfreq("MS")
    pfl3 = pf.dev.get_nestedpfline(index, nlevels=2, childcount=children)
    pfl4 = pfl3.asfreq("MS")
    # pfl1.print()
    pfl4.print()
    end = pfl1.index[5]
    pfl1 = pfl1.loc[:end]

    pfl1.plot("r", children=True)
    # pfl2.plot("r", children=False)
    plt.show()

    # pf_nested_pr.print()
    # pf_nested_pr.plot(children=True)
    # plt.show()
    print("The number of children is:", children)


plot_bars(4)
