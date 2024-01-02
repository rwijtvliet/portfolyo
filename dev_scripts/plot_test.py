"""Create several plots to see if plotting still works."""

import matplotlib.pyplot as plt
import portfolyo as pf


def plot_bars(children: int = 1):
    """Create bar chart with 4 subplots: (q) x (daily, monthly) x (children, no children)."""
    index = pf.dev.get_index(freq="D")
    pfl1 = pf.dev.get_pfline(index, nlevels=2, childcount=children)
    pfl2 = pfl1.asfreq("MS")
    pfl2.print()
    pfl2.plot("p", children=True)
    # pfl1.plot("p", children=False)
    plt.show()
    print("The number of children is:", children)


def plot_bars_to_ax():
    # Create a figure and an array of subplots (2 rows, 2 columns)
    index = pf.dev.get_index(freq="D")
    pfl1 = pf.dev.get_flatpfline(index)
    pfl2 = pfl1.asfreq("MS")

    fig, axs = plt.subplots(1, 2)
    pfl2.plot_to_ax(axs[0], "q", "bar", "")
    pfl2.plot_to_ax(axs[1], "p", "hline", "")
    plt.show()


# plot_bars(4)
plot_bars_to_ax()
