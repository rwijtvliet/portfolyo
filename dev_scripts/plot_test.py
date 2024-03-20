"""Create several plots to see if plotting still works."""

import matplotlib.pyplot as plt
import portfolyo as pf
from portfolyo.visualize.plot import (
    CategoricalValuesNotSupported,
    ContinuousValuesNotSupported,
)


def plot_pfline_to_ax(value: str, how: str, children: int = 1):
    """Plot a timeseries of the PfLine to 4 axes: (daily, monthly) x (children, no children).
    Proper values to parameters:
        "p", "step"
        "p", "hline"
        "w", "area"
        "q", "bar"
    Othwerwise, getting only 1 plot for pfline with no children since the function for plotting children use default_kwargs

        Parameters
        ----------

        value : str
            The column to plot. One of {'w', 'q', 'p', 'r'}.
        how : str
            How to plot the data. One of {'bar', 'area', 'step', 'hline'}.
        children: int
            The number of children to assign to pfline.
    """
    # Create a figure and an array of subplots (2 rows, 2 columns)
    index = pf.dev.get_index(freq="D")
    pfl1 = pf.dev.get_pfline(index, nlevels=2, childcount=children)
    pfl2 = pfl1.asfreq("MS")
    pfl2.print()
    fig, axs = plt.subplots(2, 2)
    try:
        pfl2.plot_to_ax(axs[0][0], value, how, children=True)
    except CategoricalValuesNotSupported:
        axs[0, 0].clear()
    try:
        pfl2.plot_to_ax(axs[0][1], value, how, children=False)
    except CategoricalValuesNotSupported:
        axs[0, 1].clear()

    try:
        pfl1.plot_to_ax(axs[1][0], value, how, children=True)
    except ContinuousValuesNotSupported:
        axs[1, 0].clear()
    try:
        pfl1.plot_to_ax(axs[1][1], value, how, children=False)
    except ContinuousValuesNotSupported:
        axs[1, 1].clear()

    # Set x-axis and y-axis labels dynamically
    x_labels = ["with children", "without children"]
    y_labels = ["more than a month", "less than a month"]

    # Set y-axis labels on the left for each row
    for i in range(2):
        axs[i, 0].set(ylabel=y_labels[i])
    # Set x-axis labels on top of each column
    for j in range(2):
        axs[1, j].set(xlabel=x_labels[j])
    plt.show()


plot_pfline_to_ax("q", "bar", 4)
