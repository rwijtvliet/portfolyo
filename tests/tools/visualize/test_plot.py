"""Test if portfolio line can be plotted."""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest
import numpy as np

import portfolyo as pf
from portfolyo import Kind, PfState


@pytest.mark.parametrize("levels", [1, 2, 3])
@pytest.mark.parametrize("childcount", [1, 2, 3])
@pytest.mark.parametrize("children", [True, False])
@pytest.mark.parametrize("kind", [Kind.VOLUME, Kind.PRICE, Kind.REVENUE, Kind.COMPLETE])
@pytest.mark.parametrize("freq", ["MS", "D"])
def test_pfline_plot(
    levels: int, childcount: int, children: bool, kind: Kind, freq: str
):
    """Test if data can be plotted with plot() function."""
    index = pd.date_range("2020-01-01", "2021-01-01", freq=freq, tz=None)
    pfl = pf.dev.get_pfline(index, nlevels=levels, childcount=childcount, kind=kind)
    pfl.plot(children=children)
    matplotlib.pyplot.close()


@pytest.mark.parametrize("childcount", [1, 2, 3])
@pytest.mark.parametrize("children", [True, False])
@pytest.mark.parametrize("freq", ["MS", "D"])
def test_pfstate_plot(
    childcount: int,
    children: bool,
    freq: str,
):
    """Test if pfstate can be plotted with plot() function."""
    index = pd.date_range(
        "2022-06-01", "2024-02-01", freq=freq, tz="Europe/Berlin", inclusive="left"
    )
    offtakevolume = pf.dev.get_nestedpfline(
        index, kind=Kind.VOLUME, childcount=childcount
    )
    sourced = pf.dev.get_nestedpfline(index, kind=Kind.COMPLETE, childcount=childcount)
    unsourcedprice = pf.dev.get_nestedpfline(
        index, kind=Kind.PRICE, childcount=childcount
    )
    pfs = PfState(-1 * offtakevolume, unsourcedprice, sourced)
    pfs.plot(children=children)
    matplotlib.pyplot.close()


@pytest.mark.parametrize("children", [True, False])
def test_flatpfline_plot(children: bool):
    """Test if plotting flatpfline with children attribute gives an error."""
    pfl = pf.dev.get_flatpfline()
    pfl.plot(children=children)
    matplotlib.pyplot.close()


@pytest.mark.parametrize("freq", ["MS", "D"])
@pytest.mark.parametrize("children", [True, False])
@pytest.mark.parametrize("levels", [1, 2, 3])
@pytest.mark.parametrize("childcount", [1, 2, 3])
def test_plot_to_ax(levels: int, childcount: int, children: bool, freq: str):
    """Test if frunction plot_to_ax works with every kind of pfline."""
    index = pd.date_range("2020-01-01", "2021-01-01", freq=freq, tz=None)
    pfl_compl = pf.dev.get_pfline(
        index, nlevels=levels, childcount=childcount, kind=Kind.COMPLETE
    )
    pfl_vol = pf.dev.get_pfline(
        index, nlevels=levels, childcount=childcount, kind=Kind.VOLUME
    )
    pfl_price = pf.dev.get_pfline(
        index, nlevels=levels, childcount=childcount, kind=Kind.PRICE
    )
    pfl_rev = pf.dev.get_pfline(
        index, nlevels=levels, childcount=childcount, kind=Kind.REVENUE
    )
    _, axs = plt.subplots(2, 2)
    with pytest.raises(ValueError):
        _ = pfl_compl.plot_to_ax(axs[0][0], children=children, kind=Kind.COMPLETE)
    pfl_vol.plot_to_ax(axs[0][1], children=children, kind=Kind.VOLUME)
    pfl_price.plot_to_ax(axs[1][0], children=children, kind=Kind.PRICE)
    pfl_rev.plot_to_ax(axs[1][1], children=children, kind=Kind.REVENUE)
    matplotlib.pyplot.close()


@pytest.mark.parametrize("freq", ["MS", "D"])
@pytest.mark.parametrize("children", [True, False])
@pytest.mark.parametrize(
    "kind, letter", [(Kind.VOLUME, "w"), (Kind.PRICE, "p"), (Kind.REVENUE, "r")]
)
def test_plot_with_nan(freq: str, children: bool, kind: Kind, letter: str):
    index = pd.date_range("2020-01-01", "2021-01-01", freq=freq, tz=None)
    s = pf.dev.get_series(index, letter)
    s.iloc[:] = np.nan
    pfl = pf.PfLine(s)
    if children:
        pfl2 = pf.dev.get_flatpfline(index, kind=kind)
        dict_of_children = {"southeast": pfl, "northwest": pfl2}
        pfl = pf.PfLine(dict_of_children)
    pfl.plot(children=children)
    matplotlib.pyplot.close()


@pytest.mark.parametrize("freq", ["MS", "D"])
@pytest.mark.parametrize("children", [True, False])
@pytest.mark.parametrize(
    "kind, letter", [(Kind.VOLUME, "w"), (Kind.PRICE, "p"), (Kind.REVENUE, "r")]
)
def test_plot_not_all_nan(freq: str, children: bool, kind: Kind, letter: str):
    index = pd.date_range("2020-01-01", "2021-01-01", freq=freq, tz=None)
    s = pf.dev.get_series(index, letter)

    # Set the first 5 values to NaN
    s.iloc[:5] = np.nan

    pfl = pf.PfLine(s)
    if children:
        pfl2 = pf.dev.get_flatpfline(index, kind=kind)
        dict_of_children = {"southeast": pfl, "northwest": pfl2}
        pfl = pf.PfLine(dict_of_children)

    pfl.plot(children=children)
    matplotlib.pyplot.close()
