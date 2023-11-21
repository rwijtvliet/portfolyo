"""
Visualize portfolio lines, etc.
"""

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .. import tools
from .categories import Categories, Category  # noqa

mpl.style.use("seaborn-v0_8")

# Plotting philosophy for timeseries:
# Not all plot types (bar, line, step, etc.) are suitable for all quantities.
#
# Data dimensionality vs plot type
# --------------------------------
#
# For data that represents an *aggregated* value for each period, like [MWh], [Eur], it
# makes sense to plot the x-axis as *categories*, i.e., without taking the duration of
# each period into consideration. Jan and Feb have distinct durations, but are shown
# identically. This data is best plotted as bars, but markers (e.g. short horizontal lines)
# are also possible.
#
# For data that can be integrated over time (like [MW]), it makes sense to plot the
# x-axis as a *timeline*, with each datapoint having its correct duration. E.g., for monthly
# values: Jan taking up more x-space than Feb. This data can be plotted as a hline, step,
# or area graph. The area graph underlines that the area under the curve has a meaning;
# in this case, MWh.
#
# For data that is neither (like [Eur/MWh]), both categorical and timeline plots can be
# used, though the area graph does not make much sense.
#
#
# Frequency vs plot type
# ----------------------
#
# A categorical x-axis loses its clarity if there are many datapoints. We are most
# commonly dealing with timeseries spanning at least a quarter. If this data is plotted
# with a frequency of daily or shorter, the number of datapoints is very high, and we should
# try to use timeline x-axis. This means that we cannot (easily) plot [MWh] and [Eur] values,
# and by default we should only plot [MW] and [Eur/MWh] values.
# If the data is plotted with a frequency of monthly or longer, we usually want to see
# aggregated values and must therefore plot on categorical x-axes. This means we cannot
# (easily) plot [MW] values, and by default we should only plot [MWh] and [Eur/MWh] values.
#
#
# This module is able to plot all graph types on both timeline x-axis and on categorical
# x-axis, and add the data labels if there are not too many. It is up to the caller to
# pick the correct graph type.
#

MAX_XLABELS = 20


def use_categories(ax: plt.Axes, s: pd.Series, cat: bool = None) -> bool:
    """Determine if plot should be made with category axis (True) or datetime axis (False)."""
    # We use categorical data if...
    if (ax.lines or ax.collections or ax.containers) and ax.xaxis.have_units():
        return True  # ...ax already has category axis; or
    elif cat is None and tools.freq.shortest(s.index.freq, "MS") == "MS":
        return True  # ...it's the default for the given frequency; or
    elif cat is True:
        return True  # ...user wants it.
    return False


docstringliteral_plotparameters = """
Other parameters
----------------
labelfmt : str, optional (default: '')
    Labels are added to each datapoint in the specified format. ('' to add no labels)
cat : bool, optional
    If False, plots x-axis as timeline with timestamps spaced according to their
    duration. If True, plots x-axis categorically, with timestamps spaced equally.
    Disregarded if ``ax`` already has values (then: use whatever is already set).
    Default: use True if ``s`` has a monthly frequency or longer, False if the frequency
    is shorter than monthly.
**kwargs : any formatting are passed to the Axes plot method being used."""


def append_to_doc(text):
    def decorator(fn):
        fn.__doc__ += f"\n{text}"
        return fn

    return decorator


@append_to_doc(docstringliteral_plotparameters)
def plot_timeseries_as_jagged(
    ax: plt.Axes, s: pd.Series, labelfmt: str = "", cat: bool = None, **kwargs
) -> None:
    """Plot timeseries ``s`` to axis ``ax``, as jagged line and/or as markers. Use kwargs
    ``linestyle`` and ``marker`` to specify line style and/or marker style. (Default: line only).
    """
    s = prepare_ax_and_s(ax, s)  # ensure unit compatibility (if possible)

    if use_categories(ax, s, cat):
        categories = Categories(s)
        ax.plot(categories.x(), categories.y(), **kwargs)
        ax.set_xticks(categories.x(MAX_XLABELS), categories.labels(MAX_XLABELS))
        set_data_labels(ax, categories.x(), categories.y(), labelfmt, False)

    else:
        ax.plot(s.index, s.values, **kwargs)
        set_data_labels(ax, s.index, s.values, labelfmt, False)


@append_to_doc(docstringliteral_plotparameters)
def plot_timeseries_as_bar(
    ax: plt.Axes, s: pd.Series, labelfmt: str = "", cat: bool = None, **kwargs
) -> None:
    """Plot timeseries ``s`` to axis ``ax``, as bars. Ideally, only used for plots with
    categorical (i.e, non-time) x-axis."""
    s = prepare_ax_and_s(ax, s)  # ensure unit compatibility (if possible)

    if use_categories(ax, s, cat):
        categories = Categories(s)
        ax.bar(categories.x(), categories.y(), 0.8, **kwargs)
        ax.set_xticks(categories.x(MAX_XLABELS), categories.labels(MAX_XLABELS))
        set_data_labels(ax, categories.x(), categories.y(), labelfmt, True)

    else:
        # Bad combination: bar graph on time-axis. But allow anyway.

        # This is slow if there are many elements.
        # x = s.index + 0.5 * (s.index.right - s.index)
        # width = pd.Timedelta(hours=s.index.duration.median().to("h").magnitude * 0.8)
        # ax.bar(x.values, s.values, width, **kwargs)

        # This is faster.
        delta = s.index.right - s.index
        x = np.array(list(zip(s.index + 0.1 * delta, s.index + 0.9 * delta))).flatten()
        magnitudes = np.array([[v, 0] for v in s.values.quantity.magnitude]).flatten()
        values = tools.unit.PA_(magnitudes, s.values.quantity.units)
        ax.fill_between(x, 0, values, step="post", **kwargs)

        set_data_labels(ax, s.index + 0.5 * delta, s.values, labelfmt, True)


@append_to_doc(docstringliteral_plotparameters)
def plot_timeseries_as_area(
    ax: plt.Axes, s: pd.Series, labelfmt: str = "", cat: bool = None, **kwargs
) -> None:
    """Plot timeseries ``s`` to axis ``ax``, as stepped area between 0 and value. Ideally,
    only used for plots with time (i.e., non-categorical) axis."""
    s = prepare_ax_and_s(ax, s)  # ensure unit compatibility (if possible)

    splot = s.copy()  # modified with additional (repeated) datapoint
    splot[splot.index.right[-1]] = splot.values[-1]

    if use_categories(ax, s, cat):
        # Bad combination: area graph on categorical axis. But allow anyway.

        categories = Categories(s)
        ctgr_extra = Categories(splot)
        # Center around x-tick:
        ax.fill_between(ctgr_extra.x() - 0.5, 0, ctgr_extra.y(), step="post", **kwargs)
        ax.set_xticks(categories.x(MAX_XLABELS), categories.labels(MAX_XLABELS))
        set_data_labels(ax, categories.x(), categories.y(), labelfmt, True)

    else:
        ax.fill_between(splot.index, 0, splot.values, step="post", **kwargs)
        delta = s.index.right - s.index
        set_data_labels(ax, s.index + 0.5 * delta, s.values, labelfmt, True)


@append_to_doc(docstringliteral_plotparameters)
def plot_timeseries_as_step(
    ax: plt.Axes, s: pd.Series, labelfmt: str = "", cat: bool = None, **kwargs
) -> None:
    """Plot timeseries ``s`` to axis ``ax``, as stepped line (horizontal and vertical lines).
    Ideally, only used for plots with time (i.e., non-categorical) axis."""
    s = prepare_ax_and_s(ax, s)  # ensure unit compatibility (if possible)

    splot = s.copy()  # modified with additional (repeated) datapoint
    splot[splot.index.right[-1]] = splot.values[-1]

    if use_categories(ax, s, cat):
        # Bad combination: step graph on categorical axis. But allow anyway.

        categories = Categories(s)
        ctgr_extra = Categories(splot)
        # Center around x-tick:
        ax.step(ctgr_extra.x() - 0.5, ctgr_extra.y(), where="post", **kwargs)
        ax.set_xticks(categories.x(MAX_XLABELS), categories.labels(MAX_XLABELS))
        set_data_labels(ax, categories.x(), categories.y(), labelfmt, True)

    else:
        ax.step(splot.index, splot.values, where="post", **kwargs)
        delta = s.index.right - s.index
        set_data_labels(ax, s.index + 0.5 * delta, s.values, labelfmt, True)


@append_to_doc(docstringliteral_plotparameters)
def plot_timeseries_as_hline(
    ax: plt.Axes, s: pd.Series, labelfmt: str = "", cat: bool = None, **kwargs
) -> None:
    """Plot timeseries ``s`` to axis ``ax``, as horizontal lines. Ideally, only used for
    plots with time (i.e., non-categorical) axis."""
    s = prepare_ax_and_s(ax, s)  # ensure unit compatibility (if possible)

    if use_categories(ax, s, cat):
        # Bad combination: hline graph on categorical axis. But allow anyway.

        categories = Categories(s)
        # Center around x-tick:
        ax.hlines(categories.y(), categories.x() - 0.5, categories.x() + 0.5, **kwargs)
        ax.set_xticks(categories.x(MAX_XLABELS), categories.labels(MAX_XLABELS))
        set_data_labels(ax, categories.x(), categories.y(), labelfmt, True)

    else:
        delta = s.index.right - s.index
        ax.hlines(s.values, s.index, s.index.right, **kwargs)
        set_data_labels(ax, s.index + 0.5 * delta, s.values, labelfmt, False)


def plot_timeseries(
    ax: plt.Axes,
    s: pd.Series,
    how: str = "jagged",
    labelfmt: str = None,
    cat: bool = None,
    **kwargs,
) -> None:
    """Plot timeseries to given axis.

    Parameters
    ----------
    ax : plt.Axes
        Axes to plot to.
    s : pd.Series
        Timeseries to plot
    how : str, optional (default: 'jagged')
        How to plot the data; one of {'jagged', 'bar', 'area', 'step', 'hline'}.
    labelfmt : str, optional (default: '')
        Labels are added to each datapoint in the specified format. ('' to add no labels)
    cat : bool, optional (default: True if frequency is monthly or larger)
        Plot as categorical x-axis.
    """
    if how == "jagged":
        plot_timeseries_as_jagged(ax, s, labelfmt, cat, **kwargs)
    elif how == "bar":
        plot_timeseries_as_bar(ax, s, labelfmt, cat, **kwargs)
    elif how == "area":
        plot_timeseries_as_area(ax, s, labelfmt, cat, **kwargs)
    elif how == "step":
        plot_timeseries_as_step(ax, s, labelfmt, cat, **kwargs)
    elif how == "hline":
        plot_timeseries_as_hline(ax, s, labelfmt, cat, **kwargs)
    else:
        raise ValueError(
            f"Parameter ``how`` must be one of 'jagged', 'bar', 'area', 'step', 'hline'; got {how}."
        )


def prepare_ax_and_s(ax: plt.Axes, s: pd.Series, unit=None) -> pd.Series:
    """Ensure axes ``ax`` has unit associated with it and checks compatibility with series
    ``s``. Also sets y label as unit abbreviation.

    Notes
    -----
    - If axes already has unit, convert series to same unit. If not, convert series to pint
    base units and set that as axis unit. Assumes that series with float or int values is
    dimensionless.
    - Changes ``ax`` in-place.
    - In order to plot in not-base-units, provide a custom value for the ``unit`` parameter
    when calling this function for the first time for a given ``ax``. Ignored on subsequent
    calls and a unit is already associated with the axes.
    """
    # Make sure series has pint unit.
    if s.dtype == float or s.dtype == int:
        s = s.astype("pint[1]")
    # Find preexisting unit.
    axunit = getattr(ax, "_unit", None)

    # Axes already has unit. Convert series to that unit; ignore any supplied custom unit.
    if axunit is not None:
        if axunit.dimensionality != s.pint.dimensionality:  # check compatibilty
            raise ValueError(
                f"Cannot plot series with units {s.pint.units} on axes with units {axunit}."
            )
        return s.astype(axunit)

    # Axes does not have unit.
    if unit is not None:
        # Custom unit is provided. Convert series to that unit.
        if unit.dimensionality != s.pint.dimensionality:  # check compatibility
            raise ValueError(
                f"Cannot convert series with units {s.pint.units} to units {unit}."
            )
        s = s.astype(unit)
        setattr(ax, "_unit", unit)
    else:
        # No custom unit provided. Convert series to base units.
        s = s.pint.to_base_units()
        setattr(ax, "_unit", s.pint.units)

    ax.set_ylabel(f"{ax._unit:~P}")
    return s


def set_data_labels(
    ax: plt.Axes, xx, yy, labelfmt, outside: bool = False, maxcount: int = 24
):
    """Add labels to axis ``ax``, at locations (``xx``, ``yy``), formatted with
    ``labelfmt``. Don't add labels if more than ``maxcount`` datapoints. If ``outside``,
    put labels of negative values *below* the datapoint."""
    # Don't label if no formatting speficied.
    if not labelfmt:
        return
    # Don't label if too many numbers.
    if len(xx) > maxcount:
        return

    # Add labels.
    for x, y in zip(xx, yy):
        lbl = labelfmt.format(y.magnitude).replace(",", " ")
        xytext = (0, -10) if outside and y.magnitude < 0 else (0, 10)
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=xytext, ha="center")

    # Increase axis range to give label space to stay inside box.
    ylim = list(ax.get_ylim())
    if not np.isclose(ylim[0], 0) and ylim[0] < 0:
        ylim[0] *= 1.1
    if not np.isclose(ylim[1], 0) and ylim[1] > 0:
        ylim[1] *= 1.1
    ax.set_ylim(*ylim)
