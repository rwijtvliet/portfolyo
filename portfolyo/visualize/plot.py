"""
Visualize portfolio lines, etc.
"""

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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


class ContinuousValuesNotSupported(Exception):
    pass


class CategoricalValuesNotSupported(Exception):
    pass


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
def plot_timeseries_as_bar(
    ax: plt.Axes,
    s: pd.Series,
    labelfmt: str = "",
    cat: bool = None,  # don't need
    width: float = 0.8,
    **kwargs,
) -> None:
    """Plot timeseries ``s`` to axis ``ax``, as bars. Ideally, only used for plots with
    categorical (i.e, non-time) x-axis."""
    if not is_categorical(s):
        raise ContinuousValuesNotSupported(
            "This plot is not compatible with continous values"
        )
    check_ax_s_compatible(ax, s)
    s = prepare_ax_and_s(ax, s)  # ensure unit compatibility (if possible)

    categories = Categories(s)
    ax.bar(categories.x(), categories.y(), width=width, **kwargs)
    ax.set_xticks(categories.x(MAX_XLABELS), categories.labels(MAX_XLABELS))
    set_data_labels(ax, categories.x(), categories.y(), labelfmt, True)
    ax.autoscale()


@append_to_doc(docstringliteral_plotparameters)
def plot_timeseries_as_area(
    ax: plt.Axes,
    s: pd.Series,
    labelfmt: str = "",
    cat: bool = None,
    **kwargs,
) -> None:
    """Plot timeseries ``s`` to axis ``ax``, as stepped area between 0 and value. Ideally,
    only used for plots with time (i.e., non-categorical) axis."""
    if is_categorical(s):
        raise CategoricalValuesNotSupported(
            "This plot is not compatible with categorical values"
        )
    check_ax_s_compatible(ax, s)
    s = prepare_ax_and_s(ax, s)  # ensure unit compatibility (if possible)

    splot = s.copy()  # modified with additional (repeated) datapoint
    splot[splot.index.right[-1]] = splot.values[-1]

    if "bottom" in kwargs:
        bottom = kwargs["bottom"]
        bottom.append(bottom[-1])
        del kwargs["bottom"]
    else:
        bottom = [0.0 for i in range(0, splot.size)]
    # make bottom into pintarray
    bottom = bottom * splot.values[0].units
    # bottom = bottom * splot.pint.units

    ax.fill_between(
        splot.index,
        bottom,
        [sum(x) for x in zip(bottom, splot.values)],
        step="post",
        **kwargs,
    )
    delta = s.index.right - s.index
    set_data_labels(ax, s.index + 0.5 * delta, s.values, labelfmt, True)
    ax.autoscale()


@append_to_doc(docstringliteral_plotparameters)
def plot_timeseries_as_step(
    ax: plt.Axes, s: pd.Series, labelfmt: str = "", cat: bool = None, **kwargs
) -> None:
    """Plot timeseries ``s`` to axis ``ax``, as stepped line (horizontal and vertical lines).
    Ideally, only used for plots with time (i.e., non-categorical) axis."""
    if is_categorical(s):
        raise CategoricalValuesNotSupported(
            "This plot is not compatible with categorical values"
        )
    check_ax_s_compatible(ax, s)
    s = prepare_ax_and_s(ax, s)  # ensure unit compatibility (if possible)

    splot = s.copy()  # modified with additional (repeated) datapoint
    splot[splot.index.right[-1]] = splot.values[-1]

    ax.step(splot.index, splot.values, where="post", **kwargs)
    delta = s.index.right - s.index
    set_data_labels(ax, s.index + 0.5 * delta, s.values, labelfmt, True)
    ax.autoscale()


@append_to_doc(docstringliteral_plotparameters)
def plot_timeseries_as_hline(
    ax: plt.Axes, s: pd.Series, labelfmt: str = "", cat: bool = None, **kwargs
) -> None:
    """Plot timeseries ``s`` to axis ``ax``, as horizontal lines. Ideally, only used for
    plots with time (i.e., non-categorical) axis."""
    if not is_categorical(s):
        raise ContinuousValuesNotSupported(
            "This plot is not compatible with continous values"
        )
    check_ax_s_compatible(ax, s)
    s = prepare_ax_and_s(ax, s)  # ensure unit compatibility (if possible)
    categories = Categories(s)
    # Center around x-tick:
    ax.hlines(categories.y(), categories.x() - 0.5, categories.x() + 0.5, **kwargs)
    ax.set_xticks(categories.x(MAX_XLABELS), categories.labels(MAX_XLABELS))
    set_data_labels(ax, categories.x(), categories.y(), labelfmt, True)
    ax.autoscale()
    # Adjust the margins around the plot
    ax.margins(x=0.2, y=0.2)


def plot_timeseries(
    ax: plt.Axes,
    s: pd.Series,
    how: str = "bar",
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
    how : str, optional (default: 'bar')
        How to plot the data; one of {'bar', 'area', 'step', 'hline'}.
    labelfmt : str, optional (default: '')
        Labels are added to each datapoint in the specified format. ('' to add no labels)
    cat : bool, optional (default: True if frequency is monthly or larger)
        Plot as categorical x-axis.
    """

    if how == "bar":
        plot_timeseries_as_bar(ax, s, labelfmt, cat, **kwargs)
    elif how == "area":
        plot_timeseries_as_area(ax, s, labelfmt, cat, **kwargs)
    elif how == "step":
        plot_timeseries_as_step(ax, s, labelfmt, cat, **kwargs)
    elif how == "hline":
        plot_timeseries_as_hline(ax, s, labelfmt, cat, **kwargs)
    else:
        raise ValueError(
            f"Parameter ``how`` must be one of 'bar', 'area', 'step', 'hline'; got {how}."
        )


def set_portfolyo_attr(ax, name, val):
    """
    Sets attribute ax._portfolyo which is a dictionary: ._portfolyo = {'unit': ..., 'freq': ..., ...}
    If dictionary doesn't exist yet, creates an empty dictionary
    """
    pattr = getattr(ax, "_portfolyo", {})
    pattr[name] = val
    setattr(ax, "_portfolyo", pattr)


def get_portfolyo_attr(ax, name, default_val=None):
    """
    Gets values from dictionary ax._portfolyo, if it doesn't exist returns None.
    """
    pattr = getattr(ax, "_portfolyo", {})
    return pattr.get(name, default_val)


def is_categorical(s: pd.Series) -> bool:
    """The function checks whether frequency of panda Series falls into continous or categorical group"""
    return s.index.freq in ["AS", "QS", "MS"]


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
    axunit = get_portfolyo_attr(ax, "unit", None)

    # Axes already has unit. Convert series to that unit; ignore any supplied custom unit.
    if axunit is not None:
        if axunit.dimensionality != s.pint.dimensionality:  # check compatibilty
            raise ValueError(
                f"Cannot plot series with units {s.pint.units} on axes with units {axunit}."
            )
        return s.astype(f"pint[{axunit}]")

    # Axes does not have unit.
    if unit is not None:
        # Custom unit is provided. Convert series to that unit.
        if unit.dimensionality != s.pint.dimensionality:  # check compatibility
            raise ValueError(
                f"Cannot convert series with units {s.pint.units} to units {unit}."
            )
        s = s.astype(unit)
        set_portfolyo_attr(ax, "unit", unit)
    else:
        # No custom unit provided. Convert series to base units.
        s = s.pint.to_base_units()
        set_portfolyo_attr(ax, "unit", s.pint.units)

    ax.set_ylabel(f"{get_portfolyo_attr(ax, 'unit'):~P}")
    return s


def check_ax_s_compatible(ax: plt.Axes, s: pd.Series):
    """Ensure axes ``ax`` has frequency associated with it and checks compatibility with series
    ``s``.
    If axes `ax`` has frequency, compare to pd.Series frequency. If they are equal, return s, if not equal, return error.
    If axes `ax`` doesn't have frequency, assign frequency of s to it.
    """
    # Find preexisting frequency.
    axfreq = get_portfolyo_attr(ax, "freq", None)
    series_freq = s.index.freq
    # Axes does not have frequency.
    if axfreq is None:
        set_portfolyo_attr(ax, "freq", series_freq)
    else:
        if get_portfolyo_attr(ax, "freq") != series_freq:
            raise AttributeError(
                "The frequency of PFLine is not compatible with current axes"
            )


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
