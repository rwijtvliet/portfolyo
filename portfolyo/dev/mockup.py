"""Create somewhat realistic curves."""

import numpy as np
import pandas as pd
import pint

from ..core.commodity import Commodity
from ..tools import unit  # noqa # ensure we use current ureg
from ..tools.product import germanpower_peakfn
from ..tools.types import FloatTimeSeries, PintTimeSeries


def w_offtake(
    commodity: Commodity | None,
    idx: pd.DatetimeIndex,
    avg: float = 100.0,
    year_amp: float = 0.20,
    week_amp: float = 0.10,
    day_amp: float = 0.30,
    rand_amp: float = 0.02,
) -> PintTimeSeries | FloatTimeSeries:
    """Create a more or less realistic-looking offtake timeseries.

    Parameters
    ----------
    commodity
        Used to get the correct unit for ``w``. If None, return timeseries of floats.
    idx
        Timestamps for which to create offtake.
    avg, optional (default: 100.0)
        Average offtake (without unit).
    year_amp, optional (default: 0.2)
        Yearly amplitude as fraction (of average). If positive: winter offtake > summer offtake.
    week_amp, optional (default: 0.1)
        Weekly amplitude as fraction (of average). If positive: midweek offtake > weekend offtake.
    day_amp, optional (default: 0.3)
        Day amplitude as fraction (of average). If positive: midday offtake > night
        offtake.
    rand_amp, optional (default: 0.02)
        Random amplitude as fraction of average (without unit).

    Returns
    -------
        Offtake timeseries.
    """
    if year_amp + day_amp + week_amp + rand_amp > 1:
        raise ValueError(
            f"Sum of fractional amplitudes ({year_amp:.1%} and {day_amp:.1%} and"
            f" {week_amp:.1%} and {rand_amp:.1%}) should not exceed 100%."
        )
    # year angle: 1jan0:00..1jan0:00 -> 0..2pi
    ya = idx.map(lambda ts: ts.dayofyear) / 365 * np.pi * 2
    # week angle: Sun0:00..Sun0:00 -> 0..2pi
    wa = (idx.map(lambda ts: (ts.weekday() + 1) * 24 + ts.hour) / 168) * np.pi * 2
    # day angle: 0:00..0:00 -> 0..2pi
    da = idx.map(lambda ts: (ts.hour * 60 + ts.minute)) / 1440 * np.pi * 2
    # Values: max mid-Jan, mid-week, at 15:00
    yv = year_amp * np.cos(ya - 0.28)
    wv = week_amp * (0.5 - 0.8 * np.cos(wa) - np.cos(2 * wa) / 2 - np.cos(3 * wa) / 6)
    dv = day_amp * (-0.7 * np.cos(da - 0.79) - 0.3 * np.sin(da * 2))
    rv = rand_amp * (1 + 2 * np.random.rand(len(idx)))  # TODO: Random mean-reverting walk
    s = pd.Series(avg * (1 + yv + dv + wv + rv), idx, name="w")
    return s if commodity is None else s.astype(f"pint[{commodity.col_to_units['w']}]")


def p_marketprices(
    commodity: Commodity | None,
    idx: pd.DatetimeIndex,
    avg: float = 100.0,
    year_amp: float = 0.30,
    week_amp: float = 0.05,
    peak_amp: float = 0.30,
) -> PintTimeSeries | FloatTimeSeries:
    """Create a more or less realistic-looking forward price curve timeseries.

    Parameters
    ----------
    commodity
        Used to get the correct unit for ``p``. If None, return timeseries of floats.
    idx
        Timestamps for which to create prices.
    avg, optional (default: 100.0)
        Average price (without unit).
    year_amp, optional (default: 0.3)
        Yearly amplitude as fraction of average. If positive: winter prices > summer prices.
    week_amp, optional (default: 0.05)
        Weekly amplitude as fraction of average. If positive: midweek prices > weekend prices.
    peak_amp, optional (default: 0.3)
        Peak-offpeak amplitude as fraction of average. If positive: peak prices > offpeak prices.

    Returns
    -------
        Price timeseries.
    """
    if year_amp + week_amp + peak_amp > 1:
        raise ValueError(
            f"Sum of fractional amplitudes ({year_amp:.1%} and {week_amp:.1%} and"
            f" {peak_amp:.1%}) should not exceed 100%."
        )
    # Year angle: 1jan0:00..1jan0:00 -> 0..2pi. But: uniform within month.
    year_angle = idx.map(lambda ts: ts.month) / 12 * np.pi * 2
    # Week angle: Sun0:00..Sun0:00 -> 0..2pi. But: uniform within day.
    week_angle = idx.map(lambda ts: ts.weekday() + 1) / 7 * np.pi * 2
    # Peak fraction: -1 (middle of offpeak hours) .. 1 (middle of peak hours)
    try:
        # Fill with -1 (offpeak) or 1 (peak). Fails if commodity = None, or has no peak_fn, or freq too long.
        osc = -1 + 2 * commodity.peak_fn(idx)
        # Smooth out by convolution.
        kernel = np.array([0.5, 0.8, 1, 0.8, 0.5])
        kernel = kernel[: len(idx)]  # slice in case idx very short
        peak_fraction = np.convolve(osc, kernel / sum(kernel), mode="same")
    except (AttributeError, ValueError):
        peak_fraction = np.zeros(len(idx))
    # Values.
    yv = year_amp * np.cos(year_angle - 0.35)  # max in feb
    wv = week_amp * np.cos(week_angle - 1.07)  # max on tuesday
    pv = peak_amp * peak_fraction  # max in middle of every peakperiod
    s = pd.Series(avg * (1 + yv + wv + pv), idx, name="p")
    return s if commodity is None else s.astype(f"pint[{commodity.col_to_units['p']}]")


def wp_sourced(
    w_offtake: PintTimeSeries | FloatTimeSeries,
    commodity: Commodity | None,
    freq: str = "MS",
    w_avg: float = 0.6,
    p_avg: float = 100.0,
    rand_amp: float = 0.2,
) -> tuple[PintTimeSeries, PintTimeSeries] | tuple[FloatTimeSeries, FloatTimeSeries]:
    """Create a more or less realistic-looking sourcing volume and sourcing price timeseries.

    Parameters
    ----------
    w_offtake
        Offtake volume timeseries for which to create sourced volume and price timeseries.
    commodity
        Used to get the correct units. If None, return timeseries of floats.
    freq, optional (default: 'MS')
        Frequency within which sourcing volume and price are uniform.
    w_avg, optional (default: 0.6)
        Average sourced fraction (without unit).
    p_avg, optional (default: 100.0)
        Average hedge price (without unit).
    rand_amp, optional (default: 0.2)
        Random amplitude, both of sourced fraction (absolute) and of price (as fraction).

    Returns
    -------
        Sourced volume timeseries and sourced price timeseries.
    """
    # Prepare series for resampling.
    # . Make w_offtake and commodity consistent with eachother.
    if commodity is None:
        if hasattr(w_offtake, "pint"):
            w_offtake = w_offtake.pint.m  # floats
    elif hasattr(w_offtake, "pint"):
        w_offtake = w_offtake.pint.to(commodity.col_to_units["w"])
    else:
        w_offtake = w_offtake.astype(f"pint[{commodity.col_to_units['w']}]")
    # . Split magnitude and units.
    if hasattr(w_offtake, "pint"):
        w_unit = w_offtake.pint.units
        sin = -1 * w_offtake.pint.magnitude
    else:
        w_unit = None
        sin = -1 * w_offtake

    # Do resampling.
    def calc_wp(sub_s):
        wval = sub_s.mean() * (w_avg + rand_amp * np.random.uniform(-1, 1))
        pval = p_avg * (1 + rand_amp * np.random.uniform(-1, 1))
        return pd.DataFrame({"p": pval, "w": wval}, sub_s.index)

    def group_and_calc(s):
        return s.resample(freq, group_keys=False).apply(calc_wp)

    try:
        # Fails if commodity = None, or has no peak_fn, or freq too long.
        is_peak = commodity.peak_fn(sin.index)
        df = sin.groupby(is_peak, group_keys=False).apply(group_and_calc)
    except (AttributeError, ValueError):
        df = group_and_calc(sin)
    w, p = df.w, df.p

    # Add unit if wanted.
    if commodity is not None:
        w = w.astype(f"pint[{commodity.col_to_units['w']}]")
        p = p.astype(f"pint[{commodity.col_to_units['p']}]")
    return w, p
