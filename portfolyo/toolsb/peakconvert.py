"""Module to split timeseries into peak/offpeak-values."""

import pandas as pd

from . import frame as tools_frame
from . import index as tools_index
from . import peakfn as tools_peakfn
from . import wavg as tools_wavg
from .types import Frequencylike


def _tseries2po(s: pd.Series, peak_fn: tools_peakfn.PeakFunction, is_summable: bool) -> pd.Series:
    """
    Aggregate timeseries with varying (float) values to a single (float) peak and offpeak value.

    In:

    ts_left
    2020-01-01 00:00:00+01:00    41.88
    2020-01-01 01:00:00+01:00    38.60
    2020-01-01 02:00:00+01:00    36.55
                                 ...
    2020-12-31 21:00:00+01:00    52.44
    2020-12-31 22:00:00+01:00    51.86
    2020-12-31 23:00:00+01:00    52.26
    Freq: H, Name: p, Length: 8784, dtype: float64

    Out:

    peak       51.363667
    offpeak    20.311204
    dtype: float64
    """
    is_peak = peak_fn(s.index)
    if is_summable:
        peak = s[is_peak].sum()
        offpeak = s[~is_peak].sum()
    else:
        duration = tools_index.duration(s.index).pint.m  # floats
        peak = tools_wavg.series(s[is_peak], duration[is_peak])
        offpeak = tools_wavg.series(s[~is_peak], duration[~is_peak])
    return pd.Series({"peak": peak, "offpeak": offpeak})


def tseries2poframe(
    s: pd.Series,
    peak_fn: tools_peakfn.PeakFunction,
    freq: Frequencylike,
    is_summable: bool,
) -> pd.DataFrame:
    """
    Aggregate timeseries with varying values to a dataframe with peak and offpeak
    timeseries, grouped by specified frequency.

    Parameters
    ----------
    s
        Timeseries with hourly or quarterhourly frequency.
    peak_fn
        Function that returns boolean Series indicating if timestamps in index lie in peak period.
    freq
        Target frequency; must be monthly-or-longer for most peak functions.
    is_summable
        True if data is summable, False if it is averagable.

    Returns
    -------
        Dataframe with base, peak and offpeak values (as columns). Index: downsampled
        timestamps at specified frequency.

    In:

    ts_left
    2020-01-01 00:00:00+01:00    41.88
    2020-01-01 01:00:00+01:00    38.60
    2020-01-01 02:00:00+01:00    36.55
                                 ...
    2020-12-31 21:00:00+01:00    52.44
    2020-12-31 22:00:00+01:00    51.86
    2020-12-31 23:00:00+01:00    52.26
    Freq: H, Name: p, Length: 8784, dtype: float64

    Out:

                                peak        offpeak
    ts_left
    2020-01-01 00:00:00+01:00   42.530036   30.614701
    2020-02-01 00:00:00+01:00   33.295167   15.931557
                                ...         ...
    2020-11-01 00:00:00+01:00   49.110873   33.226004
    2020-12-01 00:00:00+01:00   57.872246   35.055449
    12 rows × 3 columns
    """
    # TODO: must assert that target frequency is longer than source frequency?

    # Remove partial data.
    s = tools_frame.trim(s, freq)

    # Handle possible units.
    sin, units = (s.pint.magnitude, s.pint.units) if hasattr(s, "pint") else (s, None)

    # Do calculations.
    sout = sin.resample(freq, group_keys=True).apply(lambda s: _tseries2po(s, peak_fn, is_summable))

    # Handle possible units.
    if units is not None:
        sout = sout.astype(f"pint[{units}]")
    return sout.unstack()  # peak, offpeak as columns
