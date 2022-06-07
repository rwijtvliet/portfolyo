"""Functions to change frequency of a pandas dataframe."""

from ..tools import stamps
from pandas.core.frame import NDFrame
import pandas as pd


def _general(s: pd.Series, freq: str = "MS", is_summable: bool = True):
    """Change frequency of a Series, depending on the type of data it contains."""

    # TODO: Add tests with multiindex columns

    # Some resampling labels are right-bound by default. Change to make left-bound.
    if freq in ["M", "A", "Q"]:
        freq += "S"
    if freq not in stamps.FREQUENCIES:
        raise ValueError(
            f"Parameter ``freq`` must be one of {','.join(stamps.FREQUENCIES)}; got {freq}."
        )
    if pd.api.types.is_integer_dtype(s.dtype):
        s = s.astype(float)

    # s is now a Series with an allowed frequency and a 'float' or 'pint' dtype.

    # Empty series.
    if len(s) == 0:
        return s.resample(freq).mean()  # empty frame.

    up_or_down = stamps.freq_up_or_down(s.index.freq, freq)

    # Nothing more needed; portfolio already in desired frequency.
    if up_or_down == 0:
        return s

    # Must downsample.
    elif up_or_down == -1:
        if is_summable:
            # Downsampling is easiest for summable series: simply sum child values.
            s2 = s.resample(freq).sum()
            # Discard rows in new series that are only partially present in original.
            msk = (s2.index >= s.index[0]) & (s2.index.ts_right <= s.index.ts_right[-1])
            s2 = s2[msk]
            # Return if any values found.
            if not len(s2):
                raise ValueError("There are no 'full' time periods at this frequency.")
            return s2
        else:
            # For averagable frames: first make summable.
            summable = s.mul(s.index.duration, axis=0)  # now has a pint dtype
            summable2 = _general(summable, freq, True)
            s2 = summable2.div(summable2.index.duration, axis=0)
            return s2.astype(s.dtype).rename(s.name)

    # Must upsample.
    else:  # up_or_down == 1
        if not is_summable:
            # Upsampling is easiest for averagable frames: simply duplicate parent value.
            # We cannot simply `.resample()`, because in that case the final value is not
            # duplicated. We add a dummy value, which we eventually remove again.

            # First, add additional row...
            # (original code to add additional row, does not work if unit-aware. Maybe with future release of pint_pandas?)
            # s = s.copy()
            # s.loc[s.index.ts_right[-1]] = None
            # (Workaround: turn into dataframe, change frequency, and turn back into series.)
            df = pd.DataFrame(s)
            df.loc[s.index.ts_right[-1], :] = None
            # ... then do upsampling ...
            df2 = df.resample(freq).asfreq().ffill()  # duplicate value
            # ... and then remove final row (and turn back into series).
            return df2.iloc[:-1, 0]
        else:
            # For summable frames: first make averagable.
            avgable = s.div(s.index.duration, axis=0)  # now has a pint dtype
            avgable2 = _general(avgable, freq, False)
            s2 = avgable2.mul(avgable2.index.duration, axis=0)
            return s2.astype(s.dtype).rename(s.name)


def summable(fr: NDFrame, freq: str = "MS") -> NDFrame:
    """
    Resample and aggregate DataFrame or Series with time-summable quantities.

    Parameters
    ----------
    fr : NDFrame
        Pandas Series or DataFrame.
    freq : str, optional
        The frequency at which to resample. 'AS' (or 'A') for year, 'QS' (or 'Q')
        for quarter, 'MS' (or 'M') for month, 'D for day', 'H' for hour, '15T' for
        quarterhour. The default is 'MS'.

    Returns
    -------
    DataFrame or Series

    Notes
    -----
    A 'time-summable' quantity is one that can be summed to get to an aggregate
    value, like revenue [Eur] or energy [MWh]. Prices [Eur/MWh] and powers [MW]
    are not time-summable.
    """
    if isinstance(fr, pd.DataFrame):
        # Turn into series, change frequency, and turn back into dataframe.
        return pd.DataFrame({key: summable(value, freq) for key, value in fr.items()})

    return _general(fr, freq, True)


def averagable(fr: NDFrame, freq: str = "MS") -> NDFrame:
    """
    Resample and aggregate DataFrame or Series with time-averagable quantities.

    Parameters
    ----------
    fr : NDFrame
        Pandas Series or DataFrame.
    freq : str, optional
        The frequency at which to resample. 'AS' (or 'A') for year, 'QS' (or 'Q')
        for quarter, 'MS' (or 'M') for month, 'D for day', 'H' for hour, '15T' for
        quarterhour. The default is 'MS'.

    Returns
    -------
    DataFrame or Series

    Notes
    -----
    A 'time-averagable' quantity is one that can be averaged to an aggregate value,
    like power [MW]. When downsampling, the values are weighted with their duration.
    """
    if isinstance(fr, pd.DataFrame):
        # Turn into series, change frequency, and turn back into dataframe.
        return pd.DataFrame({key: averagable(value, freq) for key, value in fr.items()})

    return _general(fr, freq, False)
