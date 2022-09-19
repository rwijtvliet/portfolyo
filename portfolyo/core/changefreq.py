"""Functions to change frequency of a pandas dataframe."""

from typing import Callable
from ..tools import stamps, frames
from pandas.core.frame import NDFrame
import pandas as pd


def _downsample_summable_freq(s: pd.Series, freq: str) -> pd.Series:
    """Downsample a summable series using a frequency."""
    # Downsampling is easiest for summable series: simply sum child values.
    s2 = s.resample(freq).sum()
    # Discard rows in new series that are only partially present in original.
    msk = (s2.index >= s.index[0]) & (s2.index.ts_right <= s.index.ts_right[-1])
    s2 = s2[msk]
    # Return if any values found.
    # if not len(s2):
    #     raise ValueError("There are no 'full' time periods at this frequency.")
    return s2


def _downsample_summable_custom(
    s: pd.Series, grouper: Callable[[pd.Timestamp], pd.Timestamp]
) -> pd.Series:
    """Downsample a summable series using a custom grouper."""
    # Downsampling is easiest for summable series: simply sum child values.
    s2 = s.groupby(grouper).sum()
    # Discard rows in new series that are only partially present in original.
    eps = pd.Timedelta(seconds=1)
    start = s.index[0]
    if grouper(start) == grouper(start - eps):
        s2 = s2.iloc[1:]
    end = s.index[-1].ts_right
    if grouper(end - eps) == grouper(end):
        s2 = s2.iloc[:-1]
    # Return if any values found.
    # if not len(s2):
    #     raise ValueError("There are no 'full' time periods for this grouper.")
    return s2


def _downsample(
    s: pd.Series,
    freq: str,
    is_summable: bool = True,
    custom_grouper: Callable[[pd.Series], pd.Series] = None,
) -> pd.Series:
    """Downsample any series."""
    if is_summable:
        # Downsampling is easiest for summable series.
        if custom_grouper:
            s2 = _downsample_summable_custom(s, custom_grouper)
            return frames.set_frequency(s2, freq)
        else:
            return _downsample_summable_freq(s, freq)
    else:
        # For averagable frames: first make summable.
        summable = s.mul(s.index.duration, axis=0)  # now has a pint dtype
        summable2 = _downsample(summable, freq, True, custom_grouper)
        s2 = summable2.div(summable2.index.duration, axis=0)
        return s2.astype(s.dtype).rename(s.name)


def _upsample_averagable_freq(s: pd.Series, freq: str) -> pd.Series:
    """Upsample an averagable series using a frequency."""
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
    return df2.iloc[:-1, 0].rename(s.name)


def _upsample_averagable_custom(
    s: pd.Series, grouper: Callable[[pd.Timestamp], pd.Timestamp]
) -> pd.Series:
    """Upsample an averagable series using a custom grouper."""
    raise NotImplementedError("Can't upsample using a custome grouper (yet)!")


def _upsample(
    s: pd.Series,
    freq: str,
    is_summable: bool = True,
    custom_grouper: Callable[[pd.Series], pd.Series] = None,
) -> pd.Series:
    """Upsample any series."""
    if not is_summable:
        # Upsampling is easiest for averagable frames.
        if custom_grouper:
            return _upsample_averagable_custom(s, custom_grouper)
        else:
            return _upsample_averagable_freq(s, freq)
    else:
        # For summable frames: first make averagable.
        avgable = s.div(s.index.duration, axis=0)  # now has a pint dtype
        avgable2 = _upsample(avgable, freq, False, custom_grouper)
        s2 = avgable2.mul(avgable2.index.duration, axis=0)
        return s2.astype(s.dtype).rename(s.name)


def _general(
    s: pd.Series,
    freq: str = "MS",
    is_summable: bool = True,
    custom_grouper: Callable[[pd.Timestamp], pd.Timestamp] = None,
) -> pd.Series:
    """Change frequency of a Series, depending on the type of data it contains.

    Parameters
    ----------
    s : pd.Series
        Series that needs to be resampled.
    freq : str, optional (default: "MS")
        Target frequency.
    is_summable : bool, optional (default: True)
        True if the values are 'time-summable'. False if it is 'time-averagable'. See
        https://portfolyo.readthedocs.io/en/latest/specialized_topics/resampling.html
        for more information.
    custom_grouper : Callable[[pd.Timestamp], pd.Timestamp], optional (default: None)
        A custom grouper function that should be used for this frequency, instead of the
        default resampling function. (``freq`` argument still used to determine up- or
        downsampling). Callable takes in any timestamp, and returns the timestamp of the
        downsampled group.

    Returns
    -------
    pd.Series
        Resampled series at target frequency.
    """

    # TODO: Add tests with multiindex columns

    # Some values for freq cause labels to be right-bound. Change to make left-bound.
    if freq in ["M", "A", "Q"]:
        freq += "S"
    # Make sure it is a valid frequency.
    if freq not in stamps.FREQUENCIES:
        raise ValueError(
            f"Parameter ``freq`` must be one of {','.join(stamps.FREQUENCIES)}; got {freq}."
        )
    # Make sure series has valid frequency.
    if s.index.freq is None:
        raise ValueError("Series must have frequency.")

    # Eliminate integers.
    if pd.api.types.is_integer_dtype(s.dtype):
        s = s.astype(float)

    # s is now a Series with a 'float' or 'pint' dtype.

    # Empty series.
    if len(s) == 0:
        # empty frame.
        return pd.Series([], pd.date_range("2020", periods=0, freq=freq))

    up_or_down = stamps.freq_up_or_down(s.index.freq, freq)

    # Nothing more needed; portfolio already in desired frequency.
    if up_or_down == 0:
        return s

    # Must downsample.
    elif up_or_down == -1:
        return _downsample(s, freq, is_summable, custom_grouper)

    # Must upsample.
    else:  # up_or_down == 1
        return _upsample(s, freq, is_summable, custom_grouper)


def summable(
    fr: NDFrame,
    freq: str = "MS",
    custom_grouper: Callable[[pd.Timestamp], pd.Timestamp] = None,
) -> NDFrame:
    """
    Resample and aggregate DataFrame or Series with time-summable quantities.

    Parameters
    ----------
    fr : NDFrame
        Pandas Series or DataFrame.
    freq : str, optional (default: 'MS')
        The frequency at which to resample. 'AS' (or 'A') for year, 'QS' (or 'Q')
        for quarter, 'MS' (or 'M') for month, 'D for day', 'H' for hour, '15T' for
        quarterhour.
    custom_grouper : Callable[[pd.Timestamp], pd.Timestamp], optional (default: None)
        A custom grouper function that should be used for this frequency, instead of the
        default resampling function. (``freq`` argument still used to determine up- or
        downsampling). Callable takes in any timestamp, and returns the timestamp of the
        downsampled group.


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

    return _general(fr, freq, True, custom_grouper)


def averagable(
    fr: NDFrame,
    freq: str = "MS",
    custom_grouper: Callable[[pd.Timestamp], pd.Timestamp] = None,
) -> NDFrame:
    """
    Resample and aggregate DataFrame or Series with time-averagable quantities.

    Parameters
    ----------
    fr : NDFrame
        Pandas Series or DataFrame.
    freq : str, optional (default: 'MS')
        The frequency at which to resample. 'AS' (or 'A') for year, 'QS' (or 'Q')
        for quarter, 'MS' (or 'M') for month, 'D for day', 'H' for hour, '15T' for
        quarterhour.
    custom_grouper : Callable[[pd.Timestamp], pd.Timestamp], optional (default: None)
        A custom grouper function that should be used for this frequency, instead of the
        default resampling function. (``freq`` argument still used to determine up- or
        downsampling). Callable takes in any timestamp, and returns the timestamp of the
        downsampled group.

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

    return _general(fr, freq, False, custom_grouper)


def index(idx: pd.DatetimeIndex, freq: str = "MS") -> pd.DatetimeIndex:
    """Resample index.

    Parameters
    ----------
    idx : pd.DatetimeIndex
        Index to resample
    freq : str, optional (default: 'MS')
        The frequency at which to resample. 'AS' (or 'A') for year, 'QS' (or 'Q')
        for quarter, 'MS' (or 'M') for month, 'D for day', 'H' for hour, '15T' for
        quarterhour.

    Returns
    -------
    pd.DatetimeIndex
    """
    up_or_down = stamps.freq_up_or_down(idx.freq, freq)

    # Nothing more needed; index already in desired frequency.
    if up_or_down == 0:
        return idx

    # Must downsample.
    elif up_or_down == -1:
        # We must jump through a hoop: can't directly resample an Index.
        return pd.Series(0, idx).resample(freq).asfreq().index

    # Must upsample.
    else:  # up_or_down == 1
        # We must jump through additional hoops.
        # First, extend by one value...
        idx = idx.append(idx[-1:].shift())
        # ... then do upsampling ...
        idx2 = pd.Series(0, idx).resample(freq).asfreq().index
        # ... and then remove final element.
        return idx2[:-1]
