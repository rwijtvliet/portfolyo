"""Tools to deal with timezones."""

from . import stamps
from typing import Union
from pandas.core.frame import NDFrame
import pandas as pd

"""
(hourly means: hourly or shorter)
(daily means: daily or longer)

We consider 3 types of gapless datetime indices:
A: tz-aware indices. These are never ambiguous, and a freq can always be set.
B: tz-naive indices without DST. These always have 24h/day, and a freq can always be set.
C: tz-naive indices for a location with DST. There is a problem if an index includes the
DST-transition:
- When representing hourly values, there are (correctly) skipped or doubled timestamps.
However, the freq cannot be inferred.
- When represinging daily values, all timestamps are present and a freq can be inferred.
However, not all timestamps have the correct duration (e.g., 24h when day actually has
23h or 25h).

Depending on the use case, we want to work only with type A, or only with type B. But we
must accept all types as input and be able to convert to either type A or type B.

Conversions:
A -> A (different tz). Not always possible. If it is possible, it is a lossless conversion.
        Always possible for quarterhourly. Usually possible for hourly (if difference in
        UTC-offset is whole number of hours). Not possible for daily (because old periods
        do not line up with new timestamps).
        NB: if a floating conversion is wanted (e.g., the 12:00 value in timezone 1 must
        be mapped onto the 12:00 value in timezone 2), go via A -> B -> A.
B -> A. Always possible. If A has no DST, it is a lossless conversion. If has a DST,
        it is a lossing conversion: for hourly indices, we will omit/double some
        corresponding values, and for daily indices, we might need to change the
        corresponding values.
C -> A. Possible for hourly and daily (if belonging to same tz).
A -> B. See B -> A.
B -> B. No conversion.
C -> B. Go via C -> A -> B; timezone must be known.


Implementation difficulties: do we have a B or a C index?
. If it has a frequency, it can be:
. a type B.
. a type C with hourly values which happen to not include a DST-transition.
. a type C with daily values.
. If it has no frequency, it can be:
. a type B with gaps.
. a type C with gaps.
. a type C with hourly values which do include a DST-transition.
It's therefore hard to determine, which it is.
"""


def force_tzaware(
    fr: NDFrame, tz: str, *, floating: bool = False, tz_in: Union[str, None] = None
) -> NDFrame:
    """Convert/set series or dataframe to a specific timezone.

    Parameters
    ----------
    fr : NDFrame
        pandas Series or Dataframe
    tz : str
        target timezone
    floating : bool, optional (default: False)
        Only relevant for timezone-aware input. If ``fr`` must be converted to a
        *different* timezone ``tz``.
        - False: fix timestamps on universal time axis and find new local time in target timezone.
          '2020-03-01 12:00+0530 Asia/Kolkata' --> '2020-03-01 07:30+0100 Europe/Berlin'
        - True: move timestamp on universal time axis to same local time in target timezone.
          '2020-03-01 12:00+0530 Asia/Kolkata' --> '2020-03-01 12:00+0100 Europe/Berlin'
    tz_in : Union[str, None], optional (default: None)
        Only relevant for timezone-naive input.
        - str: the timezone in which the tz-naive timestamps must be interpreted. E.g.,
          if 'Europe/Berlin', the index should be missing the 2:00 timestamp on the
          last Sunday in March (if it has hourly values) and is it is assumed that that
          Sunday is only 23h long (if it has daily values). Some recalculations are
          necessary.
        - None: index is already "standardized", i.e., does not descibre a timezone with
          DST-transitions. No recalculations are necessary.

    Returns
    -------
    NDFrame
        With tz-aware index.
    """

    if not tz:
        raise ValueError(
            "No timezone was specified. To convert to standardized timezone-naive frame, "
            "use ``force_tznaive`` instead."
        )

    # Copy, try to set freq, and store original attributes.
    fr = _setfreq(fr)
    freq_input, tz_input = fr.index.freq, fr.index.tz

    if tz_input:  # input is tz-aware
        if tz_input == tz:
            # A->A same tz
            return fr
        elif not floating:
            # A->A different tz
            fr_out = fr.tz_convert(tz)
            return _setfreq(fr_out, freq_input)
        else:
            # A->A floating
            # defer to A->B and B->A
            fr_naive = force_tznaive(fr)  # A->B
            fr_out = force_tzaware(fr_naive, tz)  # B->A
            return fr_out

    elif tz_in:  # input is tz-naive, but not (necessarily) standardized
        # C->A
        fr_aware = fr.tz_localize(tz_in, ambiguous="infer")
        fr_out = force_tzaware(fr_aware, tz)
        return _setfreq(fr_out, freq_input)

    else:  # input is standardized tz-naive
        # B->A
        # Some recalculations may be necessary.
        fr_aware = swap_naiveaware(fr, tz_in)
        fr_out = force_tzaware(fr_aware, tz)
        return _setfreq(fr_out, freq_input)


def force_tznaive(fr: NDFrame, *, tz_in: Union[str, None] = None) -> NDFrame:
    """Turn a frame (series or dataframe) into "standardized" timezone-naive frame (see
    'Returns', below).

    Parameters
    ----------
    fr : NDFrame
        pandas Series or Dataframe
    tz_in : Union[str, None], optional (default: None)
        Only relevant for timezone-naive input.
        - str: the timezone in which the tz-naive timestamps must be interpreted. E.g.,
          if 'Europe/Berlin', the index should be missing the 2:00 timestamp on the
          last Sunday in March (if it has hourly values) and is it is assumed that that
          Sunday is only 23h long (if it has daily values). Some recalculations are
          necessary.
        - None: index is already "standardized", i.e., does not descibre a timezone with
          DST-transitions. No recalculations are necessary.

    Returns
    -------
    NDFrame
        With standardized tz-naive index, i.e., with 24h in each day and no DST transitions.
    """
    # Copy, try to set freq, and store original attributes.
    fr = _setfreq(fr)
    freq_input, tz_input = fr.index.freq, fr.index.tz

    if tz_input:  # input is tz-aware
        # A->B
        # Some recalculations may be necessary.
        return swap_naiveaware(fr)

    elif tz_in:  # input is tz-naive, but not (necessarily) standardized
        # C->B
        # defer to C->A and A->B
        fr = fr.tz_localize(tz_in)  # C -> A
        fr = _setfreq(fr, freq_input)
        return force_tznaive(fr)  # A -> B

    else:  # input is standardized tz-naive
        # B->B
        return fr


def _setfreq(fr: NDFrame, wanted: str = None) -> NDFrame:
    """Try to read, infer, and force frequency of frame's index, and return the frequency
    that sticks."""
    fr = fr.copy()
    if fr.index.freq:
        pass
    elif freq := pd.infer_freq(fr.index):
        fr.index.freq = freq
    elif wanted:
        fr.index.freq = wanted
    return fr


def swap_naiveaware(fr: NDFrame, tz: str = None) -> NDFrame:
    """Recalculate the values in a tz-aware series or dataframe, to get a tz-naive one
    (that has no DST-transitions and exactly 24h per day) or vice versa.

    Parameters
    ----------
    fr : NDFrame
        pandas Series or Dataframe
    tz : str
        Only relevant when converting timezone-naive input (i.e., into tz-aware output).
        The timezone of the output frame.

    Returns
    -------
    NDFrame
        with recalculated values.

    Notes
    -----
    It is assumed that the .freq attribute of the index has been set, if that is possible
    for the index of ``fr``.
    """

    freq_input, tz_input = fr.index.freq, fr.index.tz
    if not freq_input:
        raise ValueError("Cannot recalculate values if frequency is not known.")
    if not tz_input and tz is None:
        raise ValueError("Parameter ``tz`` must be specified if ``fr`` is tz-naive.")
    if tz_input:
        tz = None

    # Find index of output frame.
    start, end = fr.index[0].tz_localize(tz), fr.index[-1].tz_localize(tz)
    i_out = pd.date_range(start, end, freq=freq_input, tz=tz)

    if stamps.freq_longest(freq_input, "H") == "H":
        # Find a source value for each timestamp in the output frame. Some input values
        # may get lost, other may get duplicated.
        if tz:  # tz-naive -> tz-aware
            # There are multiple timestamps in the output that receive the same input value.
            # (and some input values will be lost). But importantly: each timestamp in
            # output exists in input.
            mapping_onto_input_index = i_out.tz_localize(None)
            return fr.loc[mapping_onto_input_index].set_axis(i_out)

        else:  # tz-aware -> tz-naive
            # There are timestamps in the output that do not exist in the input.
            # In that case, repeat the value of the previous hour.
            partly = fr.tz_localize(tz)
            partly = partly[~partly.index.duplicated()]  # remove duplicates

            def value(ts):
                try:
                    return partly.loc[ts]
                except KeyError:
                    return partly.loc[ts - pd.Timedelta(hours=1)]

            if isinstance(fr, pd.Series):
                return pd.Series([value(ts) for ts in i_out], index=i_out)
            else:
                return pd.DataFrame([value(ts) for ts in i_out], index=i_out)

    else:
        # There is a one-to-one correspondence between the timestamps in input and ouput frames.
        # --> Simply replace the index. Works best if time-averagable quantities are used,
        # like MW and Eur/MWh.
        return _setfreq(fr.set_axis(i_out), freq_input)
