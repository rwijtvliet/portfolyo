"""Tools to deal with timezones."""

from pytz import AmbiguousTimeError
from . import frames, stamps
from typing import Union
from pandas.core.frame import NDFrame
import pandas as pd

# TODO: put in documentation.
"""
Working with timezones can be a real headache. Here are some common scenarios and how
to handle them.

In general, a lot of problems and ambiguity can be removed by picking a timezone and
sticking to it - and converting all incoming timeseries into this timezone. Let's see
how we can do this for various types of information.

The most common case is where we have a timezone tied to a geographic location. In that
case, we can do all our conversions with the ``force_tzaware()`` function. In this
example we'll use the 'Europe/Berlin' timezone. Here's how we can convert any timeseries
(or dataframe) ``fr`` into this timezone:
* The series already has the correct timezone set. This can be checked with
``fr.index.tz``. In this case, nothing needs to be done (and if we pass it to
``force_tzaware()``, it is returned as-as).
* The series has a timezone set (i.e., is 'tz-aware') but to a different timezone,
let's say 'Japan'. In that case, we need to decide if we want to convert the series
while keeping the local time or while keeping the universal time.
  - Keeping the local time ('floating conversion') means that the value belonging to
    2020-01-01 15:00 in Japan is converted to 2020-01-01 15:00 in Europe/Berlin.
  - Keeping the universal time ('non-floating conversion') means that the 15:00 value
    is converted to 07:00 in Europe/Berlin. For daily (and longer) values, this is not
    possible, as e.g. the day 2020-01-01 in Japan falls on two days in Europe/Berlin.
For these conversions, we use the ``force_tzaware()`` function and specify the
``floating`` parameter.
* The series is in the correct timezone, but has 'wall time' values. This means that the
UTC-offset is not shown; the values are not tz-aware. For timezones with daylight-
savings time, like 'Europe/Berlin', this means that a timeseries with hourly values has
a missing/repeated hour at the start/end of DST. This lossless conversion is done by
``force_tzaware()`` if possible. (Alternatively, we could use the ``fr.tz_localize()``
method that pandas provides).
* The series is tz-agnostic and has exactly 24h for each day. In this case, a conversion
to the 'Europe/Berlin' timezone is lossy if we are dealing with hourly values (and
shorter) - after all, some values around the DST-transition will get dropped or
duplicated. This is also done by the ``force_tzaware()`` function.

Alternatively, we might not want to deal with timezones at all, and convert any
timeseries (or dataframe) ``fr`` into 'tz-agnostic'. The data is then no longer tied to
a specific geographic location, and therefore do not have DST and every day has exactly
24h. In that case, we can do all our conversions with the ``force_tzagnostic()``
function. Here's how we can use this function:
* The series is already 'standardized'. No conversion is needed (and passing it to
``force_tzagnostic()`` returns it as-is).
* The series has a timezone set. The conversion is done with ``force_tzagnostic()``. As
above: if the source data has hourly (or shorter) values in a timezone with a
DST-transition (like 'Europe/Berlin'), the conversion is lossy.
* The series is in a specific timezone, but is not tz-aware. Instead, it has 'wall time'
values. In this case the same conversion needs to be done as in the previous point, but
we need to specifiy the timezone in which the data should be interpreted with the
``tz_in`` parameter of the ``force_tzagnostic()`` function.

The conversion functions assume that we are dealing with "time-averable" data, such as
values in [MW] or [Eur/MWh].
Two examples:
* Daily values: 10 [MW] on 2020-10-25 (which has 24h in a standardized context) is
converted into 10 [MW] on 2020-10-25 in the Europe/Berlin timezone (which has 25h).
* Hourly values: a value of 4 [MW] on 2020-10-25 02:00-03:00 (standardized) is
duplicated to TWO values of 4 [MW] each in the Europe/Berlin timezone.

When converting "time-summable" data, such as values in [MWh] or [Eur], do the unit
conversion into [MW] or [Eur/MWh] before doing the timezone conversion.
"""

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


def force_tzaware(fr: NDFrame, tz: str, *, floating: bool = True) -> NDFrame:
    """Convert/set series or dataframe to a specific timezone.

    Parameters
    ----------
    fr : NDFrame
        pandas Series or Dataframe
    tz : str
        target timezone
    floating : bool, optional (default: True)
        Ignored if ``fr`` is not tz-aware. How to handle conversion to *another*
        timezone.
        - True: move timestamp on universal time axis to same local time in target
          timezone.
          '2020-03-01 12:00+0530 Asia/Kolkata' --> '2020-03-01 12:00+0100 Europe/Berlin'
          Lossy conversion if source and target timezones do not both have a fixed UTC-
          offset.
        - False: fix timestamps on universal time axis and find new local time in target
          timezone.
          '2020-03-01 12:00+0530 Asia/Kolkata' --> '2020-03-01 07:30+0100 Europe/Berlin'
          Conversion in many cases (e.g. daily values and longer) impossible as there is
          no 1-to-1 relation between source and target timezones.

    Returns
    -------
    NDFrame
        With tz-aware index.

    Notes
    -----
    If ``fr`` is not tz-aware, there are two accepted forms for its index:
    - The index may be the local time in the timezone ``tz``. In that case, if e.g.
    'Europe/Berlin', the index is missing the 2:00 timestamp on the last Sunday in March
    (if it has hourly values).
    - The index may be 'standardized', i.e., having exactly 24h per day and no DST-
    transitions.

    It is assumed that we are dealing with "time-averable" data, such as values in [MW]
    or [Eur/MWh]. This is especially important when converting daily (and longer) values
    between a tz-agnostic context and a tz-aware context with DST-transitions; see the
    first example:
    * Daily values: 10 [MW] on 2020-10-25 (which has 24h in a standardized context) is
    converted into 10 [MW] on 2020-10-25 in the Europe/Berlin timezone (which has 25h).
    Note that this conversion is probably not what we want if we have values in [MWh].
    * Hourly values: a value of 4 [MW] on 2020-10-25 02:00-03:00 (standardized) is
    duplicated to TWO values of 4 [MW] each in the Europe/Berlin timezone. Note that
    this conversion is probably what we want, regardless of the unit.
    """

    if not tz:
        raise ValueError(
            "No timezone was specified. To convert to standardized timezone-agnostic "
            "frame, use ``force_tzagnostic`` instead."
        )

    # Copy, try to set freq, and store original attributes.
    fr = frames.set_frequency(fr)
    freq_input, tz_input = fr.index.freq, fr.index.tz

    # Do conversion.
    if tz_input:  # input is tz-aware
        fr_out = _A_to_A(fr, tz=tz, floating=floating)

    else:  # input is tz-naive and may be standardized

        try:
            fr_out = _C_to_A(fr, tz=tz)
        except AmbiguousTimeError:
            fr_out = _B_to_A(fr, tz=tz)

    # Return: set frequency, and if succesful, check if all timestamps are valid for it.
    fr_out = frames.set_frequency(fr_out, freq_input)
    if freq := fr_out.index.freq:
        stamps.assert_boundary_ts(fr_out.index, freq)
    return fr_out


def force_tzagnostic(fr: NDFrame, *, tz_in: Union[str, None] = None) -> NDFrame:
    """Turn a frame (series or dataframe) into timezone-agnostic frame.

    Parameters
    ----------
    fr : NDFrame
        pandas Series or Dataframe
    tz_in : Union[str, None], optional (default: None)
        Ignored for tz-aware ``fr``. The timezone in which the timestamps in the index
        of ``fr`` must be interpreted.
        - str: if e.g. 'Europe/Berlin', the index should be missing the 2:00 timestamp
          on the last Sunday in March (if it has hourly values) and is it is assumed
          that that Sunday is only 23h long (if it has daily values).
        - None: index is already "standardized", i.e., does not descibre a timezone with
          DST-transitions.

    Returns
    -------
    NDFrame
        With tz-agnostic index, i.e., 24h in each day and no DST transitions.

    Notes
    -----
    It is assumed that we are dealing with "time-averable" data, such as values in [MW]
    or [Eur/MWh]. This is especially important when converting daily (and longer) values
    between a tz-agnostic context and a tz-aware context with DST-transitions; see the
    first example:
    * Daily values: 10 [MW] on 2020-10-25 (which has 24h in a standardized context) is
    converted into 10 [MW] on 2020-10-25 in the Europe/Berlin timezone (which has 25h).
    Note that this conversion is probably not what we want if we have values in [MWh].
    * Hourly values: a value of 4 [MW] on 2020-10-25 02:00-03:00 (standardized) is
    duplicated to TWO values of 4 [MW] each in the Europe/Berlin timezone. Note that
    this conversion is probably what we want, regardless of the unit.
    """
    # Copy, try to set freq, and store original attributes.
    fr = frames.set_frequency(fr)
    freq_input, tz_input = fr.index.freq, fr.index.tz

    if tz_input:  # input is tz-aware
        fr_out = _A_to_B(fr)

    elif tz_in:  # input is tz-naive, but not (necessarily) standardized
        fr_out = _C_to_B(fr, tz_in=tz_in)

    else:  # input is already standardized tz-agnostic
        fr_out = fr

    # Return: set frequency, and if succesful, check if all timestamps are valid for it.
    fr_out = frames.set_frequency(fr_out, freq_input)
    if freq := fr_out.index.freq:
        stamps.assert_boundary_ts(fr_out.index, freq)
    return fr_out


def _A_to_A(fr: NDFrame, *, tz, floating) -> NDFrame:
    if fr.index.tz == tz:  # input already has wanted tz; no conversion necessary
        return fr

    elif not floating:  # convertion using universal time
        return fr.tz_convert(tz)

    else:  # conversion by keeping local time
        # Defer to A -> B -> A
        return _B_to_A(_A_to_B(fr), tz=tz)


def _A_to_B(fr: NDFrame) -> NDFrame:
    return _aware_to_agnostic(fr)


def _B_to_A(fr: NDFrame, *, tz) -> NDFrame:
    return _agnostic_to_aware(fr, tz)


def _C_to_A(fr: NDFrame, *, tz) -> NDFrame:
    return fr.tz_localize(tz, ambiguous="infer")


def _C_to_B(fr: NDFrame, *, tz_in) -> NDFrame:
    # Defer to C -> A -> B
    return _A_to_B(_C_to_A(fr, tz=tz_in))


def _idx_after_conversion(fr: NDFrame, tz) -> pd.DatetimeIndex:
    fr = frames.set_frequency(fr)
    freq_input = fr.index.freq
    if not freq_input:
        raise ValueError("Cannot recalculate values if frequency is not known.")

    # Index of output frame.
    start, end = fr.index[0].tz_localize(tz), fr.index[-1].tz_localize(tz)
    idx = pd.date_range(start, end, freq=freq_input, tz=tz)

    return idx


def _agnostic_to_aware(fr: NDFrame, tz: str) -> NDFrame:
    """Recalculate values in tz-agnostic series or dataframe, to get a tz-aware one.
    (i.e., B to A)."""
    if tz_input := fr.index.tz:
        raise ValueError(f"``fr`` must be tz-agnostic; found timezone '{tz_input}'.")
    if not tz:
        raise ValueError("Must specify timezone ``tz`` to convert into.")

    idx_out = _idx_after_conversion(fr, tz)

    # Convert daily or longer.
    if stamps.freq_shortest(idx_out.freq, "D") == "D":
        # One-to-one correspondence between the timestamps in input and ouput frames.
        # --> Simply replace the index.
        return fr.set_axis(idx_out)

    # Convert hourly or shorter.
    # There may be multiple timestamps in output receiving same input value. (And some
    # that are lost). But importantly: each timestamp in output exists in input.
    mapping_onto_input_index = idx_out.tz_localize(None)
    return fr.loc[mapping_onto_input_index].set_axis(idx_out)


def _aware_to_agnostic(fr: NDFrame) -> NDFrame:
    """Recalculate values in tz-aware series or dataframe, to get a tz-agnostic one.
    (i.e., A to B)."""
    if not fr.index.tz:
        raise ValueError("``fr`` must be tz-aware.")

    idx_out = _idx_after_conversion(fr, None)

    # Convert daily or longer.
    if stamps.freq_shortest(idx_out.freq, "D") == "D":
        # One-to-one correspondence between the timestamps in input and ouput frames.
        # --> Simply replace the index.
        return fr.set_axis(idx_out)

    # Convert hourly or shorter.
    # There are timestamps in the output that do not exist in the input. In that case,
    # repeat the value of the previous hour.
    partly = fr.tz_localize(None)
    partly = partly[~partly.index.duplicated()]  # remove duplicates

    def value(ts):  # Take value of prev hour if current time not found in the input.
        try:
            return partly.loc[ts]
        except KeyError:
            return partly.loc[ts - pd.Timedelta(hours=1)]

    return fr.__class__([value(ts) for ts in idx_out], index=idx_out)


def _swap_agnosticaware(fr: NDFrame, tz: str = None) -> NDFrame:
    """Recalculate the values in a tz-aware series or dataframe, to get a tz-agnostic
    one (that has no DST-transitions and exactly 24h per day) or vice versa.

    Parameters
    ----------
    fr : NDFrame
        pandas Series or Dataframe
    tz : str
        The timezone of the output frame.
        Ignored for tz-aware input (i.e., tz-agnostic output).

    Returns
    -------
    NDFrame
        with recalculated values.
    """
    fr = frames.set_frequency(fr)
    freq_input, tz_input = fr.index.freq, fr.index.tz
    if not freq_input:
        raise ValueError("Cannot recalculate values if frequency is not known.")
    if not tz_input and tz is None:
        raise ValueError("Parameter ``tz`` must be specified if ``fr`` is tz-agnostic.")
    if tz_input:
        tz = None

    # Find index of output frame.
    start, end = fr.index[0].tz_localize(tz), fr.index[-1].tz_localize(tz)
    i_out = pd.date_range(start, end, freq=freq_input, tz=tz)

    if stamps.freq_longest(freq_input, "H") == "H":
        # Find a source value for each timestamp in the output frame. Some input values
        # may get lost, other may get duplicated.
        if tz:  # tz-agnostic -> tz-aware
            # There are multiple timestamps in the output that receive the same input value.
            # (and some input values will be lost). But importantly: each timestamp in
            # output exists in input.
            mapping_onto_input_index = i_out.tz_localize(None)
            return fr.loc[mapping_onto_input_index].set_axis(i_out)

        else:  # tz-aware -> tz-agnostic
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
        return frames.set_frequency(fr.set_axis(i_out), freq_input)
