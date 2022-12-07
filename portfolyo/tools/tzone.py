"""Tools to deal with timezones."""

from typing import Union

import pandas as pd
import pytz

from . import freq as tools_freq

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


def force_aware(
    fr: Union[pd.Series, pd.DataFrame], tz: str, *, floating: bool = True
) -> Union[pd.Series, pd.DataFrame]:
    """Convert/set series or dataframe to a specific timezone.

    Parameters
    ----------
    fr : pd.Series or pd.DataFrame
        with tz-aware or (standardized) tz-agnostic index.
    tz : str
        Target timezone.
    floating : bool, optional (default: True)
        Ignored if ``fr`` is tz-agnostic. How to handle conversion to *another*
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
    pd.Series or pd.DataFrame
        with tz-aware index.

    Notes
    -----
    The frequency of ``fr``'s index must be set (or inferrable). Common reasons why this
    is not possible:
    * Timestamps are missing from the index; add them with ``fr.resample().asfreq()``
      (don't forget the ``offset`` argument for daily-or-longer frequencies that do not
      have a day start at midnight).
    * The index is not tz-aware but implies a certain timezone; localize it with
      ``fr.tz_localize()``.

    It is assumed that we are dealing with "time-averable" data, such as values in [MW]
    or [Eur/MWh]. This is especially important when converting daily (and longer) values
    between a tz-agnostic context and a tz-aware context with DST-transitions; see the
    first example:
    * Daily values: 10 [MW] on 2020-10-25 (which has 24h in a standardized context) is
      converted into 10 [MW] on 2020-10-25 in the Europe/Berlin timezone (which has
      25h). Note that this conversion is probably not what we want if we have values in
      units of [MWh] or [Eur].
    * Hourly values: a value of 4 [MW] on 2020-10-25 02:00-03:00 (standardized) is
      duplicated to TWO values of 4 [MW] each in the Europe/Berlin timezone. Note that
      this conversion is probably what we want, regardless of the unit.
    """

    if not tz:
        raise ValueError(
            "No timezone was specified. To convert to standardized timezone-agnostic frame,"
            " use ``force_tzagnostic`` instead."
        )

    # Copy, try to set freq, and store original attributes.
    fr = tools_freq.set_to_frame(fr)
    freq_input, tz_input = fr.index.freq, fr.index.tz

    if not freq_input:
        raise ValueError(
            "A frequency could not be determined for this data. Localize the index (with"
            "``fr.tz_localize``) or add missing datapoints (with ``fr.resample().asfreq()``), or both."
        )

    # Do conversion.
    if tz_input:  # input is tz-aware but maybe not in the correct timezone
        fr_out = _A_to_A(fr, tz=tz, floating=floating)

    else:  # input is standardized tz-agnostic
        fr_out = _B_to_A(fr, tz=tz)

    # Return: set frequency.
    fr_out = tools_freq.set_to_frame(fr_out, freq_input)
    return fr_out


def force_agnostic(
    fr: Union[pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    """Turn a frame (series or dataframe) into timezone-agnostic frame.

    Parameters
    ----------
    fr : pd.Series or pd.DataFrame
        with tz-aware or (standardized) tz-agnostic index.

    Returns
    -------
    pd.Series or pd.DataFrame
        with standardize tz-agnostic index, i.e., 24h in each day and no DST transitions.

    Notes
    -----
    The frequency of ``fr``'s index must be set (or inferrable). Common reasons why this
    is not possible:
    * Timestamps are missing from the index; add them with ``fr.resample().asfreq()``
      (don't forget the ``offset`` argument for daily-or-longer frequencies that do not
      have a day start at midnight).
    * The index is not tz-aware but implies a certain timezone; localize it with
      ``fr.tz_localize()``.

    It is assumed that we are dealing with "time-averable" data, such as values in [MW]
    or [Eur/MWh]. This is especially important when converting daily (and longer) values
    between a tz-agnostic context and a tz-aware context with DST-transitions; see the
    first example:
    * Daily values: 10 [MW] on 2020-10-25 (which has 24h in a standardized context) is
      converted into 10 [MW] on 2020-10-25 in the Europe/Berlin timezone (which has
      25h). Note that this conversion is probably not what we want if we have values in
      units of [MWh] or [Eur].
    * Hourly values: a value of 4 [MW] on 2020-10-25 02:00-03:00 (standardized) is
      duplicated to TWO values of 4 [MW] each in the Europe/Berlin timezone. Note that
      this conversion is probably what we want, regardless of the unit.
    """
    # Copy, try to set freq, and store original attributes.
    fr = tools_freq.set_to_frame(fr)
    freq_input, tz_input = fr.index.freq, fr.index.tz

    if not freq_input:
        raise ValueError(
            "A frequency could not be determined for this data. Localize the index (with "
            "``fr.tz_localize``) or add missing datapoints (with ``fr.resample().asfreq()``), or both."
        )

    # Do conversion.
    if tz_input:  # input is tz-aware
        fr_out = _A_to_B(fr)

    else:  # input is already standardized tz-agnostic
        fr_out = fr

    # Return: set frequency.
    fr_out = tools_freq.set_to_frame(fr_out, freq_input)
    return fr_out


def _A_to_A(
    fr: Union[pd.Series, pd.DataFrame], *, tz, floating
) -> Union[pd.Series, pd.DataFrame]:
    if isinstance(tz, str):  # convert to make comparison (below) possible
        tz = pytz.timezone(tz)

    if fr.index.tz == tz:
        # input already has wanted tz; no conversion necessary
        return fr

    elif not floating:
        # convertion using universal time
        return fr.tz_convert(tz)

    else:
        # conversion by keeping local time
        # Defer to A -> B -> A
        return _B_to_A(_A_to_B(fr), tz=tz)


def _A_to_B(fr: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    return _aware_to_agnostic(fr)


def _B_to_A(
    fr: Union[pd.Series, pd.DataFrame], *, tz
) -> Union[pd.Series, pd.DataFrame]:
    return _agnostic_to_aware(fr, tz)


def _idx_after_conversion(fr: Union[pd.Series, pd.DataFrame], tz) -> pd.DatetimeIndex:
    fr = tools_freq.set_to_frame(fr)
    freq_input = fr.index.freq
    if not freq_input:
        raise ValueError("Cannot recalculate values if frequency is not known.")

    # Index of output frame.
    start, end = fr.index[0].tz_localize(tz), fr.index[-1].tz_localize(tz)
    idx = pd.date_range(start, end, freq=freq_input, tz=tz)

    return idx


def _agnostic_to_aware(
    fr: Union[pd.Series, pd.DataFrame], tz: str
) -> Union[pd.Series, pd.DataFrame]:
    """Recalculate values in tz-agnostic series or dataframe, to get a tz-aware one.
    (i.e., B to A)."""
    if tz_input := fr.index.tz:
        raise ValueError(f"``fr`` must be tz-agnostic; found timezone '{tz_input}'.")
    if not tz:
        raise ValueError("Must specify timezone ``tz`` to convert into.")

    idx_out = _idx_after_conversion(fr, tz)

    # Convert daily or longer.
    if tools_freq.shortest(idx_out.freq, "D") == "D":
        # One-to-one correspondence between the timestamps in input and ouput frames.
        # --> Simply replace the index.
        return fr.set_axis(idx_out)

    # Convert hourly or shorter.
    # There may be multiple timestamps in output receiving same input value. (And some
    # that are lost). But importantly: each timestamp in output exists in input.
    mapping_onto_input_index = idx_out.tz_localize(None)
    return fr.loc[mapping_onto_input_index].set_axis(idx_out)


def _aware_to_agnostic(
    fr: Union[pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    """Recalculate values in tz-aware series or dataframe, to get a tz-agnostic one.
    (i.e., A to B)."""
    if not fr.index.tz:
        raise ValueError("``fr`` must be tz-aware.")

    idx_out = _idx_after_conversion(fr, None)

    # Convert daily or longer.
    if tools_freq.shortest(idx_out.freq, "D") == "D":
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
