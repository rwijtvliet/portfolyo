"""
Standardizing series and dataframes to use as input for PfLine.
"""

from typing import Union

import pandas as pd
from pytz import AmbiguousTimeError, NonExistentTimeError

from . import freq as tools_freq
from . import right as tools_right
from . import righttoleft as tools_righttoleft
from . import tzone as tools_tzone


# TODO: remove 'Europe/Berlin' as default for ``tz``, use None instead.
def frame(
    fr: Union[pd.Series, pd.DataFrame],
    force: str = None,
    bound: str = "left",
    *,
    tz: str = None,
    floating: bool = True,
    force_freq: str = None,
) -> Union[pd.Series, pd.DataFrame]:
    """Standardize a series or dataframe.

    Parameters
    ----------
    fr : pd.Series or pd.DataFrame
    force : {'aware', 'agnostic'}, optional (default: None)
        Force ``fr`` to be timezone aware or timezone agnostic. If None: keep index
        as-is.
    bound : {'left', 'right'}, optional (default: 'left')
        If 'left' ('right'), specifies that input timestamps are left-(right-)bound.
    tz : str, optional (default: None)
        The timezone in which to interpret non-localized values. If ``force`` ==
        'aware': also the timezone to localize to. Ignored if ``force`` is None.
    floating : bool, optional (default: True)
        If ``force`` == 'aware': how to convert to ``tz`` if ``fr`` has other timezone.
        Keep local time (``floating`` == True) or keep universal time (``floating`` ==
        False). Ignored if ``force`` == 'agnostic' or None.
    force_freq : str, optional (default: None)
        If a frequency cannot be inferred from the data (e.g. due to gaps), it is
        resampled at this frequency and the gaps are filled with NaN-values.

    Returns
    -------
    Union[pd.Series, pd.DataFrame]
        Same type as ``fr``, with a left-bound DatetimeIndex, a valid frequency, and
        wanted timezone.

    Notes
    -----
    It is assumed that we are dealing with "time-averable" data, such as values in [MW]
    or [Eur/MWh]. This is especially important when converting daily (and longer) values
    between a tz-agnostic context and a tz-aware context with DST-transitions.

    See also
    --------
    ``portfolyo.force_aware``
    ``portfolyo.force_agnostic``
    """
    kwargs = {"tz": tz, "floating": floating, "force_freq": force_freq}

    # Set index.
    fr = fr.set_axis(pd.DatetimeIndex(fr.index))  # turn / try to turn into datetime

    # We want to cover 2 additional cases for convenience sake:
    # a. The user passes a frame that still needs to be localized (--> freq unknown)
    # b. The user passes a frame that is not left-bound (--> freq needed)

    # Make sure it has a frequency, i.e., make sure it is tz-aware or tz-agnostic.
    # Pipeline if frequency not yet found: right -> left -> localize -> tz-aware -> freq
    fr = tools_freq.set_to_frame(fr)
    freq_input, tz_input = fr.index.freq, fr.index.tz

    # The data may be right-bound.

    if bound == "right":  # right -> left
        return _fix_rightbound(fr, {**kwargs, "force": force})

    # Now the data is left-bound.
    # If the frequency is not found, and it is tz-naive, the index may need to be localized.

    if not freq_input and not tz_input and tz:  # left -> tz-aware (try)
        # fr is tz-agnostic.
        try:
            fr_aware = fr.tz_localize(tz, ambiguous="infer")
        except (AmbiguousTimeError, NonExistentTimeError):
            pass  # fr did not need / cound not be localized. Continue with fr as-is.
        else:
            # Could be localized. Again remove localization if force == 'agnostic' or None.
            force_to = force or "agnostic"
            return frame(fr_aware, force_to, "left", **kwargs)

    # All options to infer frequency have been exhausted. One may or may not have been found.
    # Does the user want to force a frequency?

    if not freq_input and force_freq:
        # No freq has been found, but user specifies which freq it should be.
        return _fix_forcefreq(
            fr, force_freq, {"tz": tz, "floating": floating, "force": force}
        )

    elif not freq_input and not force_freq:
        # No freq has been bound, and user specifies no freq either.
        raise ValueError(
            "A frequency could not be inferred for this data. Force a frequency (by passing the"
            " ``force_freq`` parameter), or localize the data in advance (with ``fr.tz_localize()``)."
        )

    elif freq_input and force_freq and force_freq != freq_input:
        # Freq has been found, but user specifies it should be a different freq.
        raise ValueError(
            f"This data seems to have a frequency {freq_input}, which is different from the frequency"
            f" the user wants to force on it {force_freq}. Note that the ``force_freq`` parameter is"
            " for filling gaps in the input data. It should not be used for resampling! If the"
            " data has e.g. daily values but you want monthly values, use ``force_freq='D'``, and"
            " pass the return value to one of the functions in the ``portfolyo.tools.changefreq`` module."
        )

    # Now the data has frequency set. It is tz-aware (possibly with wrong tz) or tz-agnostic.

    # Fix timezone.
    if force is not None:
        fr = _fix_timezone(fr, force, tz, floating)

    # Check if index is OK; otherwise raise error.
    try:
        assert_frame_standardized(fr)
    except AssertionError as e:
        raise ValueError("Could not standardize this frame") from e

    # Standardize index name.
    fr = _standardize_index_name(fr)
    # After standardizing timezone, the frequency should have been set.
    return tools_freq.set_to_frame(fr, freq_input, strict=force_freq)


def _fix_rightbound(fr, kwargs):
    for how in ["A", "B"]:
        try:
            i_left = tools_righttoleft.index(fr.index, how)
            fr_left = fr.set_axis(i_left)
            return frame(fr_left, bound="left", **kwargs)
        except ValueError:
            pass
    raise ValueError("Cannot make this frame left-bound.")


def _fix_forcefreq(fr, force_freq, kwargs):
    fr_withfreq = fr.asfreq(force_freq)
    return frame(fr_withfreq, bound="left", **kwargs)


def _fix_timezone(fr, force, tz, floating):
    if force == "aware":
        return tools_tzone.force_aware(fr, tz, floating=floating)
    elif force == "agnostic" or force == "naive":
        return tools_tzone.force_agnostic(fr)
    raise ValueError(f"Parameter ``force`` must be 'aware' or 'agnostic'; got {force}.")


def _standardize_index_name(fr: Union[pd.Series, pd.DataFrame]):
    return fr.rename_axis(index="ts_left")


def assert_frame_standardized(fr: Union[pd.Series, pd.DataFrame]):
    """Assert that series or dataframe is standardized."""
    assert_index_standardized(fr.index)


def assert_index_standardized(i: pd.DatetimeIndex, __right: bool = False):
    """Assert that index is standardized."""

    if not isinstance(i, pd.DatetimeIndex):
        raise AssertionError(f"Expecting DatetimeIndex; got {type(i)}.")

    # Check frequency.
    freq = i.freq
    if not freq:
        raise AssertionError("Index must have frequency set.")
    if freq not in (freqs := tools_freq.FREQUENCIES):
        raise AssertionError(
            f"Index frequency must be one of {', '.join(freqs)}; found '{freq}'."
        )

    # Check length.
    if not len(i):
        raise AssertionError("Index must have values; got empty index.")

    # Check hour and minute.
    if tools_freq.up_or_down(freq, "15T") <= 0:  # quarterhour
        startminute = 15 if __right else 0
        if i[0].minute != startminute:
            err = ("right-bound", "15 min past the") if __right else ("", "at a full")
            raise AssertionError(
                f"The first element in an index with {err[0]} quarterhourly values must be {err[1]} hour; found {i[0]}."
            )

        if any(not_ok := [ts.minute not in (0, 15, 30, 45) for ts in i]):
            raise AssertionError(
                "In an index with quarterhourly values, all timestamps (all periods) should"
                f" start at a full quarter-hour; found {i[not_ok]}."
            )
    else:  # longer than quarterhour
        if any(not_ok := i.minute != 0):
            raise AssertionError(
                "In an index with hourly-or-longer values, all timestamps (all periods) should"
                f" start at a full hour; found {i[not_ok]}."
            )

    # Check time-of-day.
    if tools_freq.up_or_down(freq, "H") <= 0:  # hour or shorter
        if not __right:
            start = i[0]
            end = tools_right.stamp(i[-1], i.freq)
        else:
            start = tools_righttoleft.index(i)[0]
            end = i[-1]
        if start.time() != end.time():
            raise AssertionError(
                "An index must contain full days. For hourly-or-shorter values, this means "
                f"that the start time of the first period ({start}) must equal the end time of the "
                f"last period ({end}), which is not the case."
            )
    else:  # days or longer
        if not len(times := set(i.time)) == 1:
            raise AssertionError(
                "In an index with daily-or-longer values, all timestamps (all periods) should"
                f" start at the same time. Found multiple times: {times}."
            )

    # Check day-of-X.
    if tools_freq.up_or_down(freq, "D") > 0:
        if freq == "MS":
            period, not_ok = "month", ~i.is_month_start
        elif freq == "QS":
            period, not_ok = "quarter", ~i.is_quarter_start
        elif freq == "AS":
            period, not_ok = "year", ~i.is_year_start
        if any(not_ok):
            raise AssertionError(
                f"In an index with {period}ly values, all timestamps (all {period}s) should"
                f" fall on the first day of a {period}; found {i[not_ok]}."
            )
