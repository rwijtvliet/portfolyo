"""Map series with quarterhourly, hourly, or daily values onto another index , trying to
align weekdays, holidays, and dst-changeover days. Always takes values from same calender
month (but different year)."""

# Three difficulties when changing the year:
# . Feb 29 might be present in one of the years but not in the other.
# . Daylight-savings time will likely start and end at a different day.
# . Weekdays and holidays are not at the same date.

from typing import Union

import holidays
import pandas as pd

from . import changefreq
from . import stamps


def characterize_index(
    idx: pd.DatetimeIndex, holiday_country: str = "DE"
) -> pd.DataFrame:
    """Characterise an index with daily frequency.

    Parameters
    ----------
    idx : pd.DatetimeIndex
    holiday_country : str, optional (default: "DE")
        Country or region for which to assume the holidays. See
        ``holidays.list_supported_countries()`` for allowed values.

    Returns
    -------
    pd.DataFrame
        With several field describing each timestamp in the index.
    """
    if idx.freq != "D":
        raise ValueError("Pass an index with daily frequency.")
    char = pd.DataFrame({"isoweekday": idx.map(lambda ts: ts.isoweekday())}, idx)
    char["yy"] = idx.map(lambda ts: ts.year)
    char["mm"] = idx.map(lambda ts: ts.month)
    char["dd"] = idx.map(lambda ts: ts.day)

    # add holidays
    char["holiday"] = ""
    if holiday_country:
        the_holidays = getattr(holidays, holiday_country)
        year_range = range(idx[0].year, idx[-1].year + 1)
        for date, occasion in the_holidays(years=year_range).items():
            mask = (
                (char.yy == date.year) & (char.mm == date.month) & (char.dd == date.day)
            )
            char.loc[mask, "holiday"] = occasion

    # add dst-change
    offsets = idx.map(lambda ts: ts.utcoffset())
    if offsets.isna().all():
        char["dst_change"] = False
    else:
        char["dst_change"] = [*(offsets[1:] != offsets[:-1]), False]
    return char


def map_index(
    idx_source: pd.DatetimeIndex,
    idx_target: pd.DatetimeIndex,
    holiday_country: str = None,
) -> pd.Series:
    """Map a source index with daily frequency onto a target index with same frequency.

    Source values are always taken from the same calendar month. For holidays, the same-
    named holiday is taken; if not found, uses other holiday in that month, or else any
    Sunday. For weekdays, the same weekday is taken. In all cases, repetitions are
    minimized.

    Parameters
    ----------
    idx_source : pd.DatetimeIndex
        Index of source values.
    idx_target : pd.DatetimeIndex
        Index of target values.
    holiday_country : str, optional (default: None)
        Country or region for which to assume the holidays. E.g. 'DE' (Germany), 'NL'
        (Netherlands), or 'USA'. See ``holidays.list_supported_countries()`` for allowed
        values.

    Returns
    -------
    pd.Series
        Index: target index. Values: corresponding timestamp in source index.

    Notes
    -----
    No values are recalculated; they are simply 'reshuffled' to obey the rules above.
    Within each pair of target and source timestamps, the duration is equal. (both 24h,
    both 23h, or both 25h). It can therefore also be used as a starting point to map
    hourly or quarterhourly indices.
    """
    if (tz1 := idx_source.tz) != (tz2 := idx_target.tz):
        raise ValueError(f"Indices must have same timezone, got {tz1} and {tz2}.")
    if (fr1 := idx_source.freq) != (fr2 := idx_target.freq):
        raise ValueError(f"Indices must have same frequency, got {fr1} and {fr2}.")

    freq_vs_d = stamps.freq_up_or_down("D", idx_source.freq)

    if freq_vs_d == 0:  # daily values.
        return _map_index_daily(idx_source, idx_target, holiday_country)

    elif freq_vs_d == 1:  # (quarter)hourly
        return _map_index_belowdaily(idx_source, idx_target, holiday_country)

    else:
        raise ValueError("Can only map indices with daily frequency or shorter.")


def _map_index_daily(
    idx_source: pd.DatetimeIndex,
    idx_target: pd.DatetimeIndex,
    holiday_country: str = None,
):
    source = characterize_index(idx_source, holiday_country)
    target = characterize_index(idx_target, holiday_country)
    source["in_target_count"] = 0  # keep track of which days have been used.
    target["source_day"] = None

    def set_day(candidates: pd.DataFrame, target_day: pd.Timestamp) -> None:
        if len(candidates) == 0:
            return
        if any(candidates["in_target_count"] == 0):  # find earliest
            source_day = candidates.sort_values(by=["in_target_count"]).index[0]
        else:  # find nearest
            candidates = candidates.copy()  # to stop complaining on next line
            candidates["dist"] = abs(candidates.dd - target_day.day)
            source_day = candidates.sort_values(by=["in_target_count", "dist"]).index[0]
        target.loc[target_day, "source_day"] = source_day
        source.loc[source_day, "in_target_count"] += 1

    # Map the DST days.
    for day in target[target.dst_change].itertuples():
        # Find dst-changing day in same month.
        candidates = source[source.dst_change & (source.mm == day.mm)]
        set_day(candidates, day.Index)

    # Map the non-holidays.
    for day in target[~target.dst_change & (target.holiday == "")].itertuples():
        # Find the same non-holiday weekday in the same month, while minimizing repetition.
        candidates = source[
            ~source.dst_change
            & (source.holiday == "")
            & (source.isoweekday == day.isoweekday)
            & (source.mm == day.mm)
        ]
        set_day(candidates, day.Index)

    # Map the holidays with same name.
    for day in target[~target.dst_change & (target.holiday != "")].itertuples():
        # Find same-named holiday IN THE SAME MONTH.
        candidates = source[
            ~source.dst_change & (source.holiday == day.holiday) & (source.mm == day.mm)
        ]
        set_day(candidates, day.Index)

    # Map holidays that were not yet found.
    for day in target[
        ~target.dst_change & (target.holiday != "") & target.source_day.isna()
    ].itertuples():
        # Find any other holiday or any Sunday.
        candidates = source[
            ~source.dst_change
            & ((source.holiday != "") | (source.isoweekday == 7))
            & (source.mm == day.mm)
        ]
        set_day(candidates, day.Index)

    if any(target.source_day.isna()):
        not_found = target[target.source_day.isna()]
        raise ValueError(
            f"Did not find correspondence for the following days: {not_found.index}."
        )

    return target["source_day"]


def _map_index_belowdaily(
    idx_source: pd.DatetimeIndex,
    idx_target: pd.DatetimeIndex,
    holiday_country: str = None,
):
    # Do mapping on day-level.
    idx_target_d = changefreq.index(idx_target, "D")
    idx_source_d = changefreq.index(idx_source, "D")
    mapp_d = _map_index_daily(idx_source_d, idx_target_d, holiday_country)

    # Split timestamps in 'day' and 'offset'.
    idx_target_part1 = idx_target.floor("D")
    idx_target_part2 = idx_target - idx_target_part1

    return pd.Series(mapp_d[idx_target_part1].values + idx_target_part2, idx_target)


def frame_set_index(
    source: Union[pd.Series, pd.DataFrame],
    idx_target: pd.DatetimeIndex,
    holiday_country: str = None,
) -> Union[pd.Series, pd.DataFrame]:
    """Map a Series or DataFrame onto a target index.

    This is useful when combining data that was recorded in distinct years.

    Source values are always taken from the same calendar month. For holidays, the same-
    named holiday is taken; if not found, uses other holiday or Sunday in that month. For
    weekdays, the same weekday is taken. In all cases, repetitions are minimized.

    Parameters
    ----------
    source: pd.Series or pd.DataFrame
        Source values.
    idx_target : pd.DatetimeIndex
        Index of target values.
    holiday_country : str, optional (default: None)
        Country or region for which to assume the holidays. E.g. 'DE' (Germany), 'NL'
        (Netherlands), or 'USA'. See ``holidays.list_supported_countries()`` for allowed
        values.

    Returns
    -------
    pd.Series or pd.DataFrame
        Index: target index. Values: corresponding values in source index.

    Notes
    -----
    - No values are recalculated; they are simply 'reshuffled' to obey the rules above.
    - Only works on data with a daily frequency or shorter.
    - Function is meant for data that spans full months. Using partial months may lead
      to unexpected behavior.
    """
    mapping = map_index(source.index, idx_target, holiday_country)

    if isinstance(source, pd.Series):
        return source[mapping].set_axis(mapping.index)
    else:
        series = {col: s[mapping].values for col, s in source.items()}
        return pd.DataFrame(series, mapping.index)


def frame_set_year(
    source: Union[pd.Series, pd.DataFrame],
    year_target: int,
    holiday_country: str = None,
) -> Union[pd.Series, pd.DataFrame]:
    """Change the year of a Series or DataFrame.

    This is useful when combining data that was recorded in distinct years.

    Source values are always taken from the same calendar month. For holidays, the same-
    named holiday is taken; if not found, uses other holiday or Sunday in that month. For
    weekdays, the same weekday is taken. In all cases, repetitions are minimized.

    Parameters
    ----------
    source: pd.Series or pd.DataFrame
        Source values.
    year_target : int
        Year onto which to map the data. If the source data spans multiple years, the
        same number of years is used in the return value. (The provided year is used for
        the first timestamp in ``source``.)
    holiday_country : str, optional (default: None)
        Country or region for which to assume the holidays. E.g. 'DE' (Germany), 'NL'
        (Netherlands), or 'USA'. See ``holidays.list_supported_countries()`` for allowed
        values.

    Returns
    -------
    pd.Series or pd.DataFrame
        Index: target index. Values: corresponding values in source index.

    Notes
    -----
    - No values are recalculated; they are simply 'reshuffled' to obey the rules above.
    - Only works on data with a daily frequency or shorter.
    - Function is meant for data that spans full months. Using partial months may lead
      to unexpected behavior.
    """

    def change_year(ts: pd.Timestamp, y: int) -> pd.Timestamp:
        return pd.Timestamp(
            year=y,
            month=ts.month,
            day=ts.day,
            hour=ts.hour,
            minute=ts.minute,
            freq=ts.freq,
            tz=ts.tz,
        )

    freq, tz = source.index.freq, source.index.tz
    if freq is None:
        raise ValueError("Source has no frequency.")
    if not stamps.freq_longest(freq, "D") == "D":
        raise ValueError("Can only handle frequencies of Daily or shorter.")

    target_start = change_year(source.index[0], year_target)
    target_end = change_year(source.index.ts_right[-1], year_target)
    target_index = pd.date_range(
        target_start, target_end, freq=freq, closed="left", tz=tz
    )

    return frame_set_index(source, target_index, holiday_country)
