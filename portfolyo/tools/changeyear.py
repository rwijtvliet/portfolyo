"""Map series with quarterhourly, hourly, or daily values onto another index or year,
trying to align weekdays, holidays, and dst-changeover days. Always takes values from
same calender month (but different year)."""

# Three difficulties when changing the year:
# . Feb 29 might be present in one of the years but not in the other.
# . Daylight-savings time will likely start and end at a different day.
# . Weekdays and holidays are not at the same date.

from typing import Callable, Union

import holidays
import pandas as pd

from . import changefreq as tools_changefreq
from . import freq as tools_freq
from . import right as tools_right
from . import unit as tools_unit

docstringliteral_notes = """
* Function is meant for data that spans full months. Using partial months may lead
  to unexpected behavior.
* No values are recalculated; they are simply 'reshuffled' to obey the rules below.
* The mapping is always between the same calendar month (e.g., April to April).
* The mapping is between fixed year-month pairs (e.g., all values for 2025-April are
  taken from 2020-April). If a year-month pair exists in both source and target, it is
  mapped onto itself without modification.
* When mapping daily (or shorter) values, the mapping is always between periods of equal
  duration (i.e. from a 24h-day to another 24h-day, or from a 23h-day to another 23h-day)
  and equal 'type' (i.e., from a Saturday to another Saturday). For holidays, the same-
  named holiday is taken; if not found, another holiday or Sunday in that month is used.
* When mapping monthly (or longer) values, the mapping is more trivial. E.g., a leapyear
  Feb is mapped to a non-leapyear-Feb, and the number of workdays or holidays in a month
  is also disregarded.
* Repetitions are minimized.
* There is a subjective trade-off to be made if the mapping is not perfect. One example
  of a trade-of: a day at the end of the target month might be mapped onto a day at the
  beginning of the source month or onto a day in the source month that was already used.
  Another trade-of: a Wednesday at the end of the target month could be mapped to a
  Wednesday at the beginning of the source month or to a Monday at the end of the source
  month. (In both cases, the first option is implemented.)"""


def additional_notes(fn: Callable) -> Callable:
    fn.__doc__ += f"\n{docstringliteral_notes}"
    return fn


def characterize_index(
    idx: pd.DatetimeIndex, holiday_country: str = "DE"
) -> pd.DataFrame:
    """Characterise an index with daily frequency.

    Parameters
    ----------
    idx : pd.DatetimeIndex
    holiday_country : str, optional (default: "DE")
        Country or region for which to assume the holidays. E.g. 'DE' (Germany), 'NL'
        (Netherlands), or 'USA'. See ``holidays.list_supported_countries()`` for allowed
        values.

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
        char["dst_change"] = abs(
            char.index.duration - tools_unit.Q_(24, "h")
        ) > tools_unit.Q_(0.1, "h")
    return char


@additional_notes
def map_index_to_index(
    idx_source: pd.DatetimeIndex,
    idx_target: pd.DatetimeIndex,
    holiday_country: str = None,
) -> pd.Series:
    """Map a source index onto a target index according to certain rules (See Notes).

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
    * If the target index spans more months than the source, some source months are mapped
      from multiple times. If it spans fewer months, not all source months are mapped from.
    """
    if (tz1 := idx_source.tz) != (tz2 := idx_target.tz):
        raise ValueError(f"Indices must have same timezone, got {tz1} and {tz2}.")
    if (fr1 := idx_source.freq) != (fr2 := idx_target.freq):
        raise ValueError(f"Indices must have same frequency, got {fr1} and {fr2}.")

    if tools_freq.shortest(idx_source.freq, "MS") == "MS":
        return _map_index_to_index_monthlyandlonger(idx_source, idx_target)

    elif idx_source.freq == "D":
        return _map_index_to_index_daily(idx_source, idx_target, holiday_country)

    else:  # (quarter)hourly
        return _map_index_to_index_hourlyandshorter(
            idx_source, idx_target, holiday_country
        )


def _map_index_to_index_monthlyandlonger(
    idx_source: pd.DatetimeIndex, idx_target: pd.DatetimeIndex
) -> pd.Series:
    source = pd.DataFrame({"in_target_count": 0}, idx_source)
    source["yy"] = idx_source.map(lambda ts: ts.year)
    source["mm"] = idx_source.map(lambda ts: ts.month)  # also works for QS and AS freq
    target = pd.DataFrame({"source_month": None}, idx_target)

    def set_month(candidates: pd.DataFrame, target_month: pd.Timestamp) -> None:
        if len(candidates) == 0:
            return
        if len(candidates) == 1:
            source_month = candidates.index[0]
        else:  # pick one
            source_month = candidates.sort_values(by=["in_target_count", "yy"]).index[0]
        target.loc[target_month, "source_month"] = source_month
        source.loc[source_month, "in_target_count"] += 1

    # First match the years that appear in both indices. (No mapping needed.)
    for month in target.itertuples():
        candidates = source[
            (source.mm == month.Index.month) & (source.yy == month.Index.year)
        ]
        set_month(candidates, month.Index)

    # Then map the remaining years.
    for month in target[target.source_month.isna()].itertuples():
        candidates = source[source.mm == month.Index.month]
        set_month(candidates, month.Index)

    if any(target.source_month.isna()):
        not_found = target[target.source_month.isna()]
        raise ValueError(
            f"Did not find correspondence for the following months: {not_found.index}."
        )

    return target.source_month


def _map_index_to_index_daily(
    idx_source: pd.DatetimeIndex,
    idx_target: pd.DatetimeIndex,
    holiday_country: str = None,
) -> pd.Series:

    # Split into months.
    idx_source_by_m = {m: s.index for m, s in pd.Series(0, idx_source).resample("MS")}
    idx_target_by_m = {m: s.index for m, s in pd.Series(0, idx_target).resample("MS")}

    # Mapping on month-level.
    mapp_m = _map_index_to_index_monthlyandlonger(
        pd.DatetimeIndex(idx_source_by_m.keys(), freq="MS"),
        pd.DatetimeIndex(idx_target_by_m.keys(), freq="MS"),
    )

    # Mapping on daily-level.
    mapp_d_series = []
    for target_m, idx_target_partial in idx_target_by_m.items():
        idx_source_partial = idx_source_by_m[mapp_m[target_m]]
        mapp_d = _map_index_to_index_daily_samemonth(
            idx_source_partial, idx_target_partial, holiday_country
        )
        mapp_d_series.append(mapp_d)

    mapping = pd.concat(mapp_d_series)
    mapping.index.freq = "D"
    return mapping


def _map_index_to_index_daily_samemonth(
    idx_source: pd.DatetimeIndex, idx_target: pd.DatetimeIndex, holiday_country: str
) -> pd.Series:

    months = set([*[ts.month for ts in idx_source], *[ts.month for ts in idx_target]])
    if len(months) > 1:
        raise ValueError(
            f"Indices must contain timestamps of only one calender month; found {months}."
        )

    source = characterize_index(idx_source, holiday_country)
    target = characterize_index(idx_target, holiday_country)
    source["in_target_count"] = 0  # keep track of which days have been used.
    target["source_day"] = None

    def set_day(candidates: pd.DataFrame, target_day: pd.Timestamp) -> None:
        if len(candidates) == 0:
            return
        if len(candidates) == 1:
            source_day = candidates.index[0]
        else:  # find 'nearest' within month under consideration
            candidates = candidates.copy()  # to stop complaining on next line
            candidates["dist"] = abs(candidates.dd - target_day.day)
            source_day = candidates.sort_values(by=["in_target_count", "dist"]).index[0]
        target.loc[target_day, "source_day"] = source_day
        source.loc[source_day, "in_target_count"] += 1

    # Map the DST days.
    for day in target[target.dst_change].itertuples():
        # Find dst-changing day in same month.
        candidates = source[source.dst_change]
        set_day(candidates, day.Index)

    # Map the non-holidays.
    for day in target[~target.dst_change & (target.holiday == "")].itertuples():
        # Find the same non-holiday weekday in the same month, while minimizing repetition.
        candidates = source[
            ~source.dst_change
            & (source.holiday == "")
            & (source.isoweekday == day.isoweekday)
        ]
        set_day(candidates, day.Index)

    # Map the holidays with same name.
    for day in target[~target.dst_change & (target.holiday != "")].itertuples():
        # Find same-named holiday.
        candidates = source[~source.dst_change & (source.holiday == day.holiday)]
        set_day(candidates, day.Index)

    # Map holidays that were not yet found.
    for day in target[
        ~target.dst_change & (target.holiday != "") & target.source_day.isna()
    ].itertuples():
        # Find any other holiday or any Sunday.
        candidates = source[
            ~source.dst_change & ((source.holiday != "") | (source.isoweekday == 7))
        ]
        set_day(candidates, day.Index)

    if any(target.source_day.isna()):
        not_found = target[target.source_day.isna()]
        raise ValueError(
            f"Did not find correspondence for the following days: {not_found.index}."
        )

    return target.source_day


def _map_index_to_index_hourlyandshorter(
    idx_source: pd.DatetimeIndex,
    idx_target: pd.DatetimeIndex,
    holiday_country: str = None,
):
    # Do mapping on day-level.
    idx_target_d = tools_changefreq.index(idx_target, "D")
    idx_source_d = tools_changefreq.index(idx_source, "D")
    mapp_d = _map_index_to_index_daily(idx_source_d, idx_target_d, holiday_country)

    # Split timestamps in 'day' and 'offset'.
    idx_target_part1 = idx_target.floor("D")
    idx_target_part2 = idx_target - idx_target_part1

    return pd.Series(mapp_d[idx_target_part1].values + idx_target_part2, idx_target)


def index_with_year(idx_source: pd.DatetimeIndex, target_year: int) -> pd.DatetimeIndex:
    """Create new index that spans the same number of months as the original, but starting
    in a different year.

    Parameters
    ----------
    idx_source : pd.DatetimeIndex
        Source index.
    target_year : int
        Year in which to start the index. If the source index spans multiple years, the
        same number of years is used in the return index. (The provided year is used for
        the first timestamp.)

    Returns
    -------
    pd.DatetimeIndex
    """

    def change_year(ts: pd.Timestamp, y: int) -> pd.Timestamp:
        return pd.Timestamp(
            year=y,
            month=ts.month,
            day=ts.day,
            hour=ts.hour,
            minute=ts.minute,
            tz=ts.tz,
        )

    freq, tz = idx_source.freq, idx_source.tz
    start, end = idx_source[0], tools_right.index(idx_source)[-1]
    delta_years = end.year - start.year

    target_start = change_year(start, target_year)
    target_end = change_year(end, target_year + delta_years)
    return pd.date_range(target_start, target_end, freq=freq, inclusive="left", tz=tz)


@additional_notes
def map_index_to_year(
    idx_source: pd.DatetimeIndex, target_year: int, holiday_country: str
) -> pd.Series:
    """Map a source index onto a target year according to certain rules (See Notes).

    Parameters
    ----------
    idx_source : pd.DatetimeIndex
        Index of source values.
    target_year : int
        Year onto which to map the data.
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
    * The ``target_year`` is used for the first timestamp; the returned frame spans the
      same number of months as the source.
    """
    idx_target = index_with_year(idx_source, target_year)
    return map_index_to_index(idx_source, idx_target, holiday_country)


# ---


@additional_notes
def map_frame_to_index(
    source: Union[pd.Series, pd.DataFrame],
    idx_target: pd.DatetimeIndex,
    holiday_country: str = None,
) -> Union[pd.Series, pd.DataFrame]:
    """Transfer the data in a Series or DataFrame to a hypothetical other index
    according to certain rules (see Notes). What would the data have looked like if it
    had occured in a different year?

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
    * This is a lossy operation.
    * No recalculation takes place! This special care must be taken when mapping monthly
      (or longer values), as there a mapping between unequal-length periods might occur
      due to leapday and non-leapday Februaries.
    """
    mapping = map_index_to_index(source.index, idx_target, holiday_country)

    if isinstance(source, pd.Series):
        return source[mapping].set_axis(mapping.index)
    else:
        series = {col: s[mapping].values for col, s in source.items()}
        return pd.DataFrame(series, mapping.index)


@additional_notes
def map_frame_to_year(
    source: Union[pd.Series, pd.DataFrame],
    target_year: int,
    holiday_country: str = None,
) -> Union[pd.Series, pd.DataFrame]:
    """Transfer the data in a Series or DataFrame to a hypothetical other year
    according to certain rules (see Notes). What would the data have looked like if it
    had occured in a different year?

    Parameters
    ----------
    source: pd.Series or pd.DataFrame
        Source values.
    target_year : int
        Year onto which to map the data.
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
    * The ``target_year`` is used for the first timestamp; the returned frame spans the
      same number of months as the source.
    * This is a lossy operation.
    * No recalculation takes place! This special care must be taken when mapping monthly
      (or longer values), as there a mapping between unequal-length periods might occur
      due to leapday and non-leapday Februaries.
    """
    mapping = map_index_to_year(source.index, target_year, holiday_country)

    if isinstance(source, pd.Series):
        return source[mapping].set_axis(mapping.index)
    else:
        series = {col: s[mapping].values for col, s in source.items()}
        return pd.DataFrame(series, mapping.index)
