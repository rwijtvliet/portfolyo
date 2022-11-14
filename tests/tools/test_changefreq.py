import functools
from typing import Callable, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd
import pytest

from portfolyo import testing, tools

# def durationfn_yearlyfreq(ts):
#     return (366 if ts.year % 4 == 0 else 365) * 24


# def durationfn_quarterlyfreq(ts):
#     days = {1: 90, 4: 91, 7: 92, 10: 92}[ts.month]
#     if ts.month == 1 and ts.year % 4 == 0:
#         days += 1
#     hours = days * 24
#     if ts.tz == "Europe/Berlin":
#         if ts.month == 1:
#             hours -= 1
#         elif ts.month == 10:
#             hours += 1
#     return hours


# def durationfn_monthlyfreq(ts):
#     if ts.month in [4, 6, 9, 11]:
#         days = 30
#     elif ts.month == 2:
#         days = 29 if ts.year % 4 == 0 else 28
#     else:
#         days = 31
#     hours = days * 24
#     if ts.tz == "Europe/Berlin":
#         if ts.month == 3:
#             hours -= 1
#         elif ts.month == 10:
#             hours += 1
#     return hours


# def durationfn_dailyfreq(ts):
#     hours = 24
#     if ts.tz == "Europe/Berlin":
#         if ts.month == 3 and (
#             ts.year == 2020
#             and (ts.day == 29 and ts.hour < 6 or ts.day == 28 and ts.hour >= 6)
#             or ts.year == 2021
#             and (ts.day == 28 and ts.hour < 6 or ts.day == 27 and ts.hour >= 6)
#         ):
#             hours = 23
#         if ts.month == 10 and (
#             ts.year == 2020
#             and (ts.day == 25 and ts.hour < 6 or ts.day == 24 and ts.hour >= 6)
#             or ts.year == 2021
#             and (ts.day == 31 and ts.hour < 6 or ts.day == 30 and ts.hour >= 6)
#         ):
#             hours = 25
#     return hours


# def calc_durations(i: pd.DatetimeIndex) -> np.ndarray:
#     # Calculate the duration of each timestamp in an index. For indices between Nov 2019 and Feb 2022
#     freq = i.freq
#     if freq == "AS":
#         return i.map(durationfn_yearlyfreq)
#     if freq == "QS":
#         return i.map(durationfn_quarterlyfreq)
#     if freq == "MS":
#         return i.map(durationfn_monthlyfreq)
#     if freq == "D":
#         return i.map(durationfn_dailyfreq)
#     if freq == "H":
#         return i.map(lambda ts: 1)
#     if freq == "15T":
#         return i.map(lambda ts: 0.25)


@functools.lru_cache()
def idfn_midnight(
    longestfreq: str,
) -> Callable[[pd.DatetimeIndex], Iterable[Tuple[int]]]:
    def utc(i):
        if not i.tz:
            return np.repeat(0, len(i))
        else:
            return [ts.utcoffset().total_seconds() for ts in i]

    if longestfreq == "15T":
        return lambda i: pd.Index(zip(i.year, i.month, i.day, i.hour, i.minute, utc(i)))
    if longestfreq == "H":
        return lambda i: pd.Index(zip(i.year, i.month, i.day, i.hour, utc(i)))
    if longestfreq == "D":
        return lambda i: pd.Index(zip(i.year, i.month, i.day))
    if longestfreq == "MS":
        return lambda i: pd.Index(zip(i.year, i.month))
    if longestfreq == "QS":
        return lambda i: pd.Index(zip(i.year, i.quarter))
    if longestfreq == "AS":
        return lambda i: i.year


@functools.lru_cache()
def idfn_0600(longestfreq: str) -> Callable[[pd.DatetimeIndex], Iterable[Tuple[int]]]:
    def utc(i):
        if not i.tz:
            return np.repeat(0, len(i))
        else:
            return [ts.utcoffset().total_seconds() for ts in i]

    def year(i):
        return i.year - 1 * (i.hour < 6) * (i.month == 1) * (i.day == 1)

    def quarter(i):
        q = i.quarter - 1 * (i.hour < 6) * ((i.month - 1) % 3 == 0) * (i.day == 1)
        return (q - 1) % 4 + 1  # 1 - 1 = 4

    def month(i):
        m = i.month - 1 * (i.hour < 6) * (i.day == 1)
        return (m - 1) % 12 + 1  # 1 - 1 = 12

    def numofdaysinprevmonth(i):
        return month(i).map(
            lambda m: [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
        ) + (i.year == 2020) * (month(i) == 2)

    def day(i):
        d = i.day - 1 * (i.hour < 6)
        mask = d == 0
        return d + mask * numofdaysinprevmonth(i)

    if longestfreq == "15T":
        return lambda i: pd.Index(zip(i.year, i.month, i.day, i.hour, i.minute, utc(i)))
    if longestfreq == "H":
        return lambda i: pd.Index(zip(i.year, i.month, i.day, i.hour, utc(i)))
    if longestfreq == "D":
        return lambda i: pd.Index(zip(year(i), month(i), day(i)))
    if longestfreq == "MS":
        return lambda i: pd.Index(zip(year(i), month(i)))
    if longestfreq == "QS":
        return lambda i: pd.Index(zip(year(i), quarter(i)))
    if longestfreq == "AS":
        return lambda i: year(i)


@functools.lru_cache()
def idxs_and_mapping(
    startdate_shrt: str,
    starttime_shrt: str,
    end_shrt: str,
    freq_shrt: str,
    tz: str,
    freq_long: str,
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex, Mapping]:
    # Create mapping from the longer frequency to the shorter frequency.
    if tools.freq.longest(freq_long, freq_shrt) != freq_long:
        pytest.skip("This combination of frequencies is not tested here.")

    # Starting indices. Untrimmed; might have too many values.
    # Index with short frequency, as specified.
    i_shrt = pd.date_range(
        pd.Timestamp(f"{startdate_shrt} {starttime_shrt}", tz=tz),
        pd.Timestamp(end_shrt, tz=tz),
        freq=freq_shrt,
        inclusive="left",
    )
    # Get generous index with long frequency to cover at least i_shrt.
    i_long = pd.date_range(
        tools.floor.stamp(i_shrt[0], freq_long, 0, i_shrt[0].time()),
        tools.ceil.stamp(i_shrt[-1], freq_long, 0, i_shrt[0].time()),
        freq=freq_long,
    )

    # Durations.
    durs_long = tools.duration.index(i_long).pint.m.values
    durs_shrt = tools.duration.index(i_shrt).pint.m.values
    # Correspondence.
    if starttime_shrt == "00:00":
        idfn = idfn_midnight(freq_long)
    else:
        idfn = idfn_0600(freq_long)
    ids_long = idfn(i_long)
    ids_shrt = idfn(i_shrt)

    # Find mapping.
    mapping = {}  # ts_long : {ts_shrt : fraction of time in ts_long}
    for (ts_long, id_long, dur_long) in zip(i_long, ids_long, durs_long):
        mask = ids_shrt == id_long
        if dur_long > sum(durs_shrt[mask]):
            # Period not fully present; should not appear in final result.
            continue

        fracs_shrt = durs_shrt[mask] / dur_long
        tss_shrt = i_shrt[mask]
        mapping[ts_long] = {ts: f for ts, f in zip(tss_shrt, fracs_shrt)}

    # Find indices that are actually mapped onto each other.
    i_shrt_values = [ts for fractions in mapping.values() for ts in fractions.keys()]
    i_shrt_trimmed = pd.DatetimeIndex(i_shrt_values, freq=freq_shrt, tz=tz)
    i_long_trimmed = pd.DatetimeIndex(mapping.keys(), freq=freq_long, tz=tz)

    return i_long_trimmed, i_shrt_trimmed, i_shrt, mapping


def startdate(freq):
    if freq == "15T" or freq == "H" or freq == "D":
        return "2019-12-15"
    if freq == "MS":
        return "2019-12-01"
    if freq == "QS":
        return "2020-04-01"
    if freq == "AS":
        return "2020-01-01"


@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("freq_long", ["15T", "H", "D", "MS", "QS", "AS"])
@pytest.mark.parametrize("freq_shrt", ["15T", "H", "D", "MS", "QS", "AS"])
def test_downsample_index(freq_shrt: str, tz: str, freq_long: str, starttime: str):
    """Test downsampling of indices."""

    # mapping = ts_long: {ts_shrt: fraction}
    i_long, i_shrt, i_shrt_untrimmed, mapping = idxs_and_mapping(
        startdate(freq_shrt), starttime, "2022-02-15", freq_shrt, tz, freq_long
    )
    # i_long is the index with the longest frequency, so with the fewest values!

    result = tools.changefreq.index(i_shrt_untrimmed, freq_long)
    expected = i_long

    testing.assert_index_equal(result, expected)


@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("freq_long", ["15T", "H", "D", "MS", "QS", "AS"])
@pytest.mark.parametrize("freq_shrt", ["15T", "H", "D", "MS", "QS", "AS"])
def test_upsample_index(freq_shrt: str, tz: str, freq_long: str, starttime: str):
    """Test upsampling of indices."""

    # mapping = ts_long: {ts_shrt: fraction}
    i_long, i_shrt, i_shrt_untrimmed, mapping = idxs_and_mapping(
        startdate(freq_shrt), starttime, "2022-02-15", freq_shrt, tz, freq_long
    )
    # i_long is the index with the longest frequency, so with the fewest values!

    result = tools.changefreq.index(i_long, freq_shrt)
    expected = i_shrt

    testing.assert_index_equal(result, expected)


@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
@pytest.mark.parametrize("seriesordf", ["s", "s_unit", "df", "df_unit"])
@pytest.mark.parametrize("complexity", ["ones", "allnumbers"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("freq_long", ["15T", "H", "D", "MS", "QS", "AS"])
@pytest.mark.parametrize("freq_shrt", ["15T", "H", "D", "MS", "QS", "AS"])
def test_downsample_summable(
    seriesordf: str,
    freq_shrt: str,
    tz: str,
    freq_long: str,
    complexity: str,
    starttime: str,
):
    """Test downsampling of summable frames."""

    # mapping = ts_long: {ts_shrt: fraction}
    i_long, i_shrt, i_shrt_untrimmed, mapping = idxs_and_mapping(
        startdate(freq_shrt), starttime, "2022-02-15", freq_shrt, tz, freq_long
    )
    # i_long is the index with the longest frequency, so with the fewest values!

    # Get raw input and expected output.
    if complexity == "ones":
        s = pd.Series(1.0, i_shrt_untrimmed)
        s_expected = pd.Series(
            {ts_long: len(fractions) * 1.0 for ts_long, fractions in mapping.items()}
        )
    else:
        s = pd.Series(range(len(i_shrt_untrimmed)), i_shrt_untrimmed) * 1.0
        expecteddict = {}
        for ts_tg, fractions in mapping.items():
            value = s[fractions.keys()].sum()
            expecteddict[ts_tg] = value * 1.0  # make float
        s_expected = pd.Series(expecteddict)

    do_test("sum", seriesordf, s, s_expected, freq_long)


@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
@pytest.mark.parametrize("seriesordf", ["s", "s_unit", "df", "df_unit"])
@pytest.mark.parametrize("complexity", ["ones", "allnumbers"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("freq_long", ["15T", "H", "D", "MS", "QS", "AS"])
@pytest.mark.parametrize("freq_shrt", ["15T", "H", "D", "MS", "QS", "AS"])
def test_downsample_avgable(
    seriesordf: str,
    freq_shrt: str,
    tz: str,
    freq_long: str,
    complexity: str,
    starttime: str,
):
    """Test downsampling of averagable frames."""

    # mapping = ts_long: {ts_shrt: fraction}
    i_long, i_shrt, i_shrt_untrimmed, mapping = idxs_and_mapping(
        startdate(freq_shrt), starttime, "2022-02-15", freq_shrt, tz, freq_long
    )
    # i_long is the index with the longest frequency, so with the fewest values!

    # Get raw input and expected output.
    if complexity == "ones":
        s = pd.Series(1.0, i_shrt_untrimmed)
        s_expected = pd.Series(1.0, mapping.keys())
    else:
        s = pd.Series(range(len(i_shrt_untrimmed)), i_shrt_untrimmed) * 1.0
        s_expected = pd.Series(
            {
                ts_long: sum(s[ts_short] * frac for ts_short, frac in fractions.items())
                for ts_long, fractions in mapping.items()
            }
        )

    do_test("avg", seriesordf, s, s_expected, freq_long)


@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
@pytest.mark.parametrize("seriesordf", ["s", "s_unit", "df", "df_unit"])
@pytest.mark.parametrize("complexity", ["ones", "allnumbers"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("freq_shrt", ["15T", "H", "D", "MS", "QS", "AS"])
@pytest.mark.parametrize("freq_long", ["15T", "H", "D", "MS", "QS", "AS"])
def test_upsample_avgable(
    seriesordf: str,
    freq_long: str,
    tz: str,
    freq_shrt: str,
    complexity: str,
    starttime: str,
):
    """Test upsampling of averagable frames."""

    # mapping = ts_long: {ts_shrt: fraction}
    i_long, i_shrt, _, mapping = idxs_and_mapping(
        startdate(freq_shrt), starttime, "2022-02-15", freq_shrt, tz, freq_long
    )
    # i_long is the index with the longest frequency, so with the fewest values!

    # Get raw input and expected output.
    if complexity == "ones":
        s = pd.Series(1.0, i_long)
        s_expected = pd.Series(1.0, i_shrt)
    else:
        s = pd.Series(range(len(i_long)), i_long) * 1.0
        s_expected = pd.Series(
            {
                ts_shrt: s[ts_long]
                for ts_long, fractions in mapping.items()
                for ts_shrt in fractions.keys()
            }
        )

    do_test("avg", seriesordf, s, s_expected, freq_shrt)


@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
@pytest.mark.parametrize("seriesordf", ["s", "s_unit", "df", "df_unit"])
@pytest.mark.parametrize("complexity", ["ones", "allnumbers"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("freq_shrt", ["15T", "H", "D", "MS", "QS", "AS"])
@pytest.mark.parametrize("freq_long", ["15T", "H", "D", "MS", "QS", "AS"])
def test_upsample_summable(
    seriesordf: str,
    freq_long: str,
    tz: str,
    freq_shrt: str,
    complexity: str,
    starttime: str,
):
    """Test upsampling of summable frames."""

    # mapping = ts_long: {ts_shrt: fraction}
    i_long, i_shrt, _, mapping = idxs_and_mapping(
        startdate(freq_shrt), starttime, "2022-02-15", freq_shrt, tz, freq_long
    )
    # i_long is the index with the longest frequency, so with the fewest values!

    # Get raw input and expected output.
    if complexity == "ones":
        s = pd.Series(1.0, i_long)
        s_expected = pd.Series(
            {
                ts_shrt: frac
                for fractions in mapping.values()
                for ts_shrt, frac in fractions.items()
            }
        )
    else:
        s = pd.Series(range(len(i_long)), i_long) * 1.0
        s_expected = pd.Series(
            {
                ts_shrt: s[ts_long] * frac
                for ts_long, fractions in mapping.items()
                for ts_shrt, frac in fractions.items()
            }
        )

    do_test("sum", seriesordf, s, s_expected, freq_shrt)


def do_test(
    avgorsum: str, seriesordf: str, s: pd.Series, s_expected: pd.Series, tg_freq: str
):
    if avgorsum == "avg":
        fn = tools.changefreq.averagable
    else:
        fn = tools.changefreq.summable

    s_expected.index.freq = tg_freq
    if seriesordf.endswith("_unit"):
        s = s.astype("pint[MW]")
        s_expected = s_expected.astype("pint[MW]")

    if seriesordf.startswith("s"):
        fr = s
        expected = s_expected
        result = fn(fr, tg_freq)
        testing.assert_series_equal(result, expected)
    else:
        fr = pd.DataFrame({"a": s})
        expected = pd.DataFrame({"a": s_expected})
        result = fn(fr, tg_freq)
        testing.assert_frame_equal(result, expected)
