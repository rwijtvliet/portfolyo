import functools
from typing import Callable, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd
import pytest

from portfolyo import testing, tools


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
    df_long = pd.DataFrame({"dur_long": durs_long, "ts_long": i_long}, ids_long)
    df_shrt = pd.DataFrame({"dur_shrt": durs_shrt, "ts_shrt": i_shrt}, ids_shrt)
    df_mapping = df_long.merge(df_shrt, left_index=True, right_index=True)
    df_mapping["fraction"] = df_mapping.dur_shrt / df_mapping.dur_long
    df_mapping = df_mapping.drop(columns=["dur_long", "dur_shrt"])
    # . Reject periods that are not fully present.
    reject = df_mapping.fraction.groupby(df_mapping.index).sum() < 0.99
    df_mapping = df_mapping.drop(reject[reject].index)
    # . Put in nested dictionary.
    df_mapping = df_mapping.set_index("ts_long")
    # mapping = {}
    # for ts_long, df in df_mapping.groupby(df_mapping.index):
    #     mapping[ts_long] = df.set_index("ts_shrt").fraction.to_dict()

    # mapping = {}  # ts_long : {ts_shrt : fraction of time in ts_long}
    # for (ts_long, id_long, dur_long) in zip(i_long, ids_long, durs_long):
    #     mask = ids_shrt == id_long
    #     if dur_long > sum(durs_shrt[mask]) + 0.01:  # +0.01 to avoid rounding errors
    #         # Period not fully present; should not appear in final result.
    #         continue

    #     fracs_shrt = durs_shrt[mask] / dur_long
    #     tss_shrt = i_shrt[mask]
    #     mapping[ts_long] = {ts: f for ts, f in zip(tss_shrt, fracs_shrt)}

    # Find indices that are actually mapped onto each other.
    i_shrt_trimmed = pd.DatetimeIndex(
        df_mapping["ts_shrt"].sort_values(), freq=freq_shrt, tz=tz
    ).rename(None)
    i_long_trimmed = pd.DatetimeIndex(
        df_mapping.index.unique().sort_values(), freq=freq_long, tz=tz
    ).rename(None)

    return i_long_trimmed, i_shrt_trimmed, i_shrt, df_mapping


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

    if freq_long == freq_shrt:
        pytest.skip("Same frequency already tested when downsampling.")

    # mapping = ts_long: {ts_shrt: fraction}
    i_long, i_shrt, i_shrt_untrimmed, mapping = idxs_and_mapping(
        startdate(freq_shrt), starttime, "2022-02-15", freq_shrt, tz, freq_long
    )
    # i_long is the index with the longest frequency, so with the fewest values!

    result = tools.changefreq.index(i_long, freq_shrt)
    expected = i_shrt

    testing.assert_index_equal(result, expected)


@pytest.mark.only_on_pr
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
        s_expected = mapping["ts_shrt"].groupby(mapping.index).count() * 1.0
    else:
        mapping["value"] = range(len(mapping))
        s = mapping.set_index("ts_shrt")["value"] * 1.0
        s_expected = mapping["value"].groupby(mapping.index).sum() * 1.0

    do_test("sum", seriesordf, s, s_expected, freq_shrt, freq_long)


@pytest.mark.only_on_pr
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
        s_expected = pd.Series(1.0, i_long)
    else:
        mapping["value"] = range(len(mapping))
        s = mapping.set_index("ts_shrt")["value"] * 1.0
        weighted = mapping.value * mapping.fraction
        s_expected = weighted.groupby(weighted.index).sum() * 1.0

    do_test("avg", seriesordf, s, s_expected, freq_shrt, freq_long)


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

    if freq_long == freq_shrt:
        pytest.skip("Same frequency already tested when downsampling.")

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
        mapping["value"] = s[mapping.index]
        s_expected = mapping.set_index("ts_shrt")["value"]

    do_test("avg", seriesordf, s, s_expected, freq_long, freq_shrt)


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

    if freq_long == freq_shrt:
        pytest.skip("Same frequency already tested when downsampling.")

    # mapping = ts_long: {ts_shrt: fraction}
    i_long, i_shrt, _, mapping = idxs_and_mapping(
        startdate(freq_shrt), starttime, "2022-02-15", freq_shrt, tz, freq_long
    )
    # i_long is the index with the longest frequency, so with the fewest values!

    # Get raw input and expected output.
    if complexity == "ones":
        s = pd.Series(1.0, i_long, name="testseries")
        s_expected = mapping.set_index("ts_shrt")["fraction"]
    else:
        s = pd.Series(range(len(i_long)), i_long) * 1.0
        mapping["value"] = s[mapping.index]
        mapping["multiplied"] = mapping.value * mapping.fraction
        s_expected = mapping.set_index("ts_shrt")["multiplied"]

    do_test("sum", seriesordf, s, s_expected, freq_long, freq_shrt)


def do_test(
    avgorsum: str,
    seriesordf: str,
    s: pd.Series,
    s_expected: pd.Series,
    freq_source: str,
    freq_target: str,
):
    if avgorsum == "avg":
        fn = tools.changefreq.averagable
    else:
        fn = tools.changefreq.summable

    s = s.sort_index()
    s.name = "testseries"
    s.index.name = None
    s.index.freq = freq_source

    s_expected = s_expected.sort_index()
    s_expected.name = "testseries"
    s_expected.index.name = None
    s_expected.index.freq = freq_target

    if seriesordf.endswith("_unit"):
        s = s.astype("pint[MW]")
        s_expected = s_expected.astype("pint[MW]")

    if seriesordf.startswith("s"):
        fr = s
        expected = s_expected
        result = fn(fr, freq_target)
        testing.assert_series_equal(result, expected)
    else:
        fr = pd.DataFrame({"a": s})
        expected = pd.DataFrame({"a": s_expected})
        result = fn(fr, freq_target)
        testing.assert_frame_equal(result, expected)
