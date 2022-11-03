import functools
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
import pytest
from portfolyo import testing, tools

freqs_small_to_large = ["T", "5T", "15T", "30T", "H", "2H", "D", "MS", "QS", "AS"]

# No timezone.

i1 = pd.date_range("2020-01-15 12:00", "2020-01-18 12:00", freq="H", inclusive="left")
s1 = pd.Series(range(len(i1)), i1)

s1_15t_sum = pd.Series(
    [v2 for v in range(len(i1)) for v2 in [v / 4.0] * 4],
    pd.date_range("2020-01-15 12:00", "2020-01-18 12:00", freq="15T", inclusive="left"),
)
s1_15t_avg = pd.Series(
    [v2 for v in range(len(i1)) for v2 in [v * 1.0] * 4],
    pd.date_range("2020-01-15 12:00", "2020-01-18 12:00", freq="15T", inclusive="left"),
)

s1_d_sum = pd.Series([564.0, 1140], pd.date_range("2020-01-16", freq="D", periods=2))
s1_d_avg = pd.Series([23.5, 47.5], pd.date_range("2020-01-16", freq="D", periods=2))

s1_m_sum = s1_m_avg = pd.Series(
    [], pd.date_range("2020", periods=0, freq="MS"), dtype=float
)
s1_q_sum = s1_q_avg = pd.Series(
    [], pd.date_range("2020", periods=0, freq="QS"), dtype=float
)
s1_a_sum = s1_a_avg = pd.Series(
    [], pd.date_range("2020", periods=0, freq="AS"), dtype=float
)

# Timezone, no DST.


def withtz(sin):
    sout = sin.tz_localize("Europe/Berlin")
    sout.index.freq = sin.index.freq
    return sout


(
    s2,
    s2_15t_sum,
    s2_15t_avg,
    s2_d_sum,
    s2_d_avg,
    s2_m_sum,
    s2_m_avg,
    s2_q_sum,
    s2_q_avg,
    s2_a_sum,
    s2_a_avg,
) = (
    withtz(s)
    for s in (
        s1,
        s1_15t_sum,
        s1_15t_avg,
        s1_d_sum,
        s1_d_avg,
        s1_m_sum,
        s1_m_avg,
        s1_q_sum,
        s1_q_avg,
        s1_a_sum,
        s1_a_avg,
    )
)

# Timezone, start of DST.

i3_start, i3_end = "2020-03-28 12:00", "2020-03-31 12:00"
tz = "Europe/Berlin"
i3 = pd.date_range(i3_start, i3_end, freq="H", inclusive="left", tz=tz)
s3 = pd.Series(range(len(i3)), i3)

s3_15t_sum = pd.Series(
    [v2 for v in range(len(i3)) for v2 in [v / 4.0] * 4],
    pd.date_range(i3_start, i3_end, freq="15T", inclusive="left", tz=tz),
)
s3_15t_avg = pd.Series(
    [v2 for v in range(len(i3)) for v2 in [v * 1.0] * 4],
    pd.date_range(i3_start, i3_end, freq="15T", inclusive="left", tz=tz),
)

s3_d_sum = pd.Series(
    [529.0, 1116], pd.date_range("2020-03-29", freq="D", periods=2, tz=tz)
)
s3_d_avg = pd.Series(
    [23.0, 46.5], pd.date_range("2020-03-29", freq="D", periods=2, tz=tz)
)

s3_m_sum = s3_m_avg = pd.Series(
    [], pd.date_range("2020", periods=0, freq="MS", tz=tz), dtype=float
)
s3_q_sum = s3_q_avg = pd.Series(
    [], pd.date_range("2020", periods=0, freq="QS", tz=tz), dtype=float
)
s3_a_sum = s3_a_avg = pd.Series(
    [], pd.date_range("2020", periods=0, freq="AS", tz=tz), dtype=float
)

# Timezone, end of DST.

i4_start, i4_end = "2020-10-24 12:00", "2020-10-27 12:00"
i4 = pd.date_range(i4_start, i4_end, freq="H", inclusive="left", tz=tz)
s4 = pd.Series(range(len(i4)), i4)

s4_15t_sum = pd.Series(
    [v2 for v in range(len(i4)) for v2 in [v / 4.0] * 4],
    pd.date_range(i4_start, i4_end, freq="15T", inclusive="left", tz=tz),
)
s4_15t_avg = pd.Series(
    [v2 for v in range(len(i4)) for v2 in [v * 1.0] * 4],
    pd.date_range(i4_start, i4_end, freq="15T", inclusive="left", tz=tz),
)

s4_d_sum = pd.Series(
    [600.0, 1164], pd.date_range("2020-10-25", freq="D", periods=2, tz=tz)
)
s4_d_avg = pd.Series(
    [24.0, 48.5], pd.date_range("2020-10-25", freq="D", periods=2, tz=tz)
)

s4_m_sum = s4_m_avg = pd.Series(
    [], pd.date_range("2020", periods=0, freq="MS", tz=tz), dtype=float
)
s4_q_sum = s4_q_avg = pd.Series(
    [], pd.date_range("2020", periods=0, freq="QS", tz=tz), dtype=float
)
s4_a_sum = s4_a_avg = pd.Series(
    [], pd.date_range("2020", periods=0, freq="AS", tz=tz), dtype=float
)

# Months (= unequal lengths) as starting point.

i5 = pd.date_range("2020-04", "2021-06", freq="MS", inclusive="left")
s5 = pd.Series(range(len(i5)), i5)

daycount = [30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31]
s5_d_sum = pd.Series(
    [v2 for v, l in zip(range(len(i5)), daycount) for v2 in [v / l] * l],
    pd.date_range("2020-04-01", "2021-06-01", freq="D", inclusive="left"),
    dtype=float,
)
s5_d_avg = pd.Series(
    [v2 for v, l in zip(range(len(i5)), daycount) for v2 in [v] * l],
    pd.date_range("2020-04-01", "2021-06-01", freq="D", inclusive="left"),
    dtype=float,
)

s5_q_sum = pd.Series(
    [3.0, 12, 21, 30],
    pd.date_range("2020-04-01", "2021-04-01", freq="QS", inclusive="left"),
)
s5_q_avg = pd.Series(
    [1.0, 3.989130434782609, 7.0, 10.0],
    pd.date_range("2020-04-01", "2021-04-01", freq="QS", inclusive="left"),
)

s5_a_sum = s5_a_avg = pd.Series(
    [], pd.date_range("2020", periods=0, freq="AS"), dtype=float
)

# Months (= unequal lengths) as starting point, with DST

i6 = pd.date_range("2020-04", "2021-06", freq="MS", inclusive="left", tz=tz)
s6 = pd.Series(range(len(i6)), i6)

s6_d_avg = pd.Series(
    [v2 for v, l in zip(range(len(i6)), daycount) for v2 in [v] * l],
    pd.date_range("2020-04-01", "2021-06-01", freq="D", inclusive="left", tz=tz),
    dtype=float,
)
hourcount = [720, 744, 720, 744, 744, 720, 745, 720, 744, 744, 672, 743, 720, 744]
s6_d_sum_values = []
for m, (value, days, monthlen) in enumerate(zip(range(len(i6)), daycount, hourcount)):
    for d in range(days):
        if m == 6 and d == 24:  # 2020-oct-25
            daylen = 25
        elif m == 11 and d == 27:  # 2021-mar-28
            daylen = 23
        else:
            daylen = 24
        s6_d_sum_values.append(value * daylen / monthlen)
s6_d_sum = pd.Series(
    s6_d_sum_values,
    pd.date_range("2020-04-01", "2021-06-01", freq="D", inclusive="left", tz=tz),
    dtype=float,
)

s6_q_sum = pd.Series(
    [3.0, 12, 21, 30],
    pd.date_range("2020-04-01", "2021-04-01", freq="QS", inclusive="left", tz=tz),
)
s6_q_avg = pd.Series(
    [1.0, 3.989130434782609, 6.999547306473517, 9.999536822603057],
    pd.date_range("2020-04-01", "2021-04-01", freq="QS", inclusive="left", tz=tz),
)

s6_a_sum = s6_a_avg = pd.Series(
    [], pd.date_range("2020", periods=0, freq="AS", tz=tz), dtype=float
)


@pytest.mark.parametrize(
    ("s", "freq", "expected"),
    [
        (s1, "15T", s1_15t_sum),
        (s1, "D", s1_d_sum),
        (s1, "MS", s1_m_sum),
        (s1, "QS", s1_q_sum),
        (s1, "AS", s1_a_sum),
        (s2, "15T", s2_15t_sum),
        (s2, "D", s2_d_sum),
        (s2, "MS", s2_m_sum),
        (s2, "QS", s2_q_sum),
        (s2, "AS", s2_a_sum),
        (s3, "15T", s3_15t_sum),
        (s3, "D", s3_d_sum),
        (s3, "MS", s3_m_sum),
        (s3, "QS", s3_q_sum),
        (s3, "AS", s3_a_sum),
        (s4, "15T", s4_15t_sum),
        (s4, "D", s4_d_sum),
        (s4, "MS", s4_m_sum),
        (s4, "QS", s4_q_sum),
        (s4, "AS", s4_a_sum),
        (s5, "D", s5_d_sum),
        (s5, "QS", s5_q_sum),
        (s5, "AS", s5_a_sum),
        (s6, "D", s6_d_sum),
        (s6, "QS", s6_q_sum),
        (s6, "AS", s6_a_sum),
    ],
)
def test_resample_summable(s, freq, expected):
    """Test if resampling works as expected."""
    result = tools.changefreq.summable(s, freq)
    testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ("s", "freq", "expected"),
    [
        (s1, "15T", s1_15t_avg),
        (s1, "D", s1_d_avg),
        (s1, "MS", s1_m_avg),
        (s1, "QS", s1_q_avg),
        (s1, "AS", s1_a_avg),
        (s2, "15T", s2_15t_avg),
        (s2, "D", s2_d_avg),
        (s2, "MS", s2_m_avg),
        (s2, "QS", s2_q_avg),
        (s2, "AS", s2_a_avg),
        (s3, "15T", s3_15t_avg),
        (s3, "D", s3_d_avg),
        (s3, "MS", s3_m_avg),
        (s3, "QS", s3_q_avg),
        (s3, "AS", s3_a_avg),
        (s4, "15T", s4_15t_avg),
        (s4, "D", s4_d_avg),
        (s4, "MS", s4_m_avg),
        (s4, "QS", s4_q_avg),
        (s4, "AS", s4_a_avg),
        (s5, "D", s5_d_avg),
        (s5, "QS", s5_q_avg),
        (s5, "AS", s5_a_avg),
        (s6, "D", s6_d_avg),
        (s6, "QS", s6_q_avg),
        (s6, "AS", s6_a_avg),
    ],
)
def test_resample_avgable(s, freq, expected):
    """Test if resampling works as expected."""
    result = tools.changefreq.averagable(s, freq)
    testing.assert_series_equal(result, expected)


# Using gas days instead of calendar days.

# . no tz
i7 = pd.date_range("2020-01-15 12:00", "2020-01-18 12:00", freq="H", inclusive="left")
s7 = pd.Series(range(len(i7)), i7)
s7_d_sum = pd.Series([708.0, 1284], pd.date_range("2020-01-16", freq="D", periods=2))
s7_d_avg = pd.Series([29.5, 53.5], pd.date_range("2020-01-16", freq="D", periods=2))

# . start of dst
i8 = pd.date_range(
    "2020-03-26 12:00", "2020-03-30 12:00", freq="H", inclusive="left", tz=tz
)
s8 = pd.Series(range(len(i8)), i8)
s8_d_sum = pd.Series(
    [708.0, 1219, 1836], pd.date_range("2020-03-27", freq="D", periods=3, tz=tz)
)
s8_d_avg = pd.Series(
    [29.5, 53, 76.5], pd.date_range("2020-03-27", freq="D", periods=3, tz=tz)
)

# . end of dst
i9 = pd.date_range(
    "2020-10-22 12:00", "2020-10-26 12:00", freq="H", inclusive="left", tz=tz
)
s9 = pd.Series(range(len(i9)), i9)
s9_d_sum = pd.Series(
    [708.0, 1350, 1884], pd.date_range("2020-10-23", freq="D", periods=3, tz=tz)
)
s9_d_avg = pd.Series(
    [29.5, 54, 78.5], pd.date_range("2020-10-23", freq="D", periods=3, tz=tz)
)


# @pytest.mark.parametrize(
#     ("s", "freq", "grouper", "expected"),
#     [
#         (s7, "D", tools.stamp.gasday_de, s7_d_sum),
#         # (s8, "D", tools.stamp.gasday_de, s8_d_sum),
#         # (s9, "D", tools.stamp.gasday_de, s9_d_sum),
#     ],
# )
# def test_resample_summable_custom(s, freq, grouper, expected):
#     """Test if resampling also works when using custom grouper function."""
#     result = tools_bak.changefreq.summable(s, freq, grouper)
#     testing.assert_series_equal(result, expected)


# @pytest.mark.parametrize(
#     ("s", "freq", "grouper", "expected"),
#     [
#         (s7, "D", tools_bak.stamp.gasday_de, s7_d_avg),
#         # (s8, "D", tools.stamp.gasday_de, s8_d_avg),
#         # (s9, "D", tools.stamp.gasday_de, s9_d_avg),
#     ],
# )
# def test_resample_avgable_custom(s, freq, grouper, expected):
#     """Test if resampling also works when using custom grouper function."""
#     result = tools_bak.changefreq.averagable(s, freq, grouper)
#     testing.assert_series_equal(result, expected)


# @pytest.fixture(params=freqs_small_to_large)
# def freq(request):
#     return request.param


# freq1 = freq2 = freq


# @functools.lru_cache()
# def aggdata():
#     # Sample data
#     i_15T = pd.date_range(
#         "2020", "2022", freq="15T", tz="Europe/Berlin", inclusive="left"
#     )

#     def value_func(mean, ampl_a, ampl_m, ampl_d):
#         start = pd.Timestamp("2020", tz="Europe/Berlin")
#         end = pd.Timestamp("2021", tz="Europe/Berlin")

#         def value(ts):
#             angle = 2 * np.pi * (ts - start) / (end - start)
#             return (
#                 mean
#                 + ampl_a * np.cos(angle + np.pi / 12)
#                 + ampl_m * np.cos(angle * 12)
#                 + ampl_d * np.cos(angle * 365)
#             )

#         return np.vectorize(value)  # make sure it accepts arrays

#     f = value_func(500, 300, 150, 50)
#     values = np.random.normal(f(i_15T), 10)  # added noise
#     source = set_ts_index(pd.Series(values, i_15T))

#     # Seperate the values in bins for later aggregation.
#     def isstart_f(freq):
#         if freq == "AS":
#             return lambda ts: ts.floor("D") == ts and ts.is_year_start
#         if freq == "QS":
#             return lambda ts: ts.floor("D") == ts and ts.is_quarter_start
#         if freq == "MS":
#             return lambda ts: ts.floor("D") == ts and ts.is_month_start
#         if freq == "D":
#             return lambda ts: ts.floor("D") == ts
#         if freq == "H":
#             return lambda ts: ts.minute == 0
#         raise ValueError("Invalid value for `freq`.")

#     agg_data = {
#         freq: {
#             "values": [],
#             "index": [],
#             "durations": [],
#             "hours": [],
#             "new": isstart_f(freq),
#         }
#         for freq in ["H", "D", "MS", "QS", "AS"]
#     }
#     for ts, val, dur in zip(source.index, source.values, source.index.duration):
#         for freq, dic in agg_data.items():
#             if dic["new"](ts):
#                 dic["index"].append(ts)
#                 dic["values"].append([])
#                 dic["durations"].append([])
#                 dic["hours"].append([])
#             dic["values"][-1].append(val)
#             dic["durations"][-1].append(dur)
#             dic["hours"][-1].append(dur.magnitude)
#     agg_data["15T"] = {
#         "values": [[v] for v in source.values],
#         "durations": [[d] for d in source.index.duration],
#         "hours": [[d.magnitude] for d in source.index.duration],
#         "index": source.index,
#     }
#     return agg_data


# @functools.lru_cache()
# def combis_downsampling():
#     # series-pairs, where one can be turned into the other by downsampling
#     agg_data = aggdata()
#     summed, avged = {}, {}
#     for freq, dic in agg_data.items():
#         summ = [sum(vals) for vals in dic["values"]]
#         avg = [
#             wavg(pd.Series(values), durations)
#             for values, durations in zip(dic["values"], dic["durations"])
#         ]

#         for vals, coll in [(summ, summed), (avg, avged)]:
#             coll[freq] = set_ts_index(
#                 pd.Series(vals, dic["index"]).resample(freq).asfreq()
#             )

#     sumcombis, avgcombis = [], []
#     for coll, combis in [(summed, sumcombis), (avged, avgcombis)]:
#         for freq1, s1 in coll.items():
#             for freq2, s2 in coll.items():
#                 if freq_up_or_down(freq1, freq2) > 0:
#                     continue
#                 # freq1 to freq2 means downsampling
#                 combis.append((s1, s2))

#     return sumcombis, avgcombis


# @functools.lru_cache()
# def combis_upsampling():
#     # series-pairs, where one can be turned into the other by upsampling.
#     agg_data = aggdata()
#     sumcombis, avgcombis = [], []
#     for freq1, dic1 in agg_data.items():
#         for freq2, dic2 in agg_data.items():
#             if freq_up_or_down(freq1, freq2) < 0:
#                 continue
#             # freq1 to freq2 means upsampling

#             # Find the two series, value-by-value.
#             sumrecords1, sumrecords2, avgrecords1, avgrecords2 = {}, {}, {}, {}
#             i2 = 0
#             for ts1, vals1, hours1 in zip(dic1["index"], dic1["values"], dic1["hours"]):
#                 len1 = len(vals1)
#                 sumval1 = sum(vals1)
#                 avgval1 = wavg(pd.Series(vals1), hours1)

#                 # For each datapoint in long frequency, find corresponing datapoints in shorter frequency.
#                 tss2, hourss2 = [], []
#                 len2 = 0
#                 # ts1 is single timestamp; vals1 and durs1 are nonnested lists.
#                 # tss2 is list of timestamps; valss2 and durss2 are nested lists.
#                 while len2 < len1:
#                     tss2.append(dic2["index"][i2])
#                     hourss2.append(dic2["hours"][i2])
#                     len2 += len(dic2["hours"][i2])
#                     i2 += 1

#                 hours2 = np.array([sum(durs) for durs in hourss2])
#                 durfractions = hours2 / hours2.sum()
#                 sumvals2 = sumval1 * durfractions

#                 assert sum(hours1) == sum(
#                     hours2
#                 )  # just small check (not part of pytests)

#                 sumrecords1[ts1] = sumval1
#                 avgrecords1[ts1] = avgval1
#                 for ts, sumval in zip(tss2, sumvals2):
#                     sumrecords2[ts] = sumval
#                     avgrecords2[ts] = avgval1  # same value copied to all children

#             # Add the pair to the combinations
#             for combis, records1, records2 in [
#                 (sumcombis, sumrecords1, sumrecords2),
#                 (avgcombis, avgrecords1, avgrecords2),
#             ]:
#                 s1 = set_ts_index(pd.Series(records1).resample(freq1).asfreq())
#                 s2 = set_ts_index(pd.Series(records2).resample(freq2).asfreq())
#                 combis.append((s1, s2))

#     return sumcombis, avgcombis


# def summable():
#     combis = []
#     sum_up, _ = combis_upsampling()
#     sum_down, _ = combis_downsampling()
#     for key, sumcombis in (("up", sum_up), ("down", sum_down)):
#         for s1, s2 in sumcombis:
#             combis.append((s1, s2, f"{key}-s-{freq1}-{freq2}"))
#             combis.append(
#                 (
#                     pd.DataFrame({"a": s1}),
#                     pd.DataFrame({"a": s2}),
#                     f"{key}-df-{freq1}-{freq2}",
#                 )
#             )
#     return combis


# def avgable():
#     combis = []
#     _, avg_up = combis_upsampling()
#     _, avg_down = combis_downsampling()
#     for key, avgcombis in (("up", avg_up), ("down", avg_down)):
#         for s1, s2 in avgcombis:
#             combis.append((s1, s2, f"{key}-s-{freq1}-{freq2}"))
#             combis.append(
#                 (
#                     pd.DataFrame({"a": s1}),
#                     pd.DataFrame({"a": s2}),
#                     f"{key}-df-{freq1}-{freq2}",
#                 )
#             )
#     return combis


# @pytest.mark.parametrize("fr1,fr2,descr", summable())
# def test_changefreq_sum(fr1, fr2, descr):
#     testfr = changefreq.summable(fr1, fr2.index.freq)
#     if isinstance(fr1, pd.Series):
#         testing.assert_series_equal(testfr, fr2)
#     else:
#         testing.assert_frame_equal(testfr, fr2)


# @pytest.mark.parametrize("fr1,fr2,descr", avgable())
# def test_changefreq_avg(fr1, fr2, descr):
#     testfr = changefreq.averagable(fr1, fr2.index.freq)
#     if isinstance(fr1, pd.Series):
#         testing.assert_series_equal(testfr, fr2)
#     else:
#         testing.assert_frame_equal(testfr, fr2)


@functools.lru_cache()
def get_df_from_excel(freq, tz):
    tzname = {None: "None", "Europe/Berlin": "Berlin"}[tz]
    path = Path(__file__).parent / f"test_changefreq_data_{tzname}.xlsx"
    df = pd.read_excel(path, freq, header=6, index_col=0).tz_localize(
        tz, ambiguous="infer"
    )
    df.index.freq = freq
    return df


@functools.lru_cache(1000)
def get_testframes(
    source_freq: str,
    target_freq: str,
    tz: str,
    avg_or_sum: str,
    series_or_df: str,
    with_units: str,
) -> Tuple[Union[pd.DataFrame, pd.Series]]:
    source_s = get_df_from_excel(source_freq, tz)[source_freq]
    if target_freq is None:
        target_s = source_s
    else:
        target_s = get_df_from_excel(target_freq, tz)[f"{source_freq}_{avg_or_sum}"]

    if with_units == "units":
        source_s = source_s.astype("pint[MW]")
        target_s = target_s.astype("pint[MW]")

    if series_or_df == "series":
        return source_s.rename("a"), target_s.rename("a")
    else:
        return (
            pd.DataFrame({"a": source_s, "b": source_s * -0.5}),
            pd.DataFrame({"a": target_s, "b": target_s * -0.5}),
        )


@pytest.mark.parametrize("with_units", ["units", "nounits"])
@pytest.mark.parametrize("series_or_df", ["series", "df"])
@pytest.mark.parametrize("avg_or_sum", ["avg", "sum"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("target_freq", tools.freq.FREQUENCIES)
@pytest.mark.parametrize("source_freq", tools.freq.FREQUENCIES)
def test_changefreq(source_freq, target_freq, tz, avg_or_sum, series_or_df, with_units):
    """Test if frequency of series or dataframe can be correctly changed."""
    # Get test data.
    if avg_or_sum == "avg":
        testfn = tools.changefreq.averagable
    else:
        testfn = tools.changefreq.summable

    if source_freq != target_freq:
        source, expected = get_testframes(
            source_freq, target_freq, tz, avg_or_sum, series_or_df, with_units
        )
    else:
        source, expected = get_testframes(
            source_freq, None, tz, avg_or_sum, series_or_df, with_units
        )

    # Get and assert result.
    result = testfn(source, target_freq)
    if series_or_df == "series":
        testing.assert_series_equal(result, expected, check_dtype=False)
    else:
        testing.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize("with_units", ["units", "nounits"])
@pytest.mark.parametrize("series_or_df", ["series", "df"])
@pytest.mark.parametrize("avg_or_sum", ["avg", "sum"])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("target_freq", tools.stamp.FREQUENCIES)
@pytest.mark.parametrize("source_freq", tools.stamp.FREQUENCIES)
def test_changefreq_fullonly(
    source_freq, target_freq, tz, avg_or_sum, series_or_df, with_units
):
    """Test if, when downsampling, only the periods are returned that are fully present in the source."""
    if tools.freq.longest(source_freq, target_freq) == source_freq:
        pytest.skip("Only test downsampling.")

    # Get test data.
    if avg_or_sum == "avg":
        testfn = tools.changefreq.averagable
    else:
        testfn = tools.changefreq.summable

    source, expected = get_testframes(
        source_freq, target_freq, tz, avg_or_sum, series_or_df, with_units
    )
    if source_freq == target_freq:
        expected = source  # no change in frequency: expect no change.

    # Remove some data, so that source contains partial periods
    source, expected = source.iloc[1:-1], expected.iloc[1:-1]

    # # Error if no data to be returned.
    # if len(expected) == 0:
    #     with pytest.raises(ValueError):
    #         _ = testfn(source, target_freq)
    #     return

    # Get and assert result.
    result = testfn(source, target_freq)
    if series_or_df == "series":
        testing.assert_series_equal(result, expected, check_dtype=False)
    else:
        testing.assert_frame_equal(result, expected, check_dtype=False)
