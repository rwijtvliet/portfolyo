import pandas as pd
import pytest

from portfolyo import toolsb


def test_downsample_index(idx_shortfreq_untrimmed, longfreq, idx_longfreq):
    """Test downsampling of indices."""
    toolsb.testing.assert_index_equal(
        toolsb.changefreq.index(idx_shortfreq_untrimmed, longfreq), idx_longfreq
    )


def test_upsample_index(idx_shortfreq, shortfreq, idx_longfreq):
    """Test upsampling of indices."""
    toolsb.testing.assert_index_equal(
        toolsb.changefreq.index(idx_longfreq, shortfreq), idx_shortfreq
    )


@pytest.mark.only_on_pr
def test_downsample_summable(
    seriesordf: str,
    shortfreq: str,
    tz: str,
    longfreq: str,
    complexity: str,
    sod: str,
    idx_short_untrimmed,
    mapping,
):
    """Test downsampling of summable frames."""

    # mapping = ts_long: {ts_shrt: fraction}
    # i_long is the index with the longest frequency, so with the fewest values!

    # Get raw input and expected output.
    if complexity == "ones":
        s = pd.Series(1.0, idx_short_untrimmed)
        s_expected = mapping["ts_shrt"].groupby(mapping.index).count() * 1.0
    else:
        mapping["value"] = range(len(mapping))
        s = mapping.set_index("ts_shrt")["value"] * 1.0
        s_expected = mapping["value"].groupby(mapping.index).sum() * 1.0

    assert s == s_expected  # delete!
    # do_test("sum", seriesordf, s, s_expected, shortfreq, longfreq)


#
#
# @pytest.mark.only_on_pr
# @pytest.mark.parametrize("starttime", ["00:00", "06:00"])
# @pytest.mark.parametrize("seriesordf", ["s", "s_unit", "df", "df_unit"])
# @pytest.mark.parametrize("complexity", ["ones", "allnumbers"])
# @pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
# @pytest.mark.parametrize("freq_long", ["15min", "h", "D", "MS", "QS", "QS-APR", "YS"])
# @pytest.mark.parametrize("freq_shrt", ["15min", "h", "D", "MS", "QS", "QS-APR", "YS"])
# def test_downsample_avgable(
#     seriesordf: str,
#     freq_shrt: str,
#     tz: str,
#     freq_long: str,
#     complexity: str,
#     starttime: str,
# ):
#     """Test downsampling of averagable frames."""
#
#     # mapping = ts_long: {ts_shrt: fraction}
#     i_long, i_shrt, i_shrt_untrimmed, mapping = idxs_and_mapping(
#         startdate(freq_shrt), starttime, "2022-02-15", freq_shrt, tz, freq_long
#     )
#     # i_long is the index with the longest frequency, so with the fewest values!
#
#     # Get raw input and expected output.
#     if complexity == "ones":
#         s = pd.Series(1.0, i_shrt_untrimmed)
#         s_expected = pd.Series(1.0, i_long)
#     else:
#         mapping["value"] = range(len(mapping))
#         s = mapping.set_index("ts_shrt")["value"] * 1.0
#         weighted = mapping.value * mapping.fraction
#         s_expected = weighted.groupby(weighted.index).sum() * 1.0
#
#     do_test("avg", seriesordf, s, s_expected, freq_shrt, freq_long)
#
#
# @pytest.mark.parametrize("starttime", ["00:00", "06:00"])
# @pytest.mark.parametrize("seriesordf", ["s", "s_unit", "df", "df_unit"])
# @pytest.mark.parametrize("complexity", ["ones", "allnumbers"])
# @pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
# @pytest.mark.parametrize("freq_shrt", ["15min", "h", "D", "MS", "QS", "QS-APR", "YS"])
# @pytest.mark.parametrize("freq_long", ["15min", "h", "D", "MS", "QS", "QS-APR", "YS"])
# def test_upsample_avgable(
#     seriesordf: str,
#     freq_long: str,
#     tz: str,
#     freq_shrt: str,
#     complexity: str,
#     starttime: str,
# ):
#     """Test upsampling of averagable frames."""
#
#     if freq_long == freq_shrt:
#         pytest.skip("Same frequency already tested when downsampling.")
#
#     # mapping = ts_long: {ts_shrt: fraction}
#     i_long, i_shrt, _, mapping = idxs_and_mapping(
#         startdate(freq_shrt), starttime, "2022-02-15", freq_shrt, tz, freq_long
#     )
#     # i_long is the index with the longest frequency, so with the fewest values!
#
#     # Get raw input and expected output.
#     if complexity == "ones":
#         s = pd.Series(1.0, i_long)
#         s_expected = pd.Series(1.0, i_shrt)
#     else:
#         s = pd.Series(range(len(i_long)), i_long) * 1.0
#         mapping["value"] = s[mapping.index]
#         s_expected = mapping.set_index("ts_shrt")["value"]
#
#     do_test("avg", seriesordf, s, s_expected, freq_long, freq_shrt)
#
#
# @pytest.mark.parametrize("starttime", ["00:00", "06:00"])
# @pytest.mark.parametrize("seriesordf", ["s", "s_unit", "df", "df_unit"])
# @pytest.mark.parametrize("complexity", ["ones", "allnumbers"])
# @pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
# @pytest.mark.parametrize(
#     "freq_shrt", ["15min", "h", "D", "MS", "QS", "QS-APR", "YS"]
# )  # SOS!: freq 'QS' and 'QS-APR' are not converting into 1 another because they are treated as the same freq
# @pytest.mark.parametrize("freq_long", ["15min", "h", "D", "MS", "QS", "QS-APR", "YS"])
# def test_upsample_summable(
#     seriesordf: str,
#     freq_long: str,
#     tz: str,
#     freq_shrt: str,
#     complexity: str,
#     starttime: str,
# ):
#     """Test upsampling of summable frames."""
#
#     if freq_long == freq_shrt:
#         pytest.skip("Same frequency already tested when downsampling.")
#
#     # mapping = ts_long: {ts_shrt: fraction}
#     i_long, i_shrt, _, mapping = idxs_and_mapping(
#         startdate(freq_shrt), starttime, "2022-02-15", freq_shrt, tz, freq_long
#     )
#     # i_long is the index with the longest frequency, so with the fewest values!
#
#     # Get raw input and expected output.
#     if complexity == "ones":
#         s = pd.Series(1.0, i_long, name="testseries")
#         s_expected = mapping.set_index("ts_shrt")["fraction"]
#     else:
#         s = pd.Series(range(len(i_long)), i_long) * 1.0
#         mapping["value"] = s[mapping.index]
#         mapping["multiplied"] = mapping.value * mapping.fraction
#         s_expected = mapping.set_index("ts_shrt")["multiplied"]
#
#     do_test("sum", seriesordf, s, s_expected, freq_long, freq_shrt)
#
#
# def do_test(
#     avgorsum: str,
#     seriesordf: str,
#     s: pd.Series,
#     s_expected: pd.Series,
#     freq_source: str,
#     freq_target: str,
# ):
#     if avgorsum == "avg":
#         fn = tools.changefreq.averagable
#     else:
#         fn = tools.changefreq.summable
#
#     s = s.sort_index()
#     s.name = "testseries"
#     s.index.name = None
#     s.index.freq = freq_source
#
#     s_expected = s_expected.sort_index()
#     s_expected.name = "testseries"
#     s_expected.index.name = None
#     s_expected.index.freq = freq_target
#
#     if seriesordf.endswith("_unit"):
#         s = s.astype("pint[MW]")
#         s_expected = s_expected.astype("pint[MW]")
#
#     if seriesordf.startswith("s"):
#         fr = s
#         expected = s_expected
#         result = fn(fr, freq_target)
#         testing.assert_series_equal(result, expected)
#     else:
#         fr = pd.DataFrame({"a": s})
#         expected = pd.DataFrame({"a": s_expected})
#         result = fn(fr, freq_target)
#         testing.assert_dataframe_equal(result, expected)
#
#
