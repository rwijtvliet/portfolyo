from enum import Enum
from portfolyo.tools import zones, frames, stamps
from portfolyo import testing
from pathlib import Path
import pandas as pd
import pytest
import functools


class TzType(Enum):
    # (type, column name, hard-coded tz of input/output frame, implied tz)
    A = ("A", "A", "Europe/Berlin", None)
    B = ("B", "B", None, None)
    C = ("C", "C", None, "Europe/Berlin")
    A_NONFLOAT = ("A", "A_Kolkata_NonFloating", "Asia/Kolkata", None)
    A_FLOAT = ("A", "A_Kolkata_Floating", "Asia/Kolkata", None)

    typ = property(lambda self: self.value[0])  # noqa
    columnname = property(lambda self: self.value[1])  # noqa
    explicit = property(lambda self: self.value[2])  # noqa
    implicit = property(lambda self: self.value[3])  # noqa
    floating = property(lambda self: "_Floating" in self.columnname)  # noqa


@functools.cache
def get_df_fromexcel(aggfreq, tzt_in: TzType, tzt_out: TzType) -> pd.DataFrame:
    path = Path(__file__).parent / "test_zones_data.xlsx"

    if tzt_in is TzType.A_NONFLOAT and aggfreq != "15T":
        return None, None  # does not exist as starting point for our tests

    df = pd.read_excel(path, aggfreq, header=[5, 6])
    df_idx = df[[c for c in df.columns if "col" not in c[1]]]
    df_val = df[[c for c in df.columns if "col" in c[1]]]

    # Find idx and vals for source and target values.
    for c in df_idx.columns:
        if c[1] == tzt_in.columnname:
            in_group = c[0]
            idx_in = df_idx[c].values  # array
            data_in = df_val[c[0]]  # df
    for c in df_idx.columns:
        if c[1] == tzt_out.columnname:
            idx_out = df_idx[c].values  # array
            data_out = df_val[c[0] + (0 if c[0] >= in_group else 2)]  # df
            break
    else:
        idx_out = data_out = None  # cannot convert; conversion should give error.

    # Prepare dataframes.
    def prep_df(idx, data, tz):
        if idx is None:
            return None
        index = pd.DatetimeIndex([pd.Timestamp(i, tz=tz) for i in idx], name="ts_left")
        index.freq = pd.infer_freq(index)
        return data.set_axis(index)

    return (
        prep_df(idx_in, data_in, tzt_in.explicit),
        prep_df(idx_out, data_out, tzt_out.explicit),
    )


def do_conversion_test(aggfreq, tzt_in, tzt_out, series_or_df, conversion_fn):
    df_in, df_out = get_df_fromexcel(aggfreq, tzt_in, tzt_out)
    if df_in is None:
        return  # cannot convert into other tz
    fr_in = df_in if series_or_df == "df" else df_in["col1"]

    # Conversion not possible.
    if df_out is None:
        with pytest.raises(ValueError):
            result = conversion_fn(fr_in)  # either conversion_fn raises error...
            stamps.assert_boundary_ts(result.index, aggfreq)  # ... or assertion does.
        return

    # Conversion possible.
    expected = df_out if series_or_df == "df" else df_out["col1"]
    result = conversion_fn(fr_in)
    if series_or_df == "df":
        testing.assert_frame_equal(result, expected, check_names=False)
    else:
        testing.assert_series_equal(result, expected, check_names=False)


@pytest.mark.parametrize("series_or_df", ["series", "df"])
@pytest.mark.parametrize("aggfreq", ["15T", "H", "D", "MS"])
@pytest.mark.parametrize(
    ("tzt_in", "tzt_out"),
    [
        (TzType.A, TzType.A),
        (TzType.A_NONFLOAT, TzType.A),
        (TzType.A_FLOAT, TzType.A),
        (TzType.B, TzType.A),
        (TzType.C, TzType.A),
    ],
)
def test_forceaware_fromexcel(aggfreq, tzt_in, tzt_out, series_or_df):
    """Test if frames can be correctly converted to tz-aware"""

    def conversion_fn(fr):
        return zones.force_tzaware(fr, tzt_out.explicit, floating=tzt_in.floating)

    do_conversion_test(aggfreq, tzt_in, tzt_out, series_or_df, conversion_fn)


@pytest.mark.parametrize("series_or_df", ["series", "df"])
@pytest.mark.parametrize("aggfreq", ["15T", "H", "D", "MS"])
@pytest.mark.parametrize(
    ("tzt_in", "tzt_out"),
    [
        (TzType.A, TzType.B),
        (TzType.B, TzType.B),
        (TzType.C, TzType.B),
    ],
)
def test_forceagnostic_fromexcel(aggfreq, tzt_in, tzt_out, series_or_df):
    """Test if frames can be correctly converted to tz-agnostic"""

    def conversion_fn(fr):
        return zones.force_tzagnostic(fr, tz_in=tzt_in.implicit)

    do_conversion_test(aggfreq, tzt_in, tzt_out, series_or_df, conversion_fn)


@pytest.mark.parametrize("series_or_df", ["series", "df"])
@pytest.mark.parametrize("aggfreq", ["15T", "H", "D", "MS"])
@pytest.mark.parametrize(
    ("tzt_in", "tzt_out"),
    [
        (TzType.A, TzType.A),
        (TzType.A, TzType.A_NONFLOAT),
        (TzType.A, TzType.A_FLOAT),
        (TzType.A_FLOAT, TzType.A),
        (TzType.A_NONFLOAT, TzType.A),
    ],
)
def test_conversionAtoA_fromexcel(aggfreq, tzt_in, tzt_out, series_or_df):
    """Test if frames can be correctly converted from type A to type A."""

    def conversion_fn(fr):
        floating = tzt_in.floating or tzt_out.floating
        fr_out = zones._A_to_A(fr, tz=tzt_out.explicit, floating=floating)
        return frames.set_frequency(fr_out, aggfreq)

    do_conversion_test(aggfreq, tzt_in, tzt_out, series_or_df, conversion_fn)


@pytest.mark.parametrize("series_or_df", ["series", "df"])
@pytest.mark.parametrize("aggfreq", ["15T", "H", "D", "MS"])
@pytest.mark.parametrize(
    ("tzt_in", "tzt_out"), [(TzType.A, TzType.B), (TzType.A_FLOAT, TzType.B)]
)
def test_conversionAtoB_fromexcel(aggfreq, tzt_in, tzt_out, series_or_df):
    """Test if frames can be correctly converted from type A to type B."""

    def conversion_fn(fr):
        fr_out = zones._A_to_B(fr)
        return frames.set_frequency(fr_out, aggfreq)

    do_conversion_test(aggfreq, tzt_in, tzt_out, series_or_df, conversion_fn)


@pytest.mark.parametrize("series_or_df", ["series", "df"])
@pytest.mark.parametrize("aggfreq", ["15T", "H", "D", "MS"])
@pytest.mark.parametrize(("tzt_in", "tzt_out"), [(TzType.B, TzType.A)])
def test_conversionBtoA_fromexcel(aggfreq, tzt_in, tzt_out, series_or_df):
    """Test if frames can be correctly converted from type B to type A."""

    def conversion_fn(fr):
        fr_out = zones._B_to_A(fr, tz=tzt_out.explicit)
        return frames.set_frequency(fr_out, aggfreq)

    do_conversion_test(aggfreq, tzt_in, tzt_out, series_or_df, conversion_fn)


@pytest.mark.parametrize("series_or_df", ["series", "df"])
@pytest.mark.parametrize("aggfreq", ["15T", "H", "D", "MS"])
@pytest.mark.parametrize(("tzt_in", "tzt_out"), [(TzType.C, TzType.A)])
def test_conversionCtoA_fromexcel(aggfreq, tzt_in, tzt_out, series_or_df):
    """Test if frames can be correctly converted from type C to type A."""

    def conversion_fn(fr):
        fr_out = zones._C_to_A(fr, tz=tzt_in.implicit)
        return frames.set_frequency(fr_out, aggfreq)

    do_conversion_test(aggfreq, tzt_in, tzt_out, series_or_df, conversion_fn)


@pytest.mark.parametrize("series_or_df", ["series", "df"])
@pytest.mark.parametrize("aggfreq", ["15T", "H", "D", "MS"])
@pytest.mark.parametrize(("tzt_in", "tzt_out"), [(TzType.C, TzType.B)])
def test_conversionCtoB_fromexcel(aggfreq, tzt_in, tzt_out, series_or_df):
    """Test if frames can be correctly converted from type C to type B."""

    def conversion_fn(fr):
        fr_out = zones._C_to_B(fr, tz_in=tzt_in.implicit)
        return frames.set_frequency(fr_out, aggfreq)

    do_conversion_test(aggfreq, tzt_in, tzt_out, series_or_df, conversion_fn)


# @pytest.mark.parametrize("series_or_df", ["series", "df"])
# @pytest.mark.parametrize("tz_in", ["Europe/Berlin", "Asia/Kolkata"])
# @pytest.mark.parametrize("do_localize", [True, False])
# @pytest.mark.parametrize("bound", ["right", "left"])
# @pytest.mark.parametrize(
#     ("values", "i"),
#     [
#         (
#             np.random.random(96),
#             pd.date_range(
#                 "2020-03-01",
#                 "2020-03-02",
#                 freq="15T",
#                 inclusive="left",
#                 tz="Europe/Berlin",
#             ),
#         ),
#         (  # Check for days incl WT->ST.
#             np.random.random(92),
#             pd.date_range(
#                 "2020-03-29",
#                 "2020-03-30",
#                 freq="15T",
#                 inclusive="left",
#                 tz="Europe/Berlin",
#             ),
#         ),
#         (  # Check for days incl ST->WT.
#             np.random.random(100),
#             pd.date_range(
#                 "2020-10-25",
#                 "2020-10-26",
#                 freq="15T",
#                 inclusive="left",
#                 tz="Europe/Berlin",
#             ),
#         ),
#     ],
# )
# def test_settsindex_1(
#     values: Iterable,
#     i: pd.DatetimeIndex,
#     bound: str,
#     do_localize: bool,
#     tz_in: str,
#     series_or_df: str,
# ):
#     """Test if index of dataframes and series is correctly standardized."""
#     # Prepare input frame.
#     if series_or_df == "df":
#         expected = pd.DataFrame({"a": values}, index=i.rename("ts_left"))
#     else:
#         expected = pd.Series(values, i.rename("ts_left"))

#     i = i.tz_convert(tz_in)
#     if not do_localize:
#         i = i.tz_localize(None)
#     else:
#         # If supplied timezone is not 'Europe/Berlin', this value is contradictory to
#         # what is in the index. Test if timezone in Index is given preference.
#         tz_in = "Europe/Berlin"

#     if bound == "right":
#         i = i + pd.Timedelta("15T")

#     if series_or_df == "df":
#         # Using expected frame: should stay the same.
#         pd.testing.assert_frame_equal(frames.set_ts_index(expected), expected)

#         # Dataframe with index.
#         result = frames.set_ts_index(
#             pd.DataFrame({"a": values}, i), bound=bound, tz_in=tz_in
#         )
#         pd.testing.assert_frame_equal(result, expected)

#         # Dataframe with column that must become index.
#         result = frames.set_ts_index(
#             pd.DataFrame({"a": values, "ts": i}), "ts", bound, tz_in
#         )
#         pd.testing.assert_frame_equal(result, expected)

#     else:
#         # Using expected frame: should stay the same.
#         pd.testing.assert_series_equal(frames.set_ts_index(expected), expected)

#         # Series.
#         result = frames.set_ts_index(pd.Series(values, i), bound=bound, tz_in=tz_in)
#         pd.testing.assert_series_equal(result, expected)
