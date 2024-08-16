import functools
from enum import Enum
from pathlib import Path

import pandas as pd
import pytest

from portfolyo import testing, tools


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


@functools.lru_cache()
def get_df_fromexcel(aggfreq, tzt_in: TzType, tzt_out: TzType) -> pd.DataFrame:
    path = Path(__file__).parent / "test_tzone_data.xlsx"

    if tzt_in is TzType.A_NONFLOAT and aggfreq != "15min":
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


@pytest.mark.only_on_pr
@pytest.mark.parametrize("seriesordf", ["series", "df"])
@pytest.mark.parametrize("aggfreq", ["15min", "h", "D", "MS"])
@pytest.mark.parametrize(
    ("tzt_in", "tzt_out"),
    [
        (TzType.A, TzType.A),
        (TzType.A_NONFLOAT, TzType.A),
        (TzType.A_FLOAT, TzType.A),
        (TzType.B, TzType.A),
    ],
)
def test_forceaware_fromexcel(aggfreq, tzt_in, tzt_out, seriesordf):
    """Test if frames can be correctly converted to tz-aware"""

    def conversion_fn(fr):
        return tools.tzone.force_aware(fr, tzt_out.explicit, floating=tzt_in.floating)

    do_test_conversion(aggfreq, tzt_in, tzt_out, seriesordf, conversion_fn)


@pytest.mark.only_on_pr
@pytest.mark.parametrize("seriesordf", ["series", "df"])
@pytest.mark.parametrize("aggfreq", ["15min", "h", "D", "MS"])
@pytest.mark.parametrize(
    ("tzt_in", "tzt_out"),
    [
        (TzType.A, TzType.B),
        (TzType.A_FLOAT, TzType.B),
        (TzType.B, TzType.B),
    ],
)
def test_forceagnostic_fromexcel(aggfreq, tzt_in, tzt_out, seriesordf):
    """Test if frames can be correctly converted to tz-agnostic"""

    def conversion_fn(fr):
        return tools.tzone.force_agnostic(fr)

    do_test_conversion(aggfreq, tzt_in, tzt_out, seriesordf, conversion_fn)


@pytest.mark.only_on_pr
@pytest.mark.parametrize("seriesordf", ["series", "df"])
@pytest.mark.parametrize("aggfreq", ["15min", "h", "D", "MS"])
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
def test_conversionAtoA_fromexcel(aggfreq, tzt_in, tzt_out, seriesordf):
    """Test if frames can be correctly converted from type A to type A."""

    def conversion_fn(fr):
        floating = tzt_in.floating or tzt_out.floating
        fr_out = tools.tzone._A_to_A(fr, tz=tzt_out.explicit, floating=floating)
        return fr_out

    do_test_conversion(aggfreq, tzt_in, tzt_out, seriesordf, conversion_fn)


@pytest.mark.parametrize("seriesordf", ["series", "df"])
@pytest.mark.parametrize("aggfreq", ["15min", "h", "D", "MS"])
@pytest.mark.parametrize(
    ("tzt_in", "tzt_out"), [(TzType.A, TzType.B), (TzType.A_FLOAT, TzType.B)]
)
def test_conversionAtoB_fromexcel(aggfreq, tzt_in, tzt_out, seriesordf):
    """Test if frames can be correctly converted from type A to type B."""

    def conversion_fn(fr):
        fr_out = tools.tzone._A_to_B(fr)
        return tools.freq.set_to_frame(fr_out, aggfreq)

    do_test_conversion(aggfreq, tzt_in, tzt_out, seriesordf, conversion_fn)


@pytest.mark.parametrize("seriesordf", ["series", "df"])
@pytest.mark.parametrize("aggfreq", ["15min", "h", "D", "MS"])
@pytest.mark.parametrize(("tzt_in", "tzt_out"), [(TzType.B, TzType.A)])
def test_conversionBtoA_fromexcel(aggfreq, tzt_in, tzt_out, seriesordf):
    """Test if frames can be correctly converted from type B to type A."""

    def conversion_fn(fr):
        fr_out = tools.tzone._B_to_A(fr, tz=tzt_out.explicit)
        return tools.freq.set_to_frame(fr_out, aggfreq)

    do_test_conversion(aggfreq, tzt_in, tzt_out, seriesordf, conversion_fn)


def do_test_conversion(aggfreq, tzt_in, tzt_out, seriesordf, conversion_fn):
    df_in, df_out = get_df_fromexcel(aggfreq, tzt_in, tzt_out)
    if df_in is None:
        pytest.skip("Cannot convert into other tz.")
    fr_in = df_in if seriesordf == "df" else df_in["col1"]

    # Conversion not possible.
    if df_out is None:
        with pytest.raises((ValueError, AssertionError)):
            _ = conversion_fn(fr_in)  # conversion_fn raises error

    # Conversion possible.
    expected = df_out if seriesordf == "df" else df_out["col1"]
    result = conversion_fn(fr_in)
    if seriesordf == "df":
        testing.assert_frame_equal(
            result, expected, check_names=False, check_freq=False
        )
    else:
        testing.assert_series_equal(
            result, expected, check_names=False, check_freq=False
        )
