from enum import Enum
from portfolyo.tools import zones
from portfolyo import testing
from pathlib import Path
import pandas as pd
import pytest
import functools

freqs_small_to_large = ["T", "5T", "15T", "30T", "H", "2H", "D", "MS", "QS", "AS"]


tzBerlin = "Europe/Berlin"
tzKolkata = "Asia/Kolkata"


class TzType(Enum):
    A_BERLIN = ("A_EuropeBerlin", "Europe/Berlin", "A")
    A_KOLKATA = ("A_AsiaKolkata", "Asia/Kolkata", "A")
    A_KOLKATA_FLOATING = ("A_AsiaKolkata_Floating", "Asia/Kolkata", "A")
    B = ("B", None, "B")
    C = ("C", "Europe/Berlin", "C")

    typ = lambda self: self.value[2]  # noqa
    tz = lambda self: self.value[1]  # noqa
    columnname = lambda self: self.value[0]  # noqa
    floating = lambda self: "Floating" in self.columnname  # noqa


@functools.cache
def get_df_fromexcel(aggfreq, tztype: TzType) -> pd.DataFrame:
    path = Path(__file__).parent / "test_zones_data.xlsx"

    for loss in ["lossless", "lossy"]:
        sheetname = f"{aggfreq}_{loss}"
        try:
            df = pd.read_excel(path, sheetname, header=6)
        except ValueError:
            continue
        if tztype.columnname in df.columns:
            df = df.set_index(tztype.columnname)[["col1", "col2"]]
            df.index = pd.DatetimeIndex(df.index, tz=tztype.tz)
            return df
    return None


@pytest.mark.parametrize("series_or_df", ["series", "df"])
@pytest.mark.parametrize("tztype_in", TzType)
@pytest.mark.parametrize(
    "tztype_out", [TzType.A_BERLIN, TzType.A_KOLKATA, TzType.A_KOLKATA_FLOATING]
)
@pytest.mark.parametrize("aggfreq", ["15T", "H", "D", "MS"])
def test_forcetzaware_fromexcel(aggfreq, tztype_in, tztype_out, series_or_df):
    """Test if frames can be correctly forced to be timezone aware."""

    # In.
    df_in = get_df_fromexcel(aggfreq, tztype_in)
    if df_in is None:
        return  # no conversion possible
    fr_in = df_in if series_or_df == "df" else df_in["col1"]

    # Out.
    expected = get_df_fromexcel(aggfreq, tztype_out)
    tz, floating = tztype_out.tz, tztype_out.floating
    if expected is None:
        with pytest.raises(ValueError):
            _ = zones.force_tzaware(fr_in, tz, floating=floating, tz_in=tztype_in.tz)
        return
    else:
        result = zones.force_tzaware(fr_in, tz, floating=floating, tz_in=tztype_in.tz)
        if series_or_df == "df":
            testing.assert_frame_equal(result, expected)
        else:
            testing.assert_series_equal(result, expected["a"])


@pytest.mark.parametrize("series_or_df", ["series", "df"])
@pytest.mark.parametrize(
    "tztype_in", [TzType.A_BERLIN, TzType.A_KOLKATA, TzType.B, TzType.C]
)
@pytest.mark.parametrize("tztype_out", [TzType.B])
@pytest.mark.parametrize("aggfreq", ["15T", "H", "D", "MS"])
def test_forcetznaive_fromexcel(aggfreq, tztype_in, tztype_out, series_or_df):
    """Test if frames can be correctly forced to be timezone naive."""

    # In.
    df_in = get_df_fromexcel(aggfreq, tztype_in)
    if df_in is None:
        return  # no conversion possible
    fr_in = df_in if series_or_df == "df" else df_in["col1"]

    # Out.
    expected = get_df_fromexcel(aggfreq, tztype_out)
    if expected is None:
        with pytest.raises(ValueError):
            _ = zones.force_tznaive(fr_in, tz_in=tztype_in.tz)
        return
    else:
        result = zones.force_tznaive(fr_in, tz_in=tztype_in.tz)
        if series_or_df == "df":
            testing.assert_frame_equal(result, expected)
        else:
            testing.assert_series_equal(result, expected["a"])
