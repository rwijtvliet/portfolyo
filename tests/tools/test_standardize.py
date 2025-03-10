import numpy as np
import pandas as pd
import pytest

from portfolyo import dev, tools

TEST_FREQUENCIES = [
    "YS",
    "YS-FEB",
    "YS-APR",
    "QS",
    "QS-FEB",
    "QS-APR",
    "MS",
    "D",
    "h",
    "15min",
]


@pytest.mark.parametrize("series_or_df", ["series", "df"])
@pytest.mark.parametrize("bound", ["right", "left"])
@pytest.mark.parametrize(
    ("in_vals_num_specialconditions", "start"),  # normal, WT->ST, ST->WT
    [(96, "2020-03-01"), (92, "2020-03-29"), (100, "2020-10-25")],
)
@pytest.mark.parametrize("in_aware", [True, False])
@pytest.mark.parametrize("in_tz", ["Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("force", ["agnostic", "aware"])
@pytest.mark.parametrize("freq", ["15min", "D"])
def test_standardize_DST(
    in_vals_num_specialconditions: int,
    start: str,
    bound: str,
    in_aware: bool,
    in_tz: str,
    series_or_df: str,
    force: str,
    freq: str,
):
    """Test if series and dataframes are correctly standardized during DST-transition.
    Using quarterhour and daily timeseries, without gaps."""

    if not in_aware and in_tz != "Europe/Berlin":
        pytest.skip("Cannot convert tz-naive fr to different timezone.")

    if freq == "D":
        in_vals_num = 200
    elif force == "agnostic" and in_tz != "Europe/Berlin":
        in_vals_num = 96
    else:
        in_vals_num = in_vals_num_specialconditions
    in_vals = np.random.random(in_vals_num)

    # Prepare expected output frame.
    out_tz = "Europe/Berlin" if force == "aware" else None
    if force == "aware" or freq == "D":
        out_vals = in_vals
    else:  # always return 96 values
        a, b = (12, -84) if in_vals_num_specialconditions == 100 else (8, -88)
        out_vals = [*in_vals[:a], *in_vals[b:]]
    iout = pd.date_range(start, freq=freq, periods=len(out_vals), tz=out_tz)
    expected = pd.Series(out_vals, iout.rename("ts_left"))
    if series_or_df == "df":
        expected = pd.DataFrame({"a": expected})

    # Prepare input frame.
    if force == "aware":
        out_tz = "Europe/Berlin"
    else:
        out_tz = in_tz
    iin = pd.date_range(start, freq=freq, periods=len(in_vals), tz=out_tz)
    if out_tz != in_tz and freq == "D":
        pytest.skip("Not at day boundary.")  # cannot test.
    iin = iin.tz_convert(in_tz).rename("the_time_stamp")
    if not in_aware:
        iin = iin.tz_localize(None)
    if bound == "right":
        td = pd.Timedelta(hours=24 if freq == "D" else 0.25)
        iin = pd.DatetimeIndex([*iin[1:], iin[-1] + td])
    kw = {"floating": False, "tz": out_tz}

    # Do actual tests.
    if isinstance(expected, pd.Series):
        # 1: Using expected frame: should stay the same.
        result = tools.standardize.frame(expected, force, **kw)
        pd.testing.assert_series_equal(result, expected)
        # 2: Series.
        # series = pd.Series(in_vals, iin)
        result = tools.standardize.frame(
            pd.Series(in_vals, iin), force, bound=bound, **kw
        )
        pd.testing.assert_series_equal(result, expected)
    else:
        # 1: Using expected frame: should stay the same.
        result = tools.standardize.frame(expected, force, **kw)
        pd.testing.assert_frame_equal(result, expected)
        # 2: Dataframe.
        result = tools.standardize.frame(
            pd.DataFrame({"a": in_vals}, iin), force, bound=bound, **kw
        )
        pd.testing.assert_frame_equal(result, expected)


@pytest.mark.only_on_pr
@pytest.mark.parametrize("series_or_df", ["series", "df"])
@pytest.mark.parametrize("in_tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("out_tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("floating", [True, False])
@pytest.mark.parametrize("bound", ["left", "right"])
@pytest.mark.parametrize("freq", TEST_FREQUENCIES)
def test_standardize_convert(freq, in_tz, floating, series_or_df, bound, out_tz):
    """Test raising errors when conversing timezones."""
    force = "aware" if out_tz else "agnostic"

    # Get index.
    i = dev.get_index(freq, in_tz, _seed=1)
    if bound == "right" and freq == "15min":  # Ensure it's a correct full-hour index
        i += pd.Timedelta(minutes=15)
    if freq == "15min" and in_tz == "Asia/Kolkata" and not floating and out_tz:
        i += pd.Timedelta(minutes=30)

    # If no timezone specified and below-daily values, the created index will have too few/many datapoints.
    # if (
    #     in_tz is None
    #     and tools.freq.up_or_down(freq, "D") == -1
    # ):
    #     pytest.skip('Too few datapoints, skip.')

    # Add values.
    fr = dev.get_series(i) if series_or_df == "series" else dev.get_dataframe(i)

    # See if error is raised.
    if (
        in_tz == "Asia/Kolkata"
        and out_tz == "Europe/Berlin"
        and tools.freq.shortest(freq, "h") == "h"
        and not floating
    ):
        # Kolkata and Berlin timezone only share 15T-boundaries. Therefore, any other
        # frequency should raise an error.
        with pytest.raises(ValueError):
            _ = tools.standardize.frame(fr, force, bound, tz=out_tz, floating=floating)
        return

    result = tools.standardize.frame(fr, force, bound, tz=out_tz, floating=floating)
    assert result.index.freq == freq


@pytest.mark.only_on_pr
@pytest.mark.parametrize("series_or_df", ["series", "df"])
@pytest.mark.parametrize("in_tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("floating", [True, False])
@pytest.mark.parametrize("force", ["agnostic", "aware"])
@pytest.mark.parametrize("freq", TEST_FREQUENCIES)
def test_standardize_freq(freq, in_tz, floating, series_or_df, force):
    """Test raising errors when passing invalid frequencies."""
    out_tz = "Europe/Berlin"

    # Get index.
    i = dev.get_index(freq, in_tz, _seed=1)

    # Add values.
    fr = dev.get_series(i) if series_or_df == "series" else dev.get_dataframe(i)

    # See if error is raised.
    try:
        tools.freq.assert_freq_valid(freq)
    except ValueError:
        # freq isn't valid, check that standardize will also raise ValueError
        with pytest.raises(ValueError):
            _ = tools.standardize.frame(fr, force, tz=out_tz, floating=floating)
        return  # freq was invalid, and standardize correctly raised ValueError

    result = tools.standardize.frame(fr, force, tz=out_tz, floating=floating)
    assert result.index.freq == freq


@pytest.mark.parametrize("series_or_df", ["series", "df"])
@pytest.mark.parametrize("remove", ["remove_some", "remove_none"])
@pytest.mark.parametrize("in_tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("freq", TEST_FREQUENCIES)
def test_standardize_gaps(freq, in_tz, remove, series_or_df):
    """Test raising errors on index with gaps. Don't test timezone-conversion."""
    force = "agnostic" if in_tz is None else "aware"
    out_tz = in_tz

    # Get index.
    i = dev.get_index(freq, in_tz, _seed=1)

    # remove timestamp from middle of index.
    if remove == "remove_some":
        i = i.delete((len(i) - 2) // 2)

    # Add values.
    fr = dev.get_series(i) if series_or_df == "series" else dev.get_dataframe(i)

    # See if error is raised.
    if (
        # fr has frequency, but it's a forbidden frequency
        (remove == "remove_none" and freq not in TEST_FREQUENCIES)
        # fr does not have frequency
        or (remove == "remove_some")
    ):
        with pytest.raises(ValueError):
            _ = tools.standardize.frame(fr, force, tz=out_tz)
        return

    result = tools.standardize.frame(fr, force, tz=out_tz)
    assert result.index.freq == freq
