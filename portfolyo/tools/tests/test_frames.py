from typing import Union
from portfolyo import dev, testing
from portfolyo.tools import frames
from numpy import nan
import portfolyo as pf
import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("series_or_df", ["series", "df"])
@pytest.mark.parametrize("bound", ["right", "left"])
@pytest.mark.parametrize(
    ("in_vals_num_specialconditions", "start"),  # normal, WT->ST, ST->WT
    [(96, "2020-03-01"), (92, "2020-03-29"), (100, "2020-10-25")],
)
@pytest.mark.parametrize("in_aware", [True, False])
@pytest.mark.parametrize("in_tz", ["Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("force", ["agnostic", "aware"])
@pytest.mark.parametrize("freq", ["15T", "D"])
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
        return  # cannot convert tz-naive fr to different timezone

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
    kw = {"bound": bound, "floating": False, "tz": out_tz}

    # Do actual tests.
    if isinstance(expected, pd.Series):
        # 1: Using expected frame: should stay the same.
        result = frames.standardize(expected, force)
        pd.testing.assert_series_equal(result, expected)
        # 2: Series.
        result = frames.standardize(pd.Series(in_vals, iin), force, **kw)
        pd.testing.assert_series_equal(result, expected)
    else:
        # 1: Using expected frame: should stay the same.
        result = frames.standardize(expected, force)
        pd.testing.assert_frame_equal(result, expected)
        # 2: Dataframe with index.
        result = frames.standardize(pd.DataFrame({"a": in_vals}, iin), force, **kw)
        pd.testing.assert_frame_equal(result, expected)
        # 3: Dataframe with column that must become index.
        result = frames.standardize(
            pd.DataFrame({"a": in_vals, "t": iin}), force, index_col="t", **kw
        )
        pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("series_or_df", ["series", "df"])
@pytest.mark.parametrize("in_tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("out_tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("floating", [True, False])
@pytest.mark.parametrize("bound", ["left", "right"])
@pytest.mark.parametrize("freq", pf.FREQUENCIES)
def test_standardize_convert(freq, in_tz, floating, series_or_df, bound, out_tz):
    """Test raising errors when conversing timezones."""
    force = "aware" if out_tz else "agnostic"

    # Get index.
    while True:
        i = dev.get_index(freq, in_tz)
        if len(i) > 10:
            break
    # If no timezone specified and below-daily values, the created index will have too few/many datapoints.
    if not in_tz and pf.freq_up_or_down(freq, "D") > 1:
        pytest.skip("Edge case: too few/too many datapoints.")  # don't check edge case

    # Add values.
    fr = dev.get_series(i) if series_or_df == "series" else dev.get_dataframe(i)

    # See if error is raised.
    if (
        in_tz == "Asia/Kolkata"
        and out_tz == "Europe/Berlin"
        and pf.freq_shortest(freq, "H") == "H"
        and not floating
    ):
        # Kolkata and Berlin timezone only share 15T-boundaries. Therefore, any other
        # frequency should raise an error.
        with pytest.raises(ValueError):
            _ = frames.standardize(fr, force, bound, tz=out_tz, floating=floating)
        return

    result = frames.standardize(fr, force, bound, tz=out_tz, floating=floating)
    assert result.index.freq == freq


@pytest.mark.parametrize("series_or_df", ["series", "df"])
@pytest.mark.parametrize("in_tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("floating", [True, False])
@pytest.mark.parametrize("force", ["agnostic", "aware"])
@pytest.mark.parametrize("freq", [*pf.FREQUENCIES, "Q", "30T", "M", "AS-FEB"])
def test_standardize_freq(freq, in_tz, floating, series_or_df, force):
    """Test raising errors when passing invalid frequencies."""
    out_tz = "Europe/Berlin"

    # Get index.
    while True:
        i = dev.get_index(freq, in_tz)
        if len(i) > 10:
            break
    # If no timezone specified and below-daily values, the created index will have too few/many datapoints.
    if not in_tz and pf.freq_up_or_down(freq, "D") > 1:
        pytest.skip("edge case: too few/too many datapoints.")  # don't check edge case

    # Add values.
    fr = dev.get_series(i) if series_or_df == "series" else dev.get_dataframe(i)

    # See if error is raised.
    if freq not in pf.FREQUENCIES:
        with pytest.raises(ValueError):
            _ = frames.standardize(fr, force, tz=out_tz, floating=floating)
        return

    result = frames.standardize(fr, force, tz=out_tz, floating=floating)
    assert result.index.freq == freq


@pytest.mark.parametrize("series_or_df", ["series", "df"])
@pytest.mark.parametrize("removesome", [0, 1, 2])  # 0=none, 1=from end, 2=from middle
@pytest.mark.parametrize("in_tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("freq", pf.FREQUENCIES)
@pytest.mark.parametrize(
    "force_freq", [*pf.FREQUENCIES[::2], "30T", "M", "AS-FEB", None]
)
def test_standardize_gaps(freq, in_tz, removesome, series_or_df, force_freq):
    """Test raising errors on index with gaps. Don't test timezone-conversion."""
    force = "agnostic" if in_tz is None else "aware"
    out_tz = in_tz

    # Get index.
    while True:
        i = dev.get_index(freq, in_tz)
        if len(i) > 10:
            break

    # If no timezone specified and below-daily values, the created index will have too few/many datapoints.
    if not in_tz and pf.freq_up_or_down(freq, "D") > 1:
        pytest.skip("edge case: too few/too many datapoints.")  # don't check edge case

    # remove timestamp from index.
    if removesome == 1:  # remove one from end or start
        i = i.delete(np.random.choice([0, len(i) - 1]))
    elif removesome == 2:  # remove max 3 from middle
        i = i.delete(np.random.randint(2, len(i) - 2, 3))

    # Add values.
    fr = dev.get_series(i) if series_or_df == "series" else dev.get_dataframe(i)

    # See if error is raised.
    if (
        # fr has frequency, but it's a forbidden frequency
        (removesome != 2 and freq not in pf.FREQUENCIES)
        # fr has frequency, but user wants to force a different frequency
        or (removesome != 2 and freq != force_freq and force_freq is not None)
        # fr does not have requency, and user does not specify a forced frequency
        or (removesome == 2 and not force_freq)
        # user wants to force a frequency, but it's a forbidden frequency
        or (force_freq is not None and force_freq not in pf.FREQUENCIES)
    ):
        with pytest.raises(ValueError):
            _ = frames.standardize(fr, force, tz=out_tz, force_freq=force_freq)
        return

    result = frames.standardize(fr, force, tz=out_tz, force_freq=force_freq)
    expected_freq = force_freq or freq
    assert result.index.freq == expected_freq


@pytest.mark.parametrize(
    ("values", "maxgap", "gapvalues"),
    [
        ([1, 2, 3, 4, 25, 7, 8], 1, []),
        ([1, 2, 3, 4, nan, 7, 8], 1, [5.5]),
        ([1, 2, 3, 4, nan, 7, 8], 2, [5.5]),
        ([1, 2, 3, 4, nan, 7, 8], 3, [5.5]),
        ([3, 2, 1, nan, nan, 7, 8], 1, [nan, nan]),
        ([3, 2, 1, nan, nan, 7, 8], 2, [3, 5]),
        ([3, 2, 1, nan, nan, 7, 8], 3, [3, 5]),
        ([nan, 2, 1, nan, nan, 7, nan], 1, [nan, nan, nan, nan]),
        ([nan, 2, 1, nan, nan, 7, nan], 2, [nan, 3, 5, nan]),
    ],
)
@pytest.mark.parametrize(
    ("index", "tol"),
    [
        (range(7), 0),
        (range(-3, 4), 0),
        (pd.date_range("2020", periods=7, freq="D"), 0),
        (pd.date_range("2020", periods=7, freq="M", tz="Europe/Berlin"), 0.04),
    ],
)
def test_fill_gaps(values, index, maxgap, gapvalues, tol):
    """Test if gaps are correctly interpolated."""
    # Test as Series.
    s = pd.Series(values, index)
    s_new = frames.fill_gaps(s, maxgap)
    s[s.isna()] = gapvalues
    pd.testing.assert_series_equal(s_new, s, rtol=tol)
    # Test as DataFrame.
    df = pd.DataFrame({"a": values}, index)
    df_new = frames.fill_gaps(df, maxgap)
    df[df.isna()] = gapvalues
    pd.testing.assert_frame_equal(df_new, df, rtol=tol)


@pytest.mark.parametrize(
    ("df_columns", "header", "expected_columns"),
    [
        (["A"], "B", [("B", "A")]),
        (["A1", "A2"], "B", [("B", "A1"), ("B", "A2")]),
        (pd.MultiIndex.from_tuples([("B", "A")]), "C", [("C", "B", "A")]),
        (
            pd.MultiIndex.from_product([["B"], ["A1", "A2"]]),
            "C",
            [("C", "B", "A1"), ("C", "B", "A2")],
        ),
        (
            pd.MultiIndex.from_tuples([("B1", "A1"), ("B2", "A2")]),
            "C",
            [("C", "B1", "A1"), ("C", "B2", "A2")],
        ),
    ],
)
def test_addheader_tocolumns(df_columns, header, expected_columns):
    """Test if header can be added to the columns of a dataframe."""
    i = dev.get_index()
    df_in = pd.DataFrame(np.random.rand(len(i), len(df_columns)), i, df_columns)
    result_columns = frames.add_header(df_in, header).columns.to_list()
    assert np.array_equal(result_columns, expected_columns)


# TODO: put in ... fixture (?)
test_index_D = dev.get_index("D")
test_index_D_deconstructed = test_index_D.map(lambda ts: (ts.year, ts.month, ts.day))
test_index_H = dev.get_index("H")
test_index_H_deconstructed = test_index_H.map(lambda ts: (ts.year, ts.month, ts.day))


@pytest.mark.parametrize(
    ("df_index", "header", "expected_index"),
    [
        (test_index_D, "test", [("test", i) for i in test_index_D]),
        (
            test_index_D_deconstructed,
            "test",
            [("test", *i) for i in test_index_D_deconstructed],
        ),
        (test_index_H, "test", [("test", i) for i in test_index_H]),
        (
            test_index_H_deconstructed,
            "test",
            [("test", *i) for i in test_index_H_deconstructed],
        ),
    ],
)
def test_addheader_torows(df_index, header, expected_index):
    """Test if header can be added to the rows of a dataframe."""
    df_in = pd.DataFrame(np.random.rand(len(df_index), 2), df_index, ["A", "B"])
    result_index = frames.add_header(df_in, header, axis=0).index.to_list()
    assert np.array_equal(result_index, expected_index)


@pytest.mark.parametrize(
    ("freq", "num", "wanted", "strict", "expected"),
    [
        # D
        # . enough
        ("D", 10, None, False, "D"),
        ("D", 10, None, True, "D"),
        ("D", 10, "MS", False, ValueError),
        ("D", 10, "MS", True, ValueError),
        ("D", 10, "D", False, "D"),
        ("D", 10, "D", True, "D"),
        # . too few
        ("D", 2, None, False, None),
        ("D", 2, None, True, ValueError),
        ("D", 2, "MS", False, ValueError),
        ("D", 2, "MS", True, ValueError),
        ("D", 2, "D", False, "D"),
        ("D", 2, "D", True, "D"),
        # 15T, too few
        ("15T", 2, None, False, None),
        ("15T", 2, None, True, ValueError),
        ("15T", 2, "MS", False, ValueError),
        ("15T", 2, "MS", True, ValueError),
        ("15T", 2, "15T", False, "15T"),
        ("15T", 2, "15T", True, "15T"),
        # invalid freq, not correctable
        # . enough
        ("2D", 10, None, False, "2D"),
        ("2D", 10, None, True, ValueError),
        ("2D", 10, "MS", False, ValueError),
        ("2D", 10, "MS", True, ValueError),
        ("2D", 10, "2D", False, "2D"),
        ("2D", 10, "2D", True, ValueError),
        # . too few
        ("2D", 2, None, False, None),
        ("2D", 2, None, True, ValueError),
        ("2D", 2, "MS", False, ValueError),
        ("2D", 2, "MS", True, ValueError),
        ("2D", 2, "2D", False, "2D"),
        ("2D", 2, "2D", True, ValueError),
        # invalid freq, correctable
        # . enough
        ("QS-APR", 10, None, False, "QS"),
        ("QS-APR", 10, None, True, "QS"),
        ("QS-APR", 10, "MS", False, ValueError),
        ("QS-APR", 10, "MS", True, ValueError),
        ("QS-APR", 10, "QS", False, "QS"),
        ("QS-APR", 10, "QS", True, "QS"),
        # . too few
        ("QS-APR", 2, None, False, None),
        ("QS-APR", 2, None, True, ValueError),
        ("QS-APR", 2, "MS", False, ValueError),
        ("QS-APR", 2, "MS", True, ValueError),
        ("QS-APR", 2, "QS", False, "QS"),
        ("QS-APR", 2, "QS", True, "QS"),
    ],
)
def test_setfreq(
    freq: str, num: int, wanted: str, strict: bool, expected: Union[str, Exception]
):
    # Create frame, with frequency, which is not set.
    fr = pd.Series(np.random.rand(num), pd.date_range("2020", periods=num, freq=freq))
    fr.index.freq = None
    # Test.
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            _ = frames.set_frequency(fr, wanted, strict)
        return
    fr2 = frames.set_frequency(fr, wanted, strict)
    assert fr2.index.freq == expected


# TODO: put in ... fixture (?)
test_values = np.random.rand(len(test_index_D), 10)
test_df_1 = pd.DataFrame(test_values[:, :2], test_index_D, ["A", "B"])
test_df_2 = pd.DataFrame(test_values[:, 2], test_index_D, ["C"])
expect_concat_12 = pd.DataFrame(test_values[:, :3], test_index_D, ["A", "B", "C"])
test_df_3 = pd.DataFrame(test_values[:, 2], test_index_D, pd.Index([("D", "C")]))
expect_concat_13 = pd.DataFrame(
    test_values[:, :3], test_index_D, pd.Index([("A", ""), ("B", ""), ("D", "C")])
)


@pytest.mark.parametrize(
    ("dfs", "axis", "expected"),
    [
        ([test_df_1, test_df_2], 1, expect_concat_12),
        ([test_df_1, test_df_3], 1, expect_concat_13),
    ],
)
def test_concat(dfs, axis, expected):
    """Test if concatenation works as expected."""
    result = frames.concat(dfs, axis)
    testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("weightsas", ["none", "list", "series"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("with_units", [True, False])
def test_wavg_valuesasseries1(weightsas: str, axis: int, with_units: bool):
    """Test if weighted average of a series is correctly calculated."""
    # Starting values.
    values = pd.Series([100, 200, 300, -150])
    weights = [10, 10, 10, 20]
    if weightsas == "none":
        weights = None
        expected_result = 112.5
    elif weightsas == "list":
        expected_result = 60
    elif weightsas == "series":
        weights = pd.Series(weights, index=[3, 2, 1, 0])  # align by index
        expected_result = 110
    # Add units.
    if with_units:
        values = values.astype("pint[Eur/MWh]")
        expected_result = pf.Q_(expected_result, "Eur/MWh")
    # Test.
    assert np.isclose(frames.wavg(values, weights, axis), expected_result)


@pytest.mark.parametrize("weightsas", ["list", "series"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("with_units", [True, False])
@pytest.mark.parametrize("indextype", [int, pd.DatetimeIndex])
def test_wavg_valuesasseries2(
    weightsas: str, axis: int, with_units: bool, indextype: type
):
    """Test if weighted average of a series is correctly calculated."""
    # Starting values.
    i = range(4) if indextype is int else pd.date_range("2020", freq="D", periods=4)
    values = pd.Series([100, 200, 300, -150], i)
    weights = [10, 0, 10, 20]
    if weightsas == "list":
        expected_result = 25
    elif weightsas == "series":
        weights = pd.Series(weights, [i[j] for j in [3, 2, 1, 0]])  # align by index
        expected_result = 62.5
    # Add units.
    if with_units:
        values = values.astype("pint[Eur/MWh]")
        expected_result = pf.Q_(expected_result, "Eur/MWh")
    # Test.
    assert np.isclose(frames.wavg(values, weights, axis), expected_result)


@pytest.mark.parametrize("weightsas", ["list", "series"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("with_units", [True, False])
def test_wavg_valuesasseries_na(weightsas: str, axis: int, with_units: bool):
    """Test if weighted average of a series is correctly identified as error,
    when all weights are 0 but not all values are equal."""
    # Starting values.
    values = pd.Series([100, 200, 300, -150])
    weights = [0, 0, 0, 0]
    if weightsas == "series":
        weights = pd.Series(weights, index=[3, 2, 1, 0])  # align by index
    # Add units.
    if with_units:
        values = values.astype("pint[Eur/MWh]")
        if weightsas == "series":
            weights = weights.astype("pint[MWh]")
    # Test.
    assert np.isnan(frames.wavg(values, weights, axis))


@pytest.mark.parametrize("weightsas", ["list", "series"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("with_units", [True, False])
def test_wavg_valuesasseries_0weights(weightsas: str, axis: int, with_units: bool):
    """Test if weighted average of a series is correctly calculated,
    when all weights are 0 and all values are equal."""
    # Starting values.
    values = pd.Series([100, 100, 100, 100])
    weights = [0, 0, 0, 0]
    expected_result = 100
    if weightsas == "series":
        weights = pd.Series(weights, index=[3, 2, 1, 0])  # align by index
    # Add units.
    if with_units:
        values = values.astype("pint[Eur/MWh]")
        expected_result = pf.Q_(expected_result, "Eur/MWh")
        if weightsas == "series":
            weights = weights.astype("pint[MWh]")
    # Test.
    assert frames.wavg(values, weights, axis) == expected_result


@pytest.mark.parametrize("weightsas", ["none", "list", "series", "dataframe"])
@pytest.mark.parametrize("axis", [0, 1])
def test_wavg_valuesasdataframe1(weightsas: str, axis: int):
    """Test if weighted average of a dataframe is correctly calculated."""
    # Starting values.
    series_a = pd.Series([100, 200, 300, -150])
    series_b = pd.Series([100, -200, 300, -150])
    values = pd.DataFrame({"a": series_a, "b": series_b})
    if weightsas == "none":
        weights = None
        if axis == 0:
            expected_result = pd.Series({"a": 112.5, "b": 12.5})
        else:
            expected_result = pd.Series([100, 0, 300, -150])
    if weightsas == "list":
        if axis == 0:
            weights = [10, 10, 10, 20]
            expected_result = pd.Series({"a": 60, "b": -20})
        else:
            weights = [10, 30]
            expected_result = pd.Series([100, -100, 300, -150])
    if weightsas == "series":
        if axis == 0:
            weights = pd.Series([10, 10, 10, 20], index=[3, 2, 1, 0])
            expected_result = pd.Series({"a": 110, "b": 30})
        else:
            weights = pd.Series({"b": 30, "a": 10})
            expected_result = pd.Series([100, -100, 300, -150])
    if weightsas == "dataframe":
        weights = pd.DataFrame({"a": [10, 10, 10, 20], "b": [10, 10, 30, 0]})
        if axis == 0:
            expected_result = pd.Series({"a": 60, "b": 160})
        else:
            expected_result = pd.Series([100, 0, 300, -150])
    # Test.
    pd.testing.assert_series_equal(
        frames.wavg(values, weights, axis), expected_result, check_dtype=False
    )


@pytest.mark.parametrize("weightsas", ["none", "list", "series", "dataframe"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("same_units", [True, False])
@pytest.mark.parametrize("weights_units", [True, False])
def test_wavg_valuesasdataframe_units(
    weightsas: str, axis: int, same_units: bool, weights_units: bool
):
    """Test if weighted average of a dataframe is correctly calculated if it has units.
    Test with float weights and weights that have a unit."""
    # Starting values.
    series_a = pd.Series([100, 200, 300, -150]).astype("pint[Eur/MWh]")
    unit_b = "Eur/MWh" if same_units else "MW"
    series_b = pd.Series([100, -200, 300, -150]).astype(f"pint[{unit_b}]")
    values = pd.DataFrame({"a": series_a, "b": series_b})

    # Filter non-tests.
    if weights_units and weightsas in ["none", "list"]:
        pytest.skip(f"If weights is {weightsas}, it cannot have a unit.")

    # Weights and exected results for 'non-error' cases (= not (axis == 1 and same_units) ).
    if axis == 0:

        def exp_res(val_a, val_b):
            if same_units:
                return pd.Series({"a": val_a, "b": val_b}, dtype="pint[Eur/MWh]")
            return pd.Series({"a": pf.Q_(val_a, "Eur/MWh"), "b": pf.Q_(val_b, unit_b)})

        if weightsas == "none":
            weights = None
            expected_result = exp_res(112.5, 12.5)
        elif weightsas == "list":
            weights = [10, 10, 10, 20]
            expected_result = exp_res(60, -20)
        elif weightsas == "series":
            weights = pd.Series([10, 10, 10, 20], index=[3, 2, 1, 0])
            if weights_units:
                weights = weights.astype("pint[h]")  # any unit will do
            expected_result = exp_res(110, 30)
        elif weightsas == "dataframe":
            weights = pd.DataFrame({"a": [10, 10, 10, 20], "b": [10, 10, 30, 0]})
            if weights_units:
                weights = pd.DataFrame(
                    {c: s.astype("pint[h]") for c, s in weights.items()}
                )
            expected_result = exp_res(60, 160)

    else:  # axis == 1
        if weightsas == "none":
            weights = None
            expected_result = pd.Series([100, 0, 300, -150]).astype("pint[Eur/MWh]")
        elif weightsas == "list":
            weights = [10, 30]
            expected_result = pd.Series([100, -100, 300, -150]).astype("pint[Eur/MWh]")
        elif weightsas == "series":
            weights = pd.Series({"b": 30, "a": 10})
            if weights_units:
                weights = weights.astype("pint[h]")
            expected_result = pd.Series([100, -100, 300, -150]).astype("pint[Eur/MWh]")
        elif weightsas == "dataframe":
            weights = pd.DataFrame({"a": [10, 10, 10, 20], "b": [10, 10, 30, 0]})
            if weights_units:
                weights = pd.DataFrame(
                    {c: s.astype("pint[h]") for c, s in weights.items()}
                )
            expected_result = pd.Series([100, 0, 300, -150]).astype("pint[Eur/MWh]")

    # Test.
    if axis == 1 and not same_units:  # error cases
        with pytest.raises(Exception):
            _ = frames.wavg(values, weights, axis)
        return
    pd.testing.assert_series_equal(
        frames.wavg(values, weights, axis), expected_result, check_dtype=False
    )


@pytest.mark.parametrize("weightsas", ["list", "series", "dataframe"])
@pytest.mark.parametrize("indextype", [int, pd.DatetimeIndex])
@pytest.mark.parametrize("axis", [0, 1])
def test_wavg_valuesasdataframe2(weightsas: str, axis: int, indextype: type):
    """Test if weighted average of a dataframe is correctly calculated."""
    # Starting values.
    i = range(4) if indextype is int else pd.date_range("2020", freq="D", periods=4)
    values = pd.DataFrame({"a": [100, 200, 200, -150], "b": [100, -200, 300, -150]}, i)
    if weightsas == "list":
        if axis == 0:
            weights = [10, 10, 0, 20]
            expected_result = pd.Series({"a": 0.0, "b": -100})
        else:
            weights = [10, 0]
            expected_result = pd.Series([100.0, 200, 200, -150], i)
    elif weightsas == "series":
        if axis == 0:
            weights = pd.Series([10, 10, 0, 20], [i[j] for j in [3, 2, 1, 0]])
            expected_result = pd.Series({"a": 62.5, "b": 87.5})
        else:
            weights = pd.Series({"b": 0, "a": 10})
            expected_result = pd.Series([100.0, 200, 200, -150], i)
    elif weightsas == "dataframe":
        weights = pd.DataFrame({"a": [10, 10, 0, 20], "b": [10, 10, 30, 0]}, i)
        if axis == 0:
            expected_result = pd.Series({"a": 0.0, "b": 160})
        else:
            expected_result = pd.Series([100.0, 0, 300, -150], i)
    # Test.
    pd.testing.assert_series_equal(frames.wavg(values, weights, axis), expected_result)


@pytest.mark.parametrize("weightsas", ["list", "series", "dataframe"])
@pytest.mark.parametrize("indextype", [int, pd.DatetimeIndex])
@pytest.mark.parametrize("axis", [0, 1])
def test_wavg_valuesasdataframe_na(weightsas: str, axis: int, indextype: type):
    """Test if weighted average of a dataframe is correctly is correctly identified as error,
    when all weights are 0 but not all values are equal."""
    # Starting values.
    i = range(4) if indextype is int else pd.date_range("2020", freq="D", periods=4)
    values = pd.DataFrame({"a": [130, 200, 200, -160], "b": [100, -200, 300, -150]}, i)
    if axis == 0:
        weights = [0, 0, 0, 0]
        expected_result = pd.Series({"a": np.nan, "b": np.nan})
    else:
        weights = [0, 0]
        expected_result = pd.Series([np.nan, np.nan, np.nan, np.nan], i)

    if weightsas == "series":
        if axis == 0:
            weights = pd.Series(weights, [i[j] for j in [3, 2, 1, 0]])
        else:
            weights = pd.Series(weights, ["a", "b"])
    elif weightsas == "dataframe":
        weights = pd.DataFrame({"a": [0, 0, 0, 0], "b": [0, 0, 0, 0]}, i)
    # Test.
    pd.testing.assert_series_equal(
        frames.wavg(values, weights, axis), expected_result, check_dtype=False
    )


@pytest.mark.parametrize("weightsas", ["list", "series", "dataframe"])
@pytest.mark.parametrize("axis", [0, 1])
def test_wavg_valuesasdataframe_0weights(weightsas: str, axis: int):
    """Test if weighted average of a dataframe is correctly is correctly identified as error,
    when all weights are 0. Some averages are calculated from identical values and should
    result in that value."""
    # Starting values.
    values = pd.DataFrame({"a": [100, 200, 200, -150], "b": [100, -200, 300, -150]})
    if axis == 0:
        weights = [0, 0, 0, 0]
        expected_result = pd.Series({"a": np.nan, "b": np.nan})
    else:
        weights = [0, 0]
        expected_result = pd.Series([100, np.nan, np.nan, -150])

    if weightsas == "series":
        if axis == 0:
            weights = pd.Series(weights, [3, 2, 1, 0])
        else:
            weights = pd.Series(weights, ["a", "b"])
    if weightsas == "dataframe":
        weights = pd.DataFrame({"a": [0, 0, 0, 0], "b": [0, 0, 0, 0]})
    # Test.
    pd.testing.assert_series_equal(
        frames.wavg(values, weights, axis), expected_result, check_dtype=False
    )


vals1 = np.array([1, 2.0, -4.1234, 0])
vals2 = np.array([1, 2.0, -4.1234, 0.5])


@pytest.mark.parametrize(
    ("s1", "s2", "expected_result"),
    [
        (pd.Series(vals1), pd.Series(vals1), True),
        (pd.Series(vals1), pd.Series(vals2), False),
        (pd.Series(vals1), pd.Series(vals1, dtype="pint[MW]"), False),
        (pd.Series(vals1).astype("pint[MW]"), pd.Series(vals1, dtype="pint[MW]"), True),
        (
            pd.Series(vals1 * 1000).astype("pint[kW]"),
            pd.Series(vals1, dtype="pint[MW]"),
            True,
        ),
        (
            pd.Series(vals1 * 1000).astype("pint[MW]"),
            pd.Series(vals1, dtype="pint[MW]"),
            False,
        ),
    ],
)
def test_series_allclose(s1, s2, expected_result):
    """Test if series can be correctly compared, even if they have a unit."""
    assert frames.series_allclose(s1, s2) == expected_result
