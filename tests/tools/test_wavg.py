import numpy as np
import pandas as pd
import pytest

import portfolyo as pf
from portfolyo import tools


@pytest.mark.parametrize("weightsas", ["none", "list", "series"])
@pytest.mark.parametrize("with_units", ["units", "nounits"])
def test_wavg_valuesasseries1(weightsas: str, with_units: str):
    """Test if weighted average of a series is correctly calculated."""
    # Starting values.
    values = pd.Series([100.0, 200, 300, -150])
    weights = [10, 10, 10, 20]
    if weightsas == "none":
        weights = None
        expected = 112.5
    elif weightsas == "list":
        expected = 60.0
    elif weightsas == "series":
        weights = pd.Series(weights, index=[3, 2, 1, 0])  # align by index
        expected = 110.0
    # Add units.
    if with_units == "units":
        values = values.astype("pint[Eur/MWh]")
        expected = pf.Q_(expected, "Eur/MWh")
    # Test.
    assert np.isclose(tools.wavg.series(values, weights), expected)


@pytest.mark.parametrize("weightsas", ["list", "series"])
@pytest.mark.parametrize("with_units", ["units", "nounits"])
@pytest.mark.parametrize("indextype", [int, pd.DatetimeIndex])
def test_wavg_valuesasseries2(weightsas: str, with_units: str, indextype: type):
    """Test if weighted average of a series is correctly calculated."""
    # Starting values.
    i = range(4) if indextype is int else pd.date_range("2020", freq="D", periods=4)
    values = pd.Series([100.0, 200, 300, -150], i)
    weights = [10.0, 0, 10, 20]
    if weightsas == "list":
        expected = 25.0
    elif weightsas == "series":
        weights = pd.Series(weights, [i[j] for j in [3, 2, 1, 0]])  # align by index
        expected = 62.5
    # Add units.
    if with_units == "units":
        values = values.astype("pint[Eur/MWh]")
        expected = pf.Q_(expected, "Eur/MWh")
    # Test.
    assert np.isclose(tools.wavg.series(values, weights), expected)


@pytest.mark.parametrize("weightsas", ["list", "series"])
@pytest.mark.parametrize("with_units", ["units", "nounits"])
def test_wavg_valuesasseries_na(weightsas: str, with_units: str):
    """Test if weighted average of a series is correctly identified as error,
    when all weights are 0 but not all values are equal."""
    # Starting values.
    values = pd.Series([100.0, 200, 300, -150])
    weights = [0.0, 0, 0, 0]
    if weightsas == "series":
        weights = pd.Series(weights, index=[3, 2, 1, 0])  # align by index
    # Add units.
    if with_units == "units":
        values = values.astype("pint[Eur/MWh]")
        if weightsas == "series":
            weights = weights.astype("pint[MWh]")
    # Test.
    assert np.isnan(tools.wavg.series(values, weights))


@pytest.mark.parametrize("weightsas", ["list", "series"])
@pytest.mark.parametrize("with_units", ["units", "nounits"])
def test_wavg_valuesasseries_0weights(weightsas: str, with_units: str):
    """Test if weighted average of a series is correctly calculated,
    when all weights are 0 and all values are equal."""
    # Starting values.
    values = pd.Series([100.0, 100, 100, 100])
    weights = [0.0, 0, 0, 0]
    expected = 100.0
    if weightsas == "series":
        weights = pd.Series(weights, index=[3, 2, 1, 0])  # align by index
    # Add units.
    if with_units == "units":
        values = values.astype("pint[Eur/MWh]")
        expected = pf.Q_(expected, "Eur/MWh")
        if weightsas == "series":
            weights = weights.astype("pint[MWh]")
    # Test.
    assert tools.wavg.series(values, weights) == expected


@pytest.mark.parametrize("weightsas", ["none", "list", "series", "dataframe"])
@pytest.mark.parametrize("axis", [0, 1])
def test_wavg_valuesasdataframe1(weightsas: str, axis: int):
    """Test if weighted average of a dataframe is correctly calculated."""
    # Starting values.
    series_a = pd.Series([100.0, 200, 300, -150])
    series_b = pd.Series([100.0, -200, 300, -150])
    values = pd.DataFrame({"a": series_a, "b": series_b})
    if weightsas == "none":
        weights = None
        if axis == 0:
            expected = pd.Series({"a": 112.5, "b": 12.5})
        else:
            expected = pd.Series([100.0, 0, 300, -150])
    if weightsas == "list":
        if axis == 0:
            weights = [10.0, 10, 10, 20]
            expected = pd.Series({"a": 60.0, "b": -20})
        else:
            weights = [10.0, 30]
            expected = pd.Series([100.0, -100, 300, -150])
    if weightsas == "series":
        if axis == 0:
            weights = pd.Series([10.0, 10, 10, 20], index=[3, 2, 1, 0])
            expected = pd.Series({"a": 110.0, "b": 30})
        else:
            weights = pd.Series({"b": 30.0, "a": 10})
            expected = pd.Series([100.0, -100, 300, -150])
    if weightsas == "dataframe":
        weights = pd.DataFrame({"a": [10.0, 10, 10, 20], "b": [10.0, 10, 30, 0]})
        if axis == 0:
            expected = pd.Series({"a": 60.0, "b": 160})
        else:
            expected = pd.Series([100.0, 0, 300, -150])
    # Test.
    result = tools.wavg.dataframe(values, weights, axis)
    pd.testing.assert_series_equal(result, expected)


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
    series_a = pd.Series([100.0, 200, 300, -150]).astype("pint[Eur/MWh]")
    unit_b = "Eur/MWh" if same_units else "MW"
    series_b = pd.Series([100.0, -200, 300, -150]).astype(f"pint[{unit_b}]")
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
            expected = exp_res(112.5, 12.5)
        elif weightsas == "list":
            weights = [10.0, 10, 10, 20]
            expected = exp_res(60.0, -20.0)
        elif weightsas == "series":
            weights = pd.Series([10.0, 10, 10, 20], index=[3, 2, 1, 0])
            if weights_units:
                weights = weights.astype("pint[h]")  # any unit will do
            expected = exp_res(110.0, 30.0)
        elif weightsas == "dataframe":
            weights = pd.DataFrame({"a": [10.0, 10, 10, 20], "b": [10.0, 10, 30, 0]})
            if weights_units:
                weights = pd.DataFrame(
                    {c: s.astype("pint[h]") for c, s in weights.items()}
                )
            expected = exp_res(60.0, 160.0)

    else:  # axis == 1
        if weightsas == "none":
            weights = None
            expected = pd.Series([100.0, 0, 300, -150]).astype("pint[Eur/MWh]")
        elif weightsas == "list":
            weights = [10.0, 30]
            expected = pd.Series([100.0, -100, 300, -150]).astype("pint[Eur/MWh]")
        elif weightsas == "series":
            weights = pd.Series({"b": 30.0, "a": 10.0})
            if weights_units:
                weights = weights.astype("pint[h]")
            expected = pd.Series([100.0, -100, 300, -150]).astype("pint[Eur/MWh]")
        elif weightsas == "dataframe":
            weights = pd.DataFrame({"a": [10.0, 10, 10, 20], "b": [10.0, 10, 30, 0]})
            if weights_units:
                weights = pd.DataFrame(
                    {c: s.astype("pint[h]") for c, s in weights.items()}
                )
            expected = pd.Series([100.0, 0, 300, -150]).astype("pint[Eur/MWh]")

    # Test.
    if axis == 1 and not same_units:  # error cases
        with pytest.raises(Exception):
            tools.wavg.dataframe(values, weights, axis)
        return
    result = tools.wavg.dataframe(values, weights, axis)
    pd.testing.assert_series_equal(result, expected)


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
            expected = pd.Series({"a": 0.0, "b": -100})
        else:
            weights = [10, 0]
            expected = pd.Series([100.0, 200, 200, -150], i)
    elif weightsas == "series":
        if axis == 0:
            weights = pd.Series([10, 10, 0, 20], [i[j] for j in [3, 2, 1, 0]])
            expected = pd.Series({"a": 62.5, "b": 87.5})
        else:
            weights = pd.Series({"b": 0, "a": 10})
            expected = pd.Series([100.0, 200, 200, -150], i)
    elif weightsas == "dataframe":
        weights = pd.DataFrame({"a": [10, 10, 0, 20], "b": [10, 10, 30, 0]}, i)
        if axis == 0:
            expected = pd.Series({"a": 0.0, "b": 160})
        else:
            expected = pd.Series([100.0, 0, 300, -150], i)
    # Test.
    result = tools.wavg.dataframe(values, weights, axis)
    pd.testing.assert_series_equal(result, expected)


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
        expected = pd.Series({"a": np.nan, "b": np.nan})
    else:
        weights = [0, 0]
        expected = pd.Series([np.nan, np.nan, np.nan, np.nan], i)

    if weightsas == "series":
        if axis == 0:
            weights = pd.Series(weights, [i[j] for j in [3, 2, 1, 0]])
        else:
            weights = pd.Series(weights, ["a", "b"])
    elif weightsas == "dataframe":
        weights = pd.DataFrame({"a": [0, 0, 0, 0], "b": [0, 0, 0, 0]}, i)
    # Test.
    result = tools.wavg.dataframe(values, weights, axis)
    pd.testing.assert_series_equal(result, expected, check_dtype=False)


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
        expected = pd.Series({"a": np.nan, "b": np.nan})
    else:
        weights = [0, 0]
        expected = pd.Series([100, np.nan, np.nan, -150])

    if weightsas == "series":
        if axis == 0:
            weights = pd.Series(weights, [3, 2, 1, 0])
        else:
            weights = pd.Series(weights, ["a", "b"])
    if weightsas == "dataframe":
        weights = pd.DataFrame({"a": [0, 0, 0, 0], "b": [0, 0, 0, 0]})
    # Test.
    result = tools.wavg.dataframe(values, weights, axis)
    pd.testing.assert_series_equal(result, expected, check_dtype=False)
