from typing import Any, Iterable

import numpy as np
import pandas as pd
import pytest

import portfolyo as pf
from portfolyo import tools


def get_weights(
    weights: Iterable[float],
    weightsas: str,
    index: Iterable = None,
    units: bool = False,
):
    """Get weights to test wavg with, if weightsas == 'none', 'list', 'dict', or 'series'."""
    # No weights.
    if weightsas == "none":
        if units:
            pytest.skip("Cannot test weights == None with units.")
        return None
    # Weights as list, no index needed.
    if weightsas == "list":
        if units:
            weights = [pf.Q_(w, "MWh") for w in weights]
        return weights
    # Weights include index value; make index if needed.
    if index is None:
        index = range(len(weights))
    weights = {i: w for w, i in zip(weights[::-1], index[::-1])}
    if weightsas == "dict":
        if units:
            weights = {i: pf.Q_(w, "MWh") for i, w in weights.items()}
        return weights
    if weightsas == "series":
        weights = pd.Series(weights)
        if units:
            weights = weights.astype("pint[MWh]")
        return weights


def get_weights_df(weights: dict, index: Iterable, units: bool = False):
    """Get weights to test wavg with, if weightsas == 'dataframe'."""
    weights = pd.DataFrame(weights, index)
    if units:
        weights = pd.DataFrame(
            {c: s.astype("pint[Eur/MWh]") for c, s in weights.items()}
        )
    return weights


def get_index(number: int, indextype: str) -> Iterable:
    if indextype == "int":
        return range(number)
    return pd.date_range("2020", freq="D", periods=number)


def do_test_series(
    values: pd.Series | pd.DataFrame, weights: Any, expected: Any, **kwargs
):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            tools.wavg.series(values, weights)
        return
    pf.testing.assert_value_equal(tools.wavg.series(values, weights), expected)


def do_test_dataframe(values: pd.DataFrame, weights: Any, expected: Any, **kwargs):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            tools.wavg.dataframe(values, weights, **kwargs)
        return
    result = tools.wavg.dataframe(values, weights, **kwargs)
    pf.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize("weightsas", ["none", "list", "dict", "series"])
@pytest.mark.parametrize("units", ["", "val", "wei", "val&wei"])
@pytest.mark.parametrize("indextype", ["int", "DatetimeIndex"])
def test_wavg_series(weightsas: str, units: str, indextype: str):
    """Test if weighted average of a series is correctly calculated."""
    # Starting values.
    i = get_index(4, indextype)
    values = pd.Series([100.0, 200, 300, -150], i)
    weights = get_weights([10.0, 0, 10, 20], weightsas, i, "wei" in units)
    if weightsas == "none":
        expected = 112.5
    else:
        expected = 25.0
    # Add units.
    if "val" in units:
        values = values.astype("pint[Eur/MWh]")
        expected = pf.Q_(expected, "Eur/MWh")
    # Test.
    do_test_series(values, weights, expected)


@pytest.mark.parametrize("weightsas", ["list", "dict", "series"])
@pytest.mark.parametrize("units", ["", "val", "wei", "val&wei"])
@pytest.mark.parametrize("indextype", ["int", "DatetimeIndex"])
def test_wavg_series_surplusvalues(weightsas: str, units: str, indextype: str):
    """Test if weighted average of a series is correctly calculated if we have more
    values and than weights."""
    # Starting values.
    i = get_index(5, indextype)
    values = pd.Series([100.0, 200, 300, -150, 100], i)
    weights = get_weights([10.0, 0, 10, 20], weightsas, i[:4], "wei" in units)
    expected = 25.0
    # Add units.
    if "val" in units:
        values = values.astype("pint[Eur/MWh]")
        expected = pf.Q_(expected, "Eur/MWh")
    if weightsas == "list":
        expected = ValueError
    # Test.
    do_test_series(values, weights, expected)


@pytest.mark.parametrize("weightsas", ["list", "dict", "series"])
@pytest.mark.parametrize("units", ["", "val", "wei", "val&wei"])
@pytest.mark.parametrize("indextype", ["int", "DatetimeIndex"])
def test_wavg_series_surplusweights(weightsas: str, units: str, indextype: str):
    """Test if error is correctly thrown if we have more weights than values."""
    # Starting values.
    i = get_index(5, indextype)
    values = pd.Series([100.0, 200, 300, -150], i[:4])
    weights = get_weights([10.0, 0, 10, 20, 10], weightsas, i, "wei" in units)
    expected = ValueError
    # Add units.
    if "val" in units:
        values = values.astype("pint[Eur/MWh]")
    # Test.
    do_test_series(values, weights, expected)


@pytest.mark.parametrize("weightsas", ["list", "dict", "series"])
@pytest.mark.parametrize("units", ["", "val", "wei", "val&wei"])
@pytest.mark.parametrize("indextype", ["int", "DatetimeIndex"])
@pytest.mark.parametrize("zerovalues", ["allzero", "sumzero"])
def test_wavg_series_0weights(
    weightsas: str, units: str, indextype: str, zerovalues: str
):
    """Test if weighted average of a series is correctly identified as error,
    when sum of weights is 0 but not all values are equal."""
    # Starting values.
    i = get_index(4, indextype)
    values = pd.Series([100.0, 200, 300, -150], i)
    weightvalues = [0.0, 0, 0, 0] if zerovalues == "allzero" else [-10.0, 10, 0, 0]
    weights = get_weights(weightvalues, weightsas, i, "wei" in units)
    expected = np.nan
    # Add units.
    if "val" in units:
        values = values.astype("pint[Eur/MWh]")
    # Test.
    do_test_series(values, weights, expected)


@pytest.mark.parametrize("weightsas", ["list", "dict", "series"])
@pytest.mark.parametrize("units", ["", "val", "wei", "val&wei"])
@pytest.mark.parametrize("indextype", ["int", "DatetimeIndex"])
@pytest.mark.parametrize("zerovalues", ["allzero", "sumzero"])
def test_wavg_onevalseries_0weights(
    weightsas: str, units: str, indextype: str, zerovalues: str
):
    """Test if weighted average of a series is correctly calculated,
    when sum of weights is 0 and all values are equal."""
    # Starting values.
    i = get_index(4, indextype)
    values = pd.Series([100.0, 100, 100, 100], i)
    weightvalues = [0.0, 0, 0, 0] if zerovalues == "allzero" else [-10.0, 10, 0, 0]
    weights = get_weights(weightvalues, weightsas, i, "wei" in units)
    expected = 100.0
    # Add units.
    if "val" in units:
        values = values.astype("pint[Eur/MWh]")
        expected = pf.Q_(expected, "Eur/MWh")
    if zerovalues == "allzero":
        expected = np.nan
    # Test.
    do_test_series(values, weights, expected)


@pytest.mark.parametrize("weightsas", ["list", "dict", "series"])
@pytest.mark.parametrize("units", ["", "val", "wei", "val&wei"])
@pytest.mark.parametrize("indextype", ["int", "DatetimeIndex"])
def test_wavg_naseries_0weights(weightsas: str, units: str, indextype: str):
    """Test if weighted average of a series is correctly calculated, when some weights
    are 0 but they have na-values."""
    # Starting values.
    i = get_index(4, indextype)
    values = pd.Series([100.0, 200, np.nan, -150], i)
    weights = get_weights([10.0, 0, 0, 0], weightsas, i, "wei" in units)
    expected = 100.0
    # Add units.
    if "val" in units:
        values = values.astype("pint[Eur/MWh]")
        expected = pf.Q_(expected, "Eur/MWh")
    # Test.
    do_test_series(values, weights, expected)


@pytest.mark.parametrize("weightsas", ["none", "list", "dict", "series", "dataframe"])
@pytest.mark.parametrize("units", ["", "val", "wei", "val&wei"])
@pytest.mark.parametrize("indextype", ["int", "DatetimeIndex"])
@pytest.mark.parametrize("axis", [0, 1])
def test_wavg_dataframe(weightsas: str, axis: int, units: str, indextype: str):
    """Test if weighted average of a dataframe is correctly calculated."""
    # Starting values.
    i = get_index(4, indextype)
    series_a = pd.Series([100.0, 200, 300, -150], i)
    series_b = pd.Series([100.0, -200, 300, -150], i)
    values = pd.DataFrame({"a": series_a, "b": series_b})

    if weightsas != "dataframe":
        if axis == 0:
            weights = get_weights([10.0, 10, 10, 20], weightsas, i, "wei" in units)
            if weightsas == "none":
                expected = pd.Series({"a": 112.5, "b": 12.5})
            else:
                expected = pd.Series({"a": 60.0, "b": -20})
        else:
            weights = get_weights([10.0, 30], weightsas, ["a", "b"], "wei" in units)
            if weightsas == "none":
                expected = pd.Series([100.0, 0, 300, -150], i)
            else:
                expected = pd.Series([100.0, -100, 300, -150], i)
    else:
        weights = get_weights_df(
            {"a": [10.0, 10, 10, 20], "b": [10.0, 10, 30, 0]}, i, "wei" in units
        )
        if axis == 0:
            expected = pd.Series({"a": 60.0, "b": 160})
        else:
            expected = pd.Series([100.0, 0, 300, -150], i)
    if "val" in units:
        values = pd.DataFrame({c: s.astype("pint[Eur/MWh]") for c, s in values.items()})
        expected = expected.astype("pint[Eur/MWh]")
    # Test.
    do_test_dataframe(values, weights, expected, axis=axis)


@pytest.mark.parametrize("weightsas", ["list", "dict", "series", "dataframe"])
@pytest.mark.parametrize("units", ["", "val", "wei", "val&wei"])
@pytest.mark.parametrize("indextype", ["int", "DatetimeIndex"])
@pytest.mark.parametrize("axis", [0, 1])
def test_wavg_dataframe_surplusvalues(
    weightsas: str, axis: int, units: str, indextype: str
):
    """Test if weighted average of a dataframe is correctly calculated if we have more
    values than weights."""
    # Starting values.
    i = get_index(5, indextype)
    series_a = pd.Series([100.0, 200, 300, -150, 99], i)
    series_b = pd.Series([100.0, -200, 300, -150, 99], i)
    series_c = pd.Series([99.0, -99, 99, -99, 99], i)
    values = pd.DataFrame({"a": series_a, "b": series_b, "c": series_c})

    if weightsas != "dataframe":
        if axis == 0:
            weights = get_weights([10.0, 10, 10, 20], weightsas, i[:4], "wei" in units)
            expected = pd.Series({"a": 60.0, "b": -20, "c": -19.8})
        else:
            weights = get_weights([10.0, 30], weightsas, ["a", "b"], "wei" in units)
            expected = pd.Series([100.0, -100, 300, -150, 99], i)
    else:
        weights = get_weights_df(
            {"a": [10.0, 10, 10, 20], "b": [10.0, 10, 30, 0]}, i[:4], "wei" in units
        )
        if axis == 0:
            expected = pd.Series({"a": 60.0, "b": 160})
        else:
            expected = pd.Series([100.0, 0, 300, -150], i[:4])
    if "val" in units:
        values = pd.DataFrame({c: s.astype("pint[Eur/MWh]") for c, s in values.items()})
        expected = expected.astype("pint[Eur/MWh]")
    if weightsas == "list":
        expected = ValueError
    # Test.
    do_test_dataframe(values, weights, expected, axis=axis)


@pytest.mark.parametrize("weightsas", ["list", "dict", "series", "dataframe"])
@pytest.mark.parametrize("units", ["", "val", "wei", "val&wei"])
@pytest.mark.parametrize("indextype", ["int", "DatetimeIndex"])
@pytest.mark.parametrize("axis", [0, 1])
def test_wavg_dataframe_surplusweights(
    weightsas: str, axis: int, units: str, indextype: str
):
    """Test if error is correctly thrown if we have more weights than values."""
    # Starting values.
    i = get_index(5, indextype)
    series_a = pd.Series([100.0, 200, 300, -150], i[:4])
    series_b = pd.Series([100.0, -200, 300, -150], i[:4])
    values = pd.DataFrame({"a": series_a, "b": series_b})

    if weightsas != "dataframe":
        if axis == 0:
            weights = get_weights([10.0, 10, 10, 20, 9], weightsas, i, "wei" in units)
        else:
            weights = get_weights(
                [10.0, 30, 9], weightsas, ["a", "b", "c"], "wei" in units
            )
    else:
        weights = get_weights_df(
            {"a": [10.0, 10, 10, 20, 9], "b": [10.0, 10, 30, 0, 9]}, i, "wei" in units
        )
    if "val" in units:
        values = pd.DataFrame({c: s.astype("pint[Eur/MWh]") for c, s in values.items()})
    # Test.
    do_test_dataframe(values, weights, ValueError, axis=axis)


@pytest.mark.parametrize("weightsas", ["none", "list", "series", "dataframe"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("compatible", ["compatible", "incompatible"])
@pytest.mark.parametrize("units", ["val", "val&wei"])
@pytest.mark.parametrize("indextype", ["int", "DatetimeIndex"])
def test_wavg_dataframe_distinctunits(
    weightsas: str, axis: int, compatible: str, units: str, indextype: str
):
    """Test if weighted average of a dataframe is correctly calculated if it has a mix
    of units."""
    # Starting values.
    i = get_index(4, indextype)
    series_a = pd.Series([100.0, 200, 300, -150], i).astype("pint[Eur/MWh]")
    if compatible == "compatible":
        series_b = pd.Series([10.0, -20, 30, -15], i).astype("pint[ctEur/kWh]")
    else:
        series_b = pd.Series([100.0, -200, 300, -150], i).astype("pint[MW]")
    values = pd.DataFrame({"a": series_a, "b": series_b})

    def exp_res(val_a, val_b):
        if compatible == "compatible":
            return pd.Series({"a": val_a, "b": val_b}, dtype="pint[Eur/MWh]")
        return pd.Series({"a": pf.Q_(val_a, "Eur/MWh"), "b": pf.Q_(val_b, "MW")})

    if weightsas != "dataframe":
        if axis == 0:
            weights = get_weights([10.0, 10, 10, 20], weightsas, i, "wei" in units)
            if weightsas == "none":
                expected = exp_res(112.5, 12.5)
            else:
                expected = exp_res(60.0, -20.0)
        else:
            weights = get_weights([10.0, 30], weightsas, ["a", "b"], "wei" in units)
            if weightsas == "none":
                expected = pd.Series([100.0, 0, 300, -150], i).astype("pint[Eur/MWh]")
            else:
                expected = pd.Series([100.0, -100, 300, -150], i).astype(
                    "pint[Eur/MWh]"
                )
    else:
        weights = get_weights_df(
            {"a": [10.0, 10, 10, 20], "b": [10.0, 10, 30, 0]}, i, "wei" in units
        )
        if axis == 0:
            expected = exp_res(60.0, 160.0)
        else:
            expected = pd.Series([100.0, 0, 300, -150], i).astype("pint[Eur/MWh]")

    # Test.
    if axis == 1 and compatible != "compatible":  # error cases
        expected = Exception
    do_test_dataframe(values, weights, expected, axis=axis)


@pytest.mark.parametrize("weightsas", ["list", "series", "dataframe"])
@pytest.mark.parametrize("indextype", ["int", "DatetimeIndex"])
@pytest.mark.parametrize("axis", [0, 1])
def test_wavg_dataframe_na(weightsas: str, axis: int, indextype: str):
    """Test if weighted average of a dataframe is correctly identified as error,
    when all weights are 0 but not all values are equal."""
    # Starting values.
    i = get_index(4, indextype)
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
    do_test_dataframe(values, weights, expected, axis=axis)


@pytest.mark.parametrize("weightsas", ["list", "series", "dataframe"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("zerovalues", ["allzero", "sumzero"])
@pytest.mark.parametrize("indextype", ["int", "DatetimeIndex"])
@pytest.mark.parametrize("units", ["", "val", "wei", "val&wei"])
def test_wavg_dataframe_0weights(
    weightsas: str, axis: int, units: str, zerovalues: str, indextype: str
):
    """Test if weighted average of a dataframe is correctly identified as error,
    when sum of weights is 0. Some averages are calculated from identical values and should
    result in that value."""
    # Starting values.
    i = get_index(4, indextype)
    values = pd.DataFrame(
        {
            "a": [100.0, 200, 150, 100],
            "b": [100.0, 100, 100, 100],
            "c": [100.0, 100, 200, 250],
        },
        i,
    )
    if weightsas != "dataframe":
        if axis == 0:
            if zerovalues == "allzero":
                weightvalues = [0.0, 0, 0, 0]
                expectedvalues = [np.nan, 100.0, np.NaN]
            else:
                weightvalues = [-10.0, 10, 0, 0]
                expectedvalues = [np.nan, 100.0, 100]
            weights = get_weights(weightvalues, weightsas, i, "wei" in units)
            expected = pd.Series(expectedvalues, list("abc"))
        else:
            if zerovalues == "allzero":
                weightvalues = [0.0, 0, 0]
                expectedvalues = [100.0, np.nan, np.nan, np.nan]
            else:
                weightvalues = [-10.0, 10, 0]
                expectedvalues = [100.0, np.nan, np.nan, 100.0]
            weights = get_weights(weightvalues, weightsas, list("abc"), "wei" in units)
            expected = pd.Series(expectedvalues, i)
    else:
        if zerovalues == "allzero":
            pytest.skip("Testing both zero cases together.")
        weights = get_weights_df(
            {"a": [10.0, 0, 0, -10], "b": [0.0, 0, 0, 0], "c": [-10, 0, 0, 10]},
            i,
            "wei" in units,
        )
        if axis == 0:
            expected = pd.Series([100.0, 100, np.nan], list("abc"))
        else:
            expected = pd.Series([100.0, np.nan, np.nan, np.nan], i)
    if "val" in units:
        values = pd.DataFrame({c: s.astype("pint[Eur/MWh]") for c, s in values.items()})
        expected = expected.astype("pint[Eur/MWh]")
    # Test.
    do_test_dataframe(values, weights, expected, axis=axis)
