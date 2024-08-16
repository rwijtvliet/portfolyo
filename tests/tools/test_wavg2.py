import dataclasses
import string
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pytest

import portfolyo as pf
from portfolyo import tools

# PARAMETRIZATION: NUMBERS


@dataclasses.dataclass
class Testcase:
    values: np.ndarray  # 1d or 2d
    weights: None | np.ndarray  # 1d or 2d
    axis: int
    expected: Exception | float | np.ndarray  # 1d


TESTCASES = {  # values: axis: weights : (expected, id)
    # 1d values, so 0d result
    (100.0, 200, 300, -150): {
        None: {
            # 0d weights
            None: (112.5, "1dvalues_noneweights"),
            # 1d weights
            (10, 0, 10, 20): (25, "1dvalues_1dweights"),
            (10, 0, 10): (200, "1dvalues_1dweights_fewerweights"),
            (10, 10, 10, 20, 9): (ValueError, "1dvalues_1dweights_moreweights"),
            (0, 0, 0, 0): (np.nan, "1dvalues_1dweights_allzero"),
            (-10, 10, 0, 0): (np.nan, "1dvalues_1dweights_sumzero"),
        },
    },
    (100.0, 100, 100, 100): {
        None: {
            # 1d weights
            (0, 0, 0, 0): (np.nan, "1dvalues_uniform_1dweights_allzero"),
            (10, -10, 0, 0): (100, "1dvalues_uniform_1dweights_sumzero"),
        },
    },
    (100.0, 200, np.nan, -150): {
        None: {
            # 1d weights
            (0, 0, 10, 0): (np.nan, "1dvalues_withna_1dweights_nanot0weight"),
            (10, 0, 0, 0): (100, "1dvalues_withna_1dweights_na0weight"),
            (0, 0, 0, 0): (np.nan, "1dvalues_withna_1dweights_allzero"),
            (-10, 10, 0, 0): (np.nan, "1dvalues_withna_1dweights_sumzero"),
        },
    },
    # 2d values, so 1d result
    ((100, 100), (200, -200), (300, 300), (-150, -150),): {
        0: {
            # 0d weights
            None: ((112.5, 12.5), "2dvalues_noneweights_ax0"),
            # 1d weights
            (10, 10, 10, 20): ((60, -20), "2dvalues_1dweights_ax0"),
            (10, 10, 30): ((240, 160), "2dvalues_1dweights_fewerweights_ax0"),
            (10, 10, 10, 20, 9): (ValueError, "2dvalues_1dweights_moreweights_ax0"),
            # 2d weights
            ((10, 10), (10, 10), (10, 30), (20, 0),): (
                (60, 160),
                "2dvalues_2dweights_ax0",
            ),
            ((10, 10), (10, 10), (10, 30),): (
                (200, 160),
                "2dvalues_2dweights_fewerweights_ax0_A",
            ),
            ((10,), (10,), (10,), (20,),): (
                (60,),
                "2dvalues_2dweights_fewerweights_ax0_B",
            ),
            ((10, 10), (10, 10), (10, 30), (20, 0), (30, 40),): (
                ValueError,
                "2dvalues_2dweights_moreweights_ax0_A",
            ),
            ((10, 10, 10), (10, 10, 20), (10, 30, 30), (20, 0, 40),): (
                ValueError,
                "2dvalues_2dweights_moreweights_ax0_B",
            ),
        },
        1: {
            # 0d weights
            None: ((100, 0, 300, -150), "2dvalues_noneweights_ax1"),
            # 1d weights
            (10, 30): ((100, -100, 300, -150), "2dvalues_1dweights_ax1"),
            (10,): ((100, 200, 300, -150), "2dvalues_1dweights_fewerweights_ax1"),
            (10, 10, 30): (ValueError, "2dvalues_1dweights_moreweights_ax1"),
            # 2d weights
            ((10, 10), (10, 10), (10, 30), (20, 0),): (
                (100, 0, 300, -150),
                "2dvalues_2dweights_ax1",
            ),
            ((10, 10), (10, 10), (10, 30),): (
                (100, 0, 300),
                "2dvalues_2dweights_fewerweights_ax1_A",
            ),
            ((10,), (10,), (10,), (20,),): (
                (100.0, 200, 300, -150),
                "2dvalues_2dweights_fewerweights_ax1_B",
            ),
            ((10, 10), (10, 10), (10, 30), (20, 0), (30, 40),): (
                ValueError,
                "2dvalues_2dweights_moreweights_ax1_A",
            ),
            ((10, 10, 10), (10, 10, 20), (10, 30, 30), (20, 0, 40),): (
                ValueError,
                "2dvalues_2dweights_moreweights_ax1_B",
            ),
        },
    },
    ((100, 100, 100), (200, 100, 100), (150, 100, 200), (100, 100, 250),): {
        0: {
            (0, 0, 0, 0): ((np.nan, 100, np.nan), "2dvalues_1dweights_allzero_ax0"),
            (-10, 10, 0, 0): ((np.nan, 100, 100), "2dvalues_1dweights_sumzero_ax0"),
            (
                (10, 0, -10),
                (0, 0, 0),
                (0, 0, 0),
                (-10, 0, 10),
            ): ((100, 100, np.nan), "2dvalues_2dweights_zeros_ax0"),
        },
        1: {
            (0, 0, 0): (
                (100, np.nan, np.nan, np.nan),
                "2dvalues_1dweights_allzero_ax1",
            ),
            (10, -10, 0): (
                (100, np.nan, np.nan, 100),
                "2dvalues_1dweights_sumzero_ax1",
            ),
            (
                (10, 0, -10),
                (0, 0, 0),
                (0, 0, 0),
                (-10, 0, 10),
            ): ((100, np.nan, np.nan, np.nan), "2dvalues_2dweights_zeros_ax1"),
        },
    },
    (
        (100, 100, 99),
        (200, -200, -99),
        (300, 300, 99),
        (-150, -150, -99),
        (99, 99, 99),
    ): {
        0: {
            # 2d weights
            ((10, 10), (10, 10), (10, 30), (20, 0),): (
                (60, 160),
                "2dvalues_2dweights_fewerweights_ax0_C",
            ),
        },
        1: {
            # 1d weights
            (10, 30): (
                (100, -100, 300, -150, 99),
                "2dvalues_1dweights_fewerweights_ax1_C",
            ),
            # 2d weights
            ((10, 10), (10, 10), (10, 30), (20, 0),): (
                (100, 0, 300, -150),
                "2dvalues_2dweights_fewerweights_ax1_C",
            ),
        },
    },
}


@pytest.fixture(
    scope="session",
    params=[
        (values, weights, axis, expected, id)
        for values, subdict1 in TESTCASES.items()
        for axis, subdict2 in subdict1.items()
        for weights, (expected, id) in subdict2.items()
    ],
    ids=lambda tupl: tupl[-1],
)
def testcase(request) -> Testcase:
    values, weights, axis, expected, _ = request.param
    values = np.array(values)
    if weights is not None:
        weights = np.array(weights)
    if not is_exception(expected) and isinstance(expected, tuple):
        expected = np.array(expected)
    return Testcase(values, weights, axis, expected)


# Split the cases

# . Split testcases based on dimensions of weights.


@pytest.fixture(scope="session")
def weights0d_numbers(testcase) -> None:
    if testcase.weights is not None:
        pytest.skip("This test is only for weights that are None.")
    return testcase.weights  # is always None


@pytest.fixture(scope="session")
def weights1d_numbers(testcase) -> np.ndarray:
    if testcase.weights is None or len(testcase.weights.shape) != 1:
        pytest.skip("This test is only for weights that are 1 dimensional.")
    return testcase.weights


@pytest.fixture(scope="session")
def weights2d_numbers(testcase) -> np.ndarray:
    if testcase.weights is None or len(testcase.weights.shape) != 2:
        pytest.skip("This test is only for weights that are 2 dimensional.")
    return testcase.weights


# . Split test cases based on dimensions of values.


@pytest.fixture(scope="session")
def values1d_numbers(testcase) -> np.ndarray:
    if len(testcase.values.shape) != 1:
        pytest.skip(
            "This test is only for values that are 1 dimensional (i.e., and expected values that are 0 dimensional."
        )
    return testcase.values


@pytest.fixture(scope="session")
def values2d_numbers(testcase) -> np.ndarray:
    if len(testcase.values.shape) != 2:
        pytest.skip(
            "This test is only for values that are 2 dimensional (i.e., and expected values that are 1 dimensional."
        )
    return testcase.values


# Other values.


@pytest.fixture(scope="session")
def expected_numbers(testcase) -> Exception | float | np.ndarray:
    return testcase.expected


@pytest.fixture(scope="session", ids=lambda a: f"axis{a}")
def axis(testcase) -> int:
    return testcase.axis


# PARAMETRIZATION: ROW INDEX OF VALUES AND COLUMN INDEX OF VALUES


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(list(range(5)), id="intindex"),
        pytest.param(pd.date_range("2020", freq="D", periods=5), id="datetimeindex"),
        pytest.param(list(string.ascii_lowercase[:5]), id="abcindex"),
    ],
)
def rowindex(request) -> Sequence:
    # Index. Always 5 values.
    return request.param


@pytest.fixture(
    scope="session",
    params=[pytest.param(list(string.ascii_lowercase[:5]), id="abccolumns")],
)
def colindex(request) -> Sequence:
    # Columns. Always 5 values.
    return request.param


@pytest.fixture(scope="session")
def values1d_rowindex(values1d_numbers, rowindex) -> Sequence:
    return rowindex[: values1d_numbers.shape[0]]


@pytest.fixture(scope="session")
def values2d_rowindex(values2d_numbers, rowindex) -> Sequence:
    return rowindex[: values2d_numbers.shape[0]]


@pytest.fixture(scope="session")
def values2d_colindex(values2d_numbers, colindex) -> Sequence:
    return colindex[: values2d_numbers.shape[1]]


@pytest.fixture(scope="session")
def values2d_index_that_collapses(
    values2d_colindex, values2d_rowindex, axis
) -> Sequence:
    return values2d_rowindex if axis == 0 else values2d_colindex


@pytest.fixture(scope="session")
def values2d_index_that_remains(values2d_colindex, values2d_rowindex, axis) -> Sequence:
    return values2d_rowindex if axis == 1 else values2d_colindex


# PARAMETRIZATION: UNITS ON VALUES


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(True, id="valueswithunits"),
        pytest.param(False, id="valueswithoutunits"),
    ],
)
def values1d_has_units(request) -> bool:
    return request.param


@pytest.fixture(
    scope="session",
    params=[
        "valuesnounits",
        "valuesuniformunits",
        "valuescompatibleunits",
        "valuesincompatibleunits",
    ],
)
def values2d_units(request) -> str:
    return request.param


@pytest.fixture(scope="session")
def values2d_unithelper(values2d_units):
    if values2d_units in ["valuesnounits", "valuesuniformunits"]:
        return None

    first = "ctEur/kWh" if values2d_units == "valuescompatibleunits" else "MW"

    def gen():
        yield first
        while True:
            yield "Eur/MWh"

    return gen


@pytest.fixture(scope="session")
def values2d_dtypes(values2d_unithelper, values2d_colindex) -> type | str | dict:
    if values2d_units == "valuesnounits":
        return float
    elif values2d_units == "valuesunitformunits":
        return "pint[Eur/MWh]"
    return {
        c: f"pint[{unit}]]" for c, unit in zip(values2d_colindex, values2d_unithelper)
    }


# CREATE VALUES


@pytest.fixture(scope="session")
def values1d(values1d_numbers, values1d_has_units, values1d_rowindex) -> pd.Series:
    s = pd.Series(values1d_numbers, values1d_rowindex)
    return s.astype("pint[Eur/MWh]") if values1d_has_units else s


@pytest.fixture(scope="session")
def values2d(
    values2d_numbers,
    values2d_units,
    values2d_dtypes,
    values2d_rowindex,
    values2d_colindex,
) -> pd.DataFrame:
    df = pd.DataFrame(values2d_numbers, values2d_rowindex, values2d_colindex)
    if values2d_units == "valuescompatibleunits":
        firstcol = df.columns[0]
        df[firstcol] /= 10
    return df.astype(values2d_dtypes)


# PARAMETRIZATION: TYPE OF WEIGHTS


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(list, id="weightsaslist"),
        pytest.param(tuple, id="weightsastuple"),
        pytest.param(dict, id="weightsasdict"),
        pytest.param(pd.Series, id="weightsasseries"),
    ],
)
def weights1d_as(request) -> type:
    return request.param


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(list, id="weightsasnestedlist"),
        pytest.param(tuple, id="weightsasnestedtuple"),
        # pytest.param(dict, id="weightsasdictofseries"), wavg cannot figure out if dict is dict of values or dict of series
        pytest.param(pd.DataFrame, id="weightsasdf"),
    ],
)
def weights2d_as(request) -> type:
    return request.param


# PARAMETRIZATION: UNITS ON WEIGHTS


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(True, id="weightswithunits"),
        pytest.param(False, id="weightswithoutunits"),
    ],
)
def weights_has_units(request) -> bool:
    return request.param


# CREATE WEIGHTS


@pytest.fixture(scope="session")
def weights0d(weights0d_numbers) -> None:
    return weights0d_numbers


@pytest.fixture(scope="session")
def weights1d_rowindex(
    weights1d_numbers, weights1d_as, axis, rowindex, colindex
) -> Sequence | None:
    if weights1d_as not in [dict, pd.Series]:
        return None
    index = rowindex if axis == 0 else colindex
    return index[: weights1d_numbers.shape[0]]


@pytest.fixture(scope="session")
def weights1d(
    weights1d_numbers,
    weights_has_units,
    weights1d_rowindex,
    weights1d_as,
) -> Iterable:
    addunitsfn = lambda x: (pf.Q_(x, "MWh") if weights_has_units else x)  # noqa

    # Create the weights: non-mappings.
    if weights1d_as in [list, tuple]:
        numbers = [addunitsfn(num) for num in weights1d_numbers]
        if weights1d_as is list:
            return numbers
        else:  # weights1d is tuple:
            return tuple(numbers)

    # Create the weights: mappings in reverse order
    if weights1d_as is dict:
        return {
            i: addunitsfn(num)
            for i, num in zip(weights1d_rowindex[::-1], weights1d_numbers[::-1])
        }
    else:  # weights1d_as is pd.Series:
        s = pd.Series(weights1d_numbers[::-1], weights1d_rowindex[::-1])
        return s.astype("pint[MWh]") if weights_has_units else s


@pytest.fixture(scope="session")
def weights2d_rowindex(weights2d_numbers, weights2d_as, rowindex) -> Sequence | None:
    # TODO: Can we remove the weights2d_as dependency?
    if weights2d_as not in [dict, pd.DataFrame]:
        return None
    return rowindex[: weights2d_numbers.shape[0]]


@pytest.fixture(scope="session")
def weights2d_colindex(weights2d_numbers, weights2d_as, colindex) -> Sequence | None:
    # TODO: Can we remove the weights2d_as dependency?
    if weights2d_as not in [dict, pd.DataFrame]:
        return None
    return colindex[: weights2d_numbers.shape[1]]


@pytest.fixture(scope="session")
def weights2d(
    weights2d_numbers,
    weights_has_units,
    weights2d_rowindex,
    weights2d_colindex,
    weights2d_as,
) -> Iterable:
    # Create the weights: non-mappings.
    if weights2d_as in [list, tuple]:
        addunitsfn = lambda x: (pf.Q_(x, "MWh") if weights_has_units else x)  # noqa
        numbers = [[addunitsfn(num) for num in nums] for nums in weights2d_numbers]
        if weights2d_as is list:
            return numbers
        else:  # weights2d is tuple
            return tuple(tuple(nums) for nums in numbers)

    # Create the weights: mappings.
    df = pd.DataFrame(weights2d_numbers, weights2d_rowindex, weights2d_colindex)
    # shuffle
    df = df.loc[weights2d_rowindex[::-1], weights2d_colindex[::-1]]
    # add unit
    if weights_has_units:
        df = df.astype("pint[MWh]")
    if weights2d_as is pd.DataFrame:
        return df
    else:  # weights2d_as is dict
        return {**df}


# CREATE EXPECTED


def is_exception(x: Any) -> bool:
    return isinstance(x, type) and issubclass(x, Exception)


@pytest.fixture(scope="session")
def expected_for_values1d_and_weights0d(
    expected_numbers, values1d, weights0d, values1d_has_units
) -> float | pf.Q_ | type:
    # TODO: Check if we can remove dependency on values1d and weights0d
    if is_exception(expected_numbers):
        return expected_numbers

    return (
        pf.Q_(expected_numbers, "Eur/MWh") if values1d_has_units else expected_numbers
    )


@pytest.fixture(scope="session")
def expected_for_values1d_and_weights1d(
    expected_numbers, values1d, weights1d, values1d_has_units
) -> float | pf.Q_ | type:
    if is_exception(expected_numbers):
        return expected_numbers

    if isinstance(weights1d, list | tuple) and len(weights1d) != len(values1d.index):
        return ValueError
    elif isinstance(weights1d, dict) and any(
        w not in values1d.index for w in weights1d.keys()
    ):
        return ValueError
    elif isinstance(weights1d, pd.Series) and any(
        w not in values1d.index for w in weights1d.index
    ):
        return ValueError

    return (
        pf.Q_(expected_numbers, "Eur/MWh") if values1d_has_units else expected_numbers
    )


@pytest.fixture(scope="session")
def expected_for_values2d_and_weights0d(
    expected_numbers,
    values2d_index_that_remains,
    values2d_units,
    values2d_unithelper,
    axis,
) -> pd.Series | type:
    if is_exception(expected_numbers):
        return expected_numbers

    if values2d_units == "valuesincompatibleunits" and axis == 1:
        return ValueError  # can't calc wavg across columns with incompatible units

    s = pd.Series(expected_numbers, values2d_index_that_remains)
    if values2d_units == "valuesnounits":
        return s
    elif values2d_units in ["valuesuniformunits", "valuescompatibleunits"]:
        return s.astype("pint[Eur/MWh]")
    else:  # (values2d_units == "valuesincompatibleunits" and axis == 0)
        # series of quantities
        return pd.Series(
            {
                i: pf.Q_(num, unit)
                for (i, num), unit in zip(s.items(), values2d_unithelper())
            }
        )


@pytest.fixture(scope="session")
def expected_for_values2d_and_weights1d(
    expected_numbers,
    weights1d,
    values2d_index_that_remains,
    values2d_index_that_collapses,
    values2d_units,
    values2d_unithelper,
    axis,
) -> pd.Series | type:
    if is_exception(expected_numbers):
        return expected_numbers

    if isinstance(weights1d, list | tuple) and len(weights1d) != len(
        values2d_index_that_collapses
    ):
        return ValueError
    elif isinstance(weights1d, dict) and any(
        w not in values2d_index_that_collapses for w in weights1d.keys()
    ):
        return ValueError
    elif isinstance(weights1d, pd.Series) and any(
        w not in values2d_index_that_collapses for w in weights1d.index
    ):
        return ValueError
    if values2d_units == "valuesincompatibleunits" and axis == 1:
        # can't calc wavg across columns if columns have incompatible units
        return ValueError

    s = pd.Series(expected_numbers, values2d_index_that_remains)
    if values2d_units == "valuesnounits":
        return s
    elif values2d_units in ["valuesuniformunits", "valuescompatibleunits"]:
        return s.astype("pint[Eur/MWh]")
    else:  # (values2d_units == "valuesincompatibleunits" and axis == 0)
        return pd.Series(
            {
                i: pf.Q_(num, unit)
                for (i, num), unit in zip(s.items(), values2d_unithelper())
            }
        )


@pytest.fixture(scope="session")
def expected_for_values2d_and_weights2d(
    expected_numbers,
    weights2d,
    values2d_rowindex,
    values2d_colindex,
    values2d_units,
    values2d_unithelper,
    axis,
    values2d_index_that_remains,
) -> pd.Series | type:
    if is_exception(expected_numbers):
        return expected_numbers

    if isinstance(weights2d, list | tuple) and (
        len(weights2d) != len(values2d_rowindex)
        or any(len(w) != len(values2d_colindex) for w in weights2d)
    ):
        return ValueError
    elif isinstance(weights2d, dict) and any(
        coli not in values2d_colindex for coli in weights2d
    ):
        return ValueError
    elif isinstance(weights2d, pd.DataFrame) and (
        any(coli not in values2d_colindex for coli in weights2d.columns)
        or any(rowi not in values2d_rowindex for rowi in weights2d.index)
    ):
        return ValueError
    if values2d_units == "valuesincompatibleunits" and axis == 1:
        # can't calc wavg across columns if columns have incompatible units
        return ValueError

    s = pd.Series(expected_numbers, values2d_index_that_remains)
    if values2d_units == "valuesnounits":
        return s
    elif values2d_units in ["valuesuniformunits", "valuescompatibleunits"]:
        return s.astype("pint[Eur/MWh]")
    else:  # (values2d_units == "valuesincompatibleunits" and axis == 0)
        return pd.Series(
            {
                i: pf.Q_(num, unit)
                for (i, num), unit in zip(s.items(), values2d_unithelper())
            }
        )


# DO THE TESTS. FINALLY.


def test_wavg_values1d_weights0d(
    values1d, weights0d, expected_for_values1d_and_weights0d
):
    s, weights, expected = values1d, weights0d, expected_for_values1d_and_weights0d
    testfn = lambda: tools.wavg.series(s, weights)  # noqa

    if is_exception(expected):
        with pytest.raises(expected):
            testfn()
        return

    result = testfn()
    pf.testing.assert_value_equal(result, expected)


def test_wavg_values1d_weights1d(
    values1d, weights1d, expected_for_values1d_and_weights1d
):
    s, weights, expected = values1d, weights1d, expected_for_values1d_and_weights1d
    testfn = lambda: tools.wavg.series(s, weights)  # noqa

    if is_exception(expected):
        with pytest.raises(expected):
            testfn()
        return

    result = testfn()
    pf.testing.assert_value_equal(result, expected)


@pytest.mark.parametrize(
    "val,wei,exp",
    [
        pytest.param(
            "values1d",
            "weights0d",
            "expected_for_values1d_and_weights0d",
        ),
        pytest.param(
            "values1d",
            "weights1d",
            "expected_for_values1d_and_weights1d",
        ),
    ],
)
def test_wavg_values1d(request, val, wei, exp):
    s = val
    weights = request.getfixturevalue(wei)
    expected = request.getfixturevalue(exp)
    testfn = lambda: tools.wavg.series(s, weights)  # noqa

    if is_exception(expected):
        with pytest.raises(expected):
            testfn()
        return

    result = testfn()
    pf.testing.assert_value_equal(result, expected)


def test_wavg_values2d_weights0d(
    values2d, weights0d, axis, expected_for_values2d_and_weights0d
):
    df, weights, expected = values2d, weights0d, expected_for_values2d_and_weights0d
    testfn = lambda: tools.wavg.dataframe(df, weights, axis)  # noqa

    if is_exception(expected):
        with pytest.raises(expected):
            testfn()
        return

    result = testfn()
    pf.testing.assert_series_equal(result, expected, check_names=False, check_like=True)


def test_wavg_values2d_weights1d(
    values2d, weights1d, axis, expected_for_values2d_and_weights1d
):
    df, weights, expected = values2d, weights1d, expected_for_values2d_and_weights1d
    testfn = lambda: tools.wavg.dataframe(df, weights, axis)  # noqa

    if is_exception(expected):
        with pytest.raises(expected):
            testfn()
        return

    result = testfn()
    pf.testing.assert_series_equal(result, expected, check_names=False, check_like=True)


def test_wavg_values2d_weights2d(
    values2d, weights2d, axis, expected_for_values2d_and_weights2d
):
    df, weights, expected = values2d, weights2d, expected_for_values2d_and_weights2d
    testfn = lambda: tools.wavg.dataframe(df, weights, axis)  # noqa

    if is_exception(expected):
        with pytest.raises(expected):
            testfn()
        return

    result = testfn()
    pf.testing.assert_series_equal(result, expected, check_names=False, check_like=True)


@pytest.mark.parametrize(
    "val,wei,exp",
    [
        pytest.param(
            "values2d",
            "weights0d",
            "expected_for_values2d_and_weights0d",
        ),
        pytest.param(
            "values2d",
            "weights1d",
            "expected_for_values2d_and_weights1d",
        ),
        pytest.param(
            "values2d",
            "weights2d",
            "expected_for_values2d_and_weights2d",
        ),
    ],
)
def test_wavg_values2d(request, val, wei, exp, axis):
    df = val
    weights = request.getfixturevalue(wei)
    expected = request.getfixturevalue(exp)
    testfn = lambda: tools.wavg.dataframe(df, weights, axis)  # noqa

    if is_exception(expected):
        with pytest.raises(expected):
            testfn()
        return

    result = testfn()
    pf.testing.assert_series_equal(result, expected)
