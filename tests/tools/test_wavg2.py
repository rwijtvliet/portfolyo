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
    values: np.ndarray
    weights: None | np.ndarray  # 1d or 2d
    axis: int
    expected: Exception | float | np.ndarray


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
        # (
        #     (100.0, 200, 300, -150),
        #     (100.0, -200, 300, -150),
        # ): {
        0: {
            # 0d weights
            None: ((112.5, 12.5), "2dvalues_noneweights_ax0"),
            # 1d weights
            (10, 10, 10, 20): ((60, -20), "2dvalues_1dweights_ax0"),
            (10, 10, 30): ((240, 160), "2dvalues_1dweights_fewerweights_ax0"),
            (10, 10, 10, 20, 9): (ValueError, "2dvalues_1dweights_moreweights_ax0"),
            # 2d weights
            ((10, 10), (10, 10), (10, 30), (20, 0),): (
                # ((10.0, 10, 10, 20), (10.0, 10, 30, 0)): (
                (60, 160),
                "2dvalues_2dweights_ax0",
            ),
            ((10, 10), (10, 10), (10, 30),): (
                # ((10.0, 10, 10), (10.0, 10, 30)): (
                (200, 160),
                "2dvalues_2dweights_fewerweights_ax0_A",
            ),
            ((10,), (10,), (10,), (20,),): (
                (60,),
                "2dvalues_2dweights_fewerweights_ax0_B",
            ),
            ((10, 10), (10, 10), (10, 30), (20, 0), (30, 40),): (
                # ((10.0, 10, 10, 20, 30), (10.0, 10, 30, 0, 40)): (
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
                # ((10.0, 10, 10, 20), (10.0, 10, 30, 0)): (
                (100, 0, 300, -150),
                "2dvalues_2dweights_ax1",
            ),
            ((10, 10), (10, 10), (10, 30),): (
                # ((10.0, 10, 10), (10.0, 10, 30)): (
                (100, 0, 300),
                "2dvalues_2dweights_fewerweights_ax1_A",
            ),
            ((10,), (10,), (10,), (20,),): (
                # ((10.0, 10, 10, 20),): (
                (100.0, 200, 300, -150),
                "2dvalues_2dweights_fewerweights_ax1_B",
            ),
            ((10, 10), (10, 10), (10, 30), (20, 0), (30, 40),): (
                # ((10.0, 10, 10, 20, 30), (10.0, 10, 30, 0, 40)): (
                ValueError,
                "2dvalues_2dweights_moreweights_ax1_A",
            ),
            ((10, 10, 10), (10, 10, 20), (10, 30, 30), (20, 0, 40),): (
                # ((10.0, 10, 10, 20), (10.0, 10, 30, 0), (10, 20, 30, 40)): (
                ValueError,
                "2dvalues_2dweights_moreweights_ax1_B",
            ),
        },
    },
    (
        (100, 100, 99),
        (200, -200, -99),
        (300, 300, 99),
        (-150, -150, -99),
        (99, 99, 99),
    ): {
        # (
        #     (100.0, 200, 300, -150, 99),
        #     (100.0, -200, 300, -150, 99),
        #     (99.0, -99, 99, -99, 99),
        # ): {
        0: {
            # 2d weights
            ((10, 10), (10, 10), (10, 30), (20, 0),): (
                # ((10.0, 10, 10, 20), (10.0, 10, 30, 0)): (
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
                # ((10.0, 10, 10, 20), (10.0, 10, 30, 0)): (
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
def values_has_units(request) -> bool:
    return request.param


# CREATE VALUES


@pytest.fixture(scope="session")
def values1d(values1d_numbers, values_has_units, values1d_rowindex) -> pd.Series:
    s = pd.Series(values1d_numbers, values1d_rowindex)
    return s.astype("pint[Eur/MWh]") if values_has_units else s


@pytest.fixture(scope="session")
def values2d(
    values2d_numbers, values_has_units, values2d_rowindex, values2d_colindex
) -> pd.DataFrame:
    df = pd.DataFrame(values2d_numbers, values2d_rowindex, values2d_colindex)
    return df.astype("pint[Eur/MWh]") if values_has_units else df


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
    # TODO: Can be remove the weights2d_as dependency?
    if weights2d_as not in [dict, pd.DataFrame]:
        return None
    return rowindex[: weights2d_numbers.shape[0]]


@pytest.fixture(scope="session")
def weights2d_colindex(weights2d_numbers, weights2d_as, colindex) -> Sequence | None:
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
    expected_numbers, values1d, weights0d, values_has_units
) -> float | pf.Q_ | type:
    # TODO: Check if we can remove dependency on values1d and weights0d
    if is_exception(expected_numbers):
        return expected_numbers

    return pf.Q_(expected_numbers, "Eur/MWh") if values_has_units else expected_numbers


@pytest.fixture(scope="session")
def expected_for_values1d_and_weights1d(
    expected_numbers, values1d, weights1d, values_has_units
) -> float | pf.Q_ | type:
    if is_exception(expected_numbers):
        return expected_numbers

    if isinstance(weights1d, list | tuple) and len(weights1d) != len(values1d.index):
        return ValueError
    elif isinstance(weights1d, dict) and any(
        w not in values1d.index for w in weights1d
    ):
        return ValueError
    elif isinstance(weights1d, pd.Series) and any(
        w not in values1d.index for w in weights1d.index
    ):
        return ValueError

    return pf.Q_(expected_numbers, "Eur/MWh") if values_has_units else expected_numbers


@pytest.fixture(scope="session")
def expected_for_values2d_and_weights0d(
    expected_numbers,
    weights0d,
    values2d_index_that_remains,
    values_has_units,
) -> pd.Series | type:
    if is_exception(expected_numbers):
        return expected_numbers

    s = pd.Series(expected_numbers, values2d_index_that_remains)
    return s.astype("pint[Eur/MWh]") if values_has_units else s


@pytest.fixture(scope="session")
def expected_for_values2d_and_weights1d(
    expected_numbers,
    weights1d,
    values2d_index_that_remains,
    values2d_index_that_collapses,
    values_has_units,
) -> pd.Series | type:
    if is_exception(expected_numbers):
        return expected_numbers

    if isinstance(weights1d, list | tuple) and len(weights1d) != len(
        values2d_index_that_collapses
    ):
        return ValueError
    elif isinstance(weights1d, dict) and any(
        w not in values2d_index_that_collapses for w in weights1d
    ):
        return ValueError
    elif isinstance(weights1d, pd.Series) and any(
        w not in values2d_index_that_collapses for w in weights1d.index
    ):
        return ValueError
    s = pd.Series(expected_numbers, values2d_index_that_remains)
    return s.astype("pint[Eur/MWh]") if values_has_units else s


@pytest.fixture(scope="session")
def expected_for_values2d_and_weights2d(
    expected_numbers,
    weights2d,
    values2d_rowindex,
    values2d_colindex,
    values_has_units,
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
    s = pd.Series(expected_numbers, values2d_index_that_remains)
    return s.astype("pint[Eur/MWh]") if values_has_units else s


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


# # Numbers in general.
#
#
# @pytest.fixture(scope="session", params=[{"v": [100.0, 200, 300, -150], "e": 112.5}])
# def numbers_0d(request) -> Dict[str, List[float] | float]:
#     return request.param
#
#
# @pytest.fixture(
#     scope="session",
#     params=[{"v": [100.0, 200, 300, -150], "w": [10.0, 0, 10, 20], "e": 25.0}],
# )
# def numbers_1d(request) -> Dict[str, List[float] | float]:
#     return request.param
#
#
#
# # Values.
#
#
#
# @pytest.fixture(scope="session")
# def values(index: Iterable, numbers_1d: Dict, values_has_units: bool) -> Iterable:
#     return create_values(index, numbers_1d["v"], values_has_units)
#
#
# def create_values(index: Iterable, numbers: Dict, values_has_units: bool):
#     s = pd.Series(numbers, index)
#     if values_has_units:
#         return s.astype("pint[Eur/MWh]")
#     return s
#
#
# # Weigths.
#
#
#
#
# @pytest.fixture(scope="session", params=["none", "list", "dict", "series", "df"])
# def weightsas(request) -> str:
#     return request.param
#
#
# @pytest.fixture
# def weights_as_list(numbers_1d, weights_has_units) -> List:
#     if weights_has_units:
#         return [pf.Q_(w, "MWh") for w in numbers_1d["w"]]
#     return numbers_1d["w"]
#
#
# @pytest.fixture
# def weights_as_series(numbers_1d, weights_has_units, index) -> pd.Series:
#     s = pd.Series({i: w for w, i in zip(numbers_1d["w"][::-1], index[::-1])})
#     if weights_has_units:
#         return s.astype("pint[MWh]")
#     return s
#
#
# @pytest.fixture
# def weights_as_dict(numbers_1d, weights_has_units, index) -> Dict:
#     d = {i: w for w, i in zip(numbers_1d["w"][::-1], index[::-1])}
#     if weights_has_units:
#         return {i: pf.Q_(w, "MWh") for i, w in d.items()}
#     return d
#
#
# @pytest.fixture
# def weights(numbers_1d, weights_has_units, index, weightsas):
#     if weightsas == "none":
#         return None
# @pytest.fixture(scope="session")
# def testcase_values(testcase: testcase, values_has_units: bool, index: Iterable) -> pd.DataFrame:
#     s = pd.Series(pd.testcase.values, index)
#     if values_has_units:
#         return s.astype("pint[Eur/MWh]")
#     return s
#
#     if weightsas == "list":
#         return weights_as_list(numbers_1d, weights_has_units)
#     if weightsas == "series":
#         return weights_as_series(numbers_1d, weights_has_units, index)
#
#
# @pytest.fixture
# def weights_as_df(numbers_2d: dict, index: Iterable, units: bool = False):
#     """Get weights to test wavg with, if weightsas == 'dataframe'."""
#     weights = pd.DataFrame(weightsnumbers, index)
#     if units:
#         weights = pd.DataFrame(
#             {c: s.astype("pint[Eur/MWh]") for c, s in weights.items()}
#         )
#     return weights
#
#
# @pytest.fixture
# def weights(weights_as_dict, weights_as_list, weights_as_series):
#     return [weights_as_dict] + [weights_as_list] + [weights_as_series]
#
#
# # Expected.
#
#
# @pytest.fixture(scope="session")
# def expected(expectednumber: float, values_has_units: bool) -> float | pf.Q_:
#     if values_has_units:
#         return pf.Q_(expectednumber, "Eur/MWh")
#     return expectednumber
#


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


# @pytest.mark.parametrize("weightsas", ["none", "list", "dict", "series"])
# @pytest.mark.parametrize("units", ["", "val", "wei", "val&wei"])
# @pytest.mark.parametrize("indextype", ["int", "DatetimeIndex"])
# def test_wavg_series(weightsas: str, units: str, indextype: str):
#     """Test if weighted average of a series is correctly calculated."""
#     # Starting values.
#     i = get_index(4, indextype)
#     values = pd.Series([100.0, 200, 300, -150], i)
#     weights = get_weights([10.0, 0, 10, 20], weightsas, i, "wei" in units)
#     if weightsas == "none":
#         expected = 112.5
#     else:
#         expected = 25.0
#     # Add units.
#     if "val" in units:
#         values = values.astype("pint[Eur/MWh]")
#         expected = pf.Q_(expected, "Eur/MWh")
#     # Test.
#     do_test_series(values, weights, expected)


# @pytest.mark.parametrize("weightsas", ["list", "dict", "series"])
# @pytest.mark.parametrize("units", ["", "val", "wei", "val&wei"])
# @pytest.mark.parametrize("indextype", ["int", "DatetimeIndex"])
# def test_wavg_series_surplusvalues(weightsas: str, units: str, indextype: str):
# """Test if weighted average of a series is correctly calculated if we have more
# values and than weights."""
# # Starting values.
# i = get_index(5, indextype)
# values = pd.Series([100.0, 200, 300, -150, 100], i)
# weights = get_weights([10.0, 0, 10, 20], weightsas, i[:4], "wei" in units)
# expected = 25.0
# # Add units.
# if "val" in units:
#     values = values.astype("pint[Eur/MWh]")
#     expected = pf.Q_(expected, "Eur/MWh")
# if weightsas == "list":
#     expected = ValueError
# # Test.
# do_test_series(values, weights, expected)


# @pytest.mark.parametrize("weightsas", ["list", "dict", "series"])
# @pytest.mark.parametrize("units", ["", "val", "wei", "val&wei"])
# @pytest.mark.parametrize("indextype", ["int", "DatetimeIndex"])
# def test_wavg_series_surplusweights(weightsas: str, units: str, indextype: str):
#     """Test if error is correctly thrown if we have more weights than values."""
#     # Starting values.
#     i = get_index(5, indextype)
#     values = pd.Series([100.0, 200, 300, -150], i[:4])
#     weights = get_weights([10.0, 0, 10, 20, 10], weightsas, i, "wei" in units)
#     expected = ValueError
#     # Add units.
#     if "val" in units:
#         values = values.astype("pint[Eur/MWh]")
#     # Test.
#     do_test_series(values, weights, expected)


# @pytest.mark.parametrize("weightsas", ["list", "dict", "series"])
# @pytest.mark.parametrize("units", ["", "val", "wei", "val&wei"])
# @pytest.mark.parametrize("indextype", ["int", "DatetimeIndex"])
# @pytest.mark.parametrize("zerovalues", ["allzero", "sumzero"])
# def test_wavg_series_0weights(
#     weightsas: str, units: str, indextype: str, zerovalues: str
# ):
#     """Test if weighted average of a series is correctly identified as error,
#     when sum of weights is 0 but not all values are equal."""
#     # Starting values.
#     i = get_index(4, indextype)
#     values = pd.Series([100.0, 200, 300, -150], i)
#     weightvalues = [0.0, 0, 0, 0] if zerovalues == "allzero" else [-10.0, 10, 0, 0]
#     weights = get_weights(weightvalues, weightsas, i, "wei" in units)
#     expected = np.nan
#     # Add units.
#     if "val" in units:
#         values = values.astype("pint[Eur/MWh]")
#     # Test.
#     do_test_series(values, weights, expected)


# @pytest.mark.parametrize("weightsas", ["list", "dict", "series"])
# @pytest.mark.parametrize("units", ["", "val", "wei", "val&wei"])
# @pytest.mark.parametrize("indextype", ["int", "DatetimeIndex"])
# @pytest.mark.parametrize("zerovalues", ["allzero", "sumzero"])
# def test_wavg_onevalseries_0weights(
#     weightsas: str, units: str, indextype: str, zerovalues: str
# ):
#     """Test if weighted average of a series is correctly calculated,
#     when sum of weights is 0 and all values are equal."""
#     # Starting values.
#     i = get_index(4, indextype)
#     values = pd.Series([100.0, 100, 100, 100], i)
#     weightvalues = [0.0, 0, 0, 0] if zerovalues == "allzero" else [-10.0, 10, 0, 0]
#     weights = get_weights(weightvalues, weightsas, i, "wei" in units)
#     expected = 100.0
#     # Add units.
#     if "val" in units:
#         values = values.astype("pint[Eur/MWh]")
#         expected = pf.Q_(expected, "Eur/MWh")
#     if zerovalues == "allzero":
#         expected = np.nan
#     # Test.
#     do_test_series(values, weights, expected)

#
# # @pytest.mark.parametrize("weightsas", ["list", "dict", "series"])
# @pytest.mark.parametrize("units", ["", "val", "wei", "val&wei"])
# @pytest.mark.parametrize("indextype", ["int", "DatetimeIndex"])
# def test_wavg_naseries_0weights(weightsas: str, units: str, indextype: str):
#     """Test if weighted average of a series is correctly calculated, when some weights
#     are 0 but they have na-values."""
#     # Starting values.
#     i = get_index(4, indextype)
#     values = pd.Series([100.0, 200, np.nan, -150], i)
#     weights = get_weights([10.0, 0, 0, 0], weightsas, i, "wei" in units)
#     expected = 100.0
#     # Add units.
#     if "val" in units:
#         values = values.astype("pint[Eur/MWh]")
#         expected = pf.Q_(expected, "Eur/MWh")
#     # Test.
#     do_test_series(values, weights, expected)


# @pytest.mark.parametrize("weightsas", ["list", "dict", "series", "dataframe"])
# @pytest.mark.parametrize("units", ["", "val", "wei", "val&wei"])
# @pytest.mark.parametrize("indextype", ["int", "DatetimeIndex"])
# @pytest.mark.parametrize("axis", [0, 1])
# def test_wavg_dataframe_surplusweights(
#     weightsas: str, axis: int, units: str, indextype: str
# ):
#     """Test if error is correctly thrown if we have more weights than values."""
#     # Starting values.
#     i = get_index(5, indextype)
#     series_a = pd.Series([100.0, 200, 300, -150], i[:4])
#     series_b = pd.Series([100.0, -200, 300, -150], i[:4])
#     values = pd.DataFrame({"a": series_a, "b": series_b})

# if weightsas != "dataframe":
#     if axis == 0:
#         weights = get_weights([10.0, 10, 10, 20, 9], weightsas, i, "wei" in units)
#     else:
#         weights = get_weights(
#             [10.0, 30, 9], weightsas, ["a", "b", "c"], "wei" in units
#         )
# else:
#     weights = get_weights_df(
#         {"a": [10.0, 10, 10, 20, 9], "b": [10.0, 10, 30, 0, 9]}, i, "wei" in units
#     )
# if "val" in units:
#     values = pd.DataFrame({c: s.astype("pint[Eur/MWh]") for c, s in values.items()})
# # Test.
# do_test_dataframe(values, weights, ValueError, axis=axis)


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
