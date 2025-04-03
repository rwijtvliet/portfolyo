import dataclasses
from typing import Iterable, Sequence, Callable
import string
import numpy as np
import pandas as pd
import pytest
import enum
import portfolyo as pf

# PARAMETRIZATION: NUMBERS


class _Values1d(enum.Enum):
    NORMAL = (100.0, 200, 300, -150)
    UNIFORM = (100.0, 100, 100, 100)
    WITHNA = (100.0, 200, np.nan, -150)


class _Weights1d(enum.Enum):
    NORMAL = (10, 0, 10, 20)
    FEWERWEIGHTS = (10, 0, 10)
    MOREWEIGHTS = (10, 10, 10, 20, 9)
    ALLZERO = (0, 0, 0, 0)
    SUMZERO = (-10, 10, 0, 0)


class _Values2d(enum.Enum):
    NORMAL = (
        (100, 100),
        (200, -200),
        (300, 300),
        (-150, -150),
    )
    SOMEUNIFORM = (
        (100, 100, 100),
        (200, 100, 100),
        (150, 100, 200),
        (100, 100, 250),
    )


@dataclasses.dataclass
class _Numbers:
    values: np.ndarray  # 1d or 2d
    weights: None | np.ndarray  # 1d or 2d
    axis: int
    wavg: Exception | float | np.ndarray  # 1d


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(
            (_Values1d.NORMAL, None, None, 112.5),
            id="1dvalues_noneweights",
        ),
        pytest.param(
            (_Values1d.NORMAL, _Weights1d.NORMAL, None, 25),
            id="1dvalues_1dweights",
        ),
        pytest.param(
            (_Values1d.NORMAL, _Weights1d.FEWERWEIGHTS, None, 200),
            id="1dvalues_1dweights_fewerweights",
        ),
        pytest.param(
            (_Values1d.NORMAL, _Weights1d.MOREWEIGHTS, None, ValueError),
            id="1dvalues_1dweights_moreweights",
        ),
        pytest.param(
            (_Values1d.NORMAL, _Weights1d.ALLZERO, None, np.nan),
            id="1dvalues_1dweights_allzero",
        ),
        pytest.param(
            (_Values1d.NORMAL, _Weights1d.SUMZERO, None, np.nan),
            id="1dvalues_1dweights_sumzero",
        ),
        pytest.param(
            (_Values1d.UNIFORM, _Weights1d.ALLZERO, None, np.nan),
            id="1dvalues_uniform_1dweights_allzero",
        ),
        pytest.param(
            ((100.0, 100, 100, 100), (10, -10, 0, 0), None, 100),
            id="1dvalues_uniform_1dweights_sumzero",
        ),
        pytest.param(
            ((100.0, 200, np.nan, -150), (0, 0, 10, 0), None, np.nan),
            id="1dvalues_withna_1dweights_nanot0weight",
        ),
        pytest.param(
            ((100.0, 200, np.nan, -150), (10, 0, 0, 0), None, 100),
            id="1dvalues_withna_1dweights_na0weight",
        ),
        pytest.param(
            ((100.0, 200, np.nan, -150), _Weights1d.ALLZERO, None, np.nan),
            id="1dvalues_withna_1dweights_allzero",
        ),
        pytest.param(
            ((100.0, 200, np.nan, -150), _Weights1d.SUMZERO, None, np.nan),
            id="1dvalues_withna_1dweights_sumzero",
        ),
        pytest.param(
            (_Values2d.NORMAL, None, 0, (112.5, 12.5)),
            id="2dvalues_noneweights_ax0",
        ),
        pytest.param(
            (_Values2d.NORMAL, (10, 10, 10, 20), 0, (60, -20)),
            id="2dvalues_1dweights_ax0",
        ),
        pytest.param(
            (_Values2d.NORMAL, (10, 10, 30), 0, (240, 160)),
            id="2dvalues_1dweights_fewerweights_ax0",
        ),
        pytest.param(
            (_Values2d.NORMAL, (10, 10, 10, 20, 9), 0, ValueError),
            id="2dvalues_1dweights_moreweights_ax0",
        ),
        pytest.param(
            (
                _Values2d.NORMAL,
                ((10, 10), (10, 10), (10, 30), (20, 0)),
                0,
                (60, 160),
            ),
            id="2dvalues_2dweights_ax0",
        ),
        pytest.param(
            (
                _Values2d.NORMAL,
                ((10, 10), (10, 10), (10, 30)),
                0,
                (200, 160),
            ),
            id="2dvalues_2dweights_fewerweights_ax0_A",
        ),
        pytest.param(
            (
                _Values2d.NORMAL,
                ((10,), (10,), (10,), (20,)),
                0,
                (60,),
            ),
            id="2dvalues_2dweights_fewerweights_ax0_B",
        ),
        pytest.param(
            (
                _Values2d.NORMAL,
                ((10, 10), (10, 10), (10, 30), (20, 0), (30, 40)),
                0,
                ValueError,
            ),
            id="2dvalues_2dweights_moreweights_ax0_A",
        ),
        pytest.param(
            (
                _Values2d.NORMAL,
                ((10, 10, 10), (10, 10, 20), (10, 30, 30), (20, 0, 40)),
                0,
                ValueError,
            ),
            id="2dvalues_2dweights_moreweights_ax0_B",
        ),
        pytest.param(
            (_Values2d.NORMAL, None, 1, (100, 0, 300, -150)),
            id="2dvalues_noneweights_ax1",
        ),
        pytest.param(
            (_Values2d.NORMAL, (10, 30), 1, (100, -100, 300, -150)),
            id="2dvalues_1dweights_ax1",
        ),
        pytest.param(
            (
                _Values2d.NORMAL,
                (10,),
                1,
                (100, 200, 300, -150),
            ),
            id="2dvalues_1dweights_fewerweights_ax1",
        ),
        pytest.param(
            (
                _Values2d.NORMAL,
                (10, 10, 30),
                1,
                ValueError,
            ),
            id="2dvalues_1dweights_moreweights_ax1",
        ),
        pytest.param(
            (
                _Values2d.NORMAL,
                ((10, 10), (10, 10), (10, 30), (20, 0)),
                1,
                (100, 0, 300, -150),
            ),
            id="2dvalues_2dweights_ax1",
        ),
        pytest.param(
            (
                _Values2d.NORMAL,
                ((10, 10), (10, 10), (10, 30)),
                1,
                (100, 0, 300),
            ),
            id="2dvalues_2dweights_fewerweights_ax1_A",
        ),
        pytest.param(
            (
                _Values2d.NORMAL,
                ((10,), (10,), (10,), (20,)),
                1,
                _Values1d.NORMAL,
            ),
            id="2dvalues_2dweights_fewerweights_ax1_B",
        ),
        pytest.param(
            (
                _Values2d.NORMAL,
                ((10, 10), (10, 10), (10, 30), (20, 0), (30, 40)),
                1,
                ValueError,
            ),
            id="2dvalues_2dweights_moreweights_ax1_A",
        ),
        pytest.param(
            (
                _Values2d.NORMAL,
                ((10, 10, 10), (10, 10, 20), (10, 30, 30), (20, 0, 40)),
                1,
                ValueError,
            ),
            id="2dvalues_2dweights_moreweights_ax1_B",
        ),
        pytest.param(
            (
                _Values2d.SOMEUNIFORM,
                _Weights1d.ALLZERO,
                0,
                (np.nan, 100, np.nan),
            ),
            id="2dvalues_1dweights_allzero_ax0",
        ),
        pytest.param(
            (
                _Values2d.SOMEUNIFORM,
                _Weights1d.SUMZERO,
                0,
                (np.nan, 100, 100),
            ),
            id="2dvalues_1dweights_sumzero_ax0",
        ),
        pytest.param(
            (
                _Values2d.SOMEUNIFORM,
                ((10, 0, -10), (0, 0, 0), (0, 0, 0), (-10, 0, 10)),
                0,
                (100, 100, np.nan),
            ),
            id="2dvalues_2dweights_zeros_ax0",
        ),
        pytest.param(
            (
                _Values2d.SOMEUNIFORM,
                (0, 0, 0),
                1,
                (100, np.nan, np.nan, np.nan),
            ),
            id="2dvalues_1dweights_allzero_ax1",
        ),
        pytest.param(
            (
                _Values2d.SOMEUNIFORM,
                (10, -10, 0),
                1,
                (100, np.nan, np.nan, 100),
            ),
            id="2dvalues_1dweights_sumzero_ax1",
        ),
        pytest.param(
            (
                _Values2d.SOMEUNIFORM,
                ((10, 0, -10), (0, 0, 0), (0, 0, 0), (-10, 0, 10)),
                1,
                (100, np.nan, np.nan, np.nan),
            ),
            id="2dvalues_2dweights_zeros_ax1",
        ),
        pytest.param(
            (
                ((100, 100, 99), (200, -200, -99), (300, 300, 99), (-150, -150, -99), (99, 99, 99)),
                ((10, 10), (10, 10), (10, 30), (20, 0)),
                0,
                (60, 160),
            ),
            id="2dvalues_2dweights_fewerweights_ax0_C",
        ),
        pytest.param(
            (
                ((100, 100, 99), (200, -200, -99), (300, 300, 99), (-150, -150, -99), (99, 99, 99)),
                (10, 30),
                1,
                (100, -100, 300, -150, 99),
            ),
            id="2dvalues_1dweights_fewerweights_ax1_C",
        ),
        pytest.param(
            (
                ((100, 100, 99), (200, -200, -99), (300, 300, 99), (-150, -150, -99), (99, 99, 99)),
                ((10, 10), (10, 10), (10, 30), (20, 0)),
                1,
                (100, 0, 300, -150),
            ),
            id="2dvalues_2dweights_fewerweights_ax1_C",
        ),
    ],
)
def _numbers(request) -> _Numbers:
    values, weights, axis, wavg = request.param
    # Values: ensure ndarray.
    if isinstance(values, _Values1d | _Values2d):
        values = values.value
    values = np.array(values)
    # Weights: ensure None or ndarray.
    if weights is not None:
        if isinstance(weights, _Weights1d):
            weights = weights.value
        weights = np.array(weights)
    # Wavg: ensure Exception, float, or ndarray.
    if not is_exception(wavg) and isinstance(wavg, tuple):
        wavg = np.array(wavg)
    return _Numbers(values, weights, axis, wavg)


# Split the cases

# . Split testcases based on dimensions of weights.


@pytest.fixture(scope="module")
def weights0d_numbers(_numbers: _Numbers) -> None:
    if _numbers.weights is not None:
        pytest.skip("This test is only for weights that are None.")
    return _numbers.weights  # is always None


@pytest.fixture(scope="module")
def weights1d_numbers(_numbers: _Numbers) -> np.ndarray:
    if _numbers.weights is None or len(_numbers.weights.shape) != 1:
        pytest.skip("This test is only for weights that are 1 dimensional.")
    return _numbers.weights


@pytest.fixture(scope="module")
def weights2d_numbers(_numbers: _Numbers) -> np.ndarray:
    if _numbers.weights is None or len(_numbers.weights.shape) != 2:
        pytest.skip("This test is only for weights that are 2 dimensional.")
    return _numbers.weights


# . Split testcases based on dimensions of values.


@pytest.fixture(scope="module")
def values1d_numbers(_numbers: _Numbers) -> np.ndarray:
    if len(_numbers.values.shape) != 1:
        pytest.skip(
            "This test is only for values that are 1 dimensional (i.e., and expected values that are 0 dimensional."
        )
    return _numbers.values


@pytest.fixture(scope="module")
def values2d_numbers(_numbers: _Numbers) -> np.ndarray:
    if len(_numbers.values.shape) != 2:
        pytest.skip(
            "This test is only for values that are 2 dimensional (i.e., and expected values that are 1 dimensional."
        )
    return _numbers.values


# Other values.


@pytest.fixture(scope="module")
def wavg_numbers(_numbers: _Numbers) -> Exception | float | np.ndarray:
    return _numbers.wavg


@pytest.fixture(scope="module", ids=lambda a: f"axis{a}")
def axis(_numbers: _Numbers) -> int:
    return _numbers.axis


# PARAMETRIZATION: ROW INDEX OF VALUES AND COLUMN INDEX OF VALUES


@pytest.fixture(
    scope="module",
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
    scope="module",
    params=[pytest.param(list(string.ascii_lowercase[:5]), id="abccolumns")],
)
def colindex(request) -> Sequence:
    # Columns. Always 5 values.
    return request.param


@pytest.fixture(scope="module")
def values1d_rowindex(values1d_numbers, rowindex) -> Sequence:
    return rowindex[: values1d_numbers.shape[0]]


@pytest.fixture(scope="module")
def values2d_rowindex(values2d_numbers, rowindex) -> Sequence:
    return rowindex[: values2d_numbers.shape[0]]


@pytest.fixture(scope="module")
def values2d_colindex(values2d_numbers, colindex) -> Sequence:
    return colindex[: values2d_numbers.shape[1]]


@pytest.fixture(scope="module")
def values2d_index_that_collapses(values2d_colindex, values2d_rowindex, axis) -> Sequence:
    return values2d_rowindex if axis == 0 else values2d_colindex


@pytest.fixture(scope="module")
def values2d_index_that_remains(values2d_colindex, values2d_rowindex, axis) -> Sequence:
    return values2d_rowindex if axis == 1 else values2d_colindex


# PARAMETRIZATION: UNITS ON VALUES


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(True, id="valueswithunits"),
        pytest.param(False, id="valueswithoutunits"),
    ],
)
def values1d_has_units(request) -> bool:
    return request.param


class _Values2dUnits(enum.Enum):
    NOUNITS = "valuesnounits"
    UNIFORM = "valuesuniformunits"
    COMPATIBLE = "valuescompatibleunits"
    INCOMPATIBLE = "valuesincompatibleunits"


@pytest.fixture(
    scope="module",
    params=[*_Values2dUnits],
)
def values2d_units(request) -> _Values2dUnits:
    return request.param


@pytest.fixture(scope="module")
def values2d_unithelper(values2d_units):
    if values2d_units in [_Values2dUnits.NOUNITS, _Values2dUnits.UNIFORM]:

        def gen():
            while True:
                yield None

        return gen

    first = "ctEur/kWh" if values2d_units == _Values2dUnits.COMPATIBLE else "MW"

    def gen():
        yield first
        while True:
            yield "Eur/MWh"

    return gen


@pytest.fixture(scope="module")
def values2d_dtypes(values2d_units, values2d_unithelper, values2d_colindex) -> type | str | dict:
    if values2d_units is _Values2dUnits.NOUNITS:
        return float
    elif values2d_units is _Values2dUnits.UNIFORM:
        return "pint[Eur/MWh]"
    return {c: f"pint[{unit}]" for c, unit in zip(values2d_colindex, values2d_unithelper())}


# CREATE VALUES


@pytest.fixture(scope="module")
def values1d(values1d_numbers, values1d_has_units, values1d_rowindex) -> pd.Series:
    s = pd.Series(values1d_numbers, values1d_rowindex)
    return s.astype("pint[Eur/MWh]") if values1d_has_units else s


@pytest.fixture(scope="module")
def values2d(
    values2d_numbers,
    values2d_units,
    values2d_dtypes,
    values2d_rowindex,
    values2d_colindex,
) -> pd.DataFrame:
    df = pd.DataFrame(values2d_numbers, values2d_rowindex, values2d_colindex)
    if values2d_units is _Values2dUnits.COMPATIBLE:
        df[df.columns[0]] /= 10  # because changed to ctEur/kWh
    return df.astype(values2d_dtypes)


# PARAMETRIZATION: TYPE OF WEIGHTS


@pytest.fixture(
    scope="module",
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
    scope="module",
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
    scope="module",
    params=[
        pytest.param(True, id="weightswithunits"),
        pytest.param(False, id="weightswithoutunits"),
        # TODO: weightsdistinctunits? (always causes error)
    ],
)
def weights_has_units(request) -> bool:
    return request.param


# CREATE WEIGHTS


@pytest.fixture(scope="module")
def weights0d(weights0d_numbers) -> None:
    return weights0d_numbers


@pytest.fixture(scope="module")
def weights1d_rowindex(
    weights1d_numbers, weights1d_as, axis, rowindex, colindex
) -> Sequence | None:
    if weights1d_as not in [dict, pd.Series]:
        return None
    index = rowindex if axis == 0 else colindex
    return index[: weights1d_numbers.shape[0]]


@pytest.fixture(scope="module")
def weights1d(
    weights1d_numbers,
    weights_has_units,
    weights1d_rowindex,
    weights1d_as,
) -> Iterable:
    addunitsfn = lambda x: (pf.Q_(x, "MWh") if weights_has_units else x)  # noqa

    # Create the weights: non-mappings.
    if weights1d_as in [list, tuple]:
        return weights1d_as(addunitsfn(w) for w in weights1d_numbers)

    # Create the weights: mappings in reverse order.
    mapping = dict(reversed(list(zip(weights1d_rowindex, weights1d_numbers))))
    if weights1d_as is dict:
        return {k: addunitsfn(v) for k, v in mapping.items()}
    else:  # weights1d_as is pd.Series:
        s = pd.Series(mapping)
        return s.astype("pint[MWh]") if weights_has_units else s


@pytest.fixture(scope="module")
def weights2d_rowindex(weights2d_numbers, weights2d_as, rowindex) -> Sequence | None:
    if weights2d_as in [dict, pd.DataFrame]:
        return rowindex[: weights2d_numbers.shape[0]]


@pytest.fixture(scope="module")
def weights2d_colindex(weights2d_numbers, weights2d_as, colindex) -> Sequence | None:
    if weights2d_as in [dict, pd.DataFrame]:
        return colindex[: weights2d_numbers.shape[1]]


@pytest.fixture(scope="module")
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
    df = df.loc[reversed(weights2d_rowindex), reversed(weights2d_colindex)]
    # add unit
    if weights_has_units:
        df = df.astype("pint[MWh]")
    if weights2d_as is pd.DataFrame:
        return df
    else:  # weights2d_as is dict
        return {**df}


# CREATE WAVG


def is_exception(x) -> bool:
    return isinstance(x, type) and issubclass(x, Exception)


@pytest.fixture(scope="module")
def wavg_for_values1d_and_weights0d(wavg_numbers, values1d_has_units) -> pf.Q_:
    if is_exception(wavg_numbers):
        return wavg_numbers

    return pf.Q_(wavg_numbers, "Eur/MWh" if values1d_has_units else "")


def wavg_for_values1d_and_weights0d_isok(wavg_for_values1d_and_weights0d) -> bool:
    return is_exception(wavg_for_values1d_and_weights0d)


@pytest.fixture(scope="module")
def wavg_for_values1d_and_weights1d(
    wavg_numbers, values1d, weights1d, values1d_has_units
) -> float | pf.Q_ | type:
    if is_exception(wavg_numbers):
        return wavg_numbers

    if isinstance(weights1d, list | tuple) and len(weights1d) != len(values1d.index):
        return ValueError
    elif isinstance(weights1d, dict) and any(w not in values1d.index for w in weights1d.keys()):
        return ValueError
    elif isinstance(weights1d, pd.Series) and any(w not in values1d.index for w in weights1d.index):
        return ValueError

    return pf.Q_(wavg_numbers, "Eur/MWh" if values1d_has_units else "")


def wavg_for_values1d_and_weights1d_isok(wavg_for_values1d_and_weights1d) -> bool:
    return is_exception(wavg_for_values1d_and_weights1d)


@pytest.fixture(scope="module")
def wavg_for_values2d_and_weights0d(
    wavg_numbers,
    values2d_index_that_remains,
    values2d_units,
    values2d_unithelper,
    axis,
) -> pd.Series | type:
    if is_exception(wavg_numbers):
        return wavg_numbers

    if values2d_units == "valuesincompatibleunits" and axis == 1:
        return ValueError  # can't calc wavg across columns with incompatible units

    s = pd.Series(wavg_numbers, values2d_index_that_remains)
    if values2d_units == "valuesnounits":
        return s
    elif values2d_units in ["valuesuniformunits", "valuescompatibleunits"]:
        return s.astype("pint[Eur/MWh]")
    else:  # (values2d_units == "valuesincompatibleunits" and axis == 0)
        # series of quantities
        return pd.Series(
            {i: pf.Q_(num, unit) for (i, num), unit in zip(s.items(), values2d_unithelper())}
        )


def wavg_for_values2d_and_weights0d_isok(wavg_for_values2d_and_weights0d) -> bool:
    return is_exception(wavg_for_values2d_and_weights0d)


@pytest.fixture(scope="module")
def wavg_for_values2d_and_weights1d(
    wavg_numbers,
    weights1d,
    values2d_index_that_remains,
    values2d_index_that_collapses,
    values2d_units,
    values2d_unithelper,
    axis,
) -> pd.Series | type:
    if is_exception(wavg_numbers):
        return wavg_numbers

    if isinstance(weights1d, list | tuple) and len(weights1d) != len(values2d_index_that_collapses):
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

    s = pd.Series(wavg_numbers, values2d_index_that_remains)
    if values2d_units == "valuesnounits":
        return s
    elif values2d_units in ["valuesuniformunits", "valuescompatibleunits"]:
        return s.astype("pint[Eur/MWh]")
    else:  # (values2d_units == "valuesincompatibleunits" and axis == 0)
        return pd.Series(
            {i: pf.Q_(num, unit) for (i, num), unit in zip(s.items(), values2d_unithelper())}
        )


def wavg_for_values2d_and_weights1d_isok(wavg_for_values2d_and_weights1d) -> bool:
    return is_exception(wavg_for_values2d_and_weights1d)


@pytest.fixture(scope="module")
def wavg_for_values2d_and_weights2d(
    wavg_numbers,
    weights2d,
    values2d_rowindex,
    values2d_colindex,
    values2d_units,
    values2d_unithelper,
    axis,
    values2d_index_that_remains,
) -> pd.Series | type:
    if is_exception(wavg_numbers):
        return wavg_numbers

    if isinstance(weights2d, list | tuple) and (
        len(weights2d) != len(values2d_rowindex)
        or any(len(w) != len(values2d_colindex) for w in weights2d)
    ):
        return ValueError
    elif isinstance(weights2d, dict) and any(coli not in values2d_colindex for coli in weights2d):
        return ValueError
    elif isinstance(weights2d, pd.DataFrame) and (
        any(coli not in values2d_colindex for coli in weights2d.columns)
        or any(rowi not in values2d_rowindex for rowi in weights2d.index)
    ):
        return ValueError
    if values2d_units == "valuesincompatibleunits" and axis == 1:
        # can't calc wavg across columns if columns have incompatible units
        return ValueError

    s = pd.Series(wavg_numbers, values2d_index_that_remains)
    if values2d_units == "valuesnounits":
        return s
    elif values2d_units in ["valuesuniformunits", "valuescompatibleunits"]:
        return s.astype("pint[Eur/MWh]")
    else:  # (values2d_units == "valuesincompatibleunits" and axis == 0)
        return pd.Series(
            {i: pf.Q_(num, unit) for (i, num), unit in zip(s.items(), values2d_unithelper())}
        )


@pytest.fixture(scope="module")
def wavg_for_values2d_and_weights2d_isok(wavg_for_values2d_and_weights2d) -> bool:
    return is_exception(wavg_for_values2d_and_weights2d)


# FUNCTIONS TO TEST


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(pf.toolsb.wavg.series, id="seriesfn"),
        pytest.param(pf.toolsb.wavg.general, id="generalfn"),
    ],
)
def wavgfnseries(request) -> Callable:
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(pf.toolsb.wavg.dataframe, id="dataframefn"),
        pytest.param(pf.toolsb.wavg.general, id="generalfn"),
    ],
)
def wavgfndataframe(request) -> Callable:
    return request.param


# DO THE TESTS. FINALLY.


# def test_wavg_values1d_weights0d(values1d, weights0d, expected_for_values1d_and_weights0d):
#     s, weights, expected = values1d, weights0d, expected_for_values1d_and_weights0d
#     testfn = lambda: tools.wavg.series(s, weights)  # noqa
#
#     if is_exception(expected):
#         with pytest.raises(expected):
#             testfn()
#         return
#
#     result = testfn()
#     pf.testing.assert_value_equal(result, expected)
#
#
# def test_wavg_values1d_weights1d(values1d, weights1d, expected_for_values1d_and_weights1d):
#     s, weights, expected = values1d, weights1d, expected_for_values1d_and_weights1d
#     testfn = lambda: tools.wavg.series(s, weights)  # noqa
#
#     if is_exception(expected):
#         with pytest.raises(expected):
#             testfn()
#         return
#
#     result = testfn()
#     pf.testing.assert_value_equal(result, expected)
#
#
# @pytest.mark.parametrize(
#     "val,wei,exp",
#     [
#         pytest.param(
#             "values1d",
#             "weights0d",
#             "expected_for_values1d_and_weights0d",
#         ),
#         pytest.param(
#             "values1d",
#             "weights1d",
#             "expected_for_values1d_and_weights1d",
#         ),
#     ],
# )
# def test_wavg_values1d(request, val, wei, exp):
#     s = val
#     weights = request.getfixturevalue(wei)
#     expected = request.getfixturevalue(exp)
#     testfn = lambda: tools.wavg.series(s, weights)  # noqa
#
#     if is_exception(expected):
#         with pytest.raises(expected):
#             testfn()
#         return
#
#     result = testfn()
#     pf.testing.assert_value_equal(result, expected)
#
#
# def test_wavg_values2d_weights0d(values2d, weights0d, axis, expected_for_values2d_and_weights0d):
#     df, weights, expected = values2d, weights0d, expected_for_values2d_and_weights0d
#     testfn = lambda: tools.wavg.dataframe(df, weights, axis)  # noqa
#
#     if is_exception(expected):
#         with pytest.raises(expected):
#             testfn()
#         return
#
#     result = testfn()
#     pf.testing.assert_series_equal(result, expected, check_names=False, check_like=True)
#
#
# def test_wavg_values2d_weights1d(values2d, weights1d, axis, expected_for_values2d_and_weights1d):
#     df, weights, expected = values2d, weights1d, expected_for_values2d_and_weights1d
#     testfn = lambda: tools.wavg.dataframe(df, weights, axis)  # noqa
#
#     if is_exception(expected):
#         with pytest.raises(expected):
#             testfn()
#         return
#
#     result = testfn()
#     pf.testing.assert_series_equal(result, expected, check_names=False, check_like=True)
#
#
# def test_wavg_values2d_weights2d(values2d, weights2d, axis, expected_for_values2d_and_weights2d):
#     df, weights, expected = values2d, weights2d, expected_for_values2d_and_weights2d
#     testfn = lambda: tools.wavg.dataframe(df, weights, axis)  # noqa
#
#     if is_exception(expected):
#         with pytest.raises(expected):
#             testfn()
#         return
#
#     result = testfn()
#     pf.testing.assert_series_equal(result, expected, check_names=False, check_like=True)
#
#
# @pytest.mark.parametrize(
#     "val,wei,exp",
#     [
#         pytest.param(
#             "values2d",
#             "weights0d",
#             "expected_for_values2d_and_weights0d",
#         ),
#         pytest.param(
#             "values2d",
#             "weights1d",
#             "expected_for_values2d_and_weights1d",
#         ),
#         pytest.param(
#             "values2d",
#             "weights2d",
#             "expected_for_values2d_and_weights2d",
#         ),
#     ],
# )
# def test_wavg_values2d(request, val, wei, exp, axis):
#     df = val
#     weights = request.getfixturevalue(wei)
#     expected = request.getfixturevalue(exp)
#     testfn = lambda: tools.wavg.dataframe(df, weights, axis)  # noqa
#
#     if is_exception(expected):
#         with pytest.raises(expected):
#             testfn()
#         return
#
#     result = testfn()
#     pf.testing.assert_series_equal(result, expected)
