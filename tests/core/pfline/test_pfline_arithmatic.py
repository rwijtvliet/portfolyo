from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd
import pytest

import portfolyo as pf
from portfolyo import (
    FlatPfLine,
    Kind,
    NestedPfLine,
    create_flatpfline,
    create_pfline,
    dev,
    testing,
)

_seed = 3  # any fixed seed


# TODO: nestedPfLine, quantity, float, 0, negation


def id_fn(data: Any):
    """Readable id of test case"""
    if isinstance(data, Dict):
        return str({key: id_fn(val) for key, val in data.items()})
    elif isinstance(data, pd.Series):
        if isinstance(data.index, pd.DatetimeIndex):
            return f"Timeseries (dtype: {data.dtype})"
        return f"Series (idx: {''.join(str(i) for i in data.index)})"
    elif isinstance(data, pd.DataFrame):
        return f"Df (columns: {''.join(str(c) for c in data.columns)})"
    elif isinstance(data, FlatPfLine):
        return f"Singlepfline_{data.kind}"
    elif isinstance(data, NestedPfLine):
        return f"Multipfline_{data.kind}"
    elif isinstance(data, str):
        return data
    elif isinstance(data, pf.Q_):
        return f"Quantity ({data.units})"
    elif isinstance(data, type):
        return data.__name__
    elif isinstance(data, Kind):
        return str(data)
    elif isinstance(data, Case):
        return f"{id_fn(data.value1)}-{data.operation}-{id_fn(data.value2)}-{id_fn(data.expected)}"
    return type(data).__name__


# Set 1.
i1 = pd.date_range("2020", freq="MS", periods=3, tz="Europe/Berlin")
ref_series = {
    "q": pd.Series([-3.5, 5, -5], i1),
    "p": pd.Series([300.0, 150, 100], i1),
    "r": pd.Series([-1050.0, 750, -500], i1),
    "nodim": pd.Series([2, -1.5, 10], i1),
}
ref_flatset = {
    Kind.VOLUME: create_flatpfline({"q": ref_series["q"]}),
    Kind.PRICE: create_flatpfline({"p": ref_series["p"]}),
    Kind.REVENUE: create_flatpfline({"r": ref_series["r"]}),
    Kind.COMPLETE: create_flatpfline({"q": ref_series["q"], "p": ref_series["p"]}),
    "nodim": ref_series["nodim"],
}
series1 = {
    "q": pd.Series([-3.0, 4, -5], i1),
    "p": pd.Series([400.0, 150, -100], i1),
    "r": pd.Series([-1200.0, 600, 500], i1),
    "nodim": pd.Series([1.0, -10, 2], i1),
}
flatset1 = {
    Kind.VOLUME: create_flatpfline({"q": series1["q"]}),
    Kind.PRICE: create_flatpfline({"p": series1["p"]}),
    Kind.REVENUE: create_flatpfline({"r": series1["r"]}),
    Kind.COMPLETE: create_flatpfline({"q": series1["q"], "p": series1["p"]}),
    "nodim": series1["nodim"],
}

# Set 2. Partial overlap with set 1.
i2 = pd.date_range("2020-02", freq="MS", periods=3, tz="Europe/Berlin")

series2 = {
    "q": pd.Series([-15.0, -20, -4], i2),
    "p": pd.Series([400.0, 50, 50], i2),
    "r": pd.Series([4000.0, 500, 500], i2),
    "nodim": pd.Series([1.0, -2.0, 4], i2),
}
flatset2 = {
    Kind.VOLUME: create_flatpfline({"q": series2["q"]}),
    Kind.PRICE: create_flatpfline({"p": series2["p"]}),
    Kind.REVENUE: create_flatpfline({"r": series2["r"]}),
    Kind.COMPLETE: create_flatpfline({"q": series2["q"], "r": series2["r"]}),
    "nodim": series2["nodim"],
}
i12 = pd.date_range("2020-02", freq="MS", periods=2, tz="Europe/Berlin")

# Portfolio lines without overlap.
i3 = pd.date_range("2022", freq="MS", periods=3, tz="Europe/Berlin")
flatset3 = {kind: dev.get_flatpfline(i3, kind, _seed=_seed) for kind in Kind}
flatset3["nodim"] = dev.get_series(i3)

# Portfolio lines with other frequency.
i4 = pd.date_range("2020", freq="D", periods=3, tz="Europe/Berlin")
flatset4 = {kind: dev.get_flatpfline(i4, kind, _seed=_seed) for kind in Kind}
flatset4["nodim"] = dev.get_series(i4)

# Portfolio lines with wrong timezone.
i5 = pd.date_range("2022", freq="MS", periods=3, tz="Asia/Kolkata")
flatset5 = {kind: dev.get_flatpfline(i5, kind, _seed=_seed) for kind in Kind}
flatset5["nodim"] = dev.get_series(i5)
i6 = pd.date_range("2022", freq="MS", periods=3, tz=None)
flatset6 = {kind: dev.get_flatpfline(i6, kind, _seed=_seed) for kind in Kind}
flatset6["nodim"] = dev.get_series(i6)


@dataclass(frozen=True)
class Case:
    value1: Any
    operation: str
    value2: Any
    expected: Any


def dimlessseries(s: pd.Series) -> pd.Series:
    return s.astype("pint[dimensionless]").rename("fraction")


# Several expected results of arithmatic.
def additiontestcases():
    for kind in Kind:
        pfl = ref_flatset[kind]
        # . ref + 1
        series = {c: ref_series[c] + series1[c] for c in kind.summable}
        yield Case(pfl, "+", flatset1[kind], create_pfline(series))
        # . ref + 2
        series = {
            c: ref_series[c].loc[i12] + series2[c].loc[i12] for c in kind.summable
        }
        yield Case(pfl, "+", flatset2[kind], create_pfline(series))
    yield Case(ref_flatset[Kind.VOLUME], "+", flatset3[Kind.VOLUME], Exception)
    yield Case(ref_flatset[Kind.VOLUME], "+", flatset4[Kind.VOLUME], Exception)
    yield Case(ref_flatset[Kind.VOLUME], "+", flatset5[Kind.VOLUME], Exception)
    yield Case(ref_flatset[Kind.VOLUME], "+", flatset6[Kind.VOLUME], Exception)


def subtractiontestcases():
    for kind in Kind:
        pfl = ref_flatset[kind]
        # . ref - 1
        series = {c: ref_series[c] - series1[c] for c in kind.summable}
        yield Case(pfl, "-", flatset1[kind], create_pfline(series))
        # . ref - 2
        series = {
            c: ref_series[c].loc[i12] - series2[c].loc[i12] for c in kind.summable
        }
        yield Case(pfl, "-", flatset2[kind], create_pfline(series))
    yield Case(ref_flatset[Kind.VOLUME], "-", flatset3[Kind.VOLUME], Exception)
    yield Case(ref_flatset[Kind.VOLUME], "-", flatset4[Kind.VOLUME], Exception)
    yield Case(ref_flatset[Kind.VOLUME], "-", flatset5[Kind.VOLUME], Exception)
    yield Case(ref_flatset[Kind.VOLUME], "-", flatset6[Kind.VOLUME], Exception)


def multiplicationtestcases():
    # something * nodim
    for kind in Kind:
        pfl = ref_flatset[kind]
        # . ref * 1
        series = {c: ref_series[c] * series1["nodim"] for c in kind.summable}
        yield Case(pfl, "*", flatset1["nodim"], create_pfline(series))
        # . ref * 2
        series = {
            c: ref_series[c].loc[i12] * series2["nodim"].loc[i12] for c in kind.summable
        }
        yield Case(pfl, "*", flatset2["nodim"], create_pfline(series))
    yield Case(ref_flatset[Kind.VOLUME], "*", flatset1[Kind.VOLUME], Exception)
    yield Case(ref_flatset[Kind.VOLUME], "*", flatset1[Kind.REVENUE], Exception)
    yield Case(ref_flatset[Kind.VOLUME], "*", flatset3["nodim"], Exception)
    yield Case(ref_flatset[Kind.VOLUME], "*", flatset4["nodim"], Exception)
    yield Case(ref_flatset[Kind.VOLUME], "*", flatset5["nodim"], Exception)
    yield Case(ref_flatset[Kind.VOLUME], "*", flatset6["nodim"], Exception)
    # volume * price
    series = {"r": ref_series["q"] * series1["p"]}
    yield Case(
        ref_flatset[Kind.VOLUME], "*", flatset1[Kind.PRICE], create_pfline(series)
    )


def divisiontestcases():
    # something / nodim
    for kind in Kind:
        pfl = ref_flatset[kind]
        # . ref / 1
        series = {c: ref_series[c] / series1["nodim"] for c in kind.summable}
        yield Case(pfl, "/", flatset1["nodim"], create_pfline(series))
        # . ref / 2
        series = {
            c: ref_series[c].loc[i12] / series2["nodim"].loc[i12] for c in kind.summable
        }
        yield Case(pfl, "/", flatset2["nodim"], create_pfline(series))
    yield Case(ref_flatset[Kind.VOLUME], "/", flatset3["nodim"], Exception)
    yield Case(ref_flatset[Kind.VOLUME], "/", flatset4["nodim"], Exception)
    yield Case(ref_flatset[Kind.VOLUME], "/", flatset5["nodim"], Exception)
    yield Case(ref_flatset[Kind.VOLUME], "/", flatset6["nodim"], Exception)
    # something / same thing
    for kind in [Kind.VOLUME, Kind.PRICE, Kind.REVENUE]:
        c = kind.summable[0]  # only one element
        pfl = ref_flatset[kind]
        # . ref / 1
        series = dimlessseries(ref_series[c] / series1[c])
        yield Case(pfl, "/", flatset1[kind], series)
        # . ref / 2
        series = dimlessseries(ref_series[c].loc[i12] / series2[c].loc[i12])
        yield Case(pfl, "/", flatset2[kind], series)
    yield Case(ref_flatset[Kind.COMPLETE], "/", flatset1[Kind.COMPLETE], Exception)
    yield Case(ref_flatset[Kind.VOLUME], "/", flatset3[Kind.VOLUME], Exception)
    yield Case(ref_flatset[Kind.VOLUME], "/", flatset4[Kind.VOLUME], Exception)
    yield Case(ref_flatset[Kind.VOLUME], "/", flatset5[Kind.VOLUME], Exception)
    yield Case(ref_flatset[Kind.VOLUME], "/", flatset6[Kind.VOLUME], Exception)
    # revenue / something
    pfl = ref_flatset[Kind.REVENUE]
    series = {"q": ref_series["r"] / series1["p"]}
    yield Case(pfl, "/", flatset1[Kind.PRICE], create_pfline(series))
    series = {"p": ref_series["r"] / series1["q"]}
    yield Case(pfl, "/", flatset1[Kind.VOLUME], create_pfline(series))


def uniontestcases():
    # something | something else
    for kind in [Kind.VOLUME, Kind.PRICE, Kind.REVENUE]:
        pfl = ref_flatset[kind]
        c = kind.summable[0]  # only one element

        for kind_other in [Kind.VOLUME, Kind.PRICE]:
            if kind is kind_other:
                continue
            c_other = kind_other.summable[0]  # only one element

            # . ref | 1
            series = {c: ref_series[c], c_other: series1[c_other]}
            yield Case(pfl, "|", flatset1[kind_other], create_pfline(series))
            # . ref | 2
            series = {c: ref_series[c].loc[i12], c_other: series2[c_other].loc[i12]}
            yield Case(pfl, "|", flatset2[kind_other], create_pfline(series))
    yield Case(ref_flatset[Kind.VOLUME], "|", flatset1[Kind.VOLUME], Exception)
    yield Case(ref_flatset[Kind.VOLUME], "|", flatset1[Kind.COMPLETE], Exception)
    yield Case(ref_flatset[Kind.VOLUME], "|", flatset3[Kind.PRICE], Exception)
    yield Case(ref_flatset[Kind.VOLUME], "|", flatset4[Kind.PRICE], Exception)
    yield Case(ref_flatset[Kind.VOLUME], "|", flatset5[Kind.PRICE], Exception)
    yield Case(ref_flatset[Kind.VOLUME], "|", flatset6[Kind.PRICE], Exception)


@pytest.mark.parametrize("testcase", additiontestcases(), ids=id_fn)
@pytest.mark.parametrize("order", ["", "reversed"])
def test_addition(testcase: Case, order: str):
    do_test(testcase, order)


@pytest.mark.parametrize("testcase", subtractiontestcases(), ids=id_fn)
def test_subtraction(testcase: Case):
    do_test(testcase)


@pytest.mark.parametrize("testcase", multiplicationtestcases(), ids=id_fn)
@pytest.mark.parametrize("order", ["", "reversed"])
def test_multiplication(testcase: Case, order: str):
    do_test(testcase, order)


@pytest.mark.parametrize("testcase", divisiontestcases(), ids=id_fn)
def test_division(testcase: Case):
    do_test(testcase)


@pytest.mark.parametrize("testcase", uniontestcases(), ids=id_fn)
@pytest.mark.parametrize("order", ["", "reversed"])
def test_union(testcase: Case, order):
    do_test(testcase, order)


def do_test(tc: Case, order: str = ""):
    """Test if addition, subtraction, multiplication, division, union return correct values."""

    def calc():
        val1, val2 = tc.value1, tc.value2
        if order == "reversed" and not isinstance(val2, pd.Series):
            # Series causes problems; does not refer to PfLine.__radd__, .__rmul__, etc.
            val1, val2 = val2, val1
        if tc.operation == "+":
            return val1 + val2
        elif tc.operation == "-":
            return val1 - val2
        elif tc.operation == "*":
            return val1 * val2
        elif tc.operation == "/":
            return val1 / val2
        elif tc.operation == "|":
            return val1 | val2
        raise NotImplementedError()

    # Test error case.
    if isinstance(tc.expected, type) and issubclass(tc.expected, Exception):
        with pytest.raises(tc.expected):
            _ = calc()
        return

    # Test correct case.
    result = calc()
    if isinstance(result, pd.DataFrame):
        testing.assert_frame_equal(result, tc.expected)
    elif isinstance(result, pd.Series):
        testing.assert_series_equal(result, tc.expected)
    else:
        assert result == tc.expected
