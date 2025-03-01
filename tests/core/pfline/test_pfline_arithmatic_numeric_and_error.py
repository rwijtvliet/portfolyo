from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest
from utils import id_fn  # relative to /tests

import portfolyo as pf
from portfolyo import Kind, create, dev, testing

_seed = 3  # any fixed seed

# TODO: nestedPfLine, quantity, float, 0, negation

# Reference Set.
i_ref = pd.date_range("2020", freq="MS", periods=3, tz="Europe/Berlin")
series_ref = {
    "q": pd.Series([-3.5, 5, -5], i_ref),
    "p": pd.Series([300.0, 150, 100], i_ref),
    "r": pd.Series([-1050.0, 750, -500], i_ref),
    "nodim": pd.Series([2, -1.5, 10], i_ref),
}
flatset_ref = {
    Kind.VOLUME: create.flatpfline({"q": series_ref["q"]}),
    Kind.PRICE: create.flatpfline({"p": series_ref["p"]}),
    Kind.REVENUE: create.flatpfline({"r": series_ref["r"]}),
    Kind.COMPLETE: create.flatpfline({"q": series_ref["q"], "p": series_ref["p"]}),
    "nodim": series_ref["nodim"],
}
# No other values.
flatset_ref_neg = {
    Kind.VOLUME: create.flatpfline({"q": -series_ref["q"]}),
    Kind.PRICE: create.flatpfline({"p": -series_ref["p"]}),
    Kind.REVENUE: create.flatpfline({"r": -series_ref["r"]}),
    Kind.COMPLETE: create.flatpfline({"q": -series_ref["q"], "p": series_ref["p"]}),
    "nodim": -series_ref["nodim"],
}
# Constant values.
series_0 = pd.Series(0.0, i_ref)
values_0 = {
    Kind.VOLUME: [
        pf.Q_(0.0, "MW"),
        pf.Q_(0.0, "GW"),
        {"q": 0.0},
        pd.Series(0.0, i_ref, dtype="pint[GW]"),
        create.flatpfline({"w": series_0}),
    ],
    Kind.PRICE: [
        pf.Q_(0.0, "Eur/MWh"),
        pf.Q_(0.0, "ctEur/kWh"),
        pd.Series(0.0, i_ref, dtype="pint[ctEur/kWh]"),
        create.flatpfline({"p": series_0}),
    ],
    Kind.REVENUE: [
        pf.Q_(0.0, "Eur"),
        pf.Q_(0.0, "kEur"),
        pd.Series(0.0, i_ref, dtype="pint[kEur]"),
        create.flatpfline({"r": series_0}),
    ],
    Kind.COMPLETE: [
        {"q": 0.0, "r": 0.0},
        {"q": 0.0, "p": 100.0},
        create.flatpfline({"r": series_0, "q": series_0}),
    ],
    "nodim": [0, 0.0, pd.Series(0.0, i_ref)],
}
flatset_ref_times_0 = {
    Kind.VOLUME: create.flatpfline({"q": series_0}),
    Kind.PRICE: create.flatpfline({"p": series_0}),
    Kind.REVENUE: create.flatpfline({"r": series_0}),
    Kind.COMPLETE: create.flatpfline({"q": series_0, "p": series_ref["p"]}),
}
series_2 = pd.Series(2.0, i_ref)
values_2 = {
    Kind.VOLUME: [
        pf.Q_(2.0, "MWh"),
        pf.Q_(2000.0, "kWh"),
        {"q": 2.0},
        pd.Series(2.0, i_ref, dtype="pint[MWh]"),
        create.flatpfline({"q": series_2}),
    ],
    Kind.PRICE: [
        pf.Q_(2.0, "Eur/MWh"),
        pf.Q_(0.2, "ctEur/kWh"),
        pd.Series(2.0, i_ref, dtype="pint[Eur/MWh]"),
        create.flatpfline({"p": series_2}),
    ],
    Kind.REVENUE: [
        pf.Q_(2.0, "Eur"),
        pf.Q_(0.002, "kEur"),
        pd.Series(2.0, i_ref, dtype="pint[Eur]"),
        create.flatpfline({"r": series_2}),
    ],
    Kind.COMPLETE: [
        {"q": 2.0, "r": 2.0},
        create.flatpfline({"r": series_2, "q": series_2}),
    ],
    "nodim": [2, 2.0, pd.Series(2.0, i_ref)],
}
series_ref_plus_2 = {
    "q": pd.Series([-1.5, 7, -3], i_ref),
    "p": pd.Series([302.0, 152, 102], i_ref),
    "r": pd.Series([-1048.0, 752, -498], i_ref),
    "nodim": pd.Series([4, 0.5, 12], i_ref),
}
flatset_ref_plus_2 = {
    Kind.VOLUME: create.flatpfline({"q": series_ref_plus_2["q"]}),
    Kind.PRICE: create.flatpfline({"p": series_ref_plus_2["p"]}),
    Kind.REVENUE: create.flatpfline({"r": series_ref_plus_2["r"]}),
    Kind.COMPLETE: create.flatpfline({"q": series_ref_plus_2["q"], "r": series_ref_plus_2["r"]}),
    "nodim": series_ref_plus_2["nodim"],
}
series_ref_minus_2 = {
    "q": pd.Series([-5.5, 3, -7], i_ref),
    "p": pd.Series([298.0, 148, 98], i_ref),
    "r": pd.Series([-1052.0, 748, -502], i_ref),
    "nodim": pd.Series([0, -3.5, 8], i_ref),
}
flatset_ref_minus_2 = {
    Kind.VOLUME: create.flatpfline({"q": series_ref_minus_2["q"]}),
    Kind.PRICE: create.flatpfline({"p": series_ref_minus_2["p"]}),
    Kind.REVENUE: create.flatpfline({"r": series_ref_minus_2["r"]}),
    Kind.COMPLETE: create.flatpfline({"q": series_ref_minus_2["q"], "r": series_ref_minus_2["r"]}),
    "nodim": series_ref_minus_2["nodim"],
}
series_ref_times_2 = {
    "q": pd.Series([-7.0, 10, -10], i_ref),
    "p": pd.Series([600.0, 300, 200], i_ref),
    "r": pd.Series([-2100.0, 1500, -1000], i_ref),
    "nodim": pd.Series([4.0, -3, 20], i_ref),
}
flatset_ref_times_2 = {
    Kind.VOLUME: create.flatpfline({"q": series_ref_times_2["q"]}),
    Kind.PRICE: create.flatpfline({"p": series_ref_times_2["p"]}),
    Kind.REVENUE: create.flatpfline({"r": series_ref_times_2["r"]}),
    Kind.COMPLETE: create.flatpfline({"q": series_ref_times_2["q"], "p": series_ref["p"]}),
    "nodim": series_ref_times_2["nodim"],
}
flatset_ref_dividedby_0 = {
    Kind.VOLUME: create.flatpfline({"q": pd.Series([-np.inf, np.inf, -np.inf], i_ref)}),
    Kind.PRICE: create.flatpfline({"p": pd.Series([np.inf, np.inf, np.inf], i_ref)}),
    Kind.REVENUE: create.flatpfline({"r": pd.Series([-np.inf, np.inf, -np.inf], i_ref)}),
    Kind.COMPLETE: create.flatpfline(
        {"q": pd.Series([-np.inf, np.inf, -np.inf], i_ref), "p": series_ref["p"]}
    ),
}
series_ref_dividedby_2 = {
    "q": pd.Series([-1.75, 2.5, -2.5], i_ref),
    "p": pd.Series([150.0, 75, 50], i_ref),
    "r": pd.Series([-525.0, 375, -250], i_ref),
    "nodim": pd.Series([1, -0.75, 5], i_ref),
}
flatset_ref_dividedby_2 = {
    Kind.VOLUME: create.flatpfline({"q": series_ref_dividedby_2["q"]}),
    Kind.PRICE: create.flatpfline({"p": series_ref_dividedby_2["p"]}),
    Kind.REVENUE: create.flatpfline({"r": series_ref_dividedby_2["r"]}),
    Kind.COMPLETE: create.flatpfline({"q": series_ref_dividedby_2["q"], "p": series_ref["p"]}),
    "nodim": series_ref_dividedby_2["nodim"],
}
flatset_ref_unionwith_2 = {
    Kind.VOLUME: {
        Kind.PRICE: create.flatpfline({"q": series_ref["q"], "p": series_2}),
        Kind.REVENUE: create.flatpfline({"q": series_ref["q"], "r": series_2}),
    },
    Kind.PRICE: {
        Kind.VOLUME: create.flatpfline({"p": series_ref["p"], "q": series_2}),
        Kind.REVENUE: create.flatpfline({"p": series_ref["p"], "r": series_2}),
    },
    Kind.REVENUE: {
        Kind.VOLUME: create.flatpfline({"r": series_ref["r"], "q": series_2}),
        Kind.PRICE: create.flatpfline({"r": series_ref["r"], "p": series_2}),
    },
}

# Set a. Full overlap with reference set.
series_a = {
    "q": pd.Series([-3.0, 4, -5], i_ref),
    "p": pd.Series([400.0, 150, -100], i_ref),
    "r": pd.Series([-1200.0, 600, 500], i_ref),
    "nodim": pd.Series([1.0, -10, 2], i_ref),
}
flatset_a = {
    Kind.VOLUME: create.flatpfline({"q": series_a["q"]}),
    Kind.PRICE: create.flatpfline({"p": series_a["p"]}),
    Kind.REVENUE: create.flatpfline({"r": series_a["r"]}),
    Kind.COMPLETE: create.flatpfline({"q": series_a["q"], "p": series_a["p"]}),
    "nodim": series_a["nodim"],
}

# Set b. Partial overlap with reference set.
i_b = pd.date_range("2020-02", freq="MS", periods=3, tz="Europe/Berlin")

series_b = {
    "q": pd.Series([-15.0, -20, -4], i_b),
    "p": pd.Series([400.0, 50, 50], i_b),
    "r": pd.Series([4000.0, 500, 500], i_b),
    "nodim": pd.Series([1.0, -2.0, 4], i_b),
}
flatset_b = {
    Kind.VOLUME: create.flatpfline({"q": series_b["q"]}),
    Kind.PRICE: create.flatpfline({"p": series_b["p"]}),
    Kind.REVENUE: create.flatpfline({"r": series_b["r"]}),
    Kind.COMPLETE: create.flatpfline({"q": series_b["q"], "r": series_b["r"]}),
    "nodim": series_b["nodim"],
}
i_ab = pd.date_range("2020-02", freq="MS", periods=2, tz="Europe/Berlin")

# Set c. Portfolio lines without overlap.
i_c = pd.date_range("2022", freq="MS", periods=3, tz="Europe/Berlin")
flatset_c = {kind: dev.get_flatpfline(i_c, kind, _seed=_seed) for kind in Kind}
flatset_c["nodim"] = dev.get_series(i_c)

# Set d. Portfolio lines with other frequency.
i_d = pd.date_range("2020", freq="D", periods=3, tz="Europe/Berlin")
flatset_d = {kind: dev.get_flatpfline(i_d, kind, _seed=_seed) for kind in Kind}
flatset_d["nodim"] = dev.get_series(i_d)

# Set e and f. Portfolio lines with wrong timezone.
i_e = pd.date_range("2022", freq="MS", periods=3, tz="Asia/Kolkata")
flatset_e = {kind: dev.get_flatpfline(i_e, kind, _seed=_seed) for kind in Kind}
flatset_e["nodim"] = dev.get_series(i_e)
i_f = pd.date_range("2022", freq="MS", periods=3, tz=None)
flatset_f = {kind: dev.get_flatpfline(i_f, kind, _seed=_seed) for kind in Kind}
flatset_f["nodim"] = dev.get_series(i_f)

# TODO: Portfolio lines with wrong start-of-day.


@dataclass(frozen=True)
class Case:
    value1: Any
    operation: str
    value2: Any
    expected: Any

    def __repr__(self):
        return f"Case({id_fn(self.value1)},{self.operation},{id_fn(self.value2)},{id_fn(self.expected)})"


def dimlessseries(s: pd.Series) -> pd.Series:
    return s.astype("pint[dimensionless]").rename("fraction")


# Several expected results of arithmatic.
def negationtestcases():
    for kind in Kind:
        pfl = flatset_ref[kind]
        # . -ref
        yield Case(pfl, "neg", "", flatset_ref_neg[kind])


def additiontestcases():
    # kind + nothing
    for kind in Kind:
        pfl = flatset_ref[kind]
        for val in values_0["nodim"]:  # necessary also for correct sum([PfLines])
            yield Case(pfl, "+", val, pfl)
        yield Case(pfl, "+", None, pfl)
    # kind + same kind
    for kind in Kind:
        pfl = flatset_ref[kind]
        # . ref + nothing (correct dimension)
        for val in values_0[kind]:
            yield Case(pfl, "+", val, pfl)
        # . ref + 2
        for val in values_2[kind]:
            yield Case(pfl, "+", val, flatset_ref_plus_2[kind])
        # . ref + a
        series = {c: series_ref[c] + series_a[c] for c in kind.summable}
        yield Case(pfl, "+", flatset_a[kind], pf.PfLine(series))
        # . ref + b
        series = {c: series_ref[c].loc[i_ab] + series_b[c].loc[i_ab] for c in kind.summable}
        yield Case(pfl, "+", flatset_b[kind], pf.PfLine(series))
    # . ref + other index
    yield Case(flatset_ref[Kind.VOLUME], "+", flatset_c[Kind.VOLUME], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "+", flatset_d[Kind.VOLUME], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "+", flatset_e[Kind.VOLUME], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "+", flatset_f[Kind.VOLUME], Exception)
    # kind + other kind
    yield Case(flatset_ref[Kind.VOLUME], "+", flatset_a[Kind.PRICE], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "+", flatset_a["nodim"], Exception)


def subtractiontestcases():
    # kind - nothing
    for kind in Kind:
        pfl = flatset_ref[kind]
        for val in values_0["nodim"]:  # necessary also for correct sum([PfLines])
            yield Case(pfl, "-", val, pfl)
        yield Case(pfl, "-", None, pfl)
    # kind - same kind
    for kind in Kind:
        pfl = flatset_ref[kind]
        # . ref - nothing (correct dimension)
        for val in values_0[kind]:
            yield Case(pfl, "-", val, pfl)
        # . ref - 2
        for val in values_2[kind]:
            yield Case(pfl, "-", val, flatset_ref_minus_2[kind])
        # . ref - a
        series = {c: series_ref[c] - series_a[c] for c in kind.summable}
        yield Case(pfl, "-", flatset_a[kind], pf.PfLine(series))
        # . ref - b
        series = {c: series_ref[c].loc[i_ab] - series_b[c].loc[i_ab] for c in kind.summable}
        yield Case(pfl, "-", flatset_b[kind], pf.PfLine(series))
    # This one is the issue
    yield Case(flatset_ref[Kind.VOLUME], "-", flatset_c[Kind.VOLUME], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "-", flatset_d[Kind.VOLUME], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "-", flatset_e[Kind.VOLUME], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "-", flatset_f[Kind.VOLUME], Exception)
    # kind - incompatible kind
    yield Case(flatset_ref[Kind.VOLUME], "-", flatset_a[Kind.PRICE], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "-", flatset_a["nodim"], Exception)


def multiplicationtestcases():
    # kind * nodim
    for kind in Kind:
        pfl = flatset_ref[kind]
        # . ref * 0
        for val in values_0["nodim"]:
            yield Case(pfl, "*", val, flatset_ref_times_0[kind])
        # . ref * 2
        for val in values_2["nodim"]:
            yield Case(pfl, "*", val, flatset_ref_times_2[kind])
        # . ref * a
        series = {c: series_ref[c] * series_a["nodim"] for c in kind.summable}
        yield Case(pfl, "*", flatset_a["nodim"], pf.PfLine(series))
        # . ref * b
        series = {c: series_ref[c].loc[i_ab] * series_b["nodim"].loc[i_ab] for c in kind.summable}
        yield Case(pfl, "*", flatset_b["nodim"], pf.PfLine(series))
    yield Case(flatset_ref[Kind.VOLUME], "*", flatset_c["nodim"], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "*", flatset_d["nodim"], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "*", flatset_e["nodim"], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "*", flatset_f["nodim"], Exception)
    # volume * price
    series = {"r": series_ref["q"] * series_a["p"]}
    yield Case(flatset_ref[Kind.VOLUME], "*", flatset_a[Kind.PRICE], pf.PfLine(series))
    for val in values_0[Kind.PRICE]:
        yield Case(flatset_ref[Kind.VOLUME], "*", val, flatset_ref_times_0[Kind.REVENUE])
    # kind * incompatible kind
    yield Case(flatset_ref[Kind.VOLUME], "*", None, Exception)
    yield Case(flatset_ref[Kind.VOLUME], "*", flatset_a[Kind.VOLUME], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "*", flatset_a[Kind.REVENUE], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "*", flatset_a[Kind.COMPLETE], Exception)


def divisiontestcases():
    # kind / nodim
    for kind in Kind:
        pfl = flatset_ref[kind]
        # . ref / 0
        for val in values_0["nodim"]:
            yield Case(pfl, "/", val, flatset_ref_dividedby_0[kind])
        # . ref / 2
        for val in values_2["nodim"]:
            yield Case(pfl, "/", val, flatset_ref_dividedby_2[kind])
        # . ref / a
        series = {c: series_ref[c] / series_a["nodim"] for c in kind.summable}
        yield Case(pfl, "/", flatset_a["nodim"], pf.PfLine(series))
        # . ref / b
        series = {c: series_ref[c].loc[i_ab] / series_b["nodim"].loc[i_ab] for c in kind.summable}
        yield Case(pfl, "/", flatset_b["nodim"], pf.PfLine(series))
    yield Case(flatset_ref[Kind.VOLUME], "/", flatset_c["nodim"], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "/", flatset_d["nodim"], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "/", flatset_e["nodim"], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "/", flatset_f["nodim"], Exception)
    # kind / same kind
    for kind in [Kind.VOLUME, Kind.PRICE, Kind.REVENUE]:
        c = kind.summable[0]  # only one element
        pfl = flatset_ref[kind]
        # . ref / 2
        yield Case(pfl, "/", flatset_ref_dividedby_2[kind], dimlessseries(series_2))
        # . ref / a
        series = dimlessseries(series_ref[c] / series_a[c])
        yield Case(pfl, "/", flatset_a[kind], series)
        # . ref / b
        series = dimlessseries(series_ref[c].loc[i_ab] / series_b[c].loc[i_ab])
        yield Case(pfl, "/", flatset_b[kind], series)
    yield Case(flatset_ref[Kind.COMPLETE], "/", flatset_a[Kind.COMPLETE], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "/", flatset_c[Kind.VOLUME], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "/", flatset_d[Kind.VOLUME], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "/", flatset_e[Kind.VOLUME], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "/", flatset_f[Kind.VOLUME], Exception)
    # revenue / volume or price
    series = {"q": series_ref["r"] / series_a["p"]}
    yield Case(flatset_ref[Kind.REVENUE], "/", flatset_a[Kind.PRICE], pf.PfLine(series))
    series = {"p": series_ref["r"] / series_a["q"]}
    yield Case(flatset_ref[Kind.REVENUE], "/", flatset_a[Kind.VOLUME], pf.PfLine(series))
    for val in values_0[Kind.PRICE]:
        yield Case(flatset_ref[Kind.REVENUE], "/", val, flatset_ref_dividedby_0[Kind.VOLUME])
    # kind / incompatible value
    yield Case(flatset_ref[Kind.VOLUME], "/", None, Exception)
    yield Case(flatset_ref[Kind.VOLUME], "/", flatset_a[Kind.PRICE], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "/", flatset_a[Kind.REVENUE], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "/", flatset_a[Kind.COMPLETE], Exception)


def uniontestcases():
    # kind | compatible kind
    for kind in [Kind.VOLUME, Kind.PRICE, Kind.REVENUE]:
        pfl = flatset_ref[kind]
        c = kind.summable[0]  # only one element

        for kind_other in [Kind.VOLUME, Kind.PRICE]:
            if kind is kind_other:
                continue

            # . ref | 2
            for val in values_2[kind_other]:
                yield Case(pfl, "|", val, flatset_ref_unionwith_2[kind][kind_other])

            c_other = kind_other.summable[0]  # only one element
            # . ref | a
            series = {c: series_ref[c], c_other: series_a[c_other]}
            yield Case(pfl, "|", flatset_a[kind_other], pf.PfLine(series))
            # . ref | b
            series = {c: series_ref[c].loc[i_ab], c_other: series_b[c_other].loc[i_ab]}
            yield Case(pfl, "|", flatset_b[kind_other], pf.PfLine(series))
    yield Case(flatset_ref[Kind.VOLUME], "|", flatset_c[Kind.PRICE], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "|", flatset_d[Kind.PRICE], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "|", flatset_e[Kind.PRICE], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "|", flatset_f[Kind.PRICE], Exception)
    # kind | incompatible kind
    yield Case(flatset_ref[Kind.VOLUME], "|", flatset_a[Kind.VOLUME], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "|", flatset_a[Kind.COMPLETE], Exception)
    yield Case(flatset_ref[Kind.VOLUME], "|", flatset_a["nodim"], Exception)


@pytest.mark.parametrize("testcase", negationtestcases(), ids=id_fn)
def test_negation(testcase: Case):
    do_test(testcase)


@pytest.mark.parametrize("testcase", additiontestcases(), ids=id_fn)
@pytest.mark.parametrize("order", ["normal", "reversed"])
def test_addition(testcase: Case, order: str):
    do_test(testcase, order)


@pytest.mark.parametrize("testcase", list(subtractiontestcases()), ids=id_fn)
def test_subtraction(testcase: Case):
    do_test(testcase)


@pytest.mark.parametrize("testcase", multiplicationtestcases(), ids=id_fn)
@pytest.mark.parametrize("order", ["normal", "reversed"])
def test_multiplication(testcase: Case, order: str):
    do_test(testcase, order)


@pytest.mark.parametrize("testcase", divisiontestcases(), ids=id_fn)
def test_division(testcase: Case):
    do_test(testcase)


@pytest.mark.parametrize("testcase", uniontestcases(), ids=id_fn)
@pytest.mark.parametrize("order", ["normal", "reversed"])
def test_union(testcase: Case, order):
    do_test(testcase, order)


def do_test(tc: Case, order: str = ""):
    """Test if addition, subtraction, multiplication, division, union return correct values."""

    def calc():
        val1, val2 = tc.value1, tc.value2

        if order == "reversed":
            # Some classes cause problems; do not refer to PfLine.__radd__, .__rmul__, etc.
            if (
                isinstance(val2, pf.Q_)
                or isinstance(val2, pd.Series)
                or isinstance(val2, pd.DataFrame)
            ):
                pytest.skip(
                    f"{val2.__class__.__name__} objects do not (cleanly) defer reverse"
                    " methods to us, so our reverse methods are not called, or called"
                    " with incorrect objects."
                )
            val1, val2 = val2, val1
        if tc.operation == "neg":
            return -val1
        elif tc.operation == "+":
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
