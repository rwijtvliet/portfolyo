from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Iterable, List

import pandas as pd
import pytest

import portfolyo as pf
from portfolyo import Q_, FlatPfLine, Kind, NestedPfLine, PfLine, dev, testing  # noqa

# TODO: various timezones


def id_fn(data: Any):
    """Readable id of test case"""
    if isinstance(data, Dict):
        return str({key: id_fn(val) for key, val in data.items()})
    elif isinstance(data, pd.Series):
        if isinstance(data.index, pd.DatetimeIndex):
            return f"Timeseries({data.dtype})"
        return f"Series(idx:{''.join(str(i) for i in data.index)})"
    elif isinstance(data, pd.DataFrame):
        return f"Df(columns:{''.join(str(c) for c in data.columns)})"
    elif isinstance(data, FlatPfLine):
        return f"Flatpfline({data.kind})"
    elif isinstance(data, NestedPfLine):
        return f"Nestedpfline({data.kind})"
    elif isinstance(data, pf.Q_):
        return f"Q({data.units})"
    elif isinstance(data, type):
        return data.__name__
    elif isinstance(data, Kind):
        return str(data)
    elif isinstance(data, Testcase):
        return f"pfl:{id_fn(data.pfl)}..val:{id_fn(data.value)}..expct:{data.expected_result}"
    elif isinstance(data, str):
        return data
    elif data is None:
        return "None"
    return type(data).__name__


class Kind2(Enum):
    VOLUME = Kind.VOLUME
    PRICE = Kind.PRICE
    REVENUE = Kind.REVENUE
    COMPLETE = Kind.COMPLETE
    NODIM = "nodim"
    NONE = "none"


class ER(Enum):
    VOLUME = Kind.VOLUME
    PRICE = Kind.PRICE
    REVENUE = Kind.REVENUE
    COMPLETE = Kind.COMPLETE
    SERIES = "nodim"
    ERROR = "error"

    def typ(self) -> type:
        if self.value in Kind:
            return PfLine
        if self is ER.SERIES:
            return pd.Series
        return None

    def kind(self) -> Kind:
        if self.value in Kind:
            return self.value
        return None


@dataclass
class Testpfl:
    kind: Kind
    nestedness: str  # 'flat' or 'nested'
    pfl: PfLine


@dataclass
class Testvalue:
    kind: Kind2
    nestedness: str  # 'flat' or 'nested'
    value: Any


@dataclass
class Testcase:
    pfl: PfLine
    value: Any
    expected_result: ER


@dataclass(frozen=True, eq=True)
class TestcaseConfig:
    pfl_kind: Kind
    pfl_nestedness: str  # 'flat' or 'nested'
    value_kind: Kind2
    value_nestedness: str  # 'flat' or 'nested'


class Testpfls:
    def __init__(self, i: pd.DatetimeIndex):
        flat_fn, nested_fn = pf.dev.get_flatpfline, pf.dev.get_nestedpfline
        self._testpfls = [
            *[Testpfl(kind, "flat", flat_fn(i, kind, _seed=1)) for kind in Kind],
            *[Testpfl(kind, "nested", nested_fn(i, kind, _seed=1)) for kind in Kind],
        ]

    def fetch(self, kind: Kind = None, nestedness: str = None) -> Iterable[Testpfl]:
        testpfls: Iterable[Testpfl] = self._testpfls
        if kind is not None:
            testpfls = [tp for tp in testpfls if tp.kind == kind]
        if nestedness is not None:
            testpfls = [tp for tp in testpfls if tp.nestedness == nestedness]
        return testpfls


class Testvalues:
    UNIT_ALT = {"w": "GW", "q": "kWh", "p": "ctEur/kWh", "r": "MEur", "nodim": ""}

    def __init__(self, i: pd.DatetimeIndex):
        self._testvalues: Iterable[Testvalue] = [
            *self.from_noname(i),
            *self.from_1name(Kind2.NODIM, "nodim", i),
            *self.from_1name(Kind2.VOLUME, "w", i),
            *self.from_1name(Kind2.VOLUME, "q", i),
            *self.from_pfline(Kind2.VOLUME, i),
            *self.from_1name(Kind2.PRICE, "p", i),
            *self.from_pfline(Kind2.PRICE, i),
            *self.from_1name(Kind2.REVENUE, "r", i),
            *self.from_pfline(Kind2.REVENUE, i),
            *self.from_2names(Kind2.COMPLETE, "wp", i),
            *self.from_2names(Kind2.COMPLETE, "qr", i),
            *self.from_2names(Kind2.COMPLETE, "pr", i),
            *self.from_pfline(Kind2.COMPLETE, i),
        ]

    @lru_cache()
    def fetch(self, kind: Kind2 = None, nestedness: str = None) -> Iterable[Testvalue]:
        testvalues: Iterable[Testvalue] = self._testvalues
        if kind is not None:
            testvalues = [tv for tv in testvalues if tv.kind == kind]
        if nestedness is not None:
            testvalues = [tv for tv in testvalues if tv.nestedness == nestedness]
        return testvalues

    @staticmethod
    def from_noname(i: pd.DatetimeIndex) -> Iterable[Testvalue]:
        fl = pf.dev.get_value("nodim", False, _seed=2)
        flseries = pf.dev.get_series(i, "", False, _seed=2)
        return [
            Testvalue(Kind2.NONE, "flat", None),
            Testvalue(Kind2.NODIM, "flat", fl),  # Single value.
            Testvalue(Kind2.NODIM, "flat", flseries),  # Timeseries.
        ]

    @staticmethod
    def from_1name(kind: Kind2, name: str, i: pd.DatetimeIndex) -> Iterable[Testvalue]:
        """name == {'w', 'q', 'p', 'r', 'nodim'}"""
        quantity = pf.dev.get_value(name, True, _seed=2)
        altunit = Testvalues.UNIT_ALT[name]
        quseries = pf.dev.get_series(i, name, _seed=2)
        return [
            # . Single value.
            Testvalue(kind, "flat", quantity),
            Testvalue(kind, "flat", quantity * 0),
            Testvalue(kind, "flat", quantity.to(altunit)),
            Testvalue(kind, "flat", {name: quantity.m}),
            Testvalue(kind, "flat", {name: quantity}),
            Testvalue(kind, "flat", {name: quantity.to(altunit)}),
            Testvalue(kind, "flat", pd.Series({name: quantity.m})),
            Testvalue(kind, "flat", pd.Series({name: quantity})),
            # . Timeseries.
            Testvalue(kind, "flat", quseries),
            Testvalue(kind, "flat", {name: quseries.pint.m}),
            Testvalue(kind, "flat", {name: quseries}),
            Testvalue(kind, "flat", pd.DataFrame({name: quseries.pint.m})),
            Testvalue(kind, "flat", pd.DataFrame({name: quseries})),
        ]

    @staticmethod
    def from_2names(
        kind: Kind2, names: Iterable[str], i: pd.DatetimeIndex
    ) -> Iterable[Testvalue]:
        """names == two of {'w', 'q', 'p', 'r'}"""
        n1, n2 = names
        qu1, qu2 = (pf.dev.get_value(name, True, _seed=2) for name in names)
        fl1, fl2 = (qu.magnitude for qu in (qu1, qu2))
        qus1, qus2 = (pf.dev.get_series(i, name, _seed=2) for name in names)
        fls1, fls2 = (quseries.pint.m for quseries in (qus1, qus2))
        return [
            # . Single values.
            Testvalue(kind, "flat", {n1: qu1, n2: qu2}),
            Testvalue(kind, "flat", {n1: fl1, n2: fl2}),
            Testvalue(kind, "flat", {n1: fl1, n2: qu2}),
            Testvalue(kind, "flat", pd.Series({n1: qu1, n2: qu2})),
            Testvalue(kind, "flat", pd.Series({n1: fl1, n2: fl2})),
            Testvalue(kind, "flat", pd.Series({n1: fl1, n2: qu2})),
            # . Single value | timeseries.
            Testvalue(kind, "flat", {n1: qu1, n2: qus2}),
            Testvalue(kind, "flat", {n1: fl1, n2: qus2}),
            Testvalue(kind, "flat", {n1: qu1, n2: fls2}),
            Testvalue(kind, "flat", {n1: fl1, n2: fls2}),
            # . Timeseries.
            Testvalue(kind, "flat", {n1: qus1, n2: qus2}),
            Testvalue(kind, "flat", {n1: qus1, n2: fls2}),
            Testvalue(kind, "flat", pd.DataFrame({n1: qus1, n2: qus2})),
            Testvalue(kind, "flat", pd.DataFrame({n1: qus1, n2: fls2})),
        ]

    @staticmethod
    def from_pfline(kind: Kind2, i: pd.DatetimeIndex) -> Iterable[Testvalue]:
        k = kind.value
        pfl1, pfl2 = (pf.dev.get_flatpfline(i, k, _seed=s) for s in (2, 3))
        return [
            Testvalue(kind, "flat", pfl1),
            Testvalue(kind, "nested", pf.PfLine({"childA": pfl1, "childB": pfl2})),
            Testvalue(kind, "nested", {"childC": pfl1, "childD": pfl2}),
        ]


tz = "Europe/Berlin"
i = pd.date_range("2020", periods=20, freq="MS", tz=tz)  # reference


class Testcases:
    _testvalues = Testvalues(i)
    _testpfls = Testpfls(i)
    _complete_outcomedict = {
        TestcaseConfig(pfl_kind, pfl_nestedness, val_kind, val_nestedness): ER.ERROR
        for pfl_kind in Kind
        for pfl_nestedness in ["flat", "nested"]
        for val_kind in Kind2
        for val_nestedness in ["flat", "nested"]
    }

    @classmethod
    def get_pfls(
        cls, pfl_kind: Kind = None, pfl_nestedness: str = None
    ) -> Iterable[Testcase]:
        return cls._testpfls.fetch(pfl_kind, pfl_nestedness)

    @classmethod
    def all_from_nonerror(
        cls, non_error_outcomedict: Dict[TestcaseConfig, ER]
    ) -> Iterable[Testcase]:
        outcomedict = {**cls._complete_outcomedict, **non_error_outcomedict}
        return [
            testcase
            for config, er in outcomedict.items()
            for testcase in cls.from_config(config, er)
        ]

    @classmethod
    def from_config(cls, config: TestcaseConfig, er: ER) -> Iterable[Testcase]:
        return [
            Testcase(tp.pfl, tv.value, er)
            for tp in cls._testpfls.fetch(config.pfl_kind, config.pfl_nestedness)
            for tv in cls._testvalues.fetch(config.value_kind, config.value_nestedness)
        ]


@pytest.mark.parametrize("operation", ["union", "runion"])
@pytest.mark.parametrize(
    "testcase",
    Testcases.all_from_nonerror(
        {
            # Operand 1 = volume pfline.
            # . Operand 2 = None.
            TestcaseConfig(Kind.VOLUME, "flat", Kind2.NONE, "flat"): ER.VOLUME,
            TestcaseConfig(Kind.VOLUME, "nested", Kind2.NONE, "flat"): ER.VOLUME,
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            TestcaseConfig(Kind.VOLUME, "flat", Kind2.PRICE, "flat"): ER.COMPLETE,
            TestcaseConfig(Kind.VOLUME, "flat", Kind2.REVENUE, "flat"): ER.COMPLETE,
            # . Operand 2 = complete.
            # Operand 1 = price pfline.
            # . Operand 2 = None.
            TestcaseConfig(Kind.PRICE, "flat", Kind2.NONE, "flat"): ER.PRICE,
            TestcaseConfig(Kind.PRICE, "nested", Kind2.NONE, "flat"): ER.PRICE,
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            TestcaseConfig(Kind.PRICE, "flat", Kind2.VOLUME, "flat"): ER.COMPLETE,
            TestcaseConfig(Kind.PRICE, "flat", Kind2.REVENUE, "flat"): ER.COMPLETE,
            # . Operand 2 = complete.
            # Operand 1 = revenue pfline.
            # . Operand 2 = None.
            TestcaseConfig(Kind.REVENUE, "flat", Kind2.NONE, "flat"): ER.REVENUE,
            TestcaseConfig(Kind.REVENUE, "nested", Kind2.NONE, "flat"): ER.REVENUE,
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            TestcaseConfig(Kind.REVENUE, "flat", Kind2.VOLUME, "flat"): ER.COMPLETE,
            TestcaseConfig(Kind.REVENUE, "flat", Kind2.PRICE, "flat"): ER.COMPLETE,
            # . Operand 2 = complete.
            # Operand 1 = complete pfline.
            # . Operand 2 = None.
            TestcaseConfig(Kind.COMPLETE, "flat", Kind2.NONE, "flat"): ER.COMPLETE,
            TestcaseConfig(Kind.COMPLETE, "nested", Kind2.NONE, "flat"): ER.COMPLETE,
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            # . Operand 2 = complete.
        }
    ),
    ids=id_fn,
)
def test_pfl_arithmatic_kind_unionrunion(testcase: Testcase, operation: str):
    """Test if arithmatic expectedly raises Error or returns expected type/kind."""
    do_kind_test(testcase, operation)


def do_kind_test(testcase: Testcase, operation: str):
    pfl = testcase.pfl
    value = testcase.value
    expected = testcase.expected_result

    # Guard clause for skipping test.
    if operation.startswith("r"):
        what = ""
        if isinstance(value, Q_):
            what = "pint.Quantity"
        elif isinstance(value, pd.Series) or isinstance(value, pd.DataFrame):
            what = "pandas.Series and pandas.DataFrame"
        if what:
            pytest.skip(
                f"{what} objects do not (cleanly) defer reverse methods to us, so our"
                " reverse methods are not called, or called with incorrect objects."
            )

    # Check error case.
    if expected is ER.ERROR:
        with pytest.raises(Exception):
            do_operation(pfl, operation, value)
        return

    # Check non-error case.
    result = do_operation(pfl, operation, value)
    if (exp_type := expected.typ()) is not None:
        assert isinstance(result, exp_type)
    if (exp_kind := expected.kind()) is not None:
        assert result.kind is exp_kind


def do_operation(pfl: PfLine, operation: str, value: Any) -> Any:
    if operation == "add":
        return pfl + value
    if operation == "radd":
        return value + pfl
    if operation == "sub":
        return pfl - value
    if operation == "rsub":
        return value - pfl
    if operation == "mul":
        return pfl * value
    if operation == "rmul":
        return value * pfl
    if operation == "div":
        return pfl / value
    if operation == "rdiv":
        return value / pfl
    if operation == "union":
        return pfl | value
    if operation == "runion":
        return value | pfl


tz = "Europe/Berlin"
i = pd.date_range("2020", freq="MS", periods=3, tz=tz)
series0 = {
    "w": pd.Series([0.0, 0, 0], i),
    "p": pd.Series([0.0, 0, 0], i),
    "r": pd.Series([0.0, 0, 0], i),
}
pflset0 = {
    Kind.VOLUME: pf.FlatPfLine({"w": series0["w"]}),
    Kind.PRICE: pf.FlatPfLine({"p": series0["p"]}),
    Kind.REVENUE: pf.FlatPfLine({"p": series0["r"]}),
    Kind.COMPLETE: pf.FlatPfLine({"w": series0["w"], "p": series0["p"]}),
}
series_ref = {
    "w": pd.Series([3.0, 5, -4], i),
    "p": pd.Series([200.0, 100, 50], i),
    "r": pd.Series([446400.0, 348000, -148600], i),
}
pflset_ref = {
    Kind.VOLUME: pf.FlatPfLine({"w": series_ref["w"]}),
    Kind.PRICE: pf.FlatPfLine({"p": series_ref["p"]}),
    Kind.REVENUE: pf.FlatPfLine({"r": series_ref["r"]}),
    Kind.COMPLETE: pf.FlatPfLine({"w": series_ref["w"], "p": series_ref["p"]}),
}
series_mul2 = {
    **{key: series * 2 for key, series in series_ref.items()},
    "nodim": pd.Series([2, 2, 2], i),
}
pflset_mul2 = {
    Kind.VOLUME: pf.FlatPfLine({"w": series_mul2["w"]}),
    Kind.PRICE: pf.FlatPfLine({"p": series_mul2["p"]}),
    Kind.REVENUE: pf.FlatPfLine({"r": series_mul2["r"]}),
    Kind.COMPLETE: pf.FlatPfLine({"w": series_mul2["w"], "p": series_mul2["p"]}),
}
series_div2 = {
    **{key: series * 0.5 for key, series in series_ref.items()},
    "nodim": pd.Series([0.5, 0.5, 0.5], i),
}
pflset_div2 = {
    Kind.VOLUME: pf.FlatPfLine({"w": series_div2["w"]}),
    Kind.PRICE: pf.FlatPfLine({"p": series_div2["p"]}),
    Kind.REVENUE: pf.FlatPfLine({"r": series_div2["r"]}),
    Kind.COMPLETE: pf.FlatPfLine({"w": series_div2["w"], "p": series_div2["p"]}),
}
series_plus2 = {key: series + 2 for key, series in series_ref.items()}
pflset_plus2 = {
    Kind.VOLUME: pf.FlatPfLine({"w": series_plus2["w"]}),
    Kind.PRICE: pf.FlatPfLine({"p": series_plus2["p"]}),
    Kind.REVENUE: pf.FlatPfLine({"p": series_plus2["r"]}),
    Kind.COMPLETE: pf.FlatPfLine({"w": series_plus2["w"], "p": series_plus2["p"]}),
}
series_minus2 = {key: series - 2 for key, series in series_ref.items()}
pflset_minus2 = {
    Kind.VOLUME: pf.FlatPfLine({"w": series_minus2["w"]}),
    Kind.PRICE: pf.FlatPfLine({"p": series_minus2["p"]}),
    Kind.REVENUE: pf.FlatPfLine({"p": series_minus2["r"]}),
    Kind.COMPLETE: pf.FlatPfLine({"w": series_minus2["w"], "p": series_minus2["p"]}),
}
values_0 = {
    Kind2.VOLUME: [
        Q_(0.0, "MW"),
        Q_(0.0, "GW"),
        pd.Series(0.0, i, dtype="pint[GW]"),
        pf.FlatPfLine({"w": pd.Series(0.0, i)}),
    ],
    Kind2.PRICE: [
        Q_(0.0, "Eur/MWh"),
        Q_(0.0, "ctEur/kWh"),
        pd.Series(0.0, i, dtype="pint[ctEur/kWh]"),
        pf.FlatPfLine({"p": pd.Series(0.0, i)}),
    ],
    Kind2.REVENUE: [
        Q_(0.0, "Eur"),
        Q_(0.0, "kEur"),
        pd.Series(0.0, i, dtype="pint[kEur]"),
        pf.FlatPfLine({"r": pd.Series(0.0, i)}),
    ],
    Kind2.NODIM: [0, 0.0, pd.Series(0.0, i)],
}
values_2 = {
    Kind2.VOLUME: [
        Q_(2.0, "MW"),
        Q_(0.002, "GW"),
        pd.Series(0.002, i, dtype="pint[GW]"),
        pf.FlatPfLine({"w": pd.Series(2, i)}),
    ],
    Kind2.PRICE: [
        Q_(2.0, "Eur/MWh"),
        Q_(0.2, "ctEur/kWh"),
        pd.Series(0.2, i, dtype="pint[ctEur/kWh]"),
        pf.FlatPfLine({"p": pd.Series(2, i)}),
    ],
    Kind2.REVENUE: [
        Q_(2.0, "Eur"),
        Q_(0.002, "kEur"),
        pd.Series(0.005, i, dtype="pint[kEur]"),
        pf.FlatPfLine({"r": pd.Series(2, i)}),
    ],
    Kind2.NODIM: [2, pd.Series(2, i)],
}
# ---
series2 = {
    "w": pd.Series([15.0, -20, 4], i),
    "p": pd.Series([400.0, 50, 50], i),
    "r": pd.Series([4464000.0, -696000, 148600], i),
    "nodim": pd.Series([2, -1.5, 10], i),
}
pflset2 = {
    Kind.VOLUME: pf.FlatPfLine({"w": series2["w"]}),
    Kind.PRICE: pf.FlatPfLine({"p": series2["p"]}),
    Kind.COMPLETE: pf.FlatPfLine({"w": series2["w"], "p": series2["p"]}),
}
neg_volume_pfl1 = pf.FlatPfLine({"w": -series_ref["w"]})
neg_price_pfl1 = pf.FlatPfLine({"p": -series_ref["p"]})
neg_all_pfl1 = pf.FlatPfLine({"w": -series_ref["w"], "r": -series_ref["r"]})
add_volume_series = {"w": series_ref["w"] + series2["w"]}
add_volume_pfl = pf.FlatPfLine({"w": add_volume_series["w"]})
sub_volume_series = {"w": series_ref["w"] - series2["w"]}
sub_volume_pfl = pf.FlatPfLine({"w": sub_volume_series["w"]})
add_price_series = {"p": series_ref["p"] + series2["p"]}
add_price_pfl = pf.FlatPfLine({"p": add_price_series["p"]})
sub_price_series = {"p": series_ref["p"] - series2["p"]}
sub_price_pfl = pf.FlatPfLine({"p": sub_price_series["p"]})
add_all_series = {
    "w": series_ref["w"] + series2["w"],
    "r": series_ref["r"] + series2["r"],
}
add_all_pfl = pf.FlatPfLine({"w": add_all_series["w"], "r": add_all_series["r"]})
sub_all_series = {
    "w": series_ref["w"] - series2["w"],
    "r": series_ref["r"] - series2["r"],
}
sub_all_pfl = pf.FlatPfLine({"w": sub_all_series["w"], "r": sub_all_series["r"]})
mul_volume1_price2 = pf.FlatPfLine({"w": series_ref["w"], "p": series2["p"]})
mul_volume2_price1 = pf.FlatPfLine({"w": series2["w"], "p": series_ref["p"]})
div_volume1_volume2 = (series_ref["w"] / series2["w"]).astype("pint[dimensionless]")
div_price1_price2 = (series_ref["p"] / series2["p"]).astype("pint[dimensionless]")
mul_all1_dimless2 = pf.FlatPfLine(
    {"w": series_ref["w"] * series2["nodim"], "p": series_ref["p"]}
)
div_all1_dimless2 = pf.FlatPfLine(
    {"w": series_ref["w"] / series2["nodim"], "p": series_ref["p"]}
)


@pytest.mark.parametrize(
    ("pfl_in", "expected"),
    [
        (pflset_ref[Kind.VOLUME], neg_volume_pfl1),
        (pflset_ref[Kind.PRICE], neg_price_pfl1),
        (pflset_ref[Kind.COMPLETE], neg_all_pfl1),
    ],
)
def test_pfl_neg(pfl_in, expected):
    """Test if portfolio lines can be negated and give correct result."""
    result = -pfl_in
    assert result == expected


class Testcases:
    def multiple(
        pfls: Iterable[PfLine],
        operations: Iterable[str],
        values: Iterable[Any],
        expected_result: Any,
    ):
        if isinstance(pfls, PfLine):
            pfls = [pfls]
        if isinstance(operations, str):
            operations = [operations]
        if not isinstance(values, List):
            values = [values]
        return [
            Testcase(pfl, operation, value, expected_result)
            for value in values
            for pfl in pfls
            for operation in operations
        ]


# Operand 1 = volume pfline.
# . No operand 2.
Testcase(pflset_ref[Kind.VOLUME], "neg", None, neg_volume_pfl1)
# . Operand 2 = None.
Testcase(pflset_ref[Kind.VOLUME], "add", None, pflset_ref[Kind.VOLUME])
Testcase(pflset_ref[Kind.VOLUME], "radd", None, pflset_ref[Kind.VOLUME])
Testcase(pflset_ref[Kind.VOLUME], "sub", None, pflset_ref[Kind.VOLUME])
Testcase(pflset_ref[Kind.VOLUME], "rsub", None, ER.ERROR)  # !
Testcase(pflset_ref[Kind.VOLUME], "mul", None, ER.ERROR)
Testcase(pflset_ref[Kind.VOLUME], "rmul", None, ER.ERROR)
Testcase(pflset_ref[Kind.VOLUME], "div", None, ER.ERROR)
Testcase(pflset_ref[Kind.VOLUME], "rdiv", None, ER.ERROR)
Testcase(pflset_ref[Kind.VOLUME], "union", None, pflset_ref[Kind.VOLUME])
Testcase(pflset_ref[Kind.VOLUME], "runion", None, pflset_ref[Kind.VOLUME])
# . Operand 2 = dimensionless.
Testcases.multiple(
    pflset_ref[Kind.VOLUME], ["add", "radd", "sub", "rsub"], [0, 0.0, 1, 2], ER.ERROR
)
Testcases.multiple(
    pflset_ref[Kind.VOLUME], ["mul" "rmul"], [0, 0.0], pflset0[Kind.VOLUME]
)
Testcases.multiple(pflset_ref[Kind.VOLUME], ["mul" "rmul"], 1, pflset_ref[Kind.VOLUME])
Testcases.multiple(
    pflset_ref[Kind.VOLUME], ["mul" "rmul"], values_2[Kind2.NODIM], pflset2[Kind.VOLUME]
)
Testcases.multiple(pflset_ref[Kind.VOLUME], "div", [0, 0.0], ER.ERROR)
Testcase(pflset_ref[Kind.VOLUME], "div", 1, pflset_ref[Kind.VOLUME])
Testcase(pflset_ref[Kind.VOLUME], "div", 2, pflset_div2[Kind.VOLUME])
Testcases.multiple(pflset_ref[Kind.VOLUME], "rdiv", [0, 0.0, 1, 2], ER.ERROR)
Testcases.multiple(
    pflset_ref[Kind.VOLUME], ["union", "runion"], [0, 0.0, 1, 2], ER.ERROR
)
# . Operand 2 = volume, price, or revenue.
Testcases.multiple(
    pflset_ref[Kind.VOLUME],
    ["add", "radd"],
    [Q_(30, "MW"), Q_(0.03, "GW")],
    pflset_plus2[Kind.VOLUME],
)
Testcases.multiple(
    pflset_ref[Kind.VOLUME],
    "sub",
    [Q_(30, "MW"), Q_(0.03, "GW")],
    pflset_minus2[Kind.VOLUME],
)
Testcases.multiple(
    pflset_ref[Kind.VOLUME],
    ["mul", "rmul", "div", "rdiv", "union", "runion"],
    Q_(30, "MW"),
    ER.ERROR,
)


Testcase(
    pflset_ref[Kind.VOLUME], "add", pflset_ref[Kind.VOLUME], pflset_ref[Kind.VOLUME]
)
Testcase(
    pflset_ref[Kind.VOLUME], "radd", pflset_ref[Kind.VOLUME], pflset_ref[Kind.VOLUME]
)
Testcase(
    pflset_ref[Kind.VOLUME], "sub", pflset_ref[Kind.VOLUME], pflset_ref[Kind.VOLUME]
)
Testcase(pflset_ref[Kind.VOLUME], "rsub", pflset_ref[Kind.VOLUME], neg_volume_pfl1)
Testcase(pflset_ref[Kind.VOLUME], "mul", pflset_ref[Kind.VOLUME], pflset0[Kind.VOLUME])
Testcase(pflset_ref[Kind.VOLUME], "rmul", pflset_ref[Kind.VOLUME], pflset0[Kind.VOLUME])
Testcase(pflset_ref[Kind.VOLUME], "div", pflset_ref[Kind.VOLUME], ER.ERROR)
Testcase(pflset_ref[Kind.VOLUME], "rdiv", pflset_ref[Kind.VOLUME], ER.ERROR)
Testcase(pflset_ref[Kind.VOLUME], "union", pflset_ref[Kind.VOLUME], ER.ERROR)
Testcase(pflset_ref[Kind.VOLUME], "runion", pflset_ref[Kind.VOLUME], ER.ERROR)
# . Operand 2 = complete.
Testcase(pflset_ref[Kind.VOLUME], 0, pflset_ref[Kind.VOLUME])


@pytest.mark.parametrize("operation", ["+", "-"])
@pytest.mark.parametrize(
    ("pfl_in", "value", "expected_add", "expected_sub"),
    [
        # Operand 1 = volume pfline.
        # . Operand 2 = constant without unit.
        # (pflset1[Kind.VOLUME], 0, pflset1[Kind.VOLUME], pflset1[Kind.VOLUME]),
        # (pflset1[Kind.VOLUME], 0.0, pflset1[Kind.VOLUME], pflset1[Kind.VOLUME]),
        # (pflset1[Kind.VOLUME], None, pflset1[Kind.VOLUME], pflset1[Kind.VOLUME]),
        # . Operand 2 = constant with unit.
        (
            pflset_ref[Kind.VOLUME],
            Q_(12.0, "MW"),
            pf.FlatPfLine({"w": pd.Series([15.0, 17, 8], i)}),
            pf.FlatPfLine({"w": pd.Series([-9.0, -7, -16], i)}),
        ),
        (
            pflset_ref[Kind.VOLUME],
            {"w": Q_(12.0, "MW")},
            pf.FlatPfLine({"w": pd.Series([15.0, 17, 8], i)}),
            pf.FlatPfLine({"w": pd.Series([-9.0, -7, -16], i)}),
        ),
        (
            pflset_ref[Kind.VOLUME],
            {"w": 12.0},
            pf.FlatPfLine({"w": pd.Series([15.0, 17, 8], i)}),
            pf.FlatPfLine({"w": pd.Series([-9.0, -7, -16], i)}),
        ),
        # . Operand 2 = constant in different unit
        (
            pflset_ref[Kind.VOLUME],
            Q_(0.012, "GW"),
            pf.FlatPfLine({"w": pd.Series([15.0, 17, 8], i)}),
            pf.FlatPfLine({"w": pd.Series([-9.0, -7, -16], i)}),
        ),
        # . Operand 2 = constant in different dimension.
        (
            pflset_ref[Kind.VOLUME],
            Q_(12.0, "MWh"),
            pf.FlatPfLine({"q": pd.Series([2244.0, 3492, -2960], i)}),
            pf.FlatPfLine({"q": pd.Series([2220.0, 3468, -2984], i)}),
        ),
        # . Operand 2 = series without unit.
        (pflset_ref[Kind.VOLUME], series2["w"], ValueError, ValueError),
        # . Operand 2 = series without name.
        (
            pflset_ref[Kind.VOLUME],
            series2["w"].astype("pint[MW]"),
            add_volume_pfl,
            sub_volume_pfl,
        ),
        # . Operand 2 = series with useless name.
        (
            pflset_ref[Kind.VOLUME],
            series2["w"].rename("the_volume").astype("pint[MW]"),
            add_volume_pfl,
            sub_volume_pfl,
        ),
        # . Operand 2 = series without name and with different unit
        (
            pflset_ref[Kind.VOLUME],
            (series2["w"] * 1000).astype("pint[kW]"),
            add_volume_pfl,
            sub_volume_pfl,
        ),
        # . Operand 2 = series out of order.
        (
            pflset_ref[Kind.VOLUME],
            pd.Series([15.0, 4, -20], [i[0], i[2], i[1]]).astype("pint[MW]"),
            ValueError,
            ValueError,
        ),
        # . Operand 2 = dataframe without unit.
        (
            pflset_ref[Kind.VOLUME],
            pd.DataFrame({"w": series2["w"]}),
            add_volume_pfl,
            sub_volume_pfl,
        ),
        # . Operand 2 = other pfline.
        (pflset_ref[Kind.VOLUME], pflset2[Kind.VOLUME], add_volume_pfl, sub_volume_pfl),
        # Operand 1 = price pfline.
        # . Operand 2 = constant without unit.
        (pflset_ref[Kind.PRICE], 0, pflset_ref[Kind.PRICE], pflset_ref[Kind.PRICE]),
        (pflset_ref[Kind.PRICE], 0.0, pflset_ref[Kind.PRICE], pflset_ref[Kind.PRICE]),
        (pflset_ref[Kind.PRICE], None, pflset_ref[Kind.PRICE], pflset_ref[Kind.PRICE]),
        (
            pflset_ref[Kind.PRICE],
            12.0,
            pf.FlatPfLine({"p": pd.Series([212.0, 112, 62], i)}),
            pf.FlatPfLine({"p": pd.Series([188.0, 88, 38], i)}),
        ),
        # . Operand 2 = constant with default unit.
        (
            pflset_ref[Kind.PRICE],
            Q_(12.0, "Eur/MWh"),
            pf.FlatPfLine({"p": pd.Series([212.0, 112, 62], i)}),
            pf.FlatPfLine({"p": pd.Series([188.0, 88, 38], i)}),
        ),
        # . Operand 2 = constant with non-default unit.
        (
            pflset_ref[Kind.PRICE],
            Q_(1.2, "ct/kWh"),
            pf.FlatPfLine({"p": pd.Series([212.0, 112, 62], i)}),
            pf.FlatPfLine({"p": pd.Series([188.0, 88, 38], i)}),
        ),
        # . Operand 2 = other pfline.
        (pflset_ref[Kind.PRICE], pflset2[Kind.PRICE], add_price_pfl, sub_price_pfl),
        # Operand 1 = full pfline.
        # . Operand 2 = constant without unit.
        (
            pflset_ref[Kind.COMPLETE],
            0,
            pflset_ref[Kind.COMPLETE],
            pflset_ref[Kind.COMPLETE],
        ),
        (
            pflset_ref[Kind.COMPLETE],
            0.0,
            pflset_ref[Kind.COMPLETE],
            pflset_ref[Kind.COMPLETE],
        ),
        (
            pflset_ref[Kind.COMPLETE],
            None,
            pflset_ref[Kind.COMPLETE],
            pflset_ref[Kind.COMPLETE],
        ),
        (pflset_ref[Kind.COMPLETE], 12.0, ValueError, ValueError),
        # . Operand 2 = series without unit.
        (pflset_ref[Kind.COMPLETE], series2["w"], ValueError, ValueError),
        # . Operand 2 = dataframe.
        (
            pflset_ref[Kind.COMPLETE],
            pd.DataFrame({"w": series2["w"], "p": series2["p"]}),
            add_all_pfl,
            sub_all_pfl,
        ),
        # . Operand 2 = dataframe.
        (
            pflset_ref[Kind.COMPLETE],
            pd.DataFrame({"w": series2["w"], "r": series2["r"]}),
            add_all_pfl,
            sub_all_pfl,
        ),
        # . Operand 2 = other pfline.
        (pflset_ref[Kind.COMPLETE], pflset2[Kind.COMPLETE], add_all_pfl, sub_all_pfl),
    ],
    ids=id_fn,
)
def test_pfl_addsub_full(pfl_in, value, expected_add, expected_sub, operation):
    """Test if portfolio lines can be added and subtracted and give correct result."""
    expected = expected_add if operation == "+" else expected_sub

    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            _ = (pfl_in + value) if operation == "+" else (pfl_in - value)
        return

    result = (pfl_in + value) if operation == "+" else (pfl_in - value)
    assert result == expected


@pytest.mark.parametrize("operation", ["*", "/"])
@pytest.mark.parametrize(
    ("pfl_in", "value", "expected_mul", "expected_div"),
    [
        # Multiplying volume pfline.
        # . (dimension-agnostic) constant.
        (
            pflset_ref[Kind.VOLUME],
            4.0,
            pf.FlatPfLine({"w": series_ref["w"] * 4}),
            pf.FlatPfLine({"w": series_ref["w"] / 4}),
        ),
        (
            pflset_ref[Kind.VOLUME],
            {"agn": 4.0},
            pf.FlatPfLine({"w": series_ref["w"] * 4}),
            pf.FlatPfLine({"w": series_ref["w"] / 4}),
        ),
        # . Explicitly dimensionless constant.
        (
            pflset_ref[Kind.VOLUME],
            Q_(4.0, ""),
            pf.FlatPfLine({"w": series_ref["w"] * 4}),
            pf.FlatPfLine({"w": series_ref["w"] / 4}),
        ),
        (
            pflset_ref[Kind.VOLUME],
            {"nodim": Q_(4.0, "")},
            pf.FlatPfLine({"w": series_ref["w"] * 4}),
            pf.FlatPfLine({"w": series_ref["w"] / 4}),
        ),
        (
            pflset_ref[Kind.VOLUME],
            {"nodim": 4.0},
            pf.FlatPfLine({"w": series_ref["w"] * 4}),
            pf.FlatPfLine({"w": series_ref["w"] / 4}),
        ),
        # . Fixed price constant.
        (
            pflset_ref[Kind.VOLUME],
            Q_(4.0, "Eur/MWh"),
            pf.FlatPfLine({"w": series_ref["w"], "p": 4}),
            Exception,
        ),
        (
            pflset_ref[Kind.VOLUME],
            {"p": Q_(4.0, "Eur/MWh")},
            pf.FlatPfLine({"w": series_ref["w"], "p": 4}),
            Exception,
        ),
        (
            pflset_ref[Kind.VOLUME],
            {"p": Q_(0.4, "ctEur/kWh")},
            pf.FlatPfLine({"w": series_ref["w"], "p": 4}),
            Exception,
        ),
        (
            pflset_ref[Kind.VOLUME],
            {"p": 4.0},
            pf.FlatPfLine({"w": series_ref["w"], "p": 4}),
            Exception,
        ),
        (
            pflset_ref[Kind.VOLUME],
            pd.Series([Q_(4.0, "Eur/MWh")], ["p"]),
            pf.FlatPfLine({"w": series_ref["w"], "p": 4}),
            Exception,
        ),
        (
            pflset_ref[Kind.VOLUME],
            pd.Series([4.0], ["p"]),
            pf.FlatPfLine({"w": series_ref["w"], "p": 4}),
            Exception,
        ),
        (
            pflset_ref[Kind.VOLUME],
            pd.Series([4.0], ["p"]).astype("pint[Eur/MWh]"),
            pf.FlatPfLine({"w": series_ref["w"], "p": 4}),
            Exception,
        ),
        # . Fixed volume constant.
        (
            pflset_ref[Kind.VOLUME],
            {"w": Q_(4.0, "MW")},
            Exception,
            (series_ref["w"] / 4).astype("pint[dimensionless]"),
        ),
        (
            pflset_ref[Kind.VOLUME],
            {"w": 4.0},
            Exception,
            (series_ref["w"] / 4).astype("pint[dimensionless]"),
        ),
        (
            pflset_ref[Kind.VOLUME],
            pd.Series([Q_(4.0, "MW")], ["w"]),
            Exception,
            (series_ref["w"] / 4).astype("pint[dimensionless]"),
        ),
        (
            pflset_ref[Kind.VOLUME],
            pd.Series([4.0], ["w"]).astype("pint[MW]"),
            Exception,
            (series_ref["w"] / 4).astype("pint[dimensionless]"),
        ),
        (  # divide by fixed ENERGY not POWER
            pflset_ref[Kind.VOLUME],
            pd.Series([4.0], ["q"]).astype("pint[MWh]"),
            Exception,
            (pflset_ref[Kind.VOLUME].q.pint.m / 4).astype("pint[dimensionless]"),
        ),
        # . Constant with incorrect unit.
        (pflset_ref[Kind.VOLUME], {"r": 4.0}, Exception, Exception),
        (
            pflset_ref[Kind.VOLUME],
            {"q": 4.0, "w": 8.0},  # incompatible
            Exception,
            Exception,
        ),
        (pflset_ref[Kind.VOLUME], Q_(4.0, "Eur"), Exception, Exception),
        (pflset_ref[Kind.VOLUME], {"r": 4.0, "q": 12}, Exception, Exception),
        (pflset_ref[Kind.VOLUME], {"r": 4.0, "nodim": 4.0}, Exception, Exception),
        # . Dim-agnostic or dimless series.
        (
            pflset_ref[Kind.VOLUME],
            series2["w"],  # has no unit
            pf.FlatPfLine({"w": series_ref["w"] * series2["w"]}),
            pf.FlatPfLine({"w": series_ref["w"] / series2["w"]}),
        ),
        (
            pflset_ref[Kind.VOLUME],
            series2["w"].astype("pint[dimensionless]"),  # dimensionless
            pf.FlatPfLine({"w": series_ref["w"] * series2["w"]}),
            pf.FlatPfLine({"w": series_ref["w"] / series2["w"]}),
        ),
        # . Price series, dataframe, or PfLine
        (
            pflset_ref[Kind.VOLUME],
            series2["p"].astype("pint[Eur/MWh]"),
            mul_volume1_price2,
            Exception,
        ),
        (
            pflset_ref[Kind.VOLUME],
            series2["p"].rename("the_price").astype("pint[Eur/MWh]"),
            mul_volume1_price2,
            Exception,
        ),
        (
            pflset_ref[Kind.VOLUME],
            (series2["p"] * 0.1).astype("pint[ct/kWh]"),
            mul_volume1_price2,
            Exception,
        ),
        (
            pflset_ref[Kind.VOLUME],
            pd.DataFrame({"p": series2["p"]}),
            mul_volume1_price2,
            Exception,
        ),
        (
            pflset_ref[Kind.VOLUME],
            pd.DataFrame({"p": (series2["p"] / 10).astype("pint[ct/kWh]")}),
            mul_volume1_price2,
            Exception,
        ),
        (pflset_ref[Kind.VOLUME], pflset2[Kind.PRICE], mul_volume1_price2, Exception),
        (
            pflset_ref[Kind.VOLUME],
            pd.Series([50.0, 400, 50], [i[1], i[0], i[2]]).astype(
                "pint[Eur/MWh]"
            ),  # not standardized
            ValueError,
            ValueError,
        ),
        # . Volume series, dataframe, or pfline
        (
            pflset_ref[Kind.VOLUME],
            series2["w"].astype("pint[MW]"),
            Exception,
            div_volume1_volume2,
        ),
        (pflset_ref[Kind.VOLUME], pflset2[Kind.VOLUME], Exception, div_volume1_volume2),
        (
            pflset_ref[Kind.VOLUME],
            pflset2[Kind.VOLUME].q,
            Exception,
            div_volume1_volume2,
        ),
        (pflset_ref[Kind.VOLUME], {"w": series2["w"]}, Exception, div_volume1_volume2),
        (
            pflset_ref[Kind.VOLUME],
            pd.DataFrame({"w": series2["w"]}),
            Exception,
            div_volume1_volume2,
        ),
        (
            pflset_ref[Kind.VOLUME],
            pflset2[Kind.VOLUME],  # other pfline
            Exception,
            div_volume1_volume2,
        ),
        # . Incorrect series, dataframe or pfline.
        (
            pflset_ref[Kind.VOLUME],
            series2["r"].astype("pint[Eur]"),
            Exception,
            Exception,
        ),
        (
            pflset_ref[Kind.VOLUME],
            pd.DataFrame({"r": series2["r"]}),
            Exception,
            Exception,
        ),
        (
            pflset_ref[Kind.VOLUME],
            pd.DataFrame({"the_price": series2["p"].astype("pint[Eur/MWh]")}),
            KeyError,
            KeyError,
        ),
        (pflset_ref[Kind.VOLUME], pflset2[Kind.COMPLETE], Exception, Exception),
        # Multiplying price pfline.
        # . (dimension-agnostic) constant.
        (
            pflset_ref[Kind.PRICE],
            4,
            pf.FlatPfLine({"p": series_ref["p"] * 4}),
            pf.FlatPfLine({"p": series_ref["p"] / 4}),
        ),
        # . Explicitly dimensionless constant.
        (
            pflset_ref[Kind.PRICE],
            Q_(4, ""),
            pf.FlatPfLine({"p": series_ref["p"] * 4}),
            pf.FlatPfLine({"p": series_ref["p"] / 4}),
        ),
        # . Fixed price constant.
        (
            pflset_ref[Kind.PRICE],
            Q_(4, "Eur/MWh"),
            Exception,
            (series_ref["p"] / 4).astype("pint[dimensionless]"),
        ),
        (
            pflset_ref[Kind.PRICE],
            {"p": 4},
            Exception,
            (series_ref["p"] / 4).astype("pint[dimensionless]"),
        ),
        # . Fixed volume constant.
        (
            pflset_ref[Kind.PRICE],
            Q_(4, "MWh"),
            pf.FlatPfLine({"p": series_ref["p"], "q": 4}),
            Exception,
        ),
        (
            pflset_ref[Kind.PRICE],
            Q_(4, "MW"),
            pf.FlatPfLine({"p": series_ref["p"], "w": 4}),
            Exception,
        ),
        (
            pflset_ref[Kind.PRICE],
            Q_(4, "GW"),
            pf.FlatPfLine({"p": series_ref["p"], "w": 4000}),
            Exception,
        ),
        (
            pflset_ref[Kind.PRICE],
            pd.Series([4], ["w"]).astype("pint[GW]"),
            pf.FlatPfLine({"p": series_ref["p"], "w": 4000}),
            Exception,
        ),
        # . Incorrect constant.
        (pflset_ref[Kind.PRICE], Q_(4, "Eur"), Exception, Exception),
        # . Dim-agnostic or dimless series.
        (
            pflset_ref[Kind.PRICE],
            series2["w"],  # has no unit
            pf.FlatPfLine({"p": series_ref["p"] * series2["w"]}),
            pf.FlatPfLine({"p": series_ref["p"] / series2["w"]}),
        ),
        (
            pflset_ref[Kind.PRICE],
            series2["w"].astype("pint[dimensionless]"),  # dimensionless
            pf.FlatPfLine({"p": series_ref["p"] * series2["w"]}),
            pf.FlatPfLine({"p": series_ref["p"] / series2["w"]}),
        ),
        # . Price series, dataframe, or PfLine
        (
            pflset_ref[Kind.PRICE],
            series2["p"].astype("pint[Eur/MWh]"),  # series
            Exception,
            div_price1_price2,
        ),
        (
            pflset_ref[Kind.PRICE],
            pflset2[Kind.PRICE],  # pfline
            Exception,
            div_price1_price2,
        ),
        (
            pflset_ref[Kind.PRICE],
            pflset2[Kind.PRICE].p,  # series
            Exception,
            div_price1_price2,
        ),
        (
            pflset_ref[Kind.PRICE],
            {"p": series2["p"]},  # dict of series
            Exception,
            div_price1_price2,
        ),
        (
            pflset_ref[Kind.PRICE],
            pd.DataFrame({"p": series2["p"]}),  # dataframe
            Exception,
            div_price1_price2,
        ),
        (
            pflset_ref[Kind.PRICE],
            pflset2[Kind.PRICE],  # other pfline
            Exception,
            div_price1_price2,
        ),
        # . Volume series, dataframe, or pfline
        (
            pflset_ref[Kind.PRICE],
            series2["w"].astype("pint[MW]"),
            mul_volume2_price1,
            Exception,
        ),
        (
            pflset_ref[Kind.PRICE],
            (series2["w"] / 1000).astype("pint[GW]"),
            mul_volume2_price1,
            Exception,
        ),
        (
            pflset_ref[Kind.PRICE],
            series2["w"].rename("the_volume").astype("pint[MW]"),
            mul_volume2_price1,
            Exception,
        ),
        (
            pflset_ref[Kind.PRICE],
            pd.DataFrame({"w": series2["w"]}),
            mul_volume2_price1,
            Exception,
        ),
        (
            pflset_ref[Kind.PRICE],
            pd.DataFrame({"w": (series2["w"] / 1000).astype("pint[GW]")}),
            mul_volume2_price1,
            Exception,
        ),
        (pflset_ref[Kind.PRICE], pflset2[Kind.VOLUME], mul_volume2_price1, Exception),
        # . Incorrect series, dataframe or pfline.
        (
            pflset_ref[Kind.PRICE],
            series2["r"].astype("pint[Eur]"),
            Exception,
            Exception,
        ),
        (
            pflset_ref[Kind.PRICE],
            pd.DataFrame({"r": series2["r"]}),
            Exception,
            Exception,
        ),
        (
            pflset_ref[Kind.PRICE],
            pd.DataFrame({"the_price": series2["p"].astype("pint[Eur/MWh]")}),
            KeyError,
            KeyError,
        ),
        (pflset_ref[Kind.PRICE], pflset2[Kind.COMPLETE], Exception, Exception),
        # Multiplying 'complete' pfline.
        # . (dimension-agnostic) constant.
        (
            pflset_ref[Kind.COMPLETE],
            6,
            FlatPfLine({"w": series_ref["w"] * 6, "p": series_ref["p"]}),
            FlatPfLine({"w": series_ref["w"] / 6, "p": series_ref["p"]}),
        ),
        # . Explicitly dimensionless constant.
        (
            pflset_ref[Kind.COMPLETE],
            Q_(6, ""),
            FlatPfLine({"w": series_ref["w"] * 6, "p": series_ref["p"]}),
            FlatPfLine({"w": series_ref["w"] / 6, "p": series_ref["p"]}),
        ),
        (
            pflset_ref[Kind.COMPLETE],
            {"nodim": 6},
            FlatPfLine({"w": series_ref["w"] * 6, "p": series_ref["p"]}),
            FlatPfLine({"w": series_ref["w"] / 6, "p": series_ref["p"]}),
        ),
        (
            pflset_ref[Kind.COMPLETE],
            pd.Series([6], ["nodim"]),
            FlatPfLine({"w": series_ref["w"] * 6, "p": series_ref["p"]}),
            FlatPfLine({"w": series_ref["w"] / 6, "p": series_ref["p"]}),
        ),
        # . Incorrect constant.
        (pflset_ref[Kind.COMPLETE], {"r": 4.0}, Exception, Exception),
        (pflset_ref[Kind.COMPLETE], {"q": 4.0, "w": 8.0}, Exception, Exception),
        (pflset_ref[Kind.COMPLETE], Q_(4.0, "Eur"), Exception, Exception),
        (pflset_ref[Kind.COMPLETE], {"r": 4.0, "q": 12}, Exception, Exception),
        (pflset_ref[Kind.COMPLETE], {"r": 4.0, "nodim": 4.0}, Exception, Exception),
        # . Dim-agnostic or dimless series.
        (
            pflset_ref[Kind.COMPLETE],
            series2["nodim"],  # dim-agnostic
            mul_all1_dimless2,
            div_all1_dimless2,
        ),
        (
            pflset_ref[Kind.COMPLETE],
            series2["nodim"].astype("pint[dimensionless]"),  # dimless
            mul_all1_dimless2,
            div_all1_dimless2,
        ),
        (
            pflset_ref[Kind.COMPLETE],
            {"nodim": series2["nodim"]},
            mul_all1_dimless2,
            div_all1_dimless2,
        ),
        (
            pflset_ref[Kind.COMPLETE],
            pd.DataFrame({"nodim": series2["nodim"]}),
            mul_all1_dimless2,
            div_all1_dimless2,
        ),
        (
            pflset_ref[Kind.COMPLETE],
            pd.DataFrame({"nodim": series2["nodim"].astype("pint[dimensionless]")}),
            mul_all1_dimless2,
            div_all1_dimless2,
        ),
        # . Incorrect series, dataframe or pfline.
        (pflset_ref[Kind.COMPLETE], {"r": series2["p"]}, Exception, Exception),
        (
            pflset_ref[Kind.COMPLETE],
            series2["p"].astype("pint[Eur/MWh]"),
            Exception,
            Exception,
        ),
        (pflset_ref[Kind.COMPLETE], pflset2[Kind.PRICE], Exception, Exception),
        (pflset_ref[Kind.COMPLETE], pflset2[Kind.VOLUME], Exception, Exception),
        (pflset_ref[Kind.COMPLETE], pflset2[Kind.COMPLETE], Exception, Exception),
    ],
    ids=id_fn,
)
def test_pfl_muldiv_full(pfl_in, value, expected_mul, expected_div, operation):
    """Test if portfolio lines can be multiplied and divided and give correct result.
    Includes partly overlapping indices."""
    expected = expected_mul if operation == "*" else expected_div

    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            _ = (pfl_in * value) if operation == "*" else (pfl_in / value)
        return

    result = (pfl_in * value) if operation == "*" else (pfl_in / value)
    if isinstance(expected, pd.Series):
        testing.assert_series_equal(result, expected, check_names=False)
    else:
        assert result == expected
