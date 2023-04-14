"""Test if arithmatic returns correct type/kind of object and/or correctly raises error."""
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Iterable

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


class Kind2(Enum):  # Kind of value for other operand
    VOLUME = Kind.VOLUME
    PRICE = Kind.PRICE
    REVENUE = Kind.REVENUE
    COMPLETE = Kind.COMPLETE
    NODIM = "nodim"
    NONE = "none"


class ER(Enum):  # Expected result
    VOLUME = Kind.VOLUME
    PRICE = Kind.PRICE
    REVENUE = Kind.REVENUE
    COMPLETE = Kind.COMPLETE
    SERIES = "nodim"
    ERROR = "error"

    def typ(self) -> type:
        if self.value in [*Kind]:  # cannot just write `in Kind`
            return PfLine
        if self is ER.SERIES:
            return pd.Series
        return None

    def kind(self) -> Kind:
        if self.value in [*Kind]:  # cannot just write `in Kind`
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
        for config, er in outcomedict.items():
            for testcase in cls.from_config(config, er):
                yield testcase  # generator to save memory (?)

    @classmethod
    def from_config(cls, config: TestcaseConfig, er: ER) -> Iterable[Testcase]:
        for tp in cls._testpfls.fetch(config.pfl_kind, config.pfl_nestedness):
            for tv in cls._testvalues.fetch(config.value_kind, config.value_nestedness):
                yield Testcase(tp.pfl, tv.value, er)


@pytest.mark.parametrize("operation", ["add", "radd", "sub", "rsub"])
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
            TestcaseConfig(Kind.VOLUME, "flat", Kind2.VOLUME, "flat"): ER.VOLUME,
            TestcaseConfig(Kind.VOLUME, "nested", Kind2.VOLUME, "nested"): ER.VOLUME,
            # . Operand 2 = complete.
            # Operand 1 = price pfline.
            # . Operand 2 = None.
            TestcaseConfig(Kind.PRICE, "flat", Kind2.NONE, "flat"): ER.PRICE,
            TestcaseConfig(Kind.PRICE, "nested", Kind2.NONE, "flat"): ER.PRICE,
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            TestcaseConfig(Kind.PRICE, "flat", Kind2.PRICE, "flat"): ER.PRICE,
            TestcaseConfig(Kind.PRICE, "nested", Kind2.PRICE, "nested"): ER.PRICE,
            # . Operand 2 = complete.
            # Operand 1 = revenue pfline.
            # . Operand 2 = None.
            TestcaseConfig(Kind.REVENUE, "flat", Kind2.NONE, "flat"): ER.REVENUE,
            TestcaseConfig(Kind.REVENUE, "nested", Kind2.NONE, "flat"): ER.REVENUE,
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            TestcaseConfig(Kind.REVENUE, "flat", Kind2.REVENUE, "flat"): ER.REVENUE,
            TestcaseConfig(Kind.REVENUE, "nested", Kind2.REVENUE, "nested"): ER.REVENUE,
            # . Operand 2 = complete.
            # Operand 1 = complete pfline.
            # . Operand 2 = None.
            TestcaseConfig(Kind.COMPLETE, "flat", Kind2.NONE, "flat"): ER.COMPLETE,
            TestcaseConfig(Kind.COMPLETE, "nested", Kind2.NONE, "flat"): ER.COMPLETE,
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            # . Operand 2 = complete.
            TestcaseConfig(Kind.COMPLETE, "flat", Kind2.COMPLETE, "flat"): ER.COMPLETE,
            TestcaseConfig(
                Kind.COMPLETE, "nested", Kind2.COMPLETE, "nested"
            ): ER.COMPLETE,
        }
    ),
    ids=id_fn,
)
def test_pfl_arithmatic_kind_addraddsubrsub(testcase: Testcase, operation: str):
    """Test if arithmatic expectedly raises Error or returns expected type/kind."""
    do_kind_test(testcase, operation)


@pytest.mark.parametrize("operation", ["mul", "rmul"])
@pytest.mark.parametrize(
    "testcase",
    Testcases.all_from_nonerror(
        {
            # Operand 1 = volume pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            TestcaseConfig(Kind.VOLUME, "flat", Kind2.NODIM, "flat"): ER.VOLUME,
            TestcaseConfig(Kind.VOLUME, "nested", Kind2.NODIM, "flat"): ER.VOLUME,
            # . Operand 2 = volume, price, or revenue.
            TestcaseConfig(Kind.VOLUME, "flat", Kind2.PRICE, "flat"): ER.REVENUE,
            # . Operand 2 = complete.
            # Operand 1 = price pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            TestcaseConfig(Kind.PRICE, "flat", Kind2.NODIM, "flat"): ER.PRICE,
            TestcaseConfig(Kind.PRICE, "nested", Kind2.NODIM, "flat"): ER.PRICE,
            # . Operand 2 = volume, price, or revenue.
            TestcaseConfig(Kind.PRICE, "flat", Kind2.VOLUME, "flat"): ER.REVENUE,
            # . Operand 2 = complete.
            # Operand 1 = revenue pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            TestcaseConfig(Kind.REVENUE, "flat", Kind2.NODIM, "flat"): ER.REVENUE,
            TestcaseConfig(Kind.REVENUE, "nested", Kind2.NODIM, "flat"): ER.REVENUE,
            # . Operand 2 = volume, price, or revenue.
            # . Operand 2 = complete.
            # Operand 1 = complete pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            TestcaseConfig(Kind.COMPLETE, "flat", Kind2.NODIM, "flat"): ER.COMPLETE,
            TestcaseConfig(Kind.COMPLETE, "nested", Kind2.NODIM, "flat"): ER.COMPLETE,
            # . Operand 2 = volume, price, or revenue.
            # . Operand 2 = complete.
        }
    ),
    ids=id_fn,
)
def test_pfl_arithmatic_kind_mulrmul(testcase: Testcase, operation: str):
    """Test if arithmatic expectedly raises Error or returns expected type/kind."""
    do_kind_test(testcase, operation)


@pytest.mark.parametrize("operation", ["div"])
@pytest.mark.parametrize(
    "testcase",
    Testcases.all_from_nonerror(
        {
            # Operand 1 = volume pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            TestcaseConfig(Kind.VOLUME, "flat", Kind2.NODIM, "flat"): ER.VOLUME,
            TestcaseConfig(Kind.VOLUME, "nested", Kind2.NODIM, "flat"): ER.VOLUME,
            # . Operand 2 = volume, price, or revenue.
            TestcaseConfig(Kind.VOLUME, "flat", Kind2.VOLUME, "flat"): ER.SERIES,
            # . Operand 2 = complete.
            # Operand 1 = price pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            TestcaseConfig(Kind.PRICE, "flat", Kind2.NODIM, "flat"): ER.PRICE,
            TestcaseConfig(Kind.PRICE, "nested", Kind2.NODIM, "flat"): ER.PRICE,
            # . Operand 2 = volume, price, or revenue.
            TestcaseConfig(Kind.PRICE, "flat", Kind2.PRICE, "flat"): ER.SERIES,
            # . Operand 2 = complete.
            # Operand 1 = revenue pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            TestcaseConfig(Kind.REVENUE, "flat", Kind2.NODIM, "flat"): ER.REVENUE,
            TestcaseConfig(Kind.REVENUE, "nested", Kind2.NODIM, "flat"): ER.REVENUE,
            # . Operand 2 = volume, price, or revenue.
            TestcaseConfig(Kind.REVENUE, "flat", Kind2.VOLUME, "flat"): ER.PRICE,
            TestcaseConfig(Kind.REVENUE, "flat", Kind2.PRICE, "flat"): ER.VOLUME,
            TestcaseConfig(Kind.REVENUE, "flat", Kind2.REVENUE, "flat"): ER.SERIES,
            # . Operand 2 = complete.
            # Operand 1 = complete pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            TestcaseConfig(Kind.COMPLETE, "flat", Kind2.NODIM, "flat"): ER.COMPLETE,
            TestcaseConfig(Kind.COMPLETE, "nested", Kind2.NODIM, "flat"): ER.COMPLETE,
            # . Operand 2 = volume, price, or revenue.
            # . Operand 2 = complete.
        }
    ),
    ids=id_fn,
)
def test_pfl_arithmatic_kind_div(testcase: Testcase, operation: str):
    """Test if arithmatic expectedly raises Error or returns expected type/kind."""
    do_kind_test(testcase, operation)


@pytest.mark.parametrize("operation", ["rdiv"])
@pytest.mark.parametrize(
    "testcase",
    Testcases.all_from_nonerror(
        {
            # Operand 1 = volume pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            TestcaseConfig(Kind.VOLUME, "flat", Kind2.VOLUME, "flat"): ER.SERIES,
            TestcaseConfig(Kind.VOLUME, "flat", Kind2.REVENUE, "flat"): ER.PRICE,
            # . Operand 2 = complete.
            # Operand 1 = price pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            TestcaseConfig(Kind.PRICE, "flat", Kind2.PRICE, "flat"): ER.SERIES,
            TestcaseConfig(Kind.PRICE, "flat", Kind2.REVENUE, "flat"): ER.VOLUME,
            # . Operand 2 = complete.
            # Operand 1 = revenue pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            TestcaseConfig(Kind.REVENUE, "flat", Kind2.REVENUE, "flat"): ER.SERIES,
            # . Operand 2 = complete.
            # Operand 1 = complete pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            # . Operand 2 = complete.
        }
    ),
    ids=id_fn,
)
def test_pfl_arithmatic_kind_rdiv(testcase: Testcase, operation: str):
    """Test if arithmatic expectedly raises Error or returns expected type/kind."""
    do_kind_test(testcase, operation)


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
