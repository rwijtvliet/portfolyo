"""Test if arithmatic returns correct type/kind of object and/or correctly raises error."""
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Iterable

import pandas as pd
import pytest

import portfolyo as pf
from portfolyo import Q_, FlatPfLine, Kind, NestedPfLine, PfLine
from portfolyo.core.pfline import arithmatic

# TODO: various timezones

# TODO: use/change STRICT setting


arithmatic.STRICT = True


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
    elif isinstance(data, Case):
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
class Pfl:  # Testpfl
    kind: Kind
    nestedness: str  # 'flat' or 'nested'  #TODO: change to use Structure instead
    pfl: PfLine


@dataclass
class Value:  # Testvalue
    kind: Kind2
    nestedness: str  # 'flat' or 'nested'
    value: Any


@dataclass
class Case:  # Testcase
    pfl: PfLine
    value: Any
    expected_result: ER


@dataclass(frozen=True, eq=True)
class CaseConfig:  # TestcaseConfig
    pfl_kind: Kind
    pfl_nestedness: str  # 'flat' or 'nested'
    value_kind: Kind2
    value_nestedness: str  # 'flat' or 'nested'


class Pfls:  # Testpfls
    def __init__(self, i: pd.DatetimeIndex):
        flat_fn, nested_fn = pf.dev.get_flatpfline, pf.dev.get_nestedpfline
        self._testpfls = [
            *[Pfl(kind, "flat", flat_fn(i, kind, _seed=1)) for kind in Kind],
            *[Pfl(kind, "nested", nested_fn(i, kind, _seed=1)) for kind in Kind],
        ]

    def fetch(self, kind: Kind = None, nestedness: str = None) -> Iterable[Pfl]:
        testpfls: Iterable[Pfl] = self._testpfls
        if kind is not None:
            testpfls = [tp for tp in testpfls if tp.kind is kind]
        if nestedness is not None:
            testpfls = [tp for tp in testpfls if tp.nestedness == nestedness]
        return testpfls


class Values:  # Testvalues
    UNIT_ALT = {"w": "GW", "q": "kWh", "p": "ctEur/kWh", "r": "MEur", "nodim": ""}

    def __init__(self, i: pd.DatetimeIndex):
        self._testvalues: Iterable[Value] = [
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
    def fetch(self, kind: Kind2 = None, nestedness: str = None) -> Iterable[Value]:
        testvalues: Iterable[Value] = self._testvalues
        if kind is not None:
            testvalues = [tv for tv in testvalues if tv.kind is kind]
        if nestedness is not None:
            testvalues = [tv for tv in testvalues if tv.nestedness == nestedness]
        return testvalues

    @staticmethod
    def from_noname(i: pd.DatetimeIndex) -> Iterable[Value]:
        fl = pf.dev.get_value("nodim", False, _seed=2)
        flseries = pf.dev.get_series(i, "", False, _seed=2)
        return [
            Value(Kind2.NONE, "flat", None),
            Value(Kind2.NODIM, "flat", fl),  # Single value.
            Value(Kind2.NODIM, "flat", flseries),  # Timeseries.
        ]

    @staticmethod
    def from_1name(kind: Kind2, name: str, i: pd.DatetimeIndex) -> Iterable[Value]:
        """name == {'w', 'q', 'p', 'r', 'nodim'}"""
        quantity = pf.dev.get_value(name, True, _seed=2)
        altunit = Values.UNIT_ALT[name]
        quseries = pf.dev.get_series(i, name, _seed=2)
        return [
            # . Single value.
            Value(kind, "flat", quantity),
            Value(kind, "flat", quantity * 0),
            Value(kind, "flat", quantity.to(altunit)),
            Value(kind, "flat", {name: quantity.m}),
            Value(kind, "flat", {name: quantity}),
            Value(kind, "flat", {name: quantity.to(altunit)}),
            Value(kind, "flat", pd.Series({name: quantity.m})),
            Value(kind, "flat", pd.Series({name: quantity})),
            # . Timeseries.
            Value(kind, "flat", quseries),
            Value(kind, "flat", {name: quseries.pint.m}),
            Value(kind, "flat", {name: quseries}),
            Value(kind, "flat", pd.DataFrame({name: quseries.pint.m})),
            Value(kind, "flat", pd.DataFrame({name: quseries})),
        ]

    @staticmethod
    def from_2names(
        kind: Kind2, names: Iterable[str], i: pd.DatetimeIndex
    ) -> Iterable[Value]:
        """names == two of {'w', 'q', 'p', 'r'}"""
        n1, n2 = names
        qu1, qu2 = (pf.dev.get_value(name, True, _seed=2) for name in names)
        fl1, fl2 = (qu.magnitude for qu in (qu1, qu2))
        qus1, qus2 = (pf.dev.get_series(i, name, _seed=2) for name in names)
        fls1, fls2 = (quseries.pint.m for quseries in (qus1, qus2))
        return [
            # . Single values.
            Value(kind, "flat", {n1: qu1, n2: qu2}),
            Value(kind, "flat", {n1: fl1, n2: fl2}),
            Value(kind, "flat", {n1: fl1, n2: qu2}),
            Value(kind, "flat", pd.Series({n1: qu1, n2: qu2})),
            Value(kind, "flat", pd.Series({n1: fl1, n2: fl2})),
            Value(kind, "flat", pd.Series({n1: fl1, n2: qu2})),
            # . Single value | timeseries.
            Value(kind, "flat", {n1: qu1, n2: qus2}),
            Value(kind, "flat", {n1: fl1, n2: qus2}),
            Value(kind, "flat", {n1: qu1, n2: fls2}),
            Value(kind, "flat", {n1: fl1, n2: fls2}),
            # . Timeseries.
            Value(kind, "flat", {n1: qus1, n2: qus2}),
            Value(kind, "flat", {n1: qus1, n2: fls2}),
            Value(kind, "flat", pd.DataFrame({n1: qus1, n2: qus2})),
            Value(kind, "flat", pd.DataFrame({n1: qus1, n2: fls2})),
        ]

    @staticmethod
    def from_pfline(kind: Kind2, i: pd.DatetimeIndex) -> Iterable[Value]:
        k = kind.value
        pfl1, pfl2 = (pf.dev.get_flatpfline(i, k, _seed=s) for s in (2, 3))
        return [
            Value(kind, "flat", pfl1),
            Value(kind, "nested", pf.NestedPfLine({"childA": pfl1, "childB": pfl2})),
            Value(kind, "nested", {"childC": pfl1, "childD": pfl2}),
        ]


tz = "Europe/Berlin"
i = pd.date_range("2020", periods=20, freq="MS", tz=tz)  # reference


class Cases:  # Testcases
    _testvalues = Values(i)
    _testpfls = Pfls(i)
    _complete_outcomedict = {
        CaseConfig(pfl_kind, pfl_nestedness, val_kind, val_nestedness): ER.ERROR
        for pfl_kind in Kind
        for pfl_nestedness in ["flat", "nested"]
        for val_kind in Kind2
        for val_nestedness in ["flat", "nested"]
    }

    @classmethod
    def get_pfls(
        cls, pfl_kind: Kind = None, pfl_nestedness: str = None
    ) -> Iterable[Case]:
        return cls._testpfls.fetch(pfl_kind, pfl_nestedness)

    @classmethod
    def all_from_nonerror(
        cls, non_error_outcomedict: Dict[CaseConfig, ER]
    ) -> Iterable[Case]:
        outcomedict = {**cls._complete_outcomedict, **non_error_outcomedict}
        for config, er in outcomedict.items():
            for testcase in cls.from_config(config, er):
                yield testcase  # generator to save memory (?)

    @classmethod
    def from_config(cls, config: CaseConfig, er: ER) -> Iterable[Case]:
        for tp in cls._testpfls.fetch(config.pfl_kind, config.pfl_nestedness):
            for tv in cls._testvalues.fetch(config.value_kind, config.value_nestedness):
                yield Case(tp.pfl, tv.value, er)


@pytest.mark.parametrize("operation", ["add", "radd", "sub", "rsub"])
@pytest.mark.parametrize(
    "testcase",
    Cases.all_from_nonerror(
        {
            # Operand 1 = volume pfline.
            # . Operand 2 = None.
            CaseConfig(Kind.VOLUME, "flat", Kind2.NONE, "flat"): ER.VOLUME,
            CaseConfig(Kind.VOLUME, "nested", Kind2.NONE, "flat"): ER.VOLUME,
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(Kind.VOLUME, "flat", Kind2.VOLUME, "flat"): ER.VOLUME,
            CaseConfig(Kind.VOLUME, "nested", Kind2.VOLUME, "nested"): ER.VOLUME,
            # . Operand 2 = complete.
            # Operand 1 = price pfline.
            # . Operand 2 = None.
            CaseConfig(Kind.PRICE, "flat", Kind2.NONE, "flat"): ER.PRICE,
            CaseConfig(Kind.PRICE, "nested", Kind2.NONE, "flat"): ER.PRICE,
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(Kind.PRICE, "flat", Kind2.PRICE, "flat"): ER.PRICE,
            CaseConfig(Kind.PRICE, "nested", Kind2.PRICE, "nested"): ER.PRICE,
            # . Operand 2 = complete.
            # Operand 1 = revenue pfline.
            # . Operand 2 = None.
            CaseConfig(Kind.REVENUE, "flat", Kind2.NONE, "flat"): ER.REVENUE,
            CaseConfig(Kind.REVENUE, "nested", Kind2.NONE, "flat"): ER.REVENUE,
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(Kind.REVENUE, "flat", Kind2.REVENUE, "flat"): ER.REVENUE,
            CaseConfig(Kind.REVENUE, "nested", Kind2.REVENUE, "nested"): ER.REVENUE,
            # . Operand 2 = complete.
            # Operand 1 = complete pfline.
            # . Operand 2 = None.
            CaseConfig(Kind.COMPLETE, "flat", Kind2.NONE, "flat"): ER.COMPLETE,
            CaseConfig(Kind.COMPLETE, "nested", Kind2.NONE, "flat"): ER.COMPLETE,
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            # . Operand 2 = complete.
            CaseConfig(Kind.COMPLETE, "flat", Kind2.COMPLETE, "flat"): ER.COMPLETE,
            CaseConfig(Kind.COMPLETE, "nested", Kind2.COMPLETE, "nested"): ER.COMPLETE,
        }
    ),
    ids=id_fn,
)
def test_pfl_arithmatic_kind_addraddsubrsub(testcase: Case, operation: str):
    """Test if arithmatic expectedly raises Error or returns expected type/kind."""
    do_kind_test(testcase, operation)


@pytest.mark.parametrize("operation", ["mul", "rmul"])
@pytest.mark.parametrize(
    "testcase",
    Cases.all_from_nonerror(
        {
            # Operand 1 = volume pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            CaseConfig(Kind.VOLUME, "flat", Kind2.NODIM, "flat"): ER.VOLUME,
            CaseConfig(Kind.VOLUME, "nested", Kind2.NODIM, "flat"): ER.VOLUME,
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(Kind.VOLUME, "flat", Kind2.PRICE, "flat"): ER.REVENUE,
            # . Operand 2 = complete.
            # Operand 1 = price pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            CaseConfig(Kind.PRICE, "flat", Kind2.NODIM, "flat"): ER.PRICE,
            CaseConfig(Kind.PRICE, "nested", Kind2.NODIM, "flat"): ER.PRICE,
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(Kind.PRICE, "flat", Kind2.VOLUME, "flat"): ER.REVENUE,
            # . Operand 2 = complete.
            # Operand 1 = revenue pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            CaseConfig(Kind.REVENUE, "flat", Kind2.NODIM, "flat"): ER.REVENUE,
            CaseConfig(Kind.REVENUE, "nested", Kind2.NODIM, "flat"): ER.REVENUE,
            # . Operand 2 = volume, price, or revenue.
            # . Operand 2 = complete.
            # Operand 1 = complete pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            CaseConfig(Kind.COMPLETE, "flat", Kind2.NODIM, "flat"): ER.COMPLETE,
            CaseConfig(Kind.COMPLETE, "nested", Kind2.NODIM, "flat"): ER.COMPLETE,
            # . Operand 2 = volume, price, or revenue.
            # . Operand 2 = complete.
        }
    ),
    ids=id_fn,
)
def test_pfl_arithmatic_kind_mulrmul(testcase: Case, operation: str):
    """Test if arithmatic expectedly raises Error or returns expected type/kind."""
    do_kind_test(testcase, operation)


@pytest.mark.parametrize("operation", ["div"])
@pytest.mark.parametrize(
    "testcase",
    Cases.all_from_nonerror(
        {
            # Operand 1 = volume pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            CaseConfig(Kind.VOLUME, "flat", Kind2.NODIM, "flat"): ER.VOLUME,
            CaseConfig(Kind.VOLUME, "nested", Kind2.NODIM, "flat"): ER.VOLUME,
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(Kind.VOLUME, "flat", Kind2.VOLUME, "flat"): ER.SERIES,
            # . Operand 2 = complete.
            # Operand 1 = price pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            CaseConfig(Kind.PRICE, "flat", Kind2.NODIM, "flat"): ER.PRICE,
            CaseConfig(Kind.PRICE, "nested", Kind2.NODIM, "flat"): ER.PRICE,
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(Kind.PRICE, "flat", Kind2.PRICE, "flat"): ER.SERIES,
            # . Operand 2 = complete.
            # Operand 1 = revenue pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            CaseConfig(Kind.REVENUE, "flat", Kind2.NODIM, "flat"): ER.REVENUE,
            CaseConfig(Kind.REVENUE, "nested", Kind2.NODIM, "flat"): ER.REVENUE,
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(Kind.REVENUE, "flat", Kind2.VOLUME, "flat"): ER.PRICE,
            CaseConfig(Kind.REVENUE, "flat", Kind2.PRICE, "flat"): ER.VOLUME,
            CaseConfig(Kind.REVENUE, "flat", Kind2.REVENUE, "flat"): ER.SERIES,
            # . Operand 2 = complete.
            # Operand 1 = complete pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            CaseConfig(Kind.COMPLETE, "flat", Kind2.NODIM, "flat"): ER.COMPLETE,
            CaseConfig(Kind.COMPLETE, "nested", Kind2.NODIM, "flat"): ER.COMPLETE,
            # . Operand 2 = volume, price, or revenue.
            # . Operand 2 = complete.
        }
    ),
    ids=id_fn,
)
def test_pfl_arithmatic_kind_div(testcase: Case, operation: str):
    """Test if arithmatic expectedly raises Error or returns expected type/kind."""
    do_kind_test(testcase, operation)


@pytest.mark.parametrize("operation", ["rdiv"])
@pytest.mark.parametrize(
    "testcase",
    Cases.all_from_nonerror(
        {
            # Operand 1 = volume pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(Kind.VOLUME, "flat", Kind2.VOLUME, "flat"): ER.SERIES,
            CaseConfig(Kind.VOLUME, "flat", Kind2.REVENUE, "flat"): ER.PRICE,
            # . Operand 2 = complete.
            # Operand 1 = price pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(Kind.PRICE, "flat", Kind2.PRICE, "flat"): ER.SERIES,
            CaseConfig(Kind.PRICE, "flat", Kind2.REVENUE, "flat"): ER.VOLUME,
            # . Operand 2 = complete.
            # Operand 1 = revenue pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(Kind.REVENUE, "flat", Kind2.REVENUE, "flat"): ER.SERIES,
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
def test_pfl_arithmatic_kind_rdiv(testcase: Case, operation: str):
    """Test if arithmatic expectedly raises Error or returns expected type/kind."""
    do_kind_test(testcase, operation)


@pytest.mark.parametrize("operation", ["union", "runion"])
@pytest.mark.parametrize(
    "testcase",
    Cases.all_from_nonerror(
        {
            # Operand 1 = volume pfline.
            # . Operand 2 = None.
            CaseConfig(Kind.VOLUME, "flat", Kind2.NONE, "flat"): ER.VOLUME,
            CaseConfig(Kind.VOLUME, "nested", Kind2.NONE, "flat"): ER.VOLUME,
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(Kind.VOLUME, "flat", Kind2.PRICE, "flat"): ER.COMPLETE,
            CaseConfig(Kind.VOLUME, "flat", Kind2.REVENUE, "flat"): ER.COMPLETE,
            # . Operand 2 = complete.
            # Operand 1 = price pfline.
            # . Operand 2 = None.
            CaseConfig(Kind.PRICE, "flat", Kind2.NONE, "flat"): ER.PRICE,
            CaseConfig(Kind.PRICE, "nested", Kind2.NONE, "flat"): ER.PRICE,
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(Kind.PRICE, "flat", Kind2.VOLUME, "flat"): ER.COMPLETE,
            CaseConfig(Kind.PRICE, "flat", Kind2.REVENUE, "flat"): ER.COMPLETE,
            # . Operand 2 = complete.
            # Operand 1 = revenue pfline.
            # . Operand 2 = None.
            CaseConfig(Kind.REVENUE, "flat", Kind2.NONE, "flat"): ER.REVENUE,
            CaseConfig(Kind.REVENUE, "nested", Kind2.NONE, "flat"): ER.REVENUE,
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(Kind.REVENUE, "flat", Kind2.VOLUME, "flat"): ER.COMPLETE,
            CaseConfig(Kind.REVENUE, "flat", Kind2.PRICE, "flat"): ER.COMPLETE,
            # . Operand 2 = complete.
            # Operand 1 = complete pfline.
            # . Operand 2 = None.
            CaseConfig(Kind.COMPLETE, "flat", Kind2.NONE, "flat"): ER.COMPLETE,
            CaseConfig(Kind.COMPLETE, "nested", Kind2.NONE, "flat"): ER.COMPLETE,
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            # . Operand 2 = complete.
        }
    ),
    ids=id_fn,
)
def test_pfl_arithmatic_kind_unionrunion(testcase: Case, operation: str):
    """Test if arithmatic expectedly raises Error or returns expected type/kind."""
    do_kind_test(testcase, operation)


def do_kind_test(testcase: Case, operation: str):
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
