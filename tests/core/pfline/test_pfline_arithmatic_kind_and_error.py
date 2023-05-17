"""Test if arithmatic returns correct type/kind of object and/or correctly raises error."""
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, Iterable

import pandas as pd
import pytest
from utils import id_fn  # relative to /tests

import portfolyo as pf
from portfolyo import Q_, PfLine
from portfolyo.core.pfline import Kind, Structure, arithmatic, create

# TODO: various timezones

# TODO: use/change STRICT setting


arithmatic.STRICT = True


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
class Value:  # Testvalue
    kind: Kind2
    structure: Structure
    value: Any

    def __repr__(self):
        return id_fn(self.value)


@dataclass
class Case:  # Testcase
    pfl: PfLine
    value: Any
    expected_result: ER

    def __repr__(self):
        return f"Case(pfl:{id_fn(self.pfl)},val:{id_fn(self.value)},expct:{id_fn(self.expected_result)})"


@dataclass(frozen=True, eq=True)
class CaseConfig:  # TestcaseConfig
    pfl_kind: Kind
    pfl_structure: Structure
    value_kind: Kind2
    value_structure: Structure


class Pfls:  # Testpfls
    def __init__(self, i: pd.DatetimeIndex):
        self._pfls = [
            pf.dev.get_pfline(i, kind, structure, _seed=1)
            for kind in Kind
            for structure in Structure
        ]

    def fetch(self, kind: Kind = None, structure: Structure = None) -> Iterable[PfLine]:
        pfls = self._pfls
        if kind is not None:
            pfls = [pfl for pfl in pfls if pfl.kind is kind]
        if structure is not None:
            pfls = [pfl for pfl in pfls if pfl.structure is structure]
        return pfls


class Values:  # Testvalues
    UNIT_ALT = {"w": "GW", "q": "kWh", "p": "ctEur/kWh", "r": "MEur", "nodim": ""}

    def __init__(self, i: pd.DatetimeIndex):
        self._values: Iterable[Value] = [
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
    def fetch(self, kind: Kind2 = None, structure: Structure = None) -> Iterable[Value]:
        testvalues = self._values
        if kind is not None:
            testvalues = [tv for tv in testvalues if tv.kind is kind]
        if structure is not None:
            testvalues = [tv for tv in testvalues if tv.structure is structure]
        return testvalues

    @staticmethod
    def from_noname(i: pd.DatetimeIndex) -> Iterable[Value]:
        flvalue = pf.dev.get_value("nodim", False, _seed=2)
        flseries = pf.dev.get_series(i, "", False, _seed=2)
        return [
            Value(Kind2.NONE, Structure.FLAT, None),
            Value(Kind2.NODIM, Structure.FLAT, flvalue),  # Single value.
            Value(Kind2.NODIM, Structure.FLAT, flseries),  # Timeseries.
        ]

    @staticmethod
    def from_1name(kind: Kind2, name: str, i: pd.DatetimeIndex) -> Iterable[Value]:
        """name == {'w', 'q', 'p', 'r', 'nodim'}"""
        quantity = pf.dev.get_value(name, True, _seed=2)
        altunit = Values.UNIT_ALT[name]
        quseries = pf.dev.get_series(i, name, _seed=2)
        values = [
            # . Single value.
            Value(kind, Structure.FLAT, quantity),
            Value(kind, Structure.FLAT, quantity.to(altunit)),
            Value(kind, Structure.FLAT, {name: quantity.m}),
            Value(kind, Structure.FLAT, {name: quantity}),
            Value(kind, Structure.FLAT, {name: quantity.to(altunit)}),
            Value(kind, Structure.FLAT, pd.Series({name: quantity.m})),
            Value(kind, Structure.FLAT, pd.Series({name: quantity})),
            # . Timeseries.
            Value(kind, Structure.FLAT, quseries),
            Value(kind, Structure.FLAT, {name: quseries.pint.m}),
            Value(kind, Structure.FLAT, {name: quseries}),
            Value(kind, Structure.FLAT, pd.DataFrame({name: quseries.pint.m})),
            Value(kind, Structure.FLAT, pd.DataFrame({name: quseries})),
        ]
        if name != "nodim":  # single 0.0-float not a problem when adding or subtracting
            values.append(Value(kind, Structure.FLAT, quantity * 0))
        return values

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
            Value(kind, Structure.FLAT, {n1: qu1, n2: qu2}),
            Value(kind, Structure.FLAT, {n1: fl1, n2: fl2}),
            Value(kind, Structure.FLAT, {n1: fl1, n2: qu2}),
            Value(kind, Structure.FLAT, pd.Series({n1: qu1, n2: qu2})),
            Value(kind, Structure.FLAT, pd.Series({n1: fl1, n2: fl2})),
            Value(kind, Structure.FLAT, pd.Series({n1: fl1, n2: qu2})),
            # . Single value | timeseries.
            Value(kind, Structure.FLAT, {n1: qu1, n2: qus2}),
            Value(kind, Structure.FLAT, {n1: fl1, n2: qus2}),
            Value(kind, Structure.FLAT, {n1: qu1, n2: fls2}),
            Value(kind, Structure.FLAT, {n1: fl1, n2: fls2}),
            # . Timeseries.
            Value(kind, Structure.FLAT, {n1: qus1, n2: qus2}),
            Value(kind, Structure.FLAT, {n1: qus1, n2: fls2}),
            Value(kind, Structure.FLAT, pd.DataFrame({n1: qus1, n2: qus2})),
            Value(kind, Structure.FLAT, pd.DataFrame({n1: qus1, n2: fls2})),
        ]

    @staticmethod
    def from_pfline(kind: Kind2, i: pd.DatetimeIndex) -> Iterable[Value]:
        k = kind.value
        pfl1, pfl2 = (pf.dev.get_flatpfline(i, k, _seed=s) for s in (2, 3))
        return [
            Value(kind, Structure.FLAT, pfl1),
            Value(
                kind,
                Structure.NESTED,
                create.nestedpfline({"childA": pfl1, "childB": pfl2}),
            ),
            Value(kind, Structure.NESTED, {"childC": pfl1, "childD": pfl2}),
        ]


tz = "Europe/Berlin"
i = pd.date_range("2020", periods=20, freq="MS", tz=tz)  # reference


class Cases:  # Testcases
    _testvalues = Values(i)
    _testpfls = Pfls(i)
    _complete_outcomedict = {
        CaseConfig(pfl_kind, pfl_structure, val_kind, val_structure): ER.ERROR
        for pfl_kind in Kind
        for pfl_structure in Structure
        for val_kind in Kind2
        for val_structure in Structure
    }

    @classmethod
    def get_pfls(
        cls, pfl_kind: Kind = None, pfl_structure: Structure = None
    ) -> Iterable[Case]:
        return cls._testpfls.fetch(pfl_kind, pfl_structure)

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
        for pfl in cls._testpfls.fetch(config.pfl_kind, config.pfl_structure):
            for tv in cls._testvalues.fetch(config.value_kind, config.value_structure):
                yield Case(pfl, tv.value, er)


@pytest.mark.parametrize("operation", ["add", "radd", "sub", "rsub"])
@pytest.mark.parametrize(
    "testcase",
    Cases.all_from_nonerror(
        {
            # Operand 1 = volume pfline.
            # . Operand 2 = None.
            CaseConfig(
                Kind.VOLUME, Structure.FLAT, Kind2.NONE, Structure.FLAT
            ): ER.VOLUME,
            CaseConfig(
                Kind.VOLUME, Structure.NESTED, Kind2.NONE, Structure.FLAT
            ): ER.VOLUME,
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(
                Kind.VOLUME, Structure.FLAT, Kind2.VOLUME, Structure.FLAT
            ): ER.VOLUME,
            CaseConfig(
                Kind.VOLUME, Structure.NESTED, Kind2.VOLUME, Structure.NESTED
            ): ER.VOLUME,
            # . Operand 2 = complete.
            # Operand 1 = price pfline.
            # . Operand 2 = None.
            CaseConfig(
                Kind.PRICE, Structure.FLAT, Kind2.NONE, Structure.FLAT
            ): ER.PRICE,
            CaseConfig(
                Kind.PRICE, Structure.NESTED, Kind2.NONE, Structure.FLAT
            ): ER.PRICE,
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(
                Kind.PRICE, Structure.FLAT, Kind2.PRICE, Structure.FLAT
            ): ER.PRICE,
            CaseConfig(
                Kind.PRICE, Structure.NESTED, Kind2.PRICE, Structure.NESTED
            ): ER.PRICE,
            # . Operand 2 = complete.
            # Operand 1 = revenue pfline.
            # . Operand 2 = None.
            CaseConfig(
                Kind.REVENUE, Structure.FLAT, Kind2.NONE, Structure.FLAT
            ): ER.REVENUE,
            CaseConfig(
                Kind.REVENUE, Structure.NESTED, Kind2.NONE, Structure.FLAT
            ): ER.REVENUE,
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(
                Kind.REVENUE, Structure.FLAT, Kind2.REVENUE, Structure.FLAT
            ): ER.REVENUE,
            CaseConfig(
                Kind.REVENUE, Structure.NESTED, Kind2.REVENUE, Structure.NESTED
            ): ER.REVENUE,
            # . Operand 2 = complete.
            # Operand 1 = complete pfline.
            # . Operand 2 = None.
            CaseConfig(
                Kind.COMPLETE, Structure.FLAT, Kind2.NONE, Structure.FLAT
            ): ER.COMPLETE,
            CaseConfig(
                Kind.COMPLETE, Structure.NESTED, Kind2.NONE, Structure.FLAT
            ): ER.COMPLETE,
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            # . Operand 2 = complete.
            CaseConfig(
                Kind.COMPLETE, Structure.FLAT, Kind2.COMPLETE, Structure.FLAT
            ): ER.COMPLETE,
            CaseConfig(
                Kind.COMPLETE, Structure.NESTED, Kind2.COMPLETE, Structure.NESTED
            ): ER.COMPLETE,
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
            CaseConfig(
                Kind.VOLUME, Structure.FLAT, Kind2.NODIM, Structure.FLAT
            ): ER.VOLUME,
            CaseConfig(
                Kind.VOLUME, Structure.NESTED, Kind2.NODIM, Structure.FLAT
            ): ER.VOLUME,
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(
                Kind.VOLUME, Structure.FLAT, Kind2.PRICE, Structure.FLAT
            ): ER.REVENUE,
            # . Operand 2 = complete.
            # Operand 1 = price pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            CaseConfig(
                Kind.PRICE, Structure.FLAT, Kind2.NODIM, Structure.FLAT
            ): ER.PRICE,
            CaseConfig(
                Kind.PRICE, Structure.NESTED, Kind2.NODIM, Structure.FLAT
            ): ER.PRICE,
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(
                Kind.PRICE, Structure.FLAT, Kind2.VOLUME, Structure.FLAT
            ): ER.REVENUE,
            # . Operand 2 = complete.
            # Operand 1 = revenue pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            CaseConfig(
                Kind.REVENUE, Structure.FLAT, Kind2.NODIM, Structure.FLAT
            ): ER.REVENUE,
            CaseConfig(
                Kind.REVENUE, Structure.NESTED, Kind2.NODIM, Structure.FLAT
            ): ER.REVENUE,
            # . Operand 2 = volume, price, or revenue.
            # . Operand 2 = complete.
            # Operand 1 = complete pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            CaseConfig(
                Kind.COMPLETE, Structure.FLAT, Kind2.NODIM, Structure.FLAT
            ): ER.COMPLETE,
            CaseConfig(
                Kind.COMPLETE, Structure.NESTED, Kind2.NODIM, Structure.FLAT
            ): ER.COMPLETE,
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
            CaseConfig(
                Kind.VOLUME, Structure.FLAT, Kind2.NODIM, Structure.FLAT
            ): ER.VOLUME,
            CaseConfig(
                Kind.VOLUME, Structure.NESTED, Kind2.NODIM, Structure.FLAT
            ): ER.VOLUME,
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(
                Kind.VOLUME, Structure.FLAT, Kind2.VOLUME, Structure.FLAT
            ): ER.SERIES,
            # . Operand 2 = complete.
            # Operand 1 = price pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            CaseConfig(
                Kind.PRICE, Structure.FLAT, Kind2.NODIM, Structure.FLAT
            ): ER.PRICE,
            CaseConfig(
                Kind.PRICE, Structure.NESTED, Kind2.NODIM, Structure.FLAT
            ): ER.PRICE,
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(
                Kind.PRICE, Structure.FLAT, Kind2.PRICE, Structure.FLAT
            ): ER.SERIES,
            # . Operand 2 = complete.
            # Operand 1 = revenue pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            CaseConfig(
                Kind.REVENUE, Structure.FLAT, Kind2.NODIM, Structure.FLAT
            ): ER.REVENUE,
            CaseConfig(
                Kind.REVENUE, Structure.NESTED, Kind2.NODIM, Structure.FLAT
            ): ER.REVENUE,
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(
                Kind.REVENUE, Structure.FLAT, Kind2.VOLUME, Structure.FLAT
            ): ER.PRICE,
            CaseConfig(
                Kind.REVENUE, Structure.FLAT, Kind2.PRICE, Structure.FLAT
            ): ER.VOLUME,
            CaseConfig(
                Kind.REVENUE, Structure.FLAT, Kind2.REVENUE, Structure.FLAT
            ): ER.SERIES,
            # . Operand 2 = complete.
            # Operand 1 = complete pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            CaseConfig(
                Kind.COMPLETE, Structure.FLAT, Kind2.NODIM, Structure.FLAT
            ): ER.COMPLETE,
            CaseConfig(
                Kind.COMPLETE, Structure.NESTED, Kind2.NODIM, Structure.FLAT
            ): ER.COMPLETE,
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
            CaseConfig(
                Kind.VOLUME, Structure.FLAT, Kind2.VOLUME, Structure.FLAT
            ): ER.SERIES,
            CaseConfig(
                Kind.VOLUME, Structure.FLAT, Kind2.REVENUE, Structure.FLAT
            ): ER.PRICE,
            # . Operand 2 = complete.
            # Operand 1 = price pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(
                Kind.PRICE, Structure.FLAT, Kind2.PRICE, Structure.FLAT
            ): ER.SERIES,
            CaseConfig(
                Kind.PRICE, Structure.FLAT, Kind2.REVENUE, Structure.FLAT
            ): ER.VOLUME,
            # . Operand 2 = complete.
            # Operand 1 = revenue pfline.
            # . Operand 2 = None.
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(
                Kind.REVENUE, Structure.FLAT, Kind2.REVENUE, Structure.FLAT
            ): ER.SERIES,
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
            CaseConfig(
                Kind.VOLUME, Structure.FLAT, Kind2.NONE, Structure.FLAT
            ): ER.VOLUME,
            CaseConfig(
                Kind.VOLUME, Structure.NESTED, Kind2.NONE, Structure.FLAT
            ): ER.VOLUME,
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(
                Kind.VOLUME, Structure.FLAT, Kind2.PRICE, Structure.FLAT
            ): ER.COMPLETE,
            CaseConfig(
                Kind.VOLUME, Structure.FLAT, Kind2.REVENUE, Structure.FLAT
            ): ER.COMPLETE,
            # . Operand 2 = complete.
            # Operand 1 = price pfline.
            # . Operand 2 = None.
            CaseConfig(
                Kind.PRICE, Structure.FLAT, Kind2.NONE, Structure.FLAT
            ): ER.PRICE,
            CaseConfig(
                Kind.PRICE, Structure.NESTED, Kind2.NONE, Structure.FLAT
            ): ER.PRICE,
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(
                Kind.PRICE, Structure.FLAT, Kind2.VOLUME, Structure.FLAT
            ): ER.COMPLETE,
            CaseConfig(
                Kind.PRICE, Structure.FLAT, Kind2.REVENUE, Structure.FLAT
            ): ER.COMPLETE,
            # . Operand 2 = complete.
            # Operand 1 = revenue pfline.
            # . Operand 2 = None.
            CaseConfig(
                Kind.REVENUE, Structure.FLAT, Kind2.NONE, Structure.FLAT
            ): ER.REVENUE,
            CaseConfig(
                Kind.REVENUE, Structure.NESTED, Kind2.NONE, Structure.FLAT
            ): ER.REVENUE,
            # . Operand 2 = dimensionless.
            # . Operand 2 = volume, price, or revenue.
            CaseConfig(
                Kind.REVENUE, Structure.FLAT, Kind2.VOLUME, Structure.FLAT
            ): ER.COMPLETE,
            CaseConfig(
                Kind.REVENUE, Structure.FLAT, Kind2.PRICE, Structure.FLAT
            ): ER.COMPLETE,
            # . Operand 2 = complete.
            # Operand 1 = complete pfline.
            # . Operand 2 = None.
            CaseConfig(
                Kind.COMPLETE, Structure.FLAT, Kind2.NONE, Structure.FLAT
            ): ER.COMPLETE,
            CaseConfig(
                Kind.COMPLETE, Structure.NESTED, Kind2.NONE, Structure.FLAT
            ): ER.COMPLETE,
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
