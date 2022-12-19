import enum
from dataclasses import dataclass
from enum import auto
from functools import lru_cache
from typing import Any, Dict, Iterable, Union

import pandas as pd
import pytest

import portfolyo as pf
from portfolyo import Q_, FlatPfLine, Kind, NestedPfLine, PfLine, dev, testing  # noqa

# TODO: Multipfline
# TODO: various timezones


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
        return f"Flatpfline_{data.kind}"
    elif isinstance(data, NestedPfLine):
        return f"Nestedpfline_{data.kind}"
    elif isinstance(data, str):
        return data
    elif isinstance(data, pf.Q_):
        return f"Quantity ({data.units})"
    elif isinstance(data, type):
        return data.__name__
    elif isinstance(data, Kind):
        return str(data)
    elif isinstance(data, Testcase):
        return f"in:{data.kind_in},val:{id_fn(data.value)},exp:{data.expected_result}"
    return type(data).__name__


class ER(enum.Flag):  # Expected Result
    # Allowed pfline type as input...
    FLAT_ONLY = auto()
    NESTED_ONLY = auto()
    ERROR = auto()  # none allowed
    # Output type...
    VOLUME = auto()
    PRICE = auto()
    REVENUE = auto()
    COMPLETE = auto()
    SERIES = auto()

    def typ(self) -> type:
        if self in [ER.VOLUME, ER.PRICE, ER.REVENUE, ER.COMPLETE]:
            return PfLine
        if self is ER.SERIES:
            return pd.Series

    def kind(self) -> Kind:
        if self is ER.VOLUME:
            return Kind.VOLUME
        if self is ER.PRICE:
            return Kind.PRICE
        if self is ER.REVENUE:
            return Kind.REVENUE
        if self is ER.COMPLETE:
            return Kind.COMPLETE
        return None

    def __str__(self) -> str:
        return "_and_".join(er.name for er in ER if er in self)


UNIT_ALT = {"w": "GW", "q": "kWh", "p": "ctEur/kWh", "r": "MEur", "nodim": ""}

tz = "Europe/Berlin"
i = pd.date_range("2020", periods=20, freq="MS", tz=tz)  # reference
PFL = {
    "flat": {kind: pf.dev.get_flatpfline(i, kind, _seed=1) for kind in Kind},
    "nested": {kind: pf.dev.get_nestedpfline(i, kind, _seed=1) for kind in Kind},
}


@dataclass
class TestValue:
    kind: Union[Kind, str]  # 'nodim' or Kind
    is_nested: bool
    title: str
    value: Any

    def __str__(self):
        return f"{self.kind}_{'nested' if self.is_nested else 'flat'}_{self.title}_{id_fn(self.value)}"


def testvalues_from_noname(
    kind: Union[Kind, str], i: pd.DatetimeIndex
) -> Iterable[TestValue]:
    return [
        # . Single value.
        TestValue(kind, False, "fl", pf.dev.get_value("nodim", False, _seed=2)),
        # . Timeseries.
        TestValue(kind, False, "flseries", pf.dev.get_series(i, "", False, _seed=2)),
    ]


def testvalues_from_1name(
    kind: Union[Kind, str], name: str, i: pd.DatetimeIndex
) -> Iterable[TestValue]:
    """name == {'w', 'q', 'p', 'r', 'nodim'}"""
    quantity = pf.dev.get_value(name, True, _seed=2)
    altunit = UNIT_ALT[name]
    quseries = pf.dev.get_series(i, name, _seed=2)
    return [
        # . Single value.
        TestValue(kind, False, "qu", quantity),
        TestValue(kind, False, "qu_0", quantity * 0),
        TestValue(kind, False, "qu_alt", quantity.to(altunit)),
        TestValue(kind, False, "dict_fl", {name: quantity.m}),
        TestValue(kind, False, "dict_qu", {name: quantity}),
        TestValue(kind, False, "dict_qu_alt", {name: quantity.to(altunit)}),
        TestValue(kind, False, "nottseries_fl", pd.Series({name: quantity.m})),
        TestValue(kind, False, "nottseries_qu", pd.Series({name: quantity})),
        # . Timeseries.
        TestValue(kind, False, "quseries", quseries),
        TestValue(kind, False, "dict_flseries", {name: quseries.pint.m}),
        TestValue(kind, False, "dict_quseries", {name: quseries}),
        TestValue(kind, False, "df_fl", pd.DataFrame({name: quseries.pint.m})),
        TestValue(kind, False, "df_qu", pd.DataFrame({name: quseries})),
    ]


def testvalues_from_2names(
    kind: Union[Kind, str], names: Iterable[str], i: pd.DatetimeIndex
) -> Iterable[TestValue]:
    """names == two of {'w', 'q', 'p', 'r'}"""
    n1, n2 = names
    qu1, qu2 = (pf.dev.get_value(name, True, _seed=2) for name in names)
    fl1, fl2 = (qu.magnitude for qu in (qu1, qu2))
    qus1, qus2 = (pf.dev.get_series(i, name, _seed=2) for name in names)
    fls1, fls2 = (quseries.pint.m for quseries in (qus1, qus2))
    return [
        # . Single values.
        TestValue(kind, False, "dict_qu", {n1: qu1, n2: qu2}),
        TestValue(kind, False, "dict_fl", {n1: fl1, n2: fl2}),
        TestValue(kind, False, "dict_fl_and_qu", {n1: fl1, n2: qu2}),
        TestValue(kind, False, "nottseries_qu", pd.Series({n1: qu1, n2: qu2})),
        TestValue(kind, False, "nottseries_fl", pd.Series({n1: fl1, n2: fl2})),
        TestValue(kind, False, "nottseries_fl_and_qu", pd.Series({n1: fl1, n2: qu2})),
        # . Single value | timeseries.
        TestValue(kind, False, "dict_qu_and_quseries", {n1: qu1, n2: qus2}),
        TestValue(kind, False, "dict_fl_and_quseries", {n1: fl1, n2: qus2}),
        TestValue(kind, False, "dict_qu_and_flseries", {n1: qu1, n2: fls2}),
        TestValue(kind, False, "dict_fl_and_flseries", {n1: fl1, n2: fls2}),
        # . Timeseries.
        TestValue(kind, False, "dict_quseries", {n1: qus1, n2: qus2}),
        TestValue(kind, False, "dict_quseries_and_flseries", {n1: qus1, n2: fls2}),
        TestValue(kind, False, "df_quseries", pd.DataFrame({n1: qus1, n2: qus2})),
        TestValue(
            kind, False, "df_quseries_and_flseries", pd.DataFrame({n1: qus1, n2: fls2})
        ),
    ]


def testvalues_withpfline(kind: Kind, i: pd.DatetimeIndex) -> Iterable[TestValue]:
    pfl1, pfl2 = (pf.dev.get_flatpfline(i, kind, _seed=s) for s in (2, 3))
    return [
        TestValue(kind, False, "flat_pfline", pfl1),
        TestValue(
            kind, True, "nested_pfline", pf.PfLine({"childA": pfl1, "childB": pfl2})
        ),
        TestValue(kind, True, "dict_pfl", {"childC": pfl1, "childD": pfl2}),
    ]


testvalues: Iterable[TestValue] = []
testvalues.extend(testvalues_from_noname("nodim", i))
testvalues.extend(testvalues_from_1name("nodim", "nodim", i))
testvalues.extend(testvalues_from_1name(Kind.VOLUME, "w", i))
testvalues.extend(testvalues_from_1name(Kind.VOLUME, "q", i))
testvalues.extend(testvalues_withpfline(Kind.VOLUME, i))
testvalues.extend(testvalues_from_1name(Kind.PRICE, "p", i))
testvalues.extend(testvalues_withpfline(Kind.PRICE, i))
testvalues.extend(testvalues_from_1name(Kind.REVENUE, "r", i))
testvalues.extend(testvalues_withpfline(Kind.REVENUE, i))
testvalues.extend(testvalues_from_2names(Kind.COMPLETE, "wp", i))
testvalues.extend(testvalues_from_2names(Kind.COMPLETE, "qr", i))
testvalues.extend(testvalues_from_2names(Kind.COMPLETE, "pr", i))
testvalues.extend(testvalues_withpfline(Kind.COMPLETE, i))


@lru_cache()
def filtertestvalues(kind: Kind = None, is_nested: bool = None) -> Iterable[TestValue]:
    values = testvalues
    if kind is not None:
        values = [tc for tc in values if tc.kind == kind]
    if is_nested is not None:
        values = [tc for tc in values if tc.is_nested == is_nested]
    return values


@dataclass
class Testcase:
    kind_in: Kind
    expected_result: ER
    value: Any


def create_testcases(
    kind_in: Kind,
    expected_result: ER,
    value_kind: Kind = None,
    value_flatornested: str = None,
) -> Iterable[Testcase]:
    is_nested = {None: None, "flat": False, "nested": True}[value_flatornested]
    return [
        Testcase(kind_in, expected_result, tv.value)
        for tv in filtertestvalues(value_kind, is_nested)
    ]


@pytest.mark.parametrize("operation", ["add", "radd", "sub", "rsub"])
@pytest.mark.parametrize(
    "testcase",
    [
        # Operand 1 = volume pfline.
        # . Operand 2 = None.
        Testcase(Kind.VOLUME, ER.VOLUME, None),
        # . Operand 2 = dimensionless.
        *create_testcases(Kind.VOLUME, ER.ERROR, "nodim"),
        # . Operand 2 = volume, price, or revenue.
        *create_testcases(Kind.VOLUME, ER.VOLUME | ER.FLAT_ONLY, Kind.VOLUME, "flat"),
        *create_testcases(
            Kind.VOLUME, ER.VOLUME | ER.NESTED_ONLY, Kind.VOLUME, "nested"
        ),
        *create_testcases(Kind.VOLUME, ER.ERROR, Kind.PRICE),
        *create_testcases(Kind.VOLUME, ER.ERROR, Kind.REVENUE),
        # . Operand 2 = complete.
        *create_testcases(Kind.VOLUME, ER.ERROR, Kind.COMPLETE),
        # Operand 1 = price pfline.
        # . Operand 2 = None.
        Testcase(Kind.PRICE, ER.PRICE, None),
        # . Operand 2 = dimensionless.
        *create_testcases(Kind.PRICE, ER.ERROR, "nodim"),
        # . Operand 2 = volume, price, or revenue.
        *create_testcases(Kind.PRICE, ER.ERROR, Kind.VOLUME),
        *create_testcases(Kind.PRICE, ER.PRICE | ER.FLAT_ONLY, Kind.PRICE, "flat"),
        *create_testcases(Kind.PRICE, ER.PRICE | ER.NESTED_ONLY, Kind.PRICE, "nested"),
        *create_testcases(Kind.PRICE, ER.ERROR, Kind.REVENUE),
        # . Operand 2 = complete.
        *create_testcases(Kind.PRICE, ER.ERROR, Kind.COMPLETE),
        # Operand 1 = revenue pfline.
        # . Operand 2 = None.
        Testcase(Kind.REVENUE, ER.REVENUE, None),
        # . Operand 2 = dimensionless.
        *create_testcases(Kind.REVENUE, ER.ERROR, "nodim"),
        # . Operand 2 = volume, price, or revenue.
        *create_testcases(Kind.REVENUE, ER.ERROR, Kind.VOLUME),
        *create_testcases(Kind.REVENUE, ER.ERROR, Kind.PRICE),
        *create_testcases(
            Kind.REVENUE, ER.REVENUE | ER.FLAT_ONLY, Kind.REVENUE, "flat"
        ),
        *create_testcases(
            Kind.REVENUE, ER.REVENUE | ER.NESTED_ONLY, Kind.REVENUE, "nested"
        ),
        # . Operand 2 = complete.
        *create_testcases(Kind.REVENUE, ER.ERROR, Kind.COMPLETE),
        # Operand 1 = complete pfline.
        # . Operand 2 = None.
        Testcase(Kind.COMPLETE, ER.COMPLETE, None),
        # . Operand 2 = dimensionless.
        *create_testcases(Kind.COMPLETE, ER.ERROR, "nodim"),
        # . Operand 2 = volume, price, or revenue.
        *create_testcases(Kind.COMPLETE, ER.ERROR, Kind.VOLUME),
        *create_testcases(Kind.COMPLETE, ER.ERROR, Kind.PRICE),
        *create_testcases(Kind.COMPLETE, ER.ERROR, Kind.REVENUE),
        # . Operand 2 = complete.
        *create_testcases(
            Kind.COMPLETE, ER.COMPLETE | ER.FLAT_ONLY, Kind.COMPLETE, "flat"
        ),
        *create_testcases(
            Kind.COMPLETE, ER.COMPLETE | ER.NESTED_ONLY, Kind.COMPLETE, "nested"
        ),
    ],
    ids=id_fn,
)
@pytest.mark.parametrize("pfl_in_flat_or_nested", ["flat", "nested"])
def test_pfl_add_kind(testcase: Testcase, pfl_in_flat_or_nested: str, operation: str):
    """Test if arithmatic expectedly raises Error or returns expected type/kind."""
    do_kind_test(testcase, pfl_in_flat_or_nested, operation)


@pytest.mark.parametrize("operation", ["mul", "rmul"])
@pytest.mark.parametrize(
    "testcase",
    [
        # Operand 1 = volume pfline.
        # . Operand 2 = None.
        Testcase(Kind.VOLUME, ER.ERROR, None),
        # . Operand 2 = dimensionless.
        *create_testcases(Kind.VOLUME, ER.VOLUME, "nodim"),
        # . Operand 2 = volume, price, or revenue.
        *create_testcases(Kind.VOLUME, ER.ERROR, Kind.VOLUME),
        *create_testcases(Kind.VOLUME, ER.REVENUE | ER.FLAT_ONLY, Kind.PRICE, "flat"),
        *create_testcases(Kind.VOLUME, ER.ERROR, Kind.PRICE, "nested"),
        *create_testcases(Kind.VOLUME, ER.ERROR, Kind.REVENUE),
        # . Operand 2 = complete.
        *create_testcases(Kind.VOLUME, ER.ERROR, Kind.COMPLETE),
        # Operand 1 = price pfline.
        # . Operand 2 = None.
        Testcase(Kind.PRICE, ER.ERROR, None),
        # . Operand 2 = dimensionless.
        *create_testcases(Kind.PRICE, ER.PRICE, "nodim"),
        # . Operand 2 = volume, price, or revenue.
        *create_testcases(Kind.PRICE, ER.REVENUE | ER.FLAT_ONLY, Kind.VOLUME, "flat"),
        *create_testcases(Kind.PRICE, ER.ERROR, Kind.VOLUME, "nested"),
        *create_testcases(Kind.PRICE, ER.ERROR, Kind.PRICE),
        *create_testcases(Kind.PRICE, ER.ERROR, Kind.REVENUE),
        # . Operand 2 = complete.
        *create_testcases(Kind.PRICE, ER.ERROR, Kind.COMPLETE),
        # Operand 1 = revenue pfline.
        # . Operand 2 = None.
        Testcase(Kind.REVENUE, ER.ERROR, None),
        # . Operand 2 = dimensionless.
        *create_testcases(Kind.REVENUE, ER.REVENUE, "nodim"),
        # . Operand 2 = volume, price, or revenue.
        *create_testcases(Kind.REVENUE, ER.ERROR, Kind.VOLUME),
        *create_testcases(Kind.REVENUE, ER.ERROR, Kind.PRICE),
        *create_testcases(Kind.REVENUE, ER.ERROR, Kind.REVENUE),
        # . Operand 2 = complete.
        *create_testcases(Kind.REVENUE, ER.ERROR, Kind.COMPLETE),
        # Operand 1 = complete pfline.
        # . Operand 2 = None.
        Testcase(Kind.COMPLETE, ER.ERROR, None),
        # . Operand 2 = dimensionless.
        *create_testcases(Kind.COMPLETE, ER.COMPLETE, "nodim"),
        # . Operand 2 = volume, price, or revenue.
        *create_testcases(Kind.COMPLETE, ER.ERROR, Kind.VOLUME),
        *create_testcases(Kind.COMPLETE, ER.ERROR, Kind.PRICE),
        *create_testcases(Kind.COMPLETE, ER.ERROR, Kind.REVENUE),
        # . Operand 2 = complete.
        *create_testcases(Kind.COMPLETE, ER.ERROR, Kind.COMPLETE),
    ],
    ids=id_fn,
)
@pytest.mark.parametrize("pfl_in_flat_or_nested", ["flat", "nested"])
def test_pfl_mul_kind(testcase: Testcase, pfl_in_flat_or_nested: str, operation: str):
    """Test if arithmatic expectedly raises Error or returns expected type/kind."""
    do_kind_test(testcase, pfl_in_flat_or_nested, operation)


@pytest.mark.parametrize("operation", ["div"])
@pytest.mark.parametrize(
    "testcase",
    [
        # Operand 1 = volume pfline.
        # . Operand 2 = None.
        Testcase(Kind.VOLUME, ER.ERROR, None),
        # . Operand 2 = dimensionless.
        *create_testcases(Kind.VOLUME, ER.VOLUME, "nodim"),
        # . Operand 2 = volume, price, or revenue.
        *create_testcases(Kind.VOLUME, ER.SERIES | ER.FLAT_ONLY, Kind.VOLUME, "flat"),
        *create_testcases(Kind.VOLUME, ER.ERROR, Kind.VOLUME, "nested"),
        *create_testcases(Kind.VOLUME, ER.ERROR, Kind.PRICE),
        *create_testcases(Kind.VOLUME, ER.ERROR, Kind.REVENUE),
        # . Operand 2 = complete.
        *create_testcases(Kind.VOLUME, ER.ERROR, Kind.COMPLETE),
        # Operand 1 = price pfline.
        # . Operand 2 = None.
        Testcase(Kind.PRICE, ER.ERROR, None),
        # . Operand 2 = dimensionless.
        *create_testcases(Kind.PRICE, ER.PRICE, "nodim"),
        # . Operand 2 = volume, price, or revenue.
        *create_testcases(Kind.PRICE, ER.ERROR, Kind.VOLUME),
        *create_testcases(Kind.PRICE, ER.SERIES | ER.FLAT_ONLY, Kind.PRICE, "flat"),
        *create_testcases(Kind.PRICE, ER.ERROR, Kind.PRICE, "nested"),
        *create_testcases(Kind.PRICE, ER.ERROR, Kind.REVENUE),
        # . Operand 2 = complete.
        *create_testcases(Kind.PRICE, ER.ERROR, Kind.COMPLETE),
        # Operand 1 = revenue pfline.
        # . Operand 2 = None.
        Testcase(Kind.REVENUE, ER.ERROR, None),
        # . Operand 2 = dimensionless.
        *create_testcases(Kind.REVENUE, ER.REVENUE, "nodim"),
        # . Operand 2 = volume, price, or revenue.
        *create_testcases(Kind.REVENUE, ER.PRICE | ER.FLAT_ONLY, Kind.VOLUME, "flat"),
        *create_testcases(Kind.REVENUE, ER.ERROR, Kind.VOLUME, "nested"),
        *create_testcases(Kind.REVENUE, ER.VOLUME | ER.FLAT_ONLY, Kind.PRICE, "flat"),
        *create_testcases(Kind.REVENUE, ER.ERROR, Kind.PRICE, "nested"),
        *create_testcases(Kind.REVENUE, ER.SERIES | ER.FLAT_ONLY, Kind.REVENUE, "flat"),
        *create_testcases(Kind.REVENUE, ER.ERROR, Kind.REVENUE, "nested"),
        # . Operand 2 = complete.
        *create_testcases(Kind.REVENUE, ER.ERROR, Kind.COMPLETE),
        # Operand 1 = complete pfline.
        # . Operand 2 = None.
        Testcase(Kind.COMPLETE, ER.ERROR, None),
        # . Operand 2 = dimensionless.
        *create_testcases(Kind.COMPLETE, ER.COMPLETE, "nodim"),
        # . Operand 2 = volume, price, or revenue.
        *create_testcases(Kind.COMPLETE, ER.ERROR, Kind.VOLUME),
        *create_testcases(Kind.COMPLETE, ER.ERROR, Kind.PRICE),
        *create_testcases(Kind.COMPLETE, ER.ERROR, Kind.REVENUE),
        # . Operand 2 = complete.
        *create_testcases(Kind.COMPLETE, ER.ERROR, Kind.COMPLETE),
    ],
    ids=id_fn,
)
@pytest.mark.parametrize("pfl_in_flat_or_nested", ["flat", "nested"])
def test_pfl_div_kind(testcase: Testcase, pfl_in_flat_or_nested: str, operation: str):
    """Test if arithmatic expectedly raises Error or returns expected type/kind."""
    do_kind_test(testcase, pfl_in_flat_or_nested, operation)


@pytest.mark.parametrize("operation", ["rdiv"])
@pytest.mark.parametrize(
    "testcase",
    [
        # Operand 1 = volume pfline.
        # . Operand 2 = None.
        Testcase(Kind.VOLUME, ER.ERROR, None),
        # . Operand 2 = dimensionless.
        *create_testcases(Kind.VOLUME, ER.ERROR, "nodim"),
        # . Operand 2 = volume, price, or revenue.
        *create_testcases(Kind.VOLUME, ER.SERIES | ER.FLAT_ONLY, Kind.VOLUME, "flat"),
        *create_testcases(Kind.VOLUME, ER.ERROR, Kind.VOLUME, "nested"),
        *create_testcases(Kind.VOLUME, ER.ERROR, Kind.PRICE),
        *create_testcases(Kind.VOLUME, ER.PRICE | ER.FLAT_ONLY, Kind.REVENUE, "flat"),
        *create_testcases(Kind.VOLUME, ER.ERROR, Kind.REVENUE, "nested"),
        # . Operand 2 = complete.
        *create_testcases(Kind.VOLUME, ER.ERROR, Kind.COMPLETE),
        # Operand 1 = price pfline.
        # . Operand 2 = None.
        Testcase(Kind.PRICE, ER.ERROR, None),
        # . Operand 2 = dimensionless.
        *create_testcases(Kind.PRICE, ER.ERROR, "nodim"),
        # . Operand 2 = volume, price, or revenue.
        *create_testcases(Kind.PRICE, ER.ERROR, Kind.VOLUME),
        *create_testcases(Kind.PRICE, ER.SERIES | ER.FLAT_ONLY, Kind.PRICE, "flat"),
        *create_testcases(Kind.PRICE, ER.ERROR, Kind.PRICE, "nested"),
        *create_testcases(Kind.PRICE, ER.VOLUME | ER.FLAT_ONLY, Kind.REVENUE, "flat"),
        *create_testcases(Kind.PRICE, ER.ERROR, Kind.REVENUE, "nested"),
        # . Operand 2 = complete.
        *create_testcases(Kind.PRICE, ER.ERROR, Kind.COMPLETE),
        # Operand 1 = revenue pfline.
        # . Operand 2 = None.
        Testcase(Kind.REVENUE, ER.ERROR, None),
        # . Operand 2 = dimensionless.
        *create_testcases(Kind.REVENUE, ER.ERROR, "nodim"),
        # . Operand 2 = volume, price, or revenue.
        *create_testcases(Kind.REVENUE, ER.ERROR, Kind.VOLUME),
        *create_testcases(Kind.REVENUE, ER.ERROR, Kind.PRICE),
        *create_testcases(Kind.REVENUE, ER.SERIES | ER.FLAT_ONLY, Kind.REVENUE, "flat"),
        *create_testcases(Kind.REVENUE, ER.ERROR, Kind.REVENUE, "nested"),
        # . Operand 2 = complete.
        *create_testcases(Kind.REVENUE, ER.ERROR, Kind.COMPLETE),
        # Operand 1 = complete pfline.
        # . Operand 2 = None.
        Testcase(Kind.COMPLETE, ER.ERROR, None),
        # . Operand 2 = dimensionless.
        *create_testcases(Kind.COMPLETE, ER.ERROR, "nodim"),
        # . Operand 2 = volume, price, or revenue.
        *create_testcases(Kind.COMPLETE, ER.ERROR, Kind.VOLUME),
        *create_testcases(Kind.COMPLETE, ER.ERROR, Kind.PRICE),
        *create_testcases(Kind.COMPLETE, ER.ERROR, Kind.REVENUE),
        # . Operand 2 = complete.
        *create_testcases(Kind.COMPLETE, ER.ERROR, Kind.COMPLETE),
    ],
    ids=id_fn,
)
@pytest.mark.parametrize("pfl_in_flat_or_nested", ["flat", "nested"])
def test_pfl_rdiv_kind(testcase: Testcase, pfl_in_flat_or_nested: str, operation: str):
    """Test if arithmatic expectedly raises Error or returns expected type/kind."""
    do_kind_test(testcase, pfl_in_flat_or_nested, operation)


@pytest.mark.parametrize("operation", ["union", "runion"])
@pytest.mark.parametrize(
    "testcase",
    [
        # Operand 1 = volume pfline.
        # . Operand 2 = None.
        Testcase(Kind.VOLUME, ER.VOLUME, None),
        # . Operand 2 = dimensionless.
        *create_testcases(Kind.VOLUME, ER.ERROR, "nodim"),
        # . Operand 2 = volume, price, or revenue.
        *create_testcases(Kind.VOLUME, ER.ERROR, Kind.VOLUME),
        *create_testcases(Kind.VOLUME, ER.COMPLETE | ER.FLAT_ONLY, Kind.PRICE, "flat"),
        *create_testcases(Kind.VOLUME, ER.ERROR, Kind.PRICE, "nested"),
        *create_testcases(
            Kind.VOLUME, ER.COMPLETE | ER.FLAT_ONLY, Kind.REVENUE, "flat"
        ),
        *create_testcases(Kind.VOLUME, ER.ERROR, Kind.REVENUE, "nested"),
        # . Operand 2 = complete.
        *create_testcases(Kind.VOLUME, ER.ERROR, Kind.COMPLETE),
        # Operand 1 = price pfline.
        # . Operand 2 = None.
        Testcase(Kind.PRICE, ER.PRICE, None),
        # . Operand 2 = dimensionless.
        *create_testcases(Kind.PRICE, ER.ERROR, "nodim"),
        # . Operand 2 = volume, price, or revenue.
        *create_testcases(Kind.PRICE, ER.COMPLETE | ER.FLAT_ONLY, Kind.VOLUME, "flat"),
        *create_testcases(Kind.PRICE, ER.ERROR, Kind.VOLUME, "nested"),
        *create_testcases(Kind.PRICE, ER.ERROR, Kind.PRICE),
        *create_testcases(Kind.PRICE, ER.COMPLETE | ER.FLAT_ONLY, Kind.REVENUE, "flat"),
        *create_testcases(Kind.PRICE, ER.ERROR, Kind.REVENUE, "nested"),
        # . Operand 2 = complete.
        *create_testcases(Kind.PRICE, ER.ERROR, Kind.COMPLETE),
        # Operand 1 = revenue pfline.
        # . Operand 2 = None.
        Testcase(Kind.REVENUE, ER.REVENUE, None),
        # . Operand 2 = dimensionless.
        *create_testcases(Kind.REVENUE, ER.ERROR, "nodim"),
        # . Operand 2 = volume, price, or revenue.
        *create_testcases(
            Kind.REVENUE, ER.COMPLETE | ER.FLAT_ONLY, Kind.VOLUME, "flat"
        ),
        *create_testcases(Kind.REVENUE, ER.ERROR, Kind.VOLUME, "nested"),
        *create_testcases(Kind.REVENUE, ER.COMPLETE | ER.FLAT_ONLY, Kind.PRICE, "flat"),
        *create_testcases(Kind.REVENUE, ER.ERROR, Kind.PRICE, "nested"),
        *create_testcases(Kind.REVENUE, ER.ERROR, Kind.REVENUE),
        # . Operand 2 = complete.
        *create_testcases(Kind.REVENUE, ER.ERROR, Kind.COMPLETE),
        # Operand 1 = complete pfline.
        # . Operand 2 = None.
        Testcase(Kind.COMPLETE, ER.COMPLETE, None),
        # . Operand 2 = dimensionless.
        *create_testcases(Kind.COMPLETE, ER.ERROR, "nodim"),
        # . Operand 2 = volume, price, or revenue.
        *create_testcases(Kind.COMPLETE, ER.ERROR, Kind.VOLUME),
        *create_testcases(Kind.COMPLETE, ER.ERROR, Kind.PRICE),
        *create_testcases(Kind.COMPLETE, ER.ERROR, Kind.REVENUE),
        # . Operand 2 = complete.
        *create_testcases(Kind.COMPLETE, ER.ERROR, Kind.COMPLETE),
    ],
    ids=id_fn,
)
@pytest.mark.parametrize("pfl_in_flat_or_nested", ["flat", "nested"])
def test_pfl_union_kind(testcase: Testcase, pfl_in_flat_or_nested: str, operation: str):
    """Test if arithmatic expectedly raises Error or returns expected type/kind."""
    do_kind_test(testcase, pfl_in_flat_or_nested, operation)


def do_kind_test(testcase: Testcase, pfl_in_flat_or_nested: str, operation: str):
    value = testcase.value
    pfl_in = PFL[pfl_in_flat_or_nested][testcase.kind_in]
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
    if (
        ER.ERROR in expected
        or (ER.NESTED_ONLY in expected and pfl_in_flat_or_nested == "flat")
        or (ER.FLAT_ONLY in expected and pfl_in_flat_or_nested == "nested")
    ):
        with pytest.raises(Exception):
            do_operation(pfl_in, operation, value)
        return

    # Check non-error case.
    result = do_operation(pfl_in, operation, value)
    if (exp_type := expected.typ()) is not None:
        assert isinstance(result, exp_type)
    if (exp_kind := expected.kind()) is not None:
        assert result.kind is exp_kind


def do_operation(operand1, operation, operand2) -> Any:
    if operation == "add":
        return operand1 + operand2
    if operation == "radd":
        return operand2 + operand1
    if operation == "sub":
        return operand1 - operand2
    if operation == "rsub":
        return operand2 - operand1
    if operation == "mul":
        return operand1 * operand2
    if operation == "rmul":
        return operand2 * operand1
    if operation == "div":
        return operand1 / operand2
    if operation == "rdiv":
        return operand2 / operand1
    if operation == "union":
        return operand1 | operand2
    if operation == "runion":
        return operand2 | operand1


i = pd.date_range("2020", freq="MS", periods=3, tz=tz)
series1 = {
    "w": pd.Series([3.0, 5, -4], i),
    "p": pd.Series([200.0, 100, 50], i),
    "r": pd.Series([446400.0, 348000, -148600], i),
}
pflset1 = {
    Kind.VOLUME: pf.FlatPfLine({"w": series1["w"]}),
    Kind.PRICE: pf.FlatPfLine({"p": series1["p"]}),
    Kind.COMPLETE: pf.FlatPfLine({"w": series1["w"], "p": series1["p"]}),
}
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
neg_volume_pfl1 = pf.FlatPfLine({"w": -series1["w"]})
neg_price_pfl1 = pf.FlatPfLine({"p": -series1["p"]})
neg_all_pfl1 = pf.FlatPfLine({"w": -series1["w"], "r": -series1["r"]})
add_volume_series = {"w": series1["w"] + series2["w"]}
add_volume_pfl = pf.FlatPfLine({"w": add_volume_series["w"]})
sub_volume_series = {"w": series1["w"] - series2["w"]}
sub_volume_pfl = pf.FlatPfLine({"w": sub_volume_series["w"]})
add_price_series = {"p": series1["p"] + series2["p"]}
add_price_pfl = pf.FlatPfLine({"p": add_price_series["p"]})
sub_price_series = {"p": series1["p"] - series2["p"]}
sub_price_pfl = pf.FlatPfLine({"p": sub_price_series["p"]})
add_all_series = {"w": series1["w"] + series2["w"], "r": series1["r"] + series2["r"]}
add_all_pfl = pf.FlatPfLine({"w": add_all_series["w"], "r": add_all_series["r"]})
sub_all_series = {"w": series1["w"] - series2["w"], "r": series1["r"] - series2["r"]}
sub_all_pfl = pf.FlatPfLine({"w": sub_all_series["w"], "r": sub_all_series["r"]})
mul_volume1_price2 = pf.FlatPfLine({"w": series1["w"], "p": series2["p"]})
mul_volume2_price1 = pf.FlatPfLine({"w": series2["w"], "p": series1["p"]})
div_volume1_volume2 = (series1["w"] / series2["w"]).astype("pint[dimensionless]")
div_price1_price2 = (series1["p"] / series2["p"]).astype("pint[dimensionless]")
mul_all1_dimless2 = pf.FlatPfLine(
    {"w": series1["w"] * series2["nodim"], "p": series1["p"]}
)
div_all1_dimless2 = pf.FlatPfLine(
    {"w": series1["w"] / series2["nodim"], "p": series1["p"]}
)


@pytest.mark.parametrize(
    ("pfl_in", "expected"),
    [
        (pflset1[Kind.VOLUME], neg_volume_pfl1),
        (pflset1[Kind.PRICE], neg_price_pfl1),
        (pflset1[Kind.COMPLETE], neg_all_pfl1),
    ],
)
def test_pfl_neg(pfl_in, expected):
    """Test if portfolio lines can be negated and give correct result."""
    result = -pfl_in
    assert result == expected


@pytest.mark.parametrize("operation", ["+", "-"])
@pytest.mark.parametrize(
    ("pfl_in", "value", "expected_add", "expected_sub"),
    [
        # Operand 1 = volume pfline.
        # . Operand 2 = constant without unit.
        (pflset1[Kind.VOLUME], 0, pflset1[Kind.VOLUME], pflset1[Kind.VOLUME]),
        (pflset1[Kind.VOLUME], 0.0, pflset1[Kind.VOLUME], pflset1[Kind.VOLUME]),
        (pflset1[Kind.VOLUME], None, pflset1[Kind.VOLUME], pflset1[Kind.VOLUME]),
        # . Operand 2 = constant with unit.
        (
            pflset1[Kind.VOLUME],
            Q_(12.0, "MW"),
            pf.FlatPfLine({"w": pd.Series([15.0, 17, 8], i)}),
            pf.FlatPfLine({"w": pd.Series([-9.0, -7, -16], i)}),
        ),
        (
            pflset1[Kind.VOLUME],
            {"w": Q_(12.0, "MW")},
            pf.FlatPfLine({"w": pd.Series([15.0, 17, 8], i)}),
            pf.FlatPfLine({"w": pd.Series([-9.0, -7, -16], i)}),
        ),
        (
            pflset1[Kind.VOLUME],
            {"w": 12.0},
            pf.FlatPfLine({"w": pd.Series([15.0, 17, 8], i)}),
            pf.FlatPfLine({"w": pd.Series([-9.0, -7, -16], i)}),
        ),
        # . Operand 2 = constant in different unit
        (
            pflset1[Kind.VOLUME],
            Q_(0.012, "GW"),
            pf.FlatPfLine({"w": pd.Series([15.0, 17, 8], i)}),
            pf.FlatPfLine({"w": pd.Series([-9.0, -7, -16], i)}),
        ),
        # . Operand 2 = constant in different dimension.
        (
            pflset1[Kind.VOLUME],
            Q_(12.0, "MWh"),
            pf.FlatPfLine({"q": pd.Series([2244.0, 3492, -2960], i)}),
            pf.FlatPfLine({"q": pd.Series([2220.0, 3468, -2984], i)}),
        ),
        # . Operand 2 = series without unit.
        (pflset1[Kind.VOLUME], series2["w"], ValueError, ValueError),
        # . Operand 2 = series without name.
        (
            pflset1[Kind.VOLUME],
            series2["w"].astype("pint[MW]"),
            add_volume_pfl,
            sub_volume_pfl,
        ),
        # . Operand 2 = series with useless name.
        (
            pflset1[Kind.VOLUME],
            series2["w"].rename("the_volume").astype("pint[MW]"),
            add_volume_pfl,
            sub_volume_pfl,
        ),
        # . Operand 2 = series without name and with different unit
        (
            pflset1[Kind.VOLUME],
            (series2["w"] * 1000).astype("pint[kW]"),
            add_volume_pfl,
            sub_volume_pfl,
        ),
        # . Operand 2 = series out of order.
        (
            pflset1[Kind.VOLUME],
            pd.Series([15.0, 4, -20], [i[0], i[2], i[1]]).astype("pint[MW]"),
            ValueError,
            ValueError,
        ),
        # . Operand 2 = dataframe without unit.
        (
            pflset1[Kind.VOLUME],
            pd.DataFrame({"w": series2["w"]}),
            add_volume_pfl,
            sub_volume_pfl,
        ),
        # . Operand 2 = other pfline.
        (pflset1[Kind.VOLUME], pflset2[Kind.VOLUME], add_volume_pfl, sub_volume_pfl),
        # Operand 1 = price pfline.
        # . Operand 2 = constant without unit.
        (pflset1[Kind.PRICE], 0, pflset1[Kind.PRICE], pflset1[Kind.PRICE]),
        (pflset1[Kind.PRICE], 0.0, pflset1[Kind.PRICE], pflset1[Kind.PRICE]),
        (pflset1[Kind.PRICE], None, pflset1[Kind.PRICE], pflset1[Kind.PRICE]),
        (
            pflset1[Kind.PRICE],
            12.0,
            pf.FlatPfLine({"p": pd.Series([212.0, 112, 62], i)}),
            pf.FlatPfLine({"p": pd.Series([188.0, 88, 38], i)}),
        ),
        # . Operand 2 = constant with default unit.
        (
            pflset1[Kind.PRICE],
            Q_(12.0, "Eur/MWh"),
            pf.FlatPfLine({"p": pd.Series([212.0, 112, 62], i)}),
            pf.FlatPfLine({"p": pd.Series([188.0, 88, 38], i)}),
        ),
        # . Operand 2 = constant with non-default unit.
        (
            pflset1[Kind.PRICE],
            Q_(1.2, "ct/kWh"),
            pf.FlatPfLine({"p": pd.Series([212.0, 112, 62], i)}),
            pf.FlatPfLine({"p": pd.Series([188.0, 88, 38], i)}),
        ),
        # . Operand 2 = other pfline.
        (pflset1[Kind.PRICE], pflset2[Kind.PRICE], add_price_pfl, sub_price_pfl),
        # Operand 1 = full pfline.
        # . Operand 2 = constant without unit.
        (pflset1[Kind.COMPLETE], 0, pflset1[Kind.COMPLETE], pflset1[Kind.COMPLETE]),
        (pflset1[Kind.COMPLETE], 0.0, pflset1[Kind.COMPLETE], pflset1[Kind.COMPLETE]),
        (pflset1[Kind.COMPLETE], None, pflset1[Kind.COMPLETE], pflset1[Kind.COMPLETE]),
        (pflset1[Kind.COMPLETE], 12.0, ValueError, ValueError),
        # . Operand 2 = series without unit.
        (pflset1[Kind.COMPLETE], series2["w"], ValueError, ValueError),
        # . Operand 2 = dataframe.
        (
            pflset1[Kind.COMPLETE],
            pd.DataFrame({"w": series2["w"], "p": series2["p"]}),
            add_all_pfl,
            sub_all_pfl,
        ),
        # . Operand 2 = dataframe.
        (
            pflset1[Kind.COMPLETE],
            pd.DataFrame({"w": series2["w"], "r": series2["r"]}),
            add_all_pfl,
            sub_all_pfl,
        ),
        # . Operand 2 = other pfline.
        (pflset1[Kind.COMPLETE], pflset2[Kind.COMPLETE], add_all_pfl, sub_all_pfl),
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
            pflset1[Kind.VOLUME],
            4.0,
            pf.FlatPfLine({"w": series1["w"] * 4}),
            pf.FlatPfLine({"w": series1["w"] / 4}),
        ),
        (
            pflset1[Kind.VOLUME],
            {"agn": 4.0},
            pf.FlatPfLine({"w": series1["w"] * 4}),
            pf.FlatPfLine({"w": series1["w"] / 4}),
        ),
        # . Explicitly dimensionless constant.
        (
            pflset1[Kind.VOLUME],
            Q_(4.0, ""),
            pf.FlatPfLine({"w": series1["w"] * 4}),
            pf.FlatPfLine({"w": series1["w"] / 4}),
        ),
        (
            pflset1[Kind.VOLUME],
            {"nodim": Q_(4.0, "")},
            pf.FlatPfLine({"w": series1["w"] * 4}),
            pf.FlatPfLine({"w": series1["w"] / 4}),
        ),
        (
            pflset1[Kind.VOLUME],
            {"nodim": 4.0},
            pf.FlatPfLine({"w": series1["w"] * 4}),
            pf.FlatPfLine({"w": series1["w"] / 4}),
        ),
        # . Fixed price constant.
        (
            pflset1[Kind.VOLUME],
            Q_(4.0, "Eur/MWh"),
            pf.FlatPfLine({"w": series1["w"], "p": 4}),
            Exception,
        ),
        (
            pflset1[Kind.VOLUME],
            {"p": Q_(4.0, "Eur/MWh")},
            pf.FlatPfLine({"w": series1["w"], "p": 4}),
            Exception,
        ),
        (
            pflset1[Kind.VOLUME],
            {"p": Q_(0.4, "ctEur/kWh")},
            pf.FlatPfLine({"w": series1["w"], "p": 4}),
            Exception,
        ),
        (
            pflset1[Kind.VOLUME],
            {"p": 4.0},
            pf.FlatPfLine({"w": series1["w"], "p": 4}),
            Exception,
        ),
        (
            pflset1[Kind.VOLUME],
            pd.Series([Q_(4.0, "Eur/MWh")], ["p"]),
            pf.FlatPfLine({"w": series1["w"], "p": 4}),
            Exception,
        ),
        (
            pflset1[Kind.VOLUME],
            pd.Series([4.0], ["p"]),
            pf.FlatPfLine({"w": series1["w"], "p": 4}),
            Exception,
        ),
        (
            pflset1[Kind.VOLUME],
            pd.Series([4.0], ["p"]).astype("pint[Eur/MWh]"),
            pf.FlatPfLine({"w": series1["w"], "p": 4}),
            Exception,
        ),
        # . Fixed volume constant.
        (
            pflset1[Kind.VOLUME],
            {"w": Q_(4.0, "MW")},
            Exception,
            (series1["w"] / 4).astype("pint[dimensionless]"),
        ),
        (
            pflset1[Kind.VOLUME],
            {"w": 4.0},
            Exception,
            (series1["w"] / 4).astype("pint[dimensionless]"),
        ),
        (
            pflset1[Kind.VOLUME],
            pd.Series([Q_(4.0, "MW")], ["w"]),
            Exception,
            (series1["w"] / 4).astype("pint[dimensionless]"),
        ),
        (
            pflset1[Kind.VOLUME],
            pd.Series([4.0], ["w"]).astype("pint[MW]"),
            Exception,
            (series1["w"] / 4).astype("pint[dimensionless]"),
        ),
        (  # divide by fixed ENERGY not POWER
            pflset1[Kind.VOLUME],
            pd.Series([4.0], ["q"]).astype("pint[MWh]"),
            Exception,
            (pflset1[Kind.VOLUME].q.pint.m / 4).astype("pint[dimensionless]"),
        ),
        # . Constant with incorrect unit.
        (pflset1[Kind.VOLUME], {"r": 4.0}, Exception, Exception),
        (
            pflset1[Kind.VOLUME],
            {"q": 4.0, "w": 8.0},  # incompatible
            Exception,
            Exception,
        ),
        (pflset1[Kind.VOLUME], Q_(4.0, "Eur"), Exception, Exception),
        (pflset1[Kind.VOLUME], {"r": 4.0, "q": 12}, Exception, Exception),
        (pflset1[Kind.VOLUME], {"r": 4.0, "nodim": 4.0}, Exception, Exception),
        # . Dim-agnostic or dimless series.
        (
            pflset1[Kind.VOLUME],
            series2["w"],  # has no unit
            pf.FlatPfLine({"w": series1["w"] * series2["w"]}),
            pf.FlatPfLine({"w": series1["w"] / series2["w"]}),
        ),
        (
            pflset1[Kind.VOLUME],
            series2["w"].astype("pint[dimensionless]"),  # dimensionless
            pf.FlatPfLine({"w": series1["w"] * series2["w"]}),
            pf.FlatPfLine({"w": series1["w"] / series2["w"]}),
        ),
        # . Price series, dataframe, or PfLine
        (
            pflset1[Kind.VOLUME],
            series2["p"].astype("pint[Eur/MWh]"),
            mul_volume1_price2,
            Exception,
        ),
        (
            pflset1[Kind.VOLUME],
            series2["p"].rename("the_price").astype("pint[Eur/MWh]"),
            mul_volume1_price2,
            Exception,
        ),
        (
            pflset1[Kind.VOLUME],
            (series2["p"] * 0.1).astype("pint[ct/kWh]"),
            mul_volume1_price2,
            Exception,
        ),
        (
            pflset1[Kind.VOLUME],
            pd.DataFrame({"p": series2["p"]}),
            mul_volume1_price2,
            Exception,
        ),
        (
            pflset1[Kind.VOLUME],
            pd.DataFrame({"p": (series2["p"] / 10).astype("pint[ct/kWh]")}),
            mul_volume1_price2,
            Exception,
        ),
        (pflset1[Kind.VOLUME], pflset2[Kind.PRICE], mul_volume1_price2, Exception),
        (
            pflset1[Kind.VOLUME],
            pd.Series([50.0, 400, 50], [i[1], i[0], i[2]]).astype(
                "pint[Eur/MWh]"
            ),  # not standardized
            ValueError,
            ValueError,
        ),
        # . Volume series, dataframe, or pfline
        (
            pflset1[Kind.VOLUME],
            series2["w"].astype("pint[MW]"),
            Exception,
            div_volume1_volume2,
        ),
        (pflset1[Kind.VOLUME], pflset2[Kind.VOLUME], Exception, div_volume1_volume2),
        (pflset1[Kind.VOLUME], pflset2[Kind.VOLUME].q, Exception, div_volume1_volume2),
        (pflset1[Kind.VOLUME], {"w": series2["w"]}, Exception, div_volume1_volume2),
        (
            pflset1[Kind.VOLUME],
            pd.DataFrame({"w": series2["w"]}),
            Exception,
            div_volume1_volume2,
        ),
        (
            pflset1[Kind.VOLUME],
            pflset2[Kind.VOLUME],  # other pfline
            Exception,
            div_volume1_volume2,
        ),
        # . Incorrect series, dataframe or pfline.
        (pflset1[Kind.VOLUME], series2["r"].astype("pint[Eur]"), Exception, Exception),
        (pflset1[Kind.VOLUME], pd.DataFrame({"r": series2["r"]}), Exception, Exception),
        (
            pflset1[Kind.VOLUME],
            pd.DataFrame({"the_price": series2["p"].astype("pint[Eur/MWh]")}),
            KeyError,
            KeyError,
        ),
        (pflset1[Kind.VOLUME], pflset2[Kind.COMPLETE], Exception, Exception),
        # Multiplying price pfline.
        # . (dimension-agnostic) constant.
        (
            pflset1[Kind.PRICE],
            4,
            pf.FlatPfLine({"p": series1["p"] * 4}),
            pf.FlatPfLine({"p": series1["p"] / 4}),
        ),
        # . Explicitly dimensionless constant.
        (
            pflset1[Kind.PRICE],
            Q_(4, ""),
            pf.FlatPfLine({"p": series1["p"] * 4}),
            pf.FlatPfLine({"p": series1["p"] / 4}),
        ),
        # . Fixed price constant.
        (
            pflset1[Kind.PRICE],
            Q_(4, "Eur/MWh"),
            Exception,
            (series1["p"] / 4).astype("pint[dimensionless]"),
        ),
        (
            pflset1[Kind.PRICE],
            {"p": 4},
            Exception,
            (series1["p"] / 4).astype("pint[dimensionless]"),
        ),
        # . Fixed volume constant.
        (
            pflset1[Kind.PRICE],
            Q_(4, "MWh"),
            pf.FlatPfLine({"p": series1["p"], "q": 4}),
            Exception,
        ),
        (
            pflset1[Kind.PRICE],
            Q_(4, "MW"),
            pf.FlatPfLine({"p": series1["p"], "w": 4}),
            Exception,
        ),
        (
            pflset1[Kind.PRICE],
            Q_(4, "GW"),
            pf.FlatPfLine({"p": series1["p"], "w": 4000}),
            Exception,
        ),
        (
            pflset1[Kind.PRICE],
            pd.Series([4], ["w"]).astype("pint[GW]"),
            pf.FlatPfLine({"p": series1["p"], "w": 4000}),
            Exception,
        ),
        # . Incorrect constant.
        (pflset1[Kind.PRICE], Q_(4, "Eur"), Exception, Exception),
        # . Dim-agnostic or dimless series.
        (
            pflset1[Kind.PRICE],
            series2["w"],  # has no unit
            pf.FlatPfLine({"p": series1["p"] * series2["w"]}),
            pf.FlatPfLine({"p": series1["p"] / series2["w"]}),
        ),
        (
            pflset1[Kind.PRICE],
            series2["w"].astype("pint[dimensionless]"),  # dimensionless
            pf.FlatPfLine({"p": series1["p"] * series2["w"]}),
            pf.FlatPfLine({"p": series1["p"] / series2["w"]}),
        ),
        # . Price series, dataframe, or PfLine
        (
            pflset1[Kind.PRICE],
            series2["p"].astype("pint[Eur/MWh]"),  # series
            Exception,
            div_price1_price2,
        ),
        (
            pflset1[Kind.PRICE],
            pflset2[Kind.PRICE],  # pfline
            Exception,
            div_price1_price2,
        ),
        (
            pflset1[Kind.PRICE],
            pflset2[Kind.PRICE].p,  # series
            Exception,
            div_price1_price2,
        ),
        (
            pflset1[Kind.PRICE],
            {"p": series2["p"]},  # dict of series
            Exception,
            div_price1_price2,
        ),
        (
            pflset1[Kind.PRICE],
            pd.DataFrame({"p": series2["p"]}),  # dataframe
            Exception,
            div_price1_price2,
        ),
        (
            pflset1[Kind.PRICE],
            pflset2[Kind.PRICE],  # other pfline
            Exception,
            div_price1_price2,
        ),
        # . Volume series, dataframe, or pfline
        (
            pflset1[Kind.PRICE],
            series2["w"].astype("pint[MW]"),
            mul_volume2_price1,
            Exception,
        ),
        (
            pflset1[Kind.PRICE],
            (series2["w"] / 1000).astype("pint[GW]"),
            mul_volume2_price1,
            Exception,
        ),
        (
            pflset1[Kind.PRICE],
            series2["w"].rename("the_volume").astype("pint[MW]"),
            mul_volume2_price1,
            Exception,
        ),
        (
            pflset1[Kind.PRICE],
            pd.DataFrame({"w": series2["w"]}),
            mul_volume2_price1,
            Exception,
        ),
        (
            pflset1[Kind.PRICE],
            pd.DataFrame({"w": (series2["w"] / 1000).astype("pint[GW]")}),
            mul_volume2_price1,
            Exception,
        ),
        (pflset1[Kind.PRICE], pflset2[Kind.VOLUME], mul_volume2_price1, Exception),
        # . Incorrect series, dataframe or pfline.
        (pflset1[Kind.PRICE], series2["r"].astype("pint[Eur]"), Exception, Exception),
        (pflset1[Kind.PRICE], pd.DataFrame({"r": series2["r"]}), Exception, Exception),
        (
            pflset1[Kind.PRICE],
            pd.DataFrame({"the_price": series2["p"].astype("pint[Eur/MWh]")}),
            KeyError,
            KeyError,
        ),
        (pflset1[Kind.PRICE], pflset2[Kind.COMPLETE], Exception, Exception),
        # Multiplying 'complete' pfline.
        # . (dimension-agnostic) constant.
        (
            pflset1[Kind.COMPLETE],
            6,
            FlatPfLine({"w": series1["w"] * 6, "p": series1["p"]}),
            FlatPfLine({"w": series1["w"] / 6, "p": series1["p"]}),
        ),
        # . Explicitly dimensionless constant.
        (
            pflset1[Kind.COMPLETE],
            Q_(6, ""),
            FlatPfLine({"w": series1["w"] * 6, "p": series1["p"]}),
            FlatPfLine({"w": series1["w"] / 6, "p": series1["p"]}),
        ),
        (
            pflset1[Kind.COMPLETE],
            {"nodim": 6},
            FlatPfLine({"w": series1["w"] * 6, "p": series1["p"]}),
            FlatPfLine({"w": series1["w"] / 6, "p": series1["p"]}),
        ),
        (
            pflset1[Kind.COMPLETE],
            pd.Series([6], ["nodim"]),
            FlatPfLine({"w": series1["w"] * 6, "p": series1["p"]}),
            FlatPfLine({"w": series1["w"] / 6, "p": series1["p"]}),
        ),
        # . Incorrect constant.
        (pflset1[Kind.COMPLETE], {"r": 4.0}, Exception, Exception),
        (pflset1[Kind.COMPLETE], {"q": 4.0, "w": 8.0}, Exception, Exception),
        (pflset1[Kind.COMPLETE], Q_(4.0, "Eur"), Exception, Exception),
        (pflset1[Kind.COMPLETE], {"r": 4.0, "q": 12}, Exception, Exception),
        (pflset1[Kind.COMPLETE], {"r": 4.0, "nodim": 4.0}, Exception, Exception),
        # . Dim-agnostic or dimless series.
        (
            pflset1[Kind.COMPLETE],
            series2["nodim"],  # dim-agnostic
            mul_all1_dimless2,
            div_all1_dimless2,
        ),
        (
            pflset1[Kind.COMPLETE],
            series2["nodim"].astype("pint[dimensionless]"),  # dimless
            mul_all1_dimless2,
            div_all1_dimless2,
        ),
        (
            pflset1[Kind.COMPLETE],
            {"nodim": series2["nodim"]},
            mul_all1_dimless2,
            div_all1_dimless2,
        ),
        (
            pflset1[Kind.COMPLETE],
            pd.DataFrame({"nodim": series2["nodim"]}),
            mul_all1_dimless2,
            div_all1_dimless2,
        ),
        (
            pflset1[Kind.COMPLETE],
            pd.DataFrame({"nodim": series2["nodim"].astype("pint[dimensionless]")}),
            mul_all1_dimless2,
            div_all1_dimless2,
        ),
        # . Incorrect series, dataframe or pfline.
        (pflset1[Kind.COMPLETE], {"r": series2["p"]}, Exception, Exception),
        (
            pflset1[Kind.COMPLETE],
            series2["p"].astype("pint[Eur/MWh]"),
            Exception,
            Exception,
        ),
        (pflset1[Kind.COMPLETE], pflset2[Kind.PRICE], Exception, Exception),
        (pflset1[Kind.COMPLETE], pflset2[Kind.VOLUME], Exception, Exception),
        (pflset1[Kind.COMPLETE], pflset2[Kind.COMPLETE], Exception, Exception),
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
