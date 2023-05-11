"""Test initialisation of PfLine, FlatPfLine, and NestedPfLine."""

import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

import pandas as pd
import pytest

import portfolyo as pf
from portfolyo import FlatPfLine, Kind, NestedPfLine, PfLine, dev


@dataclass
class InitTestcase:
    data_in: Any  # The data to use in the initialisation
    expected_df: pd.DataFrame
    expected_kind: Kind


def _returntype(self, testtype):
    if testtype is FlatPfLine:
        return FlatPfLine if self.value[1] else Exception
    elif testtype is NestedPfLine:
        return NestedPfLine if self.value[2] else Exception
    elif self is InputTypeA.SINGLEPFLINE:
        return FlatPfLine
    elif self is InputTypeA.MULTIPFLINE:
        return NestedPfLine
    else:
        return FlatPfLine if self.value[1] else NestedPfLine


class InputTypeA(Enum):
    # for functions that take `columns` as a parameter.
    FLATDF = ("flatdf", True, False)  # name, can-be-turned-into-single, can-...-multi
    FLATDICT = ("flatdict", True, False)
    FLATSERIES = ("flatseries", True, False)
    SINGLEPFLINE = ("flatpfline", True, False)
    MULTIPFLINE = ("nestedpfline", True, True)

    returntype = _returntype


class InputTypeB(Enum):
    # for functions that take `kind` as a parameter.
    PFDICTSINGLE = ("pfdictsingle", False, True)
    PFDICTMULTI = ("pfdictmulti", False, True)
    PFDICTMIX = ("pfdictmix", False, True)
    DICTINITIABLE_NOUNIT = ("object dict", False, True)
    DICTINITIABLE_UNIT = ("object-with-unit dict", False, True)
    MULTILEVELDF_NOUNIT = ("multileveldf", False, True)
    MULTILEVELDF_UNIT = ("multileveldf-with-units", False, True)

    returntype = _returntype


def get_testcase_A(
    i: pd.DatetimeIndex, columns: Iterable[str], inputtype: InputTypeA, has_unit: bool
) -> InitTestcase:
    """Create test case that uses ``columns`` as parameter."""

    # Data.
    df = dev.get_dataframe(i, columns, has_unit)
    if inputtype is InputTypeA.FLATDF:
        data_in = df
    elif inputtype is InputTypeA.FLATDICT:
        data_in = {name: s for name, s in df.items()}
    elif inputtype is InputTypeA.FLATSERIES:
        data_in = [df[col] for col in columns]  # pass as list of series...
        if len(data_in) == 1:
            data_in = data_in[0]  # ... or as single series
        if not has_unit:
            df = Exception
    elif inputtype is InputTypeA.SINGLEPFLINE:
        data_in = pf.flatpfline(df)
    elif inputtype is InputTypeA.MULTIPFLINE:
        if columns in ["w", "q", "p", "qr", "wr"]:
            df1 = 0.4 * df
            df2 = 0.6 * df
        else:  # has price column
            othercol = columns.replace("p", "")
            df1 = df.mul({"p": 1, othercol: 0.4})
            df2 = df.mul({"p": 1, othercol: 0.6})
        data_in = pf.nestedpfline({"a": pf.flatpfline(df1), "b": pf.flatpfline(df2)})
    else:
        raise ValueError("unknown inputtype")

    # Checks.
    if columns in ["w", "q"]:
        kind = Kind.VOLUME
    elif columns in ["p"]:
        kind = Kind.PRICE
    else:
        kind = Kind.COMPLETE

    return InitTestcase(data_in, df, kind)


def get_testcase_B(
    i: pd.DatetimeIndex, kind: Kind, inputtype: InputTypeB
) -> InitTestcase:
    """Create test case that uses ``kind`` as parameter."""

    # Data.
    if inputtype is InputTypeB.PFDICTSINGLE:
        data_in = {
            "child1": dev.get_flatpfline(i, kind),
            "child2": dev.get_flatpfline(i, kind),
        }
    elif inputtype is InputTypeB.PFDICTMULTI:
        data_in = {
            "child1": dev.get_nestedpfline(i, kind),
            "child2": dev.get_nestedpfline(i, kind),
        }
    elif inputtype is InputTypeB.PFDICTMIX:
        data_in = {
            "child1": dev.get_flatpfline(i, kind),
            "child2": dev.get_nestedpfline(i, kind),
        }
    elif inputtype in [InputTypeB.MULTILEVELDF_NOUNIT, InputTypeB.MULTILEVELDF_UNIT]:
        # only check one initalisable inputtype
        has_unit = inputtype is InputTypeB.MULTILEVELDF_UNIT
        data_in = {}
        for c in range(3):
            if kind is Kind.VOLUME:
                columns = random.choice(("w", "q"))
            elif kind is Kind.PRICE:
                columns = "p"
            elif kind is Kind.REVENUE:
                columns = "r"
            else:
                columns = random.choice(["wp", "wr", "qp", "qr", "pr"])
            data_in[f"child{c}"] = (
                get_testcase_A(i, columns, InputTypeA.FLATDF, has_unit).data_in,
            )

    elif inputtype in [InputTypeB.DICTINITIABLE_NOUNIT, InputTypeB.DICTINITIABLE_UNIT]:
        has_unit = inputtype is InputTypeB.DICTINITIABLE_UNIT
        columns = {
            Kind.PRICE: "p",
            Kind.VOLUME: "w",
            Kind.REVENUE: "r",
            Kind.COMPLETE: "wp",
        }[kind]
        data_in = pd.concat(
            [
                dev.get_dataframe(i, columns, has_unit),
                dev.get_dataframe(i, columns, has_unit),
            ],
            axis=1,
            keys=["child1", "child2"],
        )

    return InitTestcase(data_in, None, kind)


def anyerror(*args):
    for a in args:
        if isinstance(a, type) and issubclass(a, Exception):
            return a
    return None


@pytest.mark.only_on_pr
@pytest.mark.parametrize("freq", pf.FREQUENCIES[::2])
@pytest.mark.parametrize("tz", ["Europe/Berlin", None])
@pytest.mark.parametrize("columns", ["w", "q", "p", "pr", "qr", "pq", "wp", "wr"])
@pytest.mark.parametrize("inputtype", InputTypeA)
@pytest.mark.parametrize("has_unit", [True, False])
@pytest.mark.parametrize("testtype", [PfLine, FlatPfLine, NestedPfLine])
def test_init_A(
    freq: str,
    tz: str,
    columns: Iterable[str],
    inputtype: InputTypeA,
    has_unit: bool,
    testtype: type,
):
    """Test if pfline can be initialized correctly from a flat testcase."""

    i = dev.get_index(freq, tz)
    itc = get_testcase_A(i, columns, inputtype, has_unit)

    expected_type = inputtype.returntype(testtype)
    if expected_error := anyerror(expected_type, itc.expected_df):
        with pytest.raises(expected_error):
            _ = testtype(itc.data_in)
        return

    result = testtype(itc.data_in)
    result_df = result.df[columns]

    assert type(result) is expected_type
    if type(itc.data_in) is type(testtype):
        assert result is itc.data_in  # assert no copy but reference.
    pf.testing.assert_frame_equal(result_df, itc.expected_df.rename_axis("ts_left"))
    assert result.kind is itc.expected_kind
    if expected_type is NestedPfLine:
        assert len(result)


@pytest.mark.only_on_pr
@pytest.mark.parametrize("freq", pf.FREQUENCIES[::2])
@pytest.mark.parametrize("tz", ["Europe/Berlin", None])
@pytest.mark.parametrize("kind", Kind)
@pytest.mark.parametrize("inputtype", InputTypeB)
@pytest.mark.parametrize("testtype", [PfLine, FlatPfLine, NestedPfLine])
def test_init_B(
    freq: str,
    tz: str,
    kind: Kind,
    inputtype: InputTypeB,
    testtype: type,
):
    """Test if pfline can be initialized correctly from a more complex testcase."""
    i = dev.get_index(freq, tz)
    itc = get_testcase_B(i, kind, inputtype)
    expected_type = inputtype.returntype(testtype)

    if expected_error := anyerror(expected_type):
        with pytest.raises(expected_error):
            _ = testtype(itc.data_in)
        return

    result = testtype(itc.data_in)

    assert type(result) is expected_type
    assert result.kind is itc.expected_kind
