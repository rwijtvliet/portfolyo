"""Test initialisation of PfLine, SinglePfLine, and MultiPfLine."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable
from portfolyo import dev, PfLine, SinglePfLine, MultiPfLine, FREQUENCIES, Kind  # noqa
from portfolyo.tools import stamps
import pandas as pd
import pytest


def id_fn(data: Any):
    """Readable id of test case"""
    if isinstance(data, Dict):
        return str({key: id_fn(val) for key, val in data.items()})
    elif isinstance(data, pd.Series):
        if isinstance(data.index, pd.DatetimeIndex):
            return "TimeSeries"
        return f"Series (idx: {''.join(str(i) for i in data.index)})"
    elif isinstance(data, pd.DataFrame):
        return f"Df (columns: {''.join(str(c) for c in data.columns)})"
    elif isinstance(data, SinglePfLine):
        return f"Singlepfline_{data.kind}"
    elif isinstance(data, MultiPfLine):
        return f"Multipfline_{data.kind}"
    elif isinstance(data, InitTestcase):
        return f"{id_fn(data.data_in)}-{id_fn(data.expected_kind)}"
    elif type(data) is type:
        return data.__name__
    return type(data).__name__


# Functions that create one object from which a PfLine can be instatiated (or not).


def dict_testcase(
    i: pd.DatetimeIndex, has_unit: bool, keys: Iterable, variation: int
) -> Dict:
    """Create dictionary to test initialisation. ``variation`` is used to specify:
    * 1: pass all as timeseries.
    * 2: pass one as timeseries, rest as values.
    * 3: pass all as values."""
    data_in = {}
    for k, key in enumerate(keys):
        if variation == 1 or variation == 2 and k == 1:
            data_in[key] = dev.get_series(i, key, has_unit)
        else:
            data_in[key] = dev.get_value(key, has_unit)
    return data_in


def df_testcase(i: pd.DatetimeIndex, has_unit: bool, cols: Iterable) -> pd.DataFrame:
    """Create dataframe to test initialisation."""
    series = {}
    for col in cols:
        series[col] = dev.get_series(i, col, has_unit)
    return pd.DataFrame(series)


def timeseries_testcase(i: pd.DataFrame, has_unit: bool, name: str) -> pd.Series:
    """Create timeseries to test initialisation."""
    return dev.get_series(i, name, has_unit)


def singlepfline_testcase(i: pd.DatetimeIndex, kind: str) -> SinglePfLine:
    """Create singlepfline to test initialisation."""
    return dev.get_singlepfline(i, kind)


def multipfline_testcase(i: pd.DatetimeIndex, kind: str) -> MultiPfLine:
    """Create multipfline to test initialisation."""
    return dev.get_multipfline(i, kind)


# Functions that create iterable of Test Case from which a PfLine can be instatiated (or not).

# Create long list of all test cases.


@dataclass
class InitTestcase:
    id: str  # Id to show for this test when running pytest
    data_in: Any  # The data to use in the initialisation
    expected_kind: Kind


all_cases = [
    f"{w}{q}{p}{r}".replace(" ", "")
    for w in "w "
    for q in "q "
    for p in "p "
    for r in "r "
    if not w == q == p == r == " "
]  # all combinations of w, q, p, r
error_cases = ["r", "wq", *[case for case in all_cases if len(case) >= 3]]


def dict_testcases(i, has_unit, typ) -> Iterable[InitTestcase]:
    """Create several testcases with a dictionary as input data."""
    testcases = []
    for keys in all_cases:
        for variation in [1, 2, 3] if len(keys) > 1 else [1, 3]:
            data_in = dict_testcase(i, has_unit, keys, variation)
            if variation == 3:  # no index
                expectedkind = Exception
            elif keys in error_cases:
                expectedkind = Exception
            elif typ is MultiPfLine:
                # under special circumstances, a multipfline can be created from dictionary of timeseries
                expectedkind = Exception
                if variation == 1 and "r" not in keys:
                    k = ""
                    if "w" in keys or "q" in keys:
                        k += "q"
                    if "p" in keys:
                        k += "p"
                    expectedkind = {
                        "p": Kind.PRICE_ONLY,
                        "q": Kind.VOLUME_ONLY,
                        "qp": Kind.ALL,
                    }[k]
            elif keys in ["w", "q"]:
                expectedkind = Kind.VOLUME_ONLY
            elif keys in ["p"]:
                expectedkind = Kind.PRICE_ONLY
            elif keys in ["wp", "wr", "qp", "qr", "pr"]:
                expectedkind = Kind.ALL
            else:
                raise ValueError("Unexpected case")
            testcases.append(InitTestcase("data:dict", data_in, expectedkind))
    return testcases


def df_testcases(i, has_unit, typ) -> Iterable[InitTestcase]:
    """Create several testcases with a dataframe as input data."""
    testcases = []
    for cols in all_cases:
        data_in = df_testcase(i, has_unit, cols)
        if typ is MultiPfLine:
            expectedkind = Exception
        elif cols in error_cases:
            expectedkind = Exception
        elif cols in ["w", "q"]:
            expectedkind = Kind.VOLUME_ONLY
        elif cols in ["p"]:
            expectedkind = Kind.PRICE_ONLY
        elif cols in ["wp", "wr", "qp", "qr", "pr"]:
            expectedkind = Kind.ALL
        else:
            raise ValueError("Unexpected case")
        testcases.append(InitTestcase("data:df", data_in, expectedkind))
    return testcases


def timeseries_testcases(i, has_unit, typ) -> Iterable[InitTestcase]:
    """Create several testcases with a timeseries as input data."""
    testcases = []
    for name in "wqpr":
        data_in = timeseries_testcase(i, has_unit, name)
        if typ is SinglePfLine and has_unit and name == "p":
            expectedkind = Kind.PRICE_ONLY
        elif typ is SinglePfLine and has_unit and name in ["w", "q"]:
            expectedkind = Kind.VOLUME_ONLY
        else:
            expectedkind = Exception
        testcases.append(InitTestcase("data:ts", data_in, expectedkind))
    return testcases


def singlepfline_testcases(i, typ) -> Iterable[InitTestcase]:
    """Create several testcases with a singlepfline as input data."""
    testcases = []
    for kind in [Kind.ALL, Kind.VOLUME_ONLY, Kind.PRICE_ONLY]:
        data_in = singlepfline_testcase(i, kind)
        if typ is MultiPfLine:
            expectedkind = Exception
        else:
            expectedkind = kind
        testcases.append(InitTestcase("data:ts", data_in, expectedkind))
    return testcases


def multipfline_testcases(i, typ) -> Iterable[InitTestcase]:
    """Create several testcases with a multipfline as input data."""
    testcases = []
    for kind in [Kind.ALL, Kind.VOLUME_ONLY, Kind.PRICE_ONLY]:
        data_in = multipfline_testcase(i, kind)
        expectedkind = kind
        testcases.append(InitTestcase("data:ts", data_in, expectedkind))
    return testcases


def all_testcases(typ) -> Iterable[InitTestcase]:
    """Create several testcases."""
    testcases = []
    # TODO: can we turn this into some kind of pytest.mark.parametrize?
    for start in ["2020", "2022-04-21 15:15"]:
        for tz in ["Europe/Berlin", None]:
            for freq in FREQUENCIES:
                i = dev.get_index(freq, tz, start)
                try:
                    stamps.assert_boundary_ts(i, freq)
                except AssertionError:
                    continue
                for has_unit in [True, False]:
                    cases = [
                        *dict_testcases(i, has_unit, typ),
                        *df_testcases(i, has_unit, typ),
                        *timeseries_testcases(i, has_unit, typ),
                        *singlepfline_testcases(i, typ),
                        *multipfline_testcases(i, typ),
                    ]
                    for case in cases:
                        case.id += f"-{id_fn(case.data_in)}"
                        case.id += f"-tz:{tz}-fr:{freq}-st:{start}-units:{has_unit}"
                    testcases.extend(cases)
    return testcases


def data_in_function(typ) -> Dict[str, Iterable]:
    """Function to collect all test cases to test initialisation of class ``typ``."""
    testcases = all_testcases(typ)
    return {"argvalues": testcases, "ids": [case.id for case in testcases]}


@pytest.mark.parametrize("itd", **data_in_function(SinglePfLine))
def test_singlepfline_init_2(itd: InitTestcase):
    """Test if SinglePfLine can be initialized correctly, and attributes return correct values."""
    data_in, expected_kind = itd.data_in, itd.expected_kind
    if type(expected_kind) is type and issubclass(expected_kind, Exception):
        with pytest.raises(expected_kind):
            _ = SinglePfLine(data_in)
        return

    result = SinglePfLine(data_in)
    assert result.kind is expected_kind


@pytest.mark.parametrize("itd", **data_in_function(MultiPfLine))
def test_multipfline_init_2(itd: InitTestcase):
    """Test if MultiPfLine can be initialized correctly, and attributes return correct values."""
    # TODO: first specify, how multipfline should be initializable
    # data_in, expected_kind = itd.data_in, itd.expected_kind
    # if type(expected_kind) is type and issubclass(expected_kind, Exception):
    #     with pytest.raises(expected_kind):
    #         _ = MultiPfLine(data_in)
    #     return

    # result = MultiPfLine(data_in)
    # assert result.kind == expected_kind
