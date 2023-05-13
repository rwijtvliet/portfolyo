from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

import portfolyo as pf
from portfolyo import Q_, Kind, PfState, dev, testing
from portfolyo.core.pfline import create, classes


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
    elif isinstance(data, classes.FlatPfLine):
        return f"Singlepfline_{data.kind}"
    elif isinstance(data, classes.NestedPfLine):
        return f"Multipfline_{data.kind}"
    elif isinstance(data, str):
        return data
    elif isinstance(data, pf.Q_):
        return f"Quantity ({data.units})"
    elif isinstance(data, type):
        return data.__name__
    elif isinstance(data, Kind):
        return str(data)
    return type(data).__name__


tz = "Europe/Berlin"

# Set 1.
i1 = pd.date_range("2020", freq="MS", periods=3, tz=tz)
dataset1 = {
    "offtake": create.flatpfline({"w": pd.Series([-3.5, -5, -5], i1)}),
    "unsourcedprice": create.flatpfline({"p": pd.Series([300.0, 150, 100], i1)}),
    "sourced": create.flatpfline(
        {"w": pd.Series([3.0, 5, 4], i1), "p": pd.Series([200.0, 100, 50], i1)}
    ),
    "unsourced": create.flatpfline(
        {"w": pd.Series([0.5, 0, 1], i1), "p": pd.Series([300.0, 150, 100], i1)}
    ),
    "nodim": pd.Series([2, -1.5, 10], i1),
}
pfs1 = PfState(dataset1["offtake"], dataset1["unsourcedprice"], dataset1["sourced"])

# Set 2. Partial overlap with set 1.
i2 = pd.date_range("2020-02", freq="MS", periods=3, tz=tz)
dataset2 = {
    "offtake": create.flatpfline({"w": pd.Series([-15.0, -20, -4], i2)}),
    "unsourcedprice": create.flatpfline({"p": pd.Series([400.0, 50, 50], i2)}),
    "sourced": create.flatpfline(
        {"w": pd.Series([12.0, 5, 4], i2), "p": pd.Series([100.0, 100, 50], i2)}
    ),
    "unsourced": create.flatpfline(
        {"w": pd.Series([3.0, 15, 0], i2), "p": pd.Series([400.0, 50, 50], i2)}
    ),
    "nodim": pd.Series([2, -1.5, 10], i2),
}
pfs2 = PfState(dataset2["offtake"], dataset2["unsourcedprice"], dataset2["sourced"])

# Portfolio state without overlap.
i3 = pd.date_range("2022", freq="MS", periods=3, tz=tz)
pfs3 = dev.get_pfstate(i3)

# Portfolio state with other frequency.
i4 = pd.date_range("2020", freq="D", periods=3, tz=tz)
pfs4 = dev.get_pfstate(i4)

# Portfolio state with wrong timezone.
i5 = pd.date_range("2022", freq="MS", periods=3, tz="Asia/Kolkata")
pfs5 = dev.get_pfstate(i5)
i6 = pd.date_range("2022", freq="MS", periods=3, tz=None)
pfs6 = dev.get_pfstate(i6)

# Several expected results of arithmatic.
mul_pfs1_2 = PfState(
    dataset1["offtake"] * 2, dataset1["unsourcedprice"], dataset1["sourced"] * 2
)
mul_pfs1_0 = PfState(
    dataset1["offtake"] * 0, dataset1["unsourcedprice"], dataset1["sourced"] * 0
)
div_pfs1_2 = PfState(
    dataset1["offtake"] / 2, dataset1["unsourcedprice"], dataset1["sourced"] / 2
)
mul_pfs1_nodim1 = PfState(
    dataset1["offtake"] * dataset1["nodim"],
    dataset1["unsourcedprice"],
    dataset1["sourced"] * dataset1["nodim"],
)
div_pfs1_nodim1 = PfState(
    dataset1["offtake"] / dataset1["nodim"],
    dataset1["unsourcedprice"],
    dataset1["sourced"] / dataset1["nodim"],
)
neg_pfs1 = PfState(
    -dataset1["offtake"], dataset1["unsourcedprice"], -dataset1["sourced"]
)
i12 = pd.date_range("2020-02", freq="MS", periods=2, tz=tz)
add_pfs1_pfs2 = PfState(
    create.flatpfline({"w": pd.Series([-20.0, -25], i12)}),
    create.flatpfline({"p": pd.Series([400, 53.125], i12)}),
    create.flatpfline(
        {"w": pd.Series([17.0, 9], i12), "p": pd.Series([100, 700 / 9], i12)}
    ),
)
div_pfs1_pfs2 = pd.DataFrame(
    {
        ("offtake", "volume"): [1 / 3, 1 / 4],
        ("sourced", "volume"): [5 / 12, 4 / 5],
        ("sourced", "price"): [1, 0.5],
        ("unsourced", "volume"): [0, 1 / 15],
        ("unsourced", "price"): [150 / 400, 2],
        ("pnl_cost", "price"): [1 / 1.6, 60 / 62.5],
    },
    i12,
    dtype="pint[dimensionless]",
)
div_pfs2_pfs1 = pd.DataFrame(
    {
        ("offtake", "volume"): [3.0, 4],
        ("sourced", "volume"): [12 / 5, 5 / 4],
        ("sourced", "price"): [1.0, 2],
        ("unsourced", "volume"): [np.nan, 15],
        ("unsourced", "price"): [400 / 150, 0.5],
        ("pnl_cost", "price"): [1.6, 62.5 / 60],
    },
    i12,
    dtype="pint[dimensionless]",
)
div_pfs1_pfs1 = pd.DataFrame(
    {
        ("offtake", "volume"): [1.0, 1, 1],
        ("sourced", "volume"): [1.0, 1, 1],
        ("sourced", "price"): [1.0, 1, 1],
        ("unsourced", "volume"): [1.0, np.nan, 1],
        ("unsourced", "price"): [1.0, 1, 1],
        ("pnl_cost", "price"): [1.0, 1, 1],
    },
    i1,
    dtype="pint[dimensionless]",
)
div_2pfs1_pfs1 = pd.DataFrame(
    {
        ("offtake", "volume"): [2.0, 2, 2],
        ("sourced", "volume"): [2.0, 2, 2],
        ("sourced", "price"): [1.0, 1, 1],
        ("unsourced", "volume"): [2.0, np.nan, 2],
        ("unsourced", "price"): [1.0, 1, 1],
        ("pnl_cost", "price"): [1.0, 1, 1],
    },
    i1,
    dtype="pint[dimensionless]",
)
emptydivision = pd.DataFrame(
    [],
    columns=(
        ("offtake", "volume"),
        ("sourced", "volume"),
        ("sourced", "price"),
        ("unsourced", "volume"),
        ("unsourced", "price"),
        ("pnl_cost", "price"),
    ),
    index=pd.DatetimeIndex([], freq="MS", tz=tz),
    dtype=float,
)


def test_pfs_negation():
    """Test if negation works correctly."""
    result = -pfs1
    assert result == neg_pfs1


@pytest.mark.parametrize(
    ("pfs_in", "operation", "value", "expected"),
    [
        # Constant without unit.
        (pfs1, "+", 2, Exception),
        (pfs1, "-", 2, Exception),
        (pfs1, "*", 2, mul_pfs1_2),
        (pfs1, "/", 2, div_pfs1_2),
        # Constant with unit.
        (pfs1, "+", Q_(4.0, "MWh"), Exception),
        (pfs1, "-", Q_(4.0, "MWh"), Exception),
        (pfs1, "*", Q_(4.0, "MWh"), Exception),
        (pfs1, "/", Q_(4.0, "MWh"), Exception),
        # Series without unit.
        (pfs1, "+", dataset1["nodim"], Exception),
        (pfs1, "-", dataset1["nodim"], Exception),
        (pfs1, "*", dataset1["nodim"], mul_pfs1_nodim1),
        (pfs1, "/", dataset1["nodim"], div_pfs1_nodim1),
        # Series with unit.
        (pfs1, "+", dataset1["nodim"].astype("pint[MW]"), Exception),
        (pfs1, "-", dataset1["nodim"].astype("pint[MW]"), Exception),
        (pfs1, "*", dataset1["nodim"].astype("pint[MW]"), Exception),
        (pfs1, "/", dataset1["nodim"].astype("pint[MW]"), Exception),
        # Other PfState.
        (pfs1, "+", pfs1, mul_pfs1_2),
        (pfs1, "+", pfs2, add_pfs1_pfs2),
        (pfs2, "+", pfs1, add_pfs1_pfs2),
        (pfs1, "-", pfs1, mul_pfs1_0),
        (pfs1, "*", pfs1, Exception),
        (pfs1, "*", pfs2, Exception),
        (pfs1, "/", pfs2, div_pfs1_pfs2),
        (pfs2, "/", pfs1, div_pfs2_pfs1),
        (pfs1, "/", pfs1, div_pfs1_pfs1),
        (mul_pfs1_2, "/", pfs1, div_2pfs1_pfs1),
        # Other PfState without overlap.
        (pfs1, "+", pfs3, Exception),
        (pfs1, "-", pfs3, Exception),
        (pfs1, "*", pfs3, Exception),
        (pfs1, "/", pfs3, ValueError),
        # Other PfState with incorrect frequency.
        (pfs1, "+", pfs4, Exception),
        (pfs1, "-", pfs4, Exception),
        (pfs1, "*", pfs4, Exception),
        (pfs1, "/", pfs4, Exception),
        # Other PfState in different timezone.
        (pfs1, "+", pfs5, Exception),
        (pfs1, "-", pfs5, Exception),
        (pfs1, "*", pfs5, Exception),
        (pfs1, "/", pfs5, Exception),
        (pfs1, "+", pfs6, Exception),
        (pfs1, "-", pfs6, Exception),
        (pfs1, "*", pfs6, Exception),
        (pfs1, "/", pfs6, Exception),
    ],
    ids=id_fn,
)
def test_pfs_arithmatic(pfs_in, value, operation, expected):
    """Test if addition, subtraction, multiplication, division return correct values."""

    def calc():
        if operation == "+":
            return pfs_in + value
        elif operation == "-":
            return pfs_in - value
        elif operation == "*":
            return pfs_in * value
        elif operation == "/":
            return pfs_in / value
        raise NotImplementedError()

    # Test error case.
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            _ = calc()
        return

    # Test correct case.
    result = calc()
    if isinstance(expected, pd.DataFrame):
        testing.assert_frame_equal(result, expected)
    else:
        assert result == expected
