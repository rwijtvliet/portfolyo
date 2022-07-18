from typing import Any, Dict
from portfolyo import (
    PfLine,
    SinglePfLine,
    MultiPfLine,
    PfState,
    Kind,
    Q_,
)  # noqa
import portfolyo as pf
import pandas as pd
import pytest


pf.dev.seed(0)  # make sure we always get the same random numbers


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
    elif isinstance(data, SinglePfLine):
        return f"Singlepfline_{data.kind}"
    elif isinstance(data, MultiPfLine):
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
# i1 = pd.date_range("2021", periods=20, freq="MS", tz=tz)  # same freq, part overlap
# i2 = pd.date_range("2022-04", periods=20, freq="MS", tz=tz)  # same freq, no overlap
# i3 = pd.date_range(
#     "2020-04", periods=8000, freq="H", tz=tz
# )  # shorter freq, part overlap
# i4 = pd.date_range("2020", periods=8, freq="QS", tz=tz)  # longer freq, part overlap


# TODO: check correct working of dunder methods. E.g. assert correct addition:
# . . pflines having same or different kind
# . . pflines having same or different frequency
# . . pflines covering same or different time periods


# Set 1.
i1 = pd.date_range("2020", freq="MS", periods=3, tz=tz)
dataset1 = {
    "offtake": PfLine({"w": pd.Series([-3.5, -5, -5], i1)}),
    "unsourcedprice": PfLine({"p": pd.Series([300.0, 150, 100], i1)}),
    "sourced": PfLine(
        {"w": pd.Series([3.0, 5, 4], i1), "p": pd.Series([200.0, 100, 50], i1)}
    ),
    "unsourced": PfLine(
        {"w": pd.Series([0.5, 0, 1], i1), "p": pd.Series([300.0, 150, 100], i1)}
    ),
    "nodim": pd.Series([2, -1.5, 10], i1),
}
pfs1 = PfState(dataset1["offtake"], dataset1["unsourcedprice"], dataset1["sourced"] / 2)

# Set 2. Partial overlap with set 1.
i2 = pd.date_range("2020-02", freq="MS", periods=3, tz=tz)
dataset2 = {
    "offtake": PfLine({"w": pd.Series([-15.0, -20, -4], i2)}),
    "unsourcedprice": PfLine({"p": pd.Series([400.0, 50, 50], i2)}),
    "sourced": PfLine(
        {"w": pd.Series([12.0, 5, 4], i2), "p": pd.Series([100.0, 100, 50], i2)}
    ),
    "unsourced": PfLine(
        {"w": pd.Series([3.0, 15, 0], i2), "p": pd.Series([400.0, 50, 50], i2)}
    ),
    "nodim": pd.Series([2, -1.5, 10], i2),
}
pfs2 = PfState(dataset2["offtake"], dataset2["unsourcedprice"], dataset2["sourced"] / 2)

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
mul_pfs1_nodim2 = PfState(
    dataset1["offtake"] * dataset2["nodim"],
    dataset1["unsourcedprice"],
    dataset1["sourced"] * dataset2["nodim"],
)
div_pfs1_nodim2 = PfState(
    dataset1["offtake"] / dataset2["nodim"],
    dataset1["unsourcedprice"],
    dataset1["sourced"] / dataset2["nodim"],
)
neg_pfs1 = PfState(
    -dataset1["offtake"], dataset1["unsourcedprice"], -dataset1["sourced"]
)
i12 = pd.date_range("2020-02", freq="MS", periods=2)
add_pfs1_pfs2 = PfState(
    PfLine({"w": pd.Series([-20.0, -25], i12)}),
    PfLine({"p": pd.Series([400, 53.125], i12)}),
    PfLine({"w": pd.Series([17.0, 9], i12), "p": pd.Series([100, 700 / 9], i12)}),
)


# neg_volume_pfl1 = pf.SinglePfLine({"w": -series1["w"]})
# neg_price_pfl1 = pf.SinglePfLine({"p": -series1["p"]})
# neg_all_pfl1 = pf.SinglePfLine({"w": -series1["w"], "r": -series1["r"]})
# add_volume_series = {"w": series1["w"] + series2["w"]}
# add_volume_pfl = pf.SinglePfLine({"w": add_volume_series["w"]})
# sub_volume_series = {"w": series1["w"] - series2["w"]}
# sub_volume_pfl = pf.SinglePfLine({"w": sub_volume_series["w"]})
# add_price_series = {"p": series1["p"] + series2["p"]}
# add_price_pfl = pf.SinglePfLine({"p": add_price_series["p"]})
# sub_price_series = {"p": series1["p"] - series2["p"]}
# sub_price_pfl = pf.SinglePfLine({"p": sub_price_series["p"]})
# add_all_series = {"w": series1["w"] + series2["w"], "r": series1["r"] + series2["r"]}
# add_all_pfl = pf.SinglePfLine({"w": add_all_series["w"], "r": add_all_series["r"]})
# sub_all_series = {"w": series1["w"] - series2["w"], "r": series1["r"] - series2["r"]}
# sub_all_pfl = pf.SinglePfLine({"w": sub_all_series["w"], "r": sub_all_series["r"]})
# mul_volume1_price2 = pf.SinglePfLine({"w": series1["w"], "p": series2["p"]})
# mul_volume2_price1 = pf.SinglePfLine({"w": series2["w"], "p": series1["p"]})
# div_volume1_volume2 = (series1["w"] / series2["w"]).astype("pint[dimensionless]")
# div_price1_price2 = (series1["p"] / series2["p"]).astype("pint[dimensionless]")
# mul_all1_dimless2 = pf.SinglePfLine(
#     {"w": series1["w"] * series2["nodim"], "p": series1["p"]}
# )
# div_all1_dimless2 = pf.SinglePfLine(
#     {"w": series1["w"] / series2["nodim"], "p": series1["p"]}
# )


@pytest.mark.parametrize(
    ("pfs_in", "operation", "value", "expected"),
    [
        # Constant without unit.
        (pfs1, "+", 2, ValueError),
        (pfs1, "-", 2, ValueError),
        (pfs1, "*", 2, mul_pfs1_2),
        (pfs1, "/", 2, div_pfs1_2),
        # Constant with unit.
        (pfs1, "+", Q_(4, "MWh"), ValueError),
        (pfs1, "-", Q_(4, "MWh"), ValueError),
        (pfs1, "*", Q_(4, "MWh"), ValueError),
        (pfs1, "/", Q_(4, "MWh"), ValueError),
        # Series without unit.
        (pfs1, "+", dataset1["nodim"], ValueError),
        (pfs1, "-", dataset1["nodim"], ValueError),
        (pfs1, "*", dataset1["nodim"], mul_pfs1_nodim2),
        (pfs1, "/", dataset1["nodim"], div_pfs1_nodim2),
        # Series with unit.
        (pfs1, "+", dataset1["nodim"].astype("pint[MW]"), ValueError),
        (pfs1, "-", dataset1["nodim"].astype("pint[MW]"), ValueError),
        (pfs1, "*", dataset1["nodim"].astype("pint[MW]"), ValueError),
        (pfs1, "/", dataset1["nodim"].astype("pint[MW]"), ValueError),
        # Other PfState.
        (pfs1, "+", pfs1, mul_pfs1_2),
        (pfs1, "+", pfs2, add_pfs1_pfs2),
        (pfs1, "-", pfs1, mul_pfs1_0),
        (pfs1, "*", pfs1, ValueError),
        (pfs1, "/", pfs1, ValueError),
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
        elif operation == "-":
            return pfs_in / value

    # Test error case.
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            _ = calc()
        return

    # Test correct case.
    result = calc()
    assert result == expected
