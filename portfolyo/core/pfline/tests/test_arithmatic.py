from portfolyo.core.pfline import PfLine, SinglePfLine, MultiPfLine  # noqa
from portfolyo.tools.nits import Q_
from portfolyo import dev, testing
from typing import Dict
import portfolyo as pf
import pandas as pd
import pytest

# TODO: Multipfline


def id_fn(data):
    if isinstance(data, Dict):
        return str({key: id_fn(val) for key, val in data.items()})
    if isinstance(data, pd.Series):
        if isinstance(data.index, pd.DatetimeIndex):
            return "ts"
        else:
            return f"series (idx: {''.join(str(i) for i in data.index)})"
    if isinstance(data, pd.DataFrame):
        return f"df (columns: {''.join(str(c) for c in data.columns)})"
    if isinstance(data, SinglePfLine):
        return f"singlepfline_{data.kind}"
    if isinstance(data, MultiPfLine):
        return f"multipfline_{data.kind}"
    return str(data)


tz = "Europe/Berlin"
i = pd.date_range("2020", periods=20, freq="MS", tz=tz)  # reference
# i1 = pd.date_range("2021", periods=20, freq="MS", tz=tz)  # same freq, part overlap
# i2 = pd.date_range("2022-04", periods=20, freq="MS", tz=tz)  # same freq, no overlap
# i3 = pd.date_range(
#     "2020-04", periods=8000, freq="H", tz=tz
# )  # shorter freq, part overlap
# i4 = pd.date_range("2020", periods=8, freq="QS", tz=tz)  # longer freq, part overlap


# . check correct working of dunder methods. E.g. assert correct addition:
# . . pflines having same or different kind
# . . pflines having same or different frequency
# . . pflines covering same or different time periods


@pytest.mark.parametrize("operation", ["plus", "minus"])
@pytest.mark.parametrize(
    ("pfl_in_i", "pfl_in_kind", "value", "returntype", "returnkind"),
    [
        # Adding to volume pfline.
        # . Add volume.
        # . . single value
        (i, "q", Q_(8.1, "Wh"), PfLine, "q"),
        (i, "q", Q_(8.1, "kWh"), PfLine, "q"),
        (i, "q", Q_(8.1, "MWh"), PfLine, "q"),
        (i, "q", Q_(8.1, "GWh"), PfLine, "q"),
        (i, "q", Q_(8.1, "W"), PfLine, "q"),
        (i, "q", Q_(8.1, "kW"), PfLine, "q"),
        (i, "q", Q_(8.1, "MW"), PfLine, "q"),
        (i, "q", Q_(8.1, "GW"), PfLine, "q"),
        (i, "q", {"w": Q_(8.1, "GW")}, PfLine, "q"),
        (i, "q", {"w": 8.1}, PfLine, "q"),
        (i, "q", pd.Series([8.1], ["q"]), PfLine, "q"),
        (i, "q", pd.Series([8.1], ["q"]).astype("pint[MWh]"), PfLine, "q"),
        (i, "q", pd.Series([Q_(8.1, "MWh")], ["q"]), PfLine, "q"),
        (i, "q", pd.Series([8.1], ["w"]), PfLine, "q"),
        (i, "q", pd.Series([8.1], ["w"]).astype("pint[MW]"), PfLine, "q"),
        (i, "q", pd.Series([Q_(8.1, "MW")], ["w"]), PfLine, "q"),
        (i, "q", Q_(8.1, "J"), PfLine, "q"),
        # . . timeseries (series, df, pfline)
        (i, "q", dev.get_series(i, "q"), PfLine, "q"),
        (i, "q", dev.get_series(i, "w"), PfLine, "q"),
        (i, "q", dev.get_dataframe(i, "q"), PfLine, "q"),
        (i, "q", dev.get_pfline(i, "q"), PfLine, "q"),
        (i, "q", dev.get_singlepfline(i, "q"), PfLine, "q"),
        (i, "q", dev.get_multipfline(i, "q"), PfLine, "q"),
        # . Add something else.
        # . . single value
        (i, "q", 8.1, None, None),
        (i, "q", Q_(8.1, ""), None, None),
        (i, "q", Q_(8.1, "Eur/MWh"), None, None),
        (i, "q", Q_(8.1, "Eur"), None, None),
        (i, "q", Q_(8.1, "h"), None, None),
        (i, "q", {"the_volume": Q_(8.1, "MWh")}, None, None),
        # . . timeseries (series, df, pfline)
        (i, "q", pd.DataFrame({"the_volume": dev.get_series(i, "w")}), None, None),
        (i, "q", dev.get_series(i, "q").pint.magnitude, None, None),
        (i, "q", dev.get_series(i, "p"), None, None),
        (i, "q", dev.get_series(i, "r"), None, None),
        (i, "q", dev.get_dataframe(i, "p"), None, None),
        (i, "q", dev.get_dataframe(i, "qr"), None, None),
        (i, "q", dev.get_dataframe(i, "qp"), None, None),
        (i, "q", dev.get_singlepfline(i, "p"), None, None),
        (i, "q", dev.get_multipfline(i, "p"), None, None),
        (i, "q", dev.get_singlepfline(i, "all"), None, None),
        (i, "q", dev.get_multipfline(i, "all"), None, None),
        # Adding to price pfline.
        # . Add dimension-agnostic.
        (i, "p", 12, PfLine, "p"),
        (i, "p", dev.get_series(i, "p").pint.magnitude, PfLine, "p"),
        # . Add price.
        (i, "p", Q_(12.0, "Eur/MWh"), PfLine, "p"),
        (i, "p", Q_(12.0, "Eur/kWh"), PfLine, "p"),
        (i, "p", Q_(12.0, "cent/kWh"), PfLine, "p"),
        (i, "p", dev.get_series(i, "p"), PfLine, "p"),
        (i, "p", dev.get_singlepfline(i, "p"), PfLine, "p"),
        (i, "p", dev.get_multipfline(i, "p"), PfLine, "p"),
        # . Add something else.
        (i, "p", Q_(12.0, ""), None, None),  # explicitly dimensionless
        (i, "p", Q_(12.0, "Eur"), None, None),
        (i, "p", Q_(12.0, "MWh"), None, None),
        (i, "p", Q_(12.0, "h"), None, None),
        (i, "p", dev.get_series(i, "q"), None, None),
        (i, "p", dev.get_singlepfline(i, "q"), None, None),
        (i, "p", dev.get_multipfline(i, "q"), None, None),
        (i, "p", dev.get_singlepfline(i, "all"), None, None),
        (i, "p", dev.get_multipfline(i, "all"), None, None),
        # Adding to 'complete' pfline.
        # . Add dimension-agnostic.
        (i, "all", 5.9, None, None),
        (i, "all", dev.get_series(i, "q").pint.magnitude, None, None),
        (i, "all", {"nodim": 5.9}, None, None),
        (i, "all", {"nodim": Q_(5.9, "")}, None, None),
        # . Add other 'all' pfline.
        (i, "all", dev.get_dataframe(i, "qr"), PfLine, "all"),
        (i, "all", dev.get_dataframe(i, "qp"), PfLine, "all"),
        (i, "all", dev.get_dataframe(i, "pr"), PfLine, "all"),
        (i, "all", dev.get_singlepfline(i, "all"), PfLine, "all"),
        (i, "all", dev.get_multipfline(i, "all"), PfLine, "all"),
        # . Add something else.
        (i, "all", Q_(6.0, "Eur"), None, None),
        (i, "all", Q_(6.0, "Eur/MWh"), None, None),
        (i, "all", Q_(6.0, "MW"), None, None),
        (i, "all", Q_(6.0, "MWh"), None, None),
        (i, "all", Q_(6.0, "h"), None, None),
        (i, "all", dev.get_series(i, "p"), None, None),
        (i, "all", dev.get_series(i, "r"), None, None),
        (i, "all", dev.get_series(i, "q"), None, None),
        (i, "all", dev.get_series(i, "w"), None, None),
        (i, "all", dev.get_dataframe(i, "p"), None, None),
        (i, "all", dev.get_dataframe(i, "r"), None, None),
        (i, "all", dev.get_dataframe(i, "q"), None, None),
        (i, "all", dev.get_dataframe(i, "w"), None, None),
        (i, "all", dev.get_dataframe(i, "wq"), None, None),
        (i, "all", dev.get_singlepfline(i, "p"), None, None),
        (i, "all", dev.get_multipfline(i, "p"), None, None),
        (i, "all", dev.get_singlepfline(i, "q"), None, None),
        (i, "all", dev.get_multipfline(i, "q"), None, None),
    ],
    ids=id_fn,
)
@pytest.mark.parametrize("pfl_in_single_or_multi", ["single", "multi"])
def test_pfl_addition_errorandreturntype(
    pfl_in_i,
    pfl_in_kind,
    pfl_in_single_or_multi,
    value,
    returntype,
    returnkind,
    operation,
):
    """Test if error is raised on impossible addition and subtraction."""
    if pfl_in_single_or_multi == "single":
        pfl_in = dev.get_singlepfline(pfl_in_i, pfl_in_kind)
    else:
        pfl_in = dev.get_multipfline(pfl_in_i, pfl_in_kind)

    # Check error is raised.
    # (Not working due to implementation issue in pint and numpy: value + pfl_in, value - pfl_in)
    if returntype is None:
        allowed_errors = (NotImplementedError, ValueError, KeyError, TypeError)
        if allowed_errors:
            with pytest.raises(allowed_errors):
                if operation == "plus":
                    _ = pfl_in + value
                else:
                    _ = pfl_in - value
            return

    # Check return type.
    # (Not working due to implementation issue in pint and numpy: value + pfl_in, value - pfl_in)
    if operation == "plus":
        result = pfl_in + value
    else:
        result = pfl_in - value
    assert isinstance(result, returntype)
    if returntype is PfLine:
        assert result.kind == returnkind


i = pd.date_range("2020", freq="MS", periods=3, tz=tz)
series1 = {
    "w": pd.Series([3.0, 5, -4], i),
    "p": pd.Series([200.0, 100, 50], i),
    "r": pd.Series([446400.0, 348000, -148600], i),
}
pflset1 = {
    "q": pf.SinglePfLine({"w": series1["w"]}),
    "p": pf.SinglePfLine({"p": series1["p"]}),
    "all": pf.SinglePfLine({"w": series1["w"], "p": series1["p"]}),
}
series2 = {
    "w": pd.Series([15.0, -20, 4], i),
    "p": pd.Series([400.0, 50, 50], i),
    "r": pd.Series([4464000.0, -696000, 148600], i),
}
pflset2 = {
    "q": pf.SinglePfLine({"w": series2["w"]}),
    "p": pf.SinglePfLine({"p": series2["p"]}),
    "all": pf.SinglePfLine({"w": series2["w"], "p": series2["p"]}),
}
neg_volume_pfl1 = pf.SinglePfLine({"w": -series1["w"]})
neg_price_pfl1 = pf.SinglePfLine({"p": -series1["p"]})
neg_all_pfl1 = pf.SinglePfLine({"w": -series1["w"], "r": -series1["r"]})
add_volume_series = {"w": series1["w"] + series2["w"]}
add_volume_pfl = pf.SinglePfLine({"w": add_volume_series["w"]})
sub_volume_series = {"w": series1["w"] - series2["w"]}
sub_volume_pfl = pf.SinglePfLine({"w": sub_volume_series["w"]})
add_price_series = {"p": series1["p"] + series2["p"]}
add_price_pfl = pf.SinglePfLine({"p": add_price_series["p"]})
sub_price_series = {"p": series1["p"] - series2["p"]}
sub_price_pfl = pf.SinglePfLine({"p": sub_price_series["p"]})
add_all_series = {"w": series1["w"] + series2["w"], "r": series1["r"] + series2["r"]}
add_all_pfl = pf.SinglePfLine({"w": add_all_series["w"], "r": add_all_series["r"]})
sub_all_series = {"w": series1["w"] - series2["w"], "r": series1["r"] - series2["r"]}
sub_all_pfl = pf.SinglePfLine({"w": sub_all_series["w"], "r": sub_all_series["r"]})
mul_volume1_price2 = pf.SinglePfLine({"w": series1["w"], "p": series2["p"]})
mul_volume2_price1 = pf.SinglePfLine({"w": series2["w"], "p": series1["p"]})
div_volume1_volume2 = (series1["w"] / series2["w"]).astype("pint[dimensionless]")


@pytest.mark.parametrize(
    ("pfl_in", "expected"),
    [
        (pflset1["q"], neg_volume_pfl1),
        (pflset1["p"], neg_price_pfl1),
        (pflset1["all"], neg_all_pfl1),
    ],
)
def test_pfl_neg(pfl_in, expected):
    """Test if portfolio lines can be negated and give correct result."""
    result = -pfl_in
    assert result == expected


@pytest.mark.parametrize("operation", ["plus", "minus"])
@pytest.mark.parametrize(
    ("pfl_in", "value", "expected_plus", "expected_minus"),
    [
        # Adding to volume pfline.
        # . Add constant.
        (
            pflset1["q"],
            Q_(12.0, "MW"),
            pf.SinglePfLine({"w": pd.Series([15.0, 17, 8], i)}),
            pf.SinglePfLine({"w": pd.Series([-9.0, -7, -16], i)}),
        ),
        (
            pflset1["q"],
            {"w": Q_(12.0, "MW")},
            pf.SinglePfLine({"w": pd.Series([15.0, 17, 8], i)}),
            pf.SinglePfLine({"w": pd.Series([-9.0, -7, -16], i)}),
        ),
        (
            pflset1["q"],
            {"w": 12.0},
            pf.SinglePfLine({"w": pd.Series([15.0, 17, 8], i)}),
            pf.SinglePfLine({"w": pd.Series([-9.0, -7, -16], i)}),
        ),
        # . Add constant in different unit
        (
            pflset1["q"],
            Q_(0.012, "GW"),
            pf.SinglePfLine({"w": pd.Series([15.0, 17, 8], i)}),
            pf.SinglePfLine({"w": pd.Series([-9.0, -7, -16], i)}),
        ),
        # . Add constant in different dimension.
        (
            pflset1["q"],
            Q_(12.0, "MWh"),
            pf.SinglePfLine({"q": pd.Series([2244.0, 3492, -2960], i)}),
            pf.SinglePfLine({"q": pd.Series([2220.0, 3468, -2984], i)}),
        ),
        # . Add series without unit.
        (
            pflset1["q"],
            series2["w"],
            ValueError,
            ValueError,
        ),
        # . Add series without name.
        (
            pflset1["q"],
            series2["w"].astype("pint[MW]"),
            add_volume_pfl,
            sub_volume_pfl,
        ),
        # . Add series with useless name.
        (
            pflset1["q"],
            series2["w"].rename("the_volume").astype("pint[MW]"),
            add_volume_pfl,
            sub_volume_pfl,
        ),
        # . Add series without name and with different unit
        (
            pflset1["q"],
            (series2["w"] * 1000).astype("pint[kW]"),
            add_volume_pfl,
            sub_volume_pfl,
        ),
        # . Add series out of order.
        (
            pflset1["q"],
            pd.Series([15.0, 4, -20], [i[0], i[2], i[1]]).astype("pint[MW]"),
            ValueError,
            ValueError,
        ),
        # . Add dataframe without unit.
        (
            pflset1["q"],
            pd.DataFrame({"w": series2["w"]}),
            add_volume_pfl,
            sub_volume_pfl,
        ),
        # . Add other pfline.
        (
            pflset1["q"],
            pflset2["q"],
            add_volume_pfl,
            sub_volume_pfl,
        ),
        # Adding to price pfline.
        # . Add constant without unit.
        (
            pflset1["p"],
            12.0,
            pf.SinglePfLine({"p": pd.Series([212.0, 112, 62], i)}),
            pf.SinglePfLine({"p": pd.Series([188.0, 88, 38], i)}),
        ),
        # . Add constant with default unit.
        (
            pflset1["p"],
            Q_(12.0, "Eur/MWh"),
            pf.SinglePfLine({"p": pd.Series([212.0, 112, 62], i)}),
            pf.SinglePfLine({"p": pd.Series([188.0, 88, 38], i)}),
        ),
        # . Add constant with non-default unit.
        (
            pflset1["p"],
            Q_(1.2, "ct/kWh"),
            pf.SinglePfLine({"p": pd.Series([212.0, 112, 62], i)}),
            pf.SinglePfLine({"p": pd.Series([188.0, 88, 38], i)}),
        ),
        # . Add other pfline.
        (
            pflset1["p"],
            pflset2["p"],
            add_price_pfl,
            sub_price_pfl,
        ),
        # Adding to full pfline.
        # . Add dataframe.
        (
            pflset1["all"],
            pd.DataFrame({"w": series2["w"], "p": series2["p"]}),
            add_all_pfl,
            sub_all_pfl,
        ),
        # . Add dataframe.
        (
            pflset1["all"],
            pd.DataFrame({"w": series2["w"], "r": series2["r"]}),
            add_all_pfl,
            sub_all_pfl,
        ),
        # . Add other pfline.
        (
            pflset1["all"],
            pflset2["all"],
            add_all_pfl,
            sub_all_pfl,
        ),
    ],
    ids=id_fn,
)
def test_pfl_addition(pfl_in, value, expected_plus, expected_minus, operation):
    """Test if portfolio lines of equal lengths can be added and subtracted and give correct result."""
    expected = expected_plus if operation == "plus" else expected_minus

    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            _ = (pfl_in + value) if operation == "plus" else (pfl_in - value)
        return

    result = (pfl_in + value) if operation == "plus" else (pfl_in - value)
    assert result == expected


@pytest.mark.parametrize("operation", ["mul", "div"])
@pytest.mark.parametrize(
    ("pfl_in", "value", "expected_mul", "expected_div"),
    [
        # Multiplying volume pfline.
        # . (dimension-agnostic) constant.
        (
            pflset1["q"],
            4.0,
            pf.SinglePfLine({"w": series1["w"] * 4}),
            pf.SinglePfLine({"w": series1["w"] / 4}),
        ),
        (
            pflset1["q"],
            {"agn": 4.0},
            pf.SinglePfLine({"w": series1["w"] * 4}),
            pf.SinglePfLine({"w": series1["w"] / 4}),
        ),
        # . Explicitly dimensionless constant.
        (
            pflset1["q"],
            Q_(4.0, ""),
            pf.SinglePfLine({"w": series1["w"] * 4}),
            pf.SinglePfLine({"w": series1["w"] / 4}),
        ),
        (
            pflset1["q"],
            {"nodim": Q_(4.0, "")},
            pf.SinglePfLine({"w": series1["w"] * 4}),
            pf.SinglePfLine({"w": series1["w"] / 4}),
        ),
        (
            pflset1["q"],
            {"nodim": 4.0},
            pf.SinglePfLine({"w": series1["w"] * 4}),
            pf.SinglePfLine({"w": series1["w"] / 4}),
        ),
        # . Fixed price constant.
        (
            pflset1["q"],
            Q_(4.0, "Eur/MWh"),
            pf.SinglePfLine({"w": series1["w"], "p": 4}),
            Exception,
        ),
        (
            pflset1["q"],
            {"p": Q_(4.0, "Eur/MWh")},
            pf.SinglePfLine({"w": series1["w"], "p": 4}),
            Exception,
        ),
        (
            pflset1["q"],
            {"p": Q_(0.4, "ctEur/kWh")},
            pf.SinglePfLine({"w": series1["w"], "p": 4}),
            Exception,
        ),
        (
            pflset1["q"],
            {"p": 4.0},
            pf.SinglePfLine({"w": series1["w"], "p": 4}),
            Exception,
        ),
        (
            pflset1["q"],
            pd.Series([Q_(4.0, "Eur/MWh")], ["p"]),
            pf.SinglePfLine({"w": series1["w"], "p": 4}),
            Exception,
        ),
        (
            pflset1["q"],
            pd.Series([4.0], ["p"]),
            pf.SinglePfLine({"w": series1["w"], "p": 4}),
            Exception,
        ),
        (
            pflset1["q"],
            pd.Series([4.0], ["p"]).astype("pint[Eur/MWh]"),
            pf.SinglePfLine({"w": series1["w"], "p": 4}),
            Exception,
        ),
        # . Fixed volume constant.
        (
            pflset1["q"],
            {"w": Q_(4.0, "MW")},
            Exception,
            (series1["w"] / 4).astype("pint[dimensionless]"),
        ),
        (
            pflset1["q"],
            {"w": 4.0},
            Exception,
            (series1["w"] / 4).astype("pint[dimensionless]"),
        ),
        (
            pflset1["q"],
            pd.Series([Q_(4.0, "MW")], ["w"]),
            Exception,
            (series1["w"] / 4).astype("pint[dimensionless]"),
        ),
        (
            pflset1["q"],
            pd.Series([4.0], ["w"]).astype("pint[MW]"),
            Exception,
            (series1["w"] / 4).astype("pint[dimensionless]"),
        ),
        (  # divide by fixed ENERGY not POWER
            pflset1["q"],
            pd.Series([4.0], ["q"]).astype("pint[MWh]"),
            Exception,
            (pflset1["q"].q.pint.m / 4).astype("pint[dimensionless]"),
        ),
        # . Constant with incorrect unit.
        (
            pflset1["q"],
            {"r": 4.0},
            Exception,
            Exception,
        ),
        (
            pflset1["q"],
            {"q": 4.0, "w": 8.0},  # incompatible
            Exception,
            Exception,
        ),
        (
            pflset1["q"],
            Q_(4.0, "Eur"),
            Exception,
            Exception,
        ),
        (
            pflset1["q"],
            {"r": 4.0, "q": 12},
            Exception,
            Exception,
        ),
        (
            pflset1["q"],
            {"r": 4.0, "nodim": 4.0},
            Exception,
            Exception,
        ),
        # . Dim-agnostic or dimless series.
        (
            pflset1["q"],
            series2["w"],  # has no unit
            pf.SinglePfLine({"w": series1["w"] * series2["w"]}),
            pf.SinglePfLine({"w": series1["w"] / series2["w"]}),
        ),
        (
            pflset1["q"],
            series2["w"].astype("pint[dimensionless]"),  # dimensionless
            pf.SinglePfLine({"w": series1["w"] * series2["w"]}),
            pf.SinglePfLine({"w": series1["w"] / series2["w"]}),
        ),
        # . Price series, dataframe, or PfLine
        (
            pflset1["q"],
            series2["p"].astype("pint[Eur/MWh]"),
            mul_volume1_price2,
            Exception,
        ),
        (
            pflset1["q"],
            series2["p"].rename("the_price").astype("pint[Eur/MWh]"),
            mul_volume1_price2,
            Exception,
        ),
        (
            pflset1["q"],
            (series2["p"] * 0.1).astype("pint[ct/kWh]"),
            mul_volume1_price2,
            Exception,
        ),
        (
            pflset1["q"],
            pd.DataFrame({"p": series2["p"]}),
            mul_volume1_price2,
            Exception,
        ),
        (
            pflset1["q"],
            pd.DataFrame({"p": (series2["p"] / 10).astype("pint[ct/kWh]")}),
            mul_volume1_price2,
            Exception,
        ),
        (
            pflset1["q"],
            pflset2["p"],
            mul_volume1_price2,
            Exception,
        ),
        (
            pflset1["q"],
            pd.Series([50.0, 400, 50], [i[1], i[0], i[2]]).astype(
                "pint[Eur/MWh]"
            ),  # not standardized
            ValueError,
            ValueError,
        ),
        # . Volume series, dataframe, or pfline
        (
            pflset1["q"],
            series2["w"].astype("pint[MW]"),
            Exception,
            div_volume1_volume2,
        ),
        (
            pflset1["q"],
            pflset2["q"],
            Exception,
            div_volume1_volume2,
        ),
        (
            pflset1["q"],
            pflset2["q"].q,
            Exception,
            div_volume1_volume2,
        ),
        (
            pflset1["q"],
            {"w": series2["w"]},
            Exception,
            div_volume1_volume2,
        ),
        (
            pflset1["q"],
            pd.DataFrame({"w": series2["w"]}),
            Exception,
            div_volume1_volume2,
        ),
        (
            pflset1["q"],
            pflset2["q"],  # other pfline
            Exception,
            div_volume1_volume2,
        ),
        # . Incorrect series, dataframe or pfline.
        (
            pflset1["q"],
            series2["r"].astype("pint[Eur]"),
            Exception,
            Exception,
        ),
        (
            pflset1["q"],
            pd.DataFrame({"r": series2["r"]}),
            Exception,
            Exception,
        ),
        (
            pflset1["q"],
            pd.DataFrame({"the_price": series2["p"].astype("pint[Eur/MWh]")}),
            KeyError,
            KeyError,
        ),
        (
            pflset1["q"],
            pflset2["all"],
            Exception,
            Exception,
        ),
        # Multiplying price pfline.
        # . (dimension-agnostic) constant.
        (
            pflset1["p"],
            4,
            pf.SinglePfLine({"p": series1["p"] * 4}),
            pf.SinglePfLine({"p": series1["p"] / 4}),
        ),
        # . Explicitly dimensionless constant.
        (
            pflset1["p"],
            Q_(4, ""),
            pf.SinglePfLine({"p": series1["p"] * 4}),
            pf.SinglePfLine({"p": series1["p"] / 4}),
        ),
        # . Fixed price constant.
        (
            pflset1["p"],
            Q_(4, "Eur/MWh"),
            Exception,
            (series1["p"] / 4).astype("pint[dimensionless]"),
        ),
        (
            pflset1["p"],
            {"p": 4},
            Exception,
            (series1["p"] / 4).astype("pint[dimensionless]"),
        ),
        # . Fixed volume constant.
        (
            pflset1["p"],
            Q_(4, "MWh"),
            pf.SinglePfLine({"p": series1["p"], "q": 4}),
            Exception,
        ),
        (
            pflset1["p"],
            Q_(4, "MW"),
            pf.SinglePfLine({"p": series1["p"], "w": 4}),
            Exception,
        ),
        (
            pflset1["p"],
            Q_(4, "GW"),
            pf.SinglePfLine({"p": series1["p"], "w": 4000}),
            Exception,
        ),
        (
            pflset1["p"],
            pd.Series([4], ["w"]).astype("pint[GW]"),
            pf.SinglePfLine({"p": series1["p"], "w": 4000}),
            Exception,
        ),
        # . Incorrect constant.
        (
            pflset1["p"],
            Q_(4, "Eur"),
            ValueError,
            ValueError,
        ),
        # . Dim-agnostic or dimless series.
        # . Price series, dataframe, or PfLine
        # . Volume series, dataframe, or pfline
        # . Incorrect series, dataframe or pfline.
        # Multiplying volume pfline.
        # . (dimension-agnostic) constant.
        # . Explicitly dimensionless constant.
        # . Fixed price constant.
        # . Fixed volume constant.
        # . Incorrect constant.
        # . Dim-agnostic or dimless series.
        # . Price series, dataframe, or PfLine
        # . Volume series, dataframe, or pfline
        # . Incorrect series, dataframe or pfline.
    ],
    ids=id_fn,
)
def test_pfl_multiplication(pfl_in, value, expected_mul, expected_div, operation):
    """Test if portfolio lines of equal lengths can be multiplied and divided and give correct result."""
    expected = expected_mul if operation == "mul" else expected_div

    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            _ = (pfl_in * value) if operation == "mul" else (pfl_in / value)
        return

    result = (pfl_in * value) if operation == "mul" else (pfl_in / value)
    if isinstance(expected, pd.Series):
        testing.assert_series_equal(result, expected, check_names=False)
    else:
        assert result == expected
