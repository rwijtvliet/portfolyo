from typing import Any, Dict

import pandas as pd
import pytest

import portfolyo as pf
from portfolyo import Q_, Kind, MultiPfLine, PfLine, SinglePfLine, dev, testing  # noqa

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
i = pd.date_range("2020", periods=20, freq="MS", tz=tz)  # reference
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


@pytest.mark.parametrize("operation", ["+", "-"])
@pytest.mark.parametrize(
    ("pfl_in_i", "pfl_in_kind", "value", "returntype", "returnkind"),
    [
        # Adding to volume pfline.
        # . Add volume.
        # . . single value
        (i, Kind.VOLUME_ONLY, Q_(0, "MWh"), PfLine, Kind.VOLUME_ONLY),
        (i, Kind.VOLUME_ONLY, Q_(8.1, "Wh"), PfLine, Kind.VOLUME_ONLY),
        (i, Kind.VOLUME_ONLY, Q_(8.1, "kWh"), PfLine, Kind.VOLUME_ONLY),
        (i, Kind.VOLUME_ONLY, Q_(8.1, "MWh"), PfLine, Kind.VOLUME_ONLY),
        (i, Kind.VOLUME_ONLY, Q_(8.1, "GWh"), PfLine, Kind.VOLUME_ONLY),
        (i, Kind.VOLUME_ONLY, Q_(8.1, "W"), PfLine, Kind.VOLUME_ONLY),
        (i, Kind.VOLUME_ONLY, Q_(8.1, "kW"), PfLine, Kind.VOLUME_ONLY),
        (i, Kind.VOLUME_ONLY, Q_(8.1, "MW"), PfLine, Kind.VOLUME_ONLY),
        (i, Kind.VOLUME_ONLY, Q_(8.1, "GW"), PfLine, Kind.VOLUME_ONLY),
        (i, Kind.VOLUME_ONLY, {"w": Q_(8.1, "GW")}, PfLine, Kind.VOLUME_ONLY),
        (i, Kind.VOLUME_ONLY, {"w": 8.1}, PfLine, Kind.VOLUME_ONLY),
        (i, Kind.VOLUME_ONLY, pd.Series([8.1], ["q"]), PfLine, Kind.VOLUME_ONLY),
        (
            i,
            Kind.VOLUME_ONLY,
            pd.Series([8.1], ["q"]).astype("pint[MWh]"),
            PfLine,
            Kind.VOLUME_ONLY,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            pd.Series([Q_(8.1, "MWh")], ["q"]),
            PfLine,
            Kind.VOLUME_ONLY,
        ),
        (i, Kind.VOLUME_ONLY, pd.Series([8.1], ["w"]), PfLine, Kind.VOLUME_ONLY),
        (
            i,
            Kind.VOLUME_ONLY,
            pd.Series([8.1], ["w"]).astype("pint[MW]"),
            PfLine,
            Kind.VOLUME_ONLY,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            pd.Series([Q_(8.1, "MW")], ["w"]),
            PfLine,
            Kind.VOLUME_ONLY,
        ),
        (i, Kind.VOLUME_ONLY, Q_(8.1, "J"), PfLine, Kind.VOLUME_ONLY),
        # . . timeseries (series, df, pfline)
        (
            i,
            Kind.VOLUME_ONLY,
            dev.get_series(i, "q", _random=False),
            PfLine,
            Kind.VOLUME_ONLY,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            dev.get_series(i, "w", _random=False),
            PfLine,
            Kind.VOLUME_ONLY,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            dev.get_dataframe(i, "q", _random=False),
            PfLine,
            Kind.VOLUME_ONLY,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            dev.get_singlepfline(i, Kind.VOLUME_ONLY, _random=False),
            PfLine,
            Kind.VOLUME_ONLY,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            dev.get_multipfline(i, Kind.VOLUME_ONLY, _random=False),
            PfLine,
            Kind.VOLUME_ONLY,
        ),
        # . Add something else.
        # . . single value
        (i, Kind.VOLUME_ONLY, 0, PfLine, Kind.VOLUME_ONLY),
        (i, Kind.VOLUME_ONLY, 0.0, PfLine, Kind.VOLUME_ONLY),
        (i, Kind.VOLUME_ONLY, None, PfLine, Kind.VOLUME_ONLY),
        (i, Kind.VOLUME_ONLY, 8.1, Exception, None),
        (i, Kind.VOLUME_ONLY, Q_(8.1, ""), Exception, None),
        (i, Kind.VOLUME_ONLY, Q_(8.1, "Eur/MWh"), Exception, None),
        (i, Kind.VOLUME_ONLY, Q_(8.1, "Eur"), Exception, None),
        (i, Kind.VOLUME_ONLY, Q_(8.1, "h"), Exception, None),
        (i, Kind.VOLUME_ONLY, Q_(0.0, "Eur/MWh"), Exception, None),
        (i, Kind.VOLUME_ONLY, {"the_volume": Q_(8.1, "MWh")}, Exception, None),
        # . . timeseries (series, df, pfline)
        (
            i,
            Kind.VOLUME_ONLY,
            pd.DataFrame({"the_volume": dev.get_series(i, "w", _random=False)}),
            Exception,
            None,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            dev.get_series(i, "q", _random=False).pint.magnitude,
            Exception,
            None,
        ),
        (i, Kind.VOLUME_ONLY, dev.get_series(i, "p", _random=False), Exception, None),
        (i, Kind.VOLUME_ONLY, dev.get_series(i, "r", _random=False), Exception, None),
        (
            i,
            Kind.VOLUME_ONLY,
            dev.get_dataframe(i, "p", _random=False),
            Exception,
            None,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            dev.get_dataframe(i, "qr", _random=False),
            Exception,
            None,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            dev.get_dataframe(i, "qp", _random=False),
            Exception,
            None,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            dev.get_singlepfline(i, Kind.PRICE_ONLY, _random=False),
            Exception,
            None,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            dev.get_multipfline(i, Kind.PRICE_ONLY, _random=False),
            Exception,
            None,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            dev.get_singlepfline(i, Kind.ALL, _random=False),
            Exception,
            None,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            dev.get_multipfline(i, Kind.ALL, _random=False),
            Exception,
            None,
        ),
        # Adding to price pfline.
        # . Add dimension-agnostic.
        (i, Kind.PRICE_ONLY, 12, PfLine, Kind.PRICE_ONLY),
        (
            i,
            Kind.PRICE_ONLY,
            dev.get_series(i, "p", _random=False).pint.magnitude,
            PfLine,
            Kind.PRICE_ONLY,
        ),
        # . Add price.
        (i, Kind.PRICE_ONLY, Q_(0, "Eur/MWh"), PfLine, Kind.PRICE_ONLY),
        (i, Kind.PRICE_ONLY, Q_(12.0, "Eur/MWh"), PfLine, Kind.PRICE_ONLY),
        (i, Kind.PRICE_ONLY, Q_(12.0, "Eur/kWh"), PfLine, Kind.PRICE_ONLY),
        (i, Kind.PRICE_ONLY, Q_(12.0, "cent/kWh"), PfLine, Kind.PRICE_ONLY),
        (
            i,
            Kind.PRICE_ONLY,
            dev.get_series(i, "p", _random=False),
            PfLine,
            Kind.PRICE_ONLY,
        ),
        (
            i,
            Kind.PRICE_ONLY,
            dev.get_singlepfline(i, Kind.PRICE_ONLY, _random=False),
            PfLine,
            Kind.PRICE_ONLY,
        ),
        (
            i,
            Kind.PRICE_ONLY,
            dev.get_multipfline(i, Kind.PRICE_ONLY, _random=False),
            PfLine,
            Kind.PRICE_ONLY,
        ),
        # . Add something else.
        (i, Kind.PRICE_ONLY, 0, PfLine, Kind.PRICE_ONLY),
        (i, Kind.PRICE_ONLY, 0.0, PfLine, Kind.PRICE_ONLY),
        (i, Kind.PRICE_ONLY, None, PfLine, Kind.PRICE_ONLY),
        (i, Kind.PRICE_ONLY, Q_(12.0, ""), Exception, None),  # explicitly dimensionless
        (i, Kind.PRICE_ONLY, Q_(12.0, "Eur"), Exception, None),
        (i, Kind.PRICE_ONLY, Q_(12.0, "MWh"), Exception, None),
        (i, Kind.PRICE_ONLY, Q_(12.0, "h"), Exception, None),
        (i, Kind.PRICE_ONLY, Q_(0.0, "MWh"), Exception, None),
        (i, Kind.PRICE_ONLY, dev.get_series(i, "q", _random=False), Exception, None),
        (
            i,
            Kind.PRICE_ONLY,
            dev.get_singlepfline(i, Kind.VOLUME_ONLY, _random=False),
            Exception,
            None,
        ),
        (
            i,
            Kind.PRICE_ONLY,
            dev.get_multipfline(i, Kind.VOLUME_ONLY, _random=False),
            Exception,
            None,
        ),
        (
            i,
            Kind.PRICE_ONLY,
            dev.get_singlepfline(i, Kind.ALL, _random=False),
            Exception,
            None,
        ),
        (
            i,
            Kind.PRICE_ONLY,
            dev.get_multipfline(i, Kind.ALL, _random=False),
            Exception,
            None,
        ),
        # Adding to 'complete' pfline.
        # . Add dimension-agnostic.
        (i, Kind.ALL, 5.9, Exception, None),
        (
            i,
            Kind.ALL,
            dev.get_series(i, "q", _random=False).pint.magnitude,
            Exception,
            None,
        ),
        (i, Kind.ALL, {"nodim": 5.9}, Exception, None),
        (i, Kind.ALL, {"nodim": Q_(5.9, "")}, Exception, None),
        # . Add other 'all' pfline.
        (
            i,
            Kind.ALL,
            dev.get_dataframe(i, "qr", _random=False),
            SinglePfLine,
            Kind.ALL,
        ),
        (
            i,
            Kind.ALL,
            dev.get_dataframe(i, "qp", _random=False),
            SinglePfLine,
            Kind.ALL,
        ),
        (
            i,
            Kind.ALL,
            dev.get_dataframe(i, "pr", _random=False),
            SinglePfLine,
            Kind.ALL,
        ),
        (
            i,
            Kind.ALL,
            dev.get_singlepfline(i, Kind.ALL, _random=False),
            SinglePfLine,
            Kind.ALL,
        ),
        (
            i,
            Kind.ALL,
            dev.get_multipfline(i, Kind.ALL, _random=False),
            PfLine,
            Kind.ALL,
        ),
        # . Add something else.
        (i, Kind.ALL, Q_(6.0, "Eur"), Exception, None),
        (i, Kind.ALL, Q_(6.0, "Eur/MWh"), Exception, None),
        (i, Kind.ALL, Q_(6.0, "MW"), Exception, None),
        (i, Kind.ALL, Q_(6.0, "MWh"), Exception, None),
        (i, Kind.ALL, Q_(6.0, "h"), Exception, None),
        (i, Kind.ALL, dev.get_series(i, "p", _random=False), Exception, None),
        (i, Kind.ALL, dev.get_series(i, "r", _random=False), Exception, None),
        (i, Kind.ALL, dev.get_series(i, "q", _random=False), Exception, None),
        (i, Kind.ALL, dev.get_series(i, "w", _random=False), Exception, None),
        (i, Kind.ALL, dev.get_dataframe(i, "p", _random=False), Exception, None),
        (i, Kind.ALL, dev.get_dataframe(i, "r", _random=False), Exception, None),
        (i, Kind.ALL, dev.get_dataframe(i, "q", _random=False), Exception, None),
        (i, Kind.ALL, dev.get_dataframe(i, "w", _random=False), Exception, None),
        (i, Kind.ALL, dev.get_dataframe(i, "wq", _random=False), Exception, None),
        (
            i,
            Kind.ALL,
            dev.get_singlepfline(i, Kind.PRICE_ONLY, _random=False),
            Exception,
            None,
        ),
        (
            i,
            Kind.ALL,
            dev.get_multipfline(i, Kind.PRICE_ONLY, _random=False),
            Exception,
            None,
        ),
        (
            i,
            Kind.ALL,
            dev.get_singlepfline(i, Kind.VOLUME_ONLY, _random=False),
            Exception,
            None,
        ),
        (
            i,
            Kind.ALL,
            dev.get_multipfline(i, Kind.VOLUME_ONLY, _random=False),
            Exception,
            None,
        ),
    ],
    ids=id_fn,
)
@pytest.mark.parametrize("pfl_in_single_or_multi", ["single", "multi"])
def test_pfl_addsub_kind(
    pfl_in_i,
    pfl_in_kind,
    pfl_in_single_or_multi,
    value,
    returntype,
    returnkind,
    operation,
):
    """Test if addition and subtraction return correct object type and kind."""
    if pfl_in_single_or_multi == "single":
        pfl_in = dev.get_singlepfline(pfl_in_i, pfl_in_kind, _random=False)
    else:
        pfl_in = dev.get_multipfline(pfl_in_i, pfl_in_kind, _random=False)
    # Check error is raised.
    # (Not working due to implementation issue in pint and numpy: value + pfl_in, value - pfl_in)
    if issubclass(returntype, Exception):
        with pytest.raises(returntype):
            _ = (pfl_in + value) if operation == "+" else (pfl_in - value)
        return

    # Check return type.
    # (Not working due to implementation issue in pint and numpy: value + pfl_in, value - pfl_in)
    result = (pfl_in + value) if operation == "+" else (pfl_in - value)
    assert isinstance(result, returntype)
    if returntype is PfLine:
        assert result.kind is returnkind


@pytest.mark.parametrize("operation", ["*", "/"])
@pytest.mark.parametrize(
    (
        "pfl_in_i",
        "pfl_in_kind",
        "value",
        "returntype_mul",
        "returnkind_mul",
        "returntype_div",
        "returnkind_div",
    ),
    [
        # Multiplying volume pfline.
        # . Multiply with dimensionless value
        # . . single value
        (i, Kind.VOLUME_ONLY, 8.1, PfLine, Kind.VOLUME_ONLY, PfLine, Kind.VOLUME_ONLY),
        (
            i,
            Kind.VOLUME_ONLY,
            Q_(8.1, ""),
            PfLine,
            Kind.VOLUME_ONLY,
            PfLine,
            Kind.VOLUME_ONLY,
        ),
        # . . timeseries (series, df)
        (
            i,
            Kind.VOLUME_ONLY,
            dev.get_series(i, "f", _random=False),
            PfLine,
            Kind.VOLUME_ONLY,
            PfLine,
            Kind.VOLUME_ONLY,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            dev.get_series(i, "f", _random=False).astype("pint[dimensionless]"),
            PfLine,
            Kind.VOLUME_ONLY,
            PfLine,
            Kind.VOLUME_ONLY,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            dev.get_dataframe(i, ["nodim"], _random=False),
            PfLine,
            Kind.VOLUME_ONLY,
            PfLine,
            Kind.VOLUME_ONLY,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            dev.get_dataframe(i, ["nodim"], _random=False).astype(
                "pint[dimensionless]"
            ),
            PfLine,
            Kind.VOLUME_ONLY,
            PfLine,
            Kind.VOLUME_ONLY,
        ),
        # . Multiply with volume
        # . . single val
        (i, Kind.VOLUME_ONLY, Q_(8.1, "Wh"), Exception, None, pd.Series, None),
        (i, Kind.VOLUME_ONLY, Q_(8.1, "kWh"), Exception, None, pd.Series, None),
        (i, Kind.VOLUME_ONLY, Q_(8.1, "MWh"), Exception, None, pd.Series, None),
        (i, Kind.VOLUME_ONLY, Q_(8.1, "GWh"), Exception, None, pd.Series, None),
        (i, Kind.VOLUME_ONLY, Q_(8.1, "W"), Exception, None, pd.Series, None),
        (i, Kind.VOLUME_ONLY, Q_(8.1, "kW"), Exception, None, pd.Series, None),
        (i, Kind.VOLUME_ONLY, Q_(8.1, "MW"), Exception, None, pd.Series, None),
        (i, Kind.VOLUME_ONLY, Q_(8.1, "GW"), Exception, None, pd.Series, None),
        (i, Kind.VOLUME_ONLY, {"w": Q_(8.1, "GW")}, Exception, None, pd.Series, None),
        (i, Kind.VOLUME_ONLY, {"w": 8.1}, Exception, None, pd.Series, None),
        (i, Kind.VOLUME_ONLY, Q_(8.1, "J"), Exception, None, pd.Series, None),
        (
            i,
            Kind.VOLUME_ONLY,
            pd.Series([8.1], ["q"]),
            Exception,
            None,
            pd.Series,
            None,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            pd.Series([8.1], ["q"]).astype("pint[MWh]"),
            Exception,
            None,
            pd.Series,
            None,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            pd.Series([Q_(8.1, "MWh")], ["q"]),
            Exception,
            None,
            pd.Series,
            None,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            pd.Series([8.1], ["w"]),
            Exception,
            None,
            pd.Series,
            None,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            pd.Series([8.1], ["w"]).astype("pint[MW]"),
            Exception,
            None,
            pd.Series,
            None,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            pd.Series([Q_(8.1, "MW")], ["w"]),
            Exception,
            None,
            pd.Series,
            None,
        ),
        # . . timeseries (series, df, pfline)
        (
            i,
            Kind.VOLUME_ONLY,
            dev.get_dataframe(i, "q", _random=False),
            Exception,
            None,
            pd.Series,
            None,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            dev.get_pfline(i, Kind.VOLUME_ONLY, _random=False),
            Exception,
            None,
            pd.Series,
            None,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            dev.get_singlepfline(i, Kind.VOLUME_ONLY, _random=False),
            Exception,
            None,
            pd.Series,
            None,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            dev.get_multipfline(i, Kind.VOLUME_ONLY, _random=False),
            Exception,
            None,
            pd.Series,
            None,
        ),
        # . Multiply with price
        # . . single val
        (i, Kind.VOLUME_ONLY, Q_(8.1, "Eur/MWh"), PfLine, Kind.ALL, Exception, None),
        (i, Kind.VOLUME_ONLY, Q_(8.1, "ctEur/kWh"), PfLine, Kind.ALL, Exception, None),
        # . . timeseries (series, df, pfline)
        (
            i,
            Kind.VOLUME_ONLY,
            dev.get_series(i, "p", _random=False),
            PfLine,
            Kind.ALL,
            Exception,
            None,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            dev.get_dataframe(i, "p", _random=False),
            PfLine,
            Kind.ALL,
            Exception,
            None,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            dev.get_singlepfline(i, Kind.PRICE_ONLY, _random=False),
            PfLine,
            Kind.ALL,
            Exception,
            None,
        ),
        (
            i,
            Kind.VOLUME_ONLY,
            dev.get_multipfline(i, Kind.PRICE_ONLY, _random=False),
            PfLine,
            Kind.ALL,
            Exception,
            None,
        ),
        # Multiplying price pfline.
        # . Multiply with dimensionless value
        # . . single value
        (i, Kind.PRICE_ONLY, 8.1, PfLine, Kind.PRICE_ONLY, PfLine, Kind.PRICE_ONLY),
        (
            i,
            Kind.PRICE_ONLY,
            Q_(8.1, ""),
            PfLine,
            Kind.PRICE_ONLY,
            PfLine,
            Kind.PRICE_ONLY,
        ),
        # . . timeseries (series, df)
        (
            i,
            Kind.PRICE_ONLY,
            dev.get_series(i, "f", _random=False),
            PfLine,
            Kind.PRICE_ONLY,
            PfLine,
            Kind.PRICE_ONLY,
        ),
        (
            i,
            Kind.PRICE_ONLY,
            dev.get_series(i, "f", _random=False).astype("pint[dimensionless]"),
            PfLine,
            Kind.PRICE_ONLY,
            PfLine,
            Kind.PRICE_ONLY,
        ),
        (
            i,
            Kind.PRICE_ONLY,
            dev.get_dataframe(i, ["nodim"], _random=False),
            PfLine,
            Kind.PRICE_ONLY,
            PfLine,
            Kind.PRICE_ONLY,
        ),
        (
            i,
            Kind.PRICE_ONLY,
            dev.get_dataframe(i, ["nodim"], _random=False).astype(
                "pint[dimensionless]"
            ),
            PfLine,
            Kind.PRICE_ONLY,
            PfLine,
            Kind.PRICE_ONLY,
        ),
        # . Multiply with volume
        # . . single val
        (i, Kind.PRICE_ONLY, Q_(8.1, "Wh"), PfLine, Kind.ALL, Exception, None),
        (i, Kind.PRICE_ONLY, Q_(8.1, "kWh"), PfLine, Kind.ALL, Exception, None),
        (i, Kind.PRICE_ONLY, Q_(8.1, "MWh"), PfLine, Kind.ALL, Exception, None),
        (i, Kind.PRICE_ONLY, Q_(8.1, "GWh"), PfLine, Kind.ALL, Exception, None),
        (i, Kind.PRICE_ONLY, Q_(8.1, "W"), PfLine, Kind.ALL, Exception, None),
        (i, Kind.PRICE_ONLY, Q_(8.1, "kW"), PfLine, Kind.ALL, Exception, None),
        (i, Kind.PRICE_ONLY, Q_(8.1, "MW"), PfLine, Kind.ALL, Exception, None),
        (i, Kind.PRICE_ONLY, Q_(8.1, "GW"), PfLine, Kind.ALL, Exception, None),
        (i, Kind.PRICE_ONLY, {"w": Q_(8.1, "GW")}, PfLine, Kind.ALL, Exception, None),
        (i, Kind.PRICE_ONLY, {"w": 8.1}, PfLine, Kind.ALL, Exception, None),
        (i, Kind.PRICE_ONLY, Q_(8.1, "J"), PfLine, Kind.ALL, Exception, None),
        (
            i,
            Kind.PRICE_ONLY,
            pd.Series([8.1], ["q"]),
            PfLine,
            Kind.ALL,
            Exception,
            None,
        ),
        (
            i,
            Kind.PRICE_ONLY,
            pd.Series([8.1], ["q"]).astype("pint[MWh]"),
            PfLine,
            Kind.ALL,
            Exception,
            None,
        ),
        (
            i,
            Kind.PRICE_ONLY,
            pd.Series([Q_(8.1, "MWh")], ["q"]),
            PfLine,
            Kind.ALL,
            Exception,
            None,
        ),
        (
            i,
            Kind.PRICE_ONLY,
            pd.Series([8.1], ["w"]),
            PfLine,
            Kind.ALL,
            Exception,
            None,
        ),
        (
            i,
            Kind.PRICE_ONLY,
            pd.Series([8.1], ["w"]).astype("pint[MW]"),
            PfLine,
            Kind.ALL,
            Exception,
            None,
        ),
        (
            i,
            Kind.PRICE_ONLY,
            pd.Series([Q_(8.1, "MW")], ["w"]),
            PfLine,
            Kind.ALL,
            Exception,
            None,
        ),
        # . . timeseries (series, df, pfline)
        (
            i,
            Kind.PRICE_ONLY,
            dev.get_dataframe(i, "q", _random=False),
            PfLine,
            Kind.ALL,
            Exception,
            None,
        ),
        (
            i,
            Kind.PRICE_ONLY,
            dev.get_pfline(i, Kind.VOLUME_ONLY, _random=False),
            PfLine,
            Kind.ALL,
            Exception,
            None,
        ),
        (
            i,
            Kind.PRICE_ONLY,
            dev.get_singlepfline(i, Kind.VOLUME_ONLY, _random=False),
            PfLine,
            Kind.ALL,
            Exception,
            None,
        ),
        (
            i,
            Kind.PRICE_ONLY,
            dev.get_multipfline(i, Kind.VOLUME_ONLY, _random=False),
            PfLine,
            Kind.ALL,
            Exception,
            None,
        ),
        # . Multiply with price
        # . . single val
        (i, Kind.PRICE_ONLY, Q_(8.1, "Eur/MWh"), Exception, None, pd.Series, None),
        (i, Kind.PRICE_ONLY, Q_(8.1, "ctEur/kWh"), Exception, None, pd.Series, None),
        # . . timeseries (series, df, pfline)
        (
            i,
            Kind.PRICE_ONLY,
            dev.get_series(i, "p", _random=False),
            Exception,
            None,
            pd.Series,
            None,
        ),
        (
            i,
            Kind.PRICE_ONLY,
            dev.get_dataframe(i, "p", _random=False),
            Exception,
            None,
            pd.Series,
            None,
        ),
        (
            i,
            Kind.PRICE_ONLY,
            dev.get_singlepfline(i, Kind.PRICE_ONLY, _random=False),
            Exception,
            None,
            pd.Series,
            None,
        ),
        (
            i,
            Kind.PRICE_ONLY,
            dev.get_multipfline(i, Kind.PRICE_ONLY, _random=False),
            Exception,
            None,
            pd.Series,
            None,
        ),
        # Multiplying 'complete' pfline.
        # . Multiply with dimensionless value
        # . . single value
        (
            i,
            Kind.ALL,
            8.1,
            PfLine,
            Kind.ALL,
            PfLine,
            Kind.ALL,
        ),
        (
            i,
            Kind.ALL,
            Q_(8.1, ""),
            PfLine,
            Kind.ALL,
            PfLine,
            Kind.ALL,
        ),
        # . . timeseries (series, df)
        (
            i,
            Kind.ALL,
            dev.get_series(i, "f", _random=False),
            PfLine,
            Kind.ALL,
            PfLine,
            Kind.ALL,
        ),
        (
            i,
            Kind.ALL,
            dev.get_series(i, "f", _random=False).astype("pint[dimensionless]"),
            PfLine,
            Kind.ALL,
            PfLine,
            Kind.ALL,
        ),
        (
            i,
            Kind.ALL,
            dev.get_dataframe(i, ["nodim"], _random=False),
            PfLine,
            Kind.ALL,
            PfLine,
            Kind.ALL,
        ),
        (
            i,
            Kind.ALL,
            dev.get_dataframe(i, ["nodim"], _random=False).astype(
                "pint[dimensionless]"
            ),
            PfLine,
            Kind.ALL,
            PfLine,
            Kind.ALL,
        ),
    ],
    ids=id_fn,
)
@pytest.mark.parametrize("pfl_in_single_or_multi", ["single", "multi"])
def test_pfl_muldiv_kind(
    pfl_in_i,
    pfl_in_kind,
    pfl_in_single_or_multi,
    value,
    returntype_mul,
    returnkind_mul,
    returntype_div,
    returnkind_div,
    operation,
):
    """Test if multiplication and division return correct object type and kind."""
    if pfl_in_single_or_multi == "single":
        pfl_in = dev.get_singlepfline(pfl_in_i, pfl_in_kind, _random=False)
    else:
        pfl_in = dev.get_multipfline(pfl_in_i, pfl_in_kind, _random=False)

    returntype = returntype_mul if operation == "*" else returntype_div
    returnkind = returnkind_mul if operation == "*" else returnkind_div

    # if returntype is PfLine:
    #     returntype = SinglePfLine if pfl_in_single_or_multi == "single" else MultiPfLine

    if issubclass(returntype, Exception):
        with pytest.raises(returntype):
            _ = (pfl_in * value) if operation == "*" else (pfl_in / value)
        return

    # Check return type.
    # (Not working due to implementation issue in pint and numpy: value * pfl_in, value / pfl_in)
    result = (pfl_in * value) if operation == "*" else (pfl_in / value)
    assert isinstance(result, returntype)
    if returntype_mul is PfLine:
        assert result.kind is returnkind


i = pd.date_range("2020", freq="MS", periods=3, tz=tz)
series1 = {
    "w": pd.Series([3.0, 5, -4], i),
    "p": pd.Series([200.0, 100, 50], i),
    "r": pd.Series([446400.0, 348000, -148600], i),
}
pflset1 = {
    Kind.VOLUME_ONLY: pf.SinglePfLine({"w": series1["w"]}),
    Kind.PRICE_ONLY: pf.SinglePfLine({"p": series1["p"]}),
    Kind.ALL: pf.SinglePfLine({"w": series1["w"], "p": series1["p"]}),
}
series2 = {
    "w": pd.Series([15.0, -20, 4], i),
    "p": pd.Series([400.0, 50, 50], i),
    "r": pd.Series([4464000.0, -696000, 148600], i),
    "nodim": pd.Series([2, -1.5, 10], i),
}
pflset2 = {
    Kind.VOLUME_ONLY: pf.SinglePfLine({"w": series2["w"]}),
    Kind.PRICE_ONLY: pf.SinglePfLine({"p": series2["p"]}),
    Kind.ALL: pf.SinglePfLine({"w": series2["w"], "p": series2["p"]}),
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
div_price1_price2 = (series1["p"] / series2["p"]).astype("pint[dimensionless]")
mul_all1_dimless2 = pf.SinglePfLine(
    {"w": series1["w"] * series2["nodim"], "p": series1["p"]}
)
div_all1_dimless2 = pf.SinglePfLine(
    {"w": series1["w"] / series2["nodim"], "p": series1["p"]}
)


@pytest.mark.parametrize(
    ("pfl_in", "expected"),
    [
        (pflset1[Kind.VOLUME_ONLY], neg_volume_pfl1),
        (pflset1[Kind.PRICE_ONLY], neg_price_pfl1),
        (pflset1[Kind.ALL], neg_all_pfl1),
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
        # Adding to volume pfline.
        # . Add constant without unit.
        (
            pflset1[Kind.VOLUME_ONLY],
            0,
            pflset1[Kind.VOLUME_ONLY],
            pflset1[Kind.VOLUME_ONLY],
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            0.0,
            pflset1[Kind.VOLUME_ONLY],
            pflset1[Kind.VOLUME_ONLY],
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            None,
            pflset1[Kind.VOLUME_ONLY],
            pflset1[Kind.VOLUME_ONLY],
        ),
        # . Add constant with unit.
        (
            pflset1[Kind.VOLUME_ONLY],
            Q_(12.0, "MW"),
            pf.SinglePfLine({"w": pd.Series([15.0, 17, 8], i)}),
            pf.SinglePfLine({"w": pd.Series([-9.0, -7, -16], i)}),
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            {"w": Q_(12.0, "MW")},
            pf.SinglePfLine({"w": pd.Series([15.0, 17, 8], i)}),
            pf.SinglePfLine({"w": pd.Series([-9.0, -7, -16], i)}),
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            {"w": 12.0},
            pf.SinglePfLine({"w": pd.Series([15.0, 17, 8], i)}),
            pf.SinglePfLine({"w": pd.Series([-9.0, -7, -16], i)}),
        ),
        # . Add constant in different unit
        (
            pflset1[Kind.VOLUME_ONLY],
            Q_(0.012, "GW"),
            pf.SinglePfLine({"w": pd.Series([15.0, 17, 8], i)}),
            pf.SinglePfLine({"w": pd.Series([-9.0, -7, -16], i)}),
        ),
        # . Add constant in different dimension.
        (
            pflset1[Kind.VOLUME_ONLY],
            Q_(12.0, "MWh"),
            pf.SinglePfLine({"q": pd.Series([2244.0, 3492, -2960], i)}),
            pf.SinglePfLine({"q": pd.Series([2220.0, 3468, -2984], i)}),
        ),
        # . Add series without unit.
        (
            pflset1[Kind.VOLUME_ONLY],
            series2["w"],
            ValueError,
            ValueError,
        ),
        # . Add series without name.
        (
            pflset1[Kind.VOLUME_ONLY],
            series2["w"].astype("pint[MW]"),
            add_volume_pfl,
            sub_volume_pfl,
        ),
        # . Add series with useless name.
        (
            pflset1[Kind.VOLUME_ONLY],
            series2["w"].rename("the_volume").astype("pint[MW]"),
            add_volume_pfl,
            sub_volume_pfl,
        ),
        # . Add series without name and with different unit
        (
            pflset1[Kind.VOLUME_ONLY],
            (series2["w"] * 1000).astype("pint[kW]"),
            add_volume_pfl,
            sub_volume_pfl,
        ),
        # . Add series out of order.
        (
            pflset1[Kind.VOLUME_ONLY],
            pd.Series([15.0, 4, -20], [i[0], i[2], i[1]]).astype("pint[MW]"),
            ValueError,
            ValueError,
        ),
        # . Add dataframe without unit.
        (
            pflset1[Kind.VOLUME_ONLY],
            pd.DataFrame({"w": series2["w"]}),
            add_volume_pfl,
            sub_volume_pfl,
        ),
        # . Add other pfline.
        (
            pflset1[Kind.VOLUME_ONLY],
            pflset2[Kind.VOLUME_ONLY],
            add_volume_pfl,
            sub_volume_pfl,
        ),
        # Adding to price pfline.
        # . Add constant without unit.
        (
            pflset1[Kind.PRICE_ONLY],
            0,
            pflset1[Kind.PRICE_ONLY],
            pflset1[Kind.PRICE_ONLY],
        ),
        (
            pflset1[Kind.PRICE_ONLY],
            0.0,
            pflset1[Kind.PRICE_ONLY],
            pflset1[Kind.PRICE_ONLY],
        ),
        (
            pflset1[Kind.PRICE_ONLY],
            None,
            pflset1[Kind.PRICE_ONLY],
            pflset1[Kind.PRICE_ONLY],
        ),
        (
            pflset1[Kind.PRICE_ONLY],
            12.0,
            pf.SinglePfLine({"p": pd.Series([212.0, 112, 62], i)}),
            pf.SinglePfLine({"p": pd.Series([188.0, 88, 38], i)}),
        ),
        # . Add constant with default unit.
        (
            pflset1[Kind.PRICE_ONLY],
            Q_(12.0, "Eur/MWh"),
            pf.SinglePfLine({"p": pd.Series([212.0, 112, 62], i)}),
            pf.SinglePfLine({"p": pd.Series([188.0, 88, 38], i)}),
        ),
        # . Add constant with non-default unit.
        (
            pflset1[Kind.PRICE_ONLY],
            Q_(1.2, "ct/kWh"),
            pf.SinglePfLine({"p": pd.Series([212.0, 112, 62], i)}),
            pf.SinglePfLine({"p": pd.Series([188.0, 88, 38], i)}),
        ),
        # . Add other pfline.
        (
            pflset1[Kind.PRICE_ONLY],
            pflset2[Kind.PRICE_ONLY],
            add_price_pfl,
            sub_price_pfl,
        ),
        # Adding to full pfline.
        # . Add constant without unit.
        (
            pflset1[Kind.ALL],
            0,
            pflset1[Kind.ALL],
            pflset1[Kind.ALL],
        ),
        (
            pflset1[Kind.ALL],
            0.0,
            pflset1[Kind.ALL],
            pflset1[Kind.ALL],
        ),
        (
            pflset1[Kind.ALL],
            None,
            pflset1[Kind.ALL],
            pflset1[Kind.ALL],
        ),
        (
            pflset1[Kind.ALL],
            12.0,
            ValueError,
            ValueError,
        ),
        # . Add series without unit.
        (
            pflset1[Kind.ALL],
            series2["w"],
            ValueError,
            ValueError,
        ),
        # . Add dataframe.
        (
            pflset1[Kind.ALL],
            pd.DataFrame({"w": series2["w"], "p": series2["p"]}),
            add_all_pfl,
            sub_all_pfl,
        ),
        # . Add dataframe.
        (
            pflset1[Kind.ALL],
            pd.DataFrame({"w": series2["w"], "r": series2["r"]}),
            add_all_pfl,
            sub_all_pfl,
        ),
        # . Add other pfline.
        (
            pflset1[Kind.ALL],
            pflset2[Kind.ALL],
            add_all_pfl,
            sub_all_pfl,
        ),
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
            pflset1[Kind.VOLUME_ONLY],
            4.0,
            pf.SinglePfLine({"w": series1["w"] * 4}),
            pf.SinglePfLine({"w": series1["w"] / 4}),
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            {"agn": 4.0},
            pf.SinglePfLine({"w": series1["w"] * 4}),
            pf.SinglePfLine({"w": series1["w"] / 4}),
        ),
        # . Explicitly dimensionless constant.
        (
            pflset1[Kind.VOLUME_ONLY],
            Q_(4.0, ""),
            pf.SinglePfLine({"w": series1["w"] * 4}),
            pf.SinglePfLine({"w": series1["w"] / 4}),
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            {"nodim": Q_(4.0, "")},
            pf.SinglePfLine({"w": series1["w"] * 4}),
            pf.SinglePfLine({"w": series1["w"] / 4}),
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            {"nodim": 4.0},
            pf.SinglePfLine({"w": series1["w"] * 4}),
            pf.SinglePfLine({"w": series1["w"] / 4}),
        ),
        # . Fixed price constant.
        (
            pflset1[Kind.VOLUME_ONLY],
            Q_(4.0, "Eur/MWh"),
            pf.SinglePfLine({"w": series1["w"], "p": 4}),
            Exception,
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            {"p": Q_(4.0, "Eur/MWh")},
            pf.SinglePfLine({"w": series1["w"], "p": 4}),
            Exception,
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            {"p": Q_(0.4, "ctEur/kWh")},
            pf.SinglePfLine({"w": series1["w"], "p": 4}),
            Exception,
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            {"p": 4.0},
            pf.SinglePfLine({"w": series1["w"], "p": 4}),
            Exception,
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            pd.Series([Q_(4.0, "Eur/MWh")], ["p"]),
            pf.SinglePfLine({"w": series1["w"], "p": 4}),
            Exception,
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            pd.Series([4.0], ["p"]),
            pf.SinglePfLine({"w": series1["w"], "p": 4}),
            Exception,
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            pd.Series([4.0], ["p"]).astype("pint[Eur/MWh]"),
            pf.SinglePfLine({"w": series1["w"], "p": 4}),
            Exception,
        ),
        # . Fixed volume constant.
        (
            pflset1[Kind.VOLUME_ONLY],
            {"w": Q_(4.0, "MW")},
            Exception,
            (series1["w"] / 4).astype("pint[dimensionless]"),
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            {"w": 4.0},
            Exception,
            (series1["w"] / 4).astype("pint[dimensionless]"),
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            pd.Series([Q_(4.0, "MW")], ["w"]),
            Exception,
            (series1["w"] / 4).astype("pint[dimensionless]"),
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            pd.Series([4.0], ["w"]).astype("pint[MW]"),
            Exception,
            (series1["w"] / 4).astype("pint[dimensionless]"),
        ),
        (  # divide by fixed ENERGY not POWER
            pflset1[Kind.VOLUME_ONLY],
            pd.Series([4.0], ["q"]).astype("pint[MWh]"),
            Exception,
            (pflset1[Kind.VOLUME_ONLY].q.pint.m / 4).astype("pint[dimensionless]"),
        ),
        # . Constant with incorrect unit.
        (
            pflset1[Kind.VOLUME_ONLY],
            {"r": 4.0},
            Exception,
            Exception,
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            {"q": 4.0, "w": 8.0},  # incompatible
            Exception,
            Exception,
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            Q_(4.0, "Eur"),
            Exception,
            Exception,
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            {"r": 4.0, "q": 12},
            Exception,
            Exception,
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            {"r": 4.0, "nodim": 4.0},
            Exception,
            Exception,
        ),
        # . Dim-agnostic or dimless series.
        (
            pflset1[Kind.VOLUME_ONLY],
            series2["w"],  # has no unit
            pf.SinglePfLine({"w": series1["w"] * series2["w"]}),
            pf.SinglePfLine({"w": series1["w"] / series2["w"]}),
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            series2["w"].astype("pint[dimensionless]"),  # dimensionless
            pf.SinglePfLine({"w": series1["w"] * series2["w"]}),
            pf.SinglePfLine({"w": series1["w"] / series2["w"]}),
        ),
        # . Price series, dataframe, or PfLine
        (
            pflset1[Kind.VOLUME_ONLY],
            series2["p"].astype("pint[Eur/MWh]"),
            mul_volume1_price2,
            Exception,
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            series2["p"].rename("the_price").astype("pint[Eur/MWh]"),
            mul_volume1_price2,
            Exception,
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            (series2["p"] * 0.1).astype("pint[ct/kWh]"),
            mul_volume1_price2,
            Exception,
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            pd.DataFrame({"p": series2["p"]}),
            mul_volume1_price2,
            Exception,
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            pd.DataFrame({"p": (series2["p"] / 10).astype("pint[ct/kWh]")}),
            mul_volume1_price2,
            Exception,
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            pflset2[Kind.PRICE_ONLY],
            mul_volume1_price2,
            Exception,
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            pd.Series([50.0, 400, 50], [i[1], i[0], i[2]]).astype(
                "pint[Eur/MWh]"
            ),  # not standardized
            ValueError,
            ValueError,
        ),
        # . Volume series, dataframe, or pfline
        (
            pflset1[Kind.VOLUME_ONLY],
            series2["w"].astype("pint[MW]"),
            Exception,
            div_volume1_volume2,
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            pflset2[Kind.VOLUME_ONLY],
            Exception,
            div_volume1_volume2,
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            pflset2[Kind.VOLUME_ONLY].q,
            Exception,
            div_volume1_volume2,
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            {"w": series2["w"]},
            Exception,
            div_volume1_volume2,
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            pd.DataFrame({"w": series2["w"]}),
            Exception,
            div_volume1_volume2,
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            pflset2[Kind.VOLUME_ONLY],  # other pfline
            Exception,
            div_volume1_volume2,
        ),
        # . Incorrect series, dataframe or pfline.
        (
            pflset1[Kind.VOLUME_ONLY],
            series2["r"].astype("pint[Eur]"),
            Exception,
            Exception,
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            pd.DataFrame({"r": series2["r"]}),
            Exception,
            Exception,
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            pd.DataFrame({"the_price": series2["p"].astype("pint[Eur/MWh]")}),
            KeyError,
            KeyError,
        ),
        (
            pflset1[Kind.VOLUME_ONLY],
            pflset2[Kind.ALL],
            Exception,
            Exception,
        ),
        # Multiplying price pfline.
        # . (dimension-agnostic) constant.
        (
            pflset1[Kind.PRICE_ONLY],
            4,
            pf.SinglePfLine({"p": series1["p"] * 4}),
            pf.SinglePfLine({"p": series1["p"] / 4}),
        ),
        # . Explicitly dimensionless constant.
        (
            pflset1[Kind.PRICE_ONLY],
            Q_(4, ""),
            pf.SinglePfLine({"p": series1["p"] * 4}),
            pf.SinglePfLine({"p": series1["p"] / 4}),
        ),
        # . Fixed price constant.
        (
            pflset1[Kind.PRICE_ONLY],
            Q_(4, "Eur/MWh"),
            Exception,
            (series1["p"] / 4).astype("pint[dimensionless]"),
        ),
        (
            pflset1[Kind.PRICE_ONLY],
            {"p": 4},
            Exception,
            (series1["p"] / 4).astype("pint[dimensionless]"),
        ),
        # . Fixed volume constant.
        (
            pflset1[Kind.PRICE_ONLY],
            Q_(4, "MWh"),
            pf.SinglePfLine({"p": series1["p"], "q": 4}),
            Exception,
        ),
        (
            pflset1[Kind.PRICE_ONLY],
            Q_(4, "MW"),
            pf.SinglePfLine({"p": series1["p"], "w": 4}),
            Exception,
        ),
        (
            pflset1[Kind.PRICE_ONLY],
            Q_(4, "GW"),
            pf.SinglePfLine({"p": series1["p"], "w": 4000}),
            Exception,
        ),
        (
            pflset1[Kind.PRICE_ONLY],
            pd.Series([4], ["w"]).astype("pint[GW]"),
            pf.SinglePfLine({"p": series1["p"], "w": 4000}),
            Exception,
        ),
        # . Incorrect constant.
        (
            pflset1[Kind.PRICE_ONLY],
            Q_(4, "Eur"),
            ValueError,
            ValueError,
        ),
        # . Dim-agnostic or dimless series.
        (
            pflset1[Kind.PRICE_ONLY],
            series2["w"],  # has no unit
            pf.SinglePfLine({"p": series1["p"] * series2["w"]}),
            pf.SinglePfLine({"p": series1["p"] / series2["w"]}),
        ),
        (
            pflset1[Kind.PRICE_ONLY],
            series2["w"].astype("pint[dimensionless]"),  # dimensionless
            pf.SinglePfLine({"p": series1["p"] * series2["w"]}),
            pf.SinglePfLine({"p": series1["p"] / series2["w"]}),
        ),
        # . Price series, dataframe, or PfLine
        (
            pflset1[Kind.PRICE_ONLY],
            series2["p"].astype("pint[Eur/MWh]"),  # series
            Exception,
            div_price1_price2,
        ),
        (
            pflset1[Kind.PRICE_ONLY],
            pflset2[Kind.PRICE_ONLY],  # pfline
            Exception,
            div_price1_price2,
        ),
        (
            pflset1[Kind.PRICE_ONLY],
            pflset2[Kind.PRICE_ONLY].p,  # series
            Exception,
            div_price1_price2,
        ),
        (
            pflset1[Kind.PRICE_ONLY],
            {"p": series2["p"]},  # dict of series
            Exception,
            div_price1_price2,
        ),
        (
            pflset1[Kind.PRICE_ONLY],
            pd.DataFrame({"p": series2["p"]}),  # dataframe
            Exception,
            div_price1_price2,
        ),
        (
            pflset1[Kind.PRICE_ONLY],
            pflset2[Kind.PRICE_ONLY],  # other pfline
            Exception,
            div_price1_price2,
        ),
        # . Volume series, dataframe, or pfline
        (
            pflset1[Kind.PRICE_ONLY],
            series2["w"].astype("pint[MW]"),
            mul_volume2_price1,
            Exception,
        ),
        (
            pflset1[Kind.PRICE_ONLY],
            (series2["w"] / 1000).astype("pint[GW]"),
            mul_volume2_price1,
            Exception,
        ),
        (
            pflset1[Kind.PRICE_ONLY],
            series2["w"].rename("the_volume").astype("pint[MW]"),
            mul_volume2_price1,
            Exception,
        ),
        (
            pflset1[Kind.PRICE_ONLY],
            pd.DataFrame({"w": series2["w"]}),
            mul_volume2_price1,
            Exception,
        ),
        (
            pflset1[Kind.PRICE_ONLY],
            pd.DataFrame({"w": (series2["w"] / 1000).astype("pint[GW]")}),
            mul_volume2_price1,
            Exception,
        ),
        (
            pflset1[Kind.PRICE_ONLY],
            pflset2[Kind.VOLUME_ONLY],
            mul_volume2_price1,
            Exception,
        ),
        # . Incorrect series, dataframe or pfline.
        (
            pflset1[Kind.PRICE_ONLY],
            series2["r"].astype("pint[Eur]"),
            Exception,
            Exception,
        ),
        (
            pflset1[Kind.PRICE_ONLY],
            pd.DataFrame({"r": series2["r"]}),
            Exception,
            Exception,
        ),
        (
            pflset1[Kind.PRICE_ONLY],
            pd.DataFrame({"the_price": series2["p"].astype("pint[Eur/MWh]")}),
            KeyError,
            KeyError,
        ),
        (
            pflset1[Kind.PRICE_ONLY],
            pflset2[Kind.ALL],
            Exception,
            Exception,
        ),
        # Multiplying 'complete' pfline.
        # . (dimension-agnostic) constant.
        (
            pflset1[Kind.ALL],
            6,
            SinglePfLine({"w": series1["w"] * 6, "p": series1["p"]}),
            SinglePfLine({"w": series1["w"] / 6, "p": series1["p"]}),
        ),
        # . Explicitly dimensionless constant.
        (
            pflset1[Kind.ALL],
            Q_(6, ""),
            SinglePfLine({"w": series1["w"] * 6, "p": series1["p"]}),
            SinglePfLine({"w": series1["w"] / 6, "p": series1["p"]}),
        ),
        (
            pflset1[Kind.ALL],
            {"nodim": 6},
            SinglePfLine({"w": series1["w"] * 6, "p": series1["p"]}),
            SinglePfLine({"w": series1["w"] / 6, "p": series1["p"]}),
        ),
        (
            pflset1[Kind.ALL],
            pd.Series([6], ["nodim"]),
            SinglePfLine({"w": series1["w"] * 6, "p": series1["p"]}),
            SinglePfLine({"w": series1["w"] / 6, "p": series1["p"]}),
        ),
        # . Incorrect constant.
        (
            pflset1[Kind.ALL],
            {"r": 4.0},
            Exception,
            Exception,
        ),
        (
            pflset1[Kind.ALL],
            {"q": 4.0, "w": 8.0},
            Exception,
            Exception,
        ),
        (
            pflset1[Kind.ALL],
            Q_(4.0, "Eur"),
            Exception,
            Exception,
        ),
        (
            pflset1[Kind.ALL],
            {"r": 4.0, "q": 12},
            Exception,
            Exception,
        ),
        (
            pflset1[Kind.ALL],
            {"r": 4.0, "nodim": 4.0},
            Exception,
            Exception,
        ),
        # . Dim-agnostic or dimless series.
        (
            pflset1[Kind.ALL],
            series2["nodim"],  # dim-agnostic
            mul_all1_dimless2,
            div_all1_dimless2,
        ),
        (
            pflset1[Kind.ALL],
            series2["nodim"].astype("pint[dimensionless]"),  # dimless
            mul_all1_dimless2,
            div_all1_dimless2,
        ),
        (
            pflset1[Kind.ALL],
            {"nodim": series2["nodim"]},
            mul_all1_dimless2,
            div_all1_dimless2,
        ),
        (
            pflset1[Kind.ALL],
            pd.DataFrame({"nodim": series2["nodim"]}),
            mul_all1_dimless2,
            div_all1_dimless2,
        ),
        (
            pflset1[Kind.ALL],
            pd.DataFrame({"nodim": series2["nodim"].astype("pint[dimensionless]")}),
            mul_all1_dimless2,
            div_all1_dimless2,
        ),
        # . Incorrect series, dataframe or pfline.
        (
            pflset1[Kind.ALL],
            {"r": series2["p"]},
            Exception,
            Exception,
        ),
        (
            pflset1[Kind.ALL],
            series2["p"].astype("pint[Eur/MWh]"),
            Exception,
            Exception,
        ),
        (
            pflset1[Kind.ALL],
            pflset2[Kind.PRICE_ONLY],
            Exception,
            Exception,
        ),
        (
            pflset1[Kind.ALL],
            pflset2[Kind.VOLUME_ONLY],
            Exception,
            Exception,
        ),
        (
            pflset1[Kind.ALL],
            pflset2[Kind.ALL],
            Exception,
            Exception,
        ),
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
