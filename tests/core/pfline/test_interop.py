from typing import Dict

import numpy as np
import pandas as pd
import pytest
from pint import DimensionalityError, UndefinedUnitError

from portfolyo.core.pfline import interop as io
from portfolyo.tools.unit import Q_

idx1 = pd.date_range("2020", freq="MS", periods=12)
val1 = 100 + 20 * np.random.random(len(idx1))
s1 = pd.Series(val1, idx1)
idx2 = pd.date_range("2020-08", freq="MS", periods=12)
val2 = 200 + 50 * np.random.random(len(idx2))
s2 = pd.Series(val2, idx2)
idx_i = idx1.intersection(idx2).sort_values()
s1_i = s1.loc[idx_i]
s2_i = s2.loc[idx_i]
idx_u = idx1.union(idx2).sort_values()
s1_u = pd.Series((s1.get(i) for i in idx_u), idx_u)
s2_u = pd.Series((s2.get(i) for i in idx_u), idx_u)


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
    if isinstance(data, io.InOp):
        return ""
    return str(data)


@pytest.mark.parametrize(
    ("data_in", "expected_io", "expected_io2"),
    [
        # One value
        # . unit-agnostic
        (23.0, io.InOp(agn=23.0), ValueError),
        # . unitless
        (
            Q_(23.0, ""),
            io.InOp(nodim=23.0),
            ValueError,
        ),
        # . known unit
        (
            Q_(-120.0, "MW"),
            io.InOp(w=-120),
            ValueError,
        ),
        (
            Q_(120e-3, "GW"),
            io.InOp(w=120),
            ValueError,
        ),
        (
            Q_(432e9, "J/h"),
            io.InOp(w=120),
            ValueError,
        ),
        (
            Q_(90_000.0, "MWh"),
            io.InOp(q=90_000),
            ValueError,
        ),
        (
            Q_(90.0, "GWh"),
            io.InOp(q=90_000),
            ValueError,
        ),
        (
            Q_(50.0, "Eur/MWh"),
            io.InOp(p=50),
            ValueError,
        ),
        (
            Q_(5.0, "ctEur/kWh"),
            io.InOp(p=50),
            ValueError,
        ),
        (
            Q_(4_500_000.0, "Eur"),
            io.InOp(r=4_500_000),
            ValueError,
        ),
        (
            Q_(4.5, "MEur"),
            io.InOp(r=4_500_000),
            ValueError,
        ),
        # . unknown unit
        (
            Q_(4.5, "MWh/Eur"),
            UndefinedUnitError,
            None,
        ),
        # One or several values
        # . name but no unit
        (
            {"nodim": 120.0},
            io.InOp(nodim=120),
            ValueError,
        ),
        (
            pd.Series({"nodim": 120.0}),
            io.InOp(nodim=120),
            ValueError,
        ),
        (
            {"w": 120.0},
            io.InOp(w=120),
            ValueError,
        ),
        (
            pd.Series({"w": 120.0}),
            io.InOp(w=120),
            ValueError,
        ),
        (
            {"q": -90_000.0},
            io.InOp(q=-90_000),
            ValueError,
        ),
        (
            pd.Series({"q": -90_000.0}),
            io.InOp(q=-90_000),
            ValueError,
        ),
        (
            {"p": 50.0},
            io.InOp(p=50),
            ValueError,
        ),
        (
            pd.Series({"p": 50.0}),
            io.InOp(p=50),
            ValueError,
        ),
        (
            {"r": 4.5e6},
            io.InOp(r=4_500_000),
            ValueError,
        ),
        (
            pd.Series({"r": 4.5e6}),
            io.InOp(r=4_500_000),
            ValueError,
        ),
        (
            {"w": 120.0, "q": -90_000},
            io.InOp(w=120, q=-90_000),
            ValueError,
        ),
        (
            pd.Series({"w": 120.0, "q": -90_000}),
            io.InOp(w=120.0, q=-90_000),
            ValueError,
        ),
        (
            {"w": 120.0, "p": 50},
            io.InOp(w=120.0, p=50),
            ValueError,
        ),
        (
            pd.Series({"w": 120.0, "p": 50}),
            io.InOp(w=120.0, p=50),
            ValueError,
        ),
        (
            {"w": 120.0, "p": 50.0, "r": 4.5e6},
            io.InOp(w=120.0, p=50.0, r=4.5e6),
            ValueError,
        ),
        (
            pd.Series({"w": 120.0, "p": 50.0, "r": 4.5e6}),
            io.InOp(w=120.0, p=50.0, r=4.5e6),
            ValueError,
        ),
        (
            {"w": 120.0, "p": 50.0, "r": 4.5e6},
            io.InOp(w=120.0, p=50.0, r=4.5e6),
            ValueError,
        ),
        (
            pd.Series({"w": 120.0, "p": 50.0, "r": 4.5e6}),
            io.InOp(w=120.0, p=50.0, r=4.5e6),
            ValueError,
        ),
        # . name and correct unit
        (
            {"p": Q_(50.0, "Eur/MWh")},
            io.InOp(p=50),
            ValueError,
        ),
        (
            pd.Series({"p": Q_(50.0, "Eur/MWh")}),
            io.InOp(p=50),
            ValueError,
        ),
        (
            pd.Series({"p": 50}).astype("pint[Eur/MWh]"),
            io.InOp(p=50),
            ValueError,
        ),
        (
            {"r": Q_(4.5, "MEur")},
            io.InOp(r=4_500_000),
            ValueError,
        ),
        (
            pd.Series({"r": Q_(4.5, "MEur")}),
            io.InOp(r=4_500_000),
            ValueError,
        ),
        (
            pd.Series({"r": 4.5}).astype("pint[MEur]"),
            io.InOp(r=4_500_000),
            ValueError,
        ),
        (
            {"w": 120.0, "q": Q_(-90_000.0, "MWh")},
            io.InOp(w=120.0, q=-90_000),
            ValueError,
        ),
        (
            pd.Series({"w": 120.0, "q": Q_(-90_000.0, "MWh")}),
            io.InOp(w=120.0, q=-90_000),
            ValueError,
        ),
        (
            pd.Series({"w": 120.0, "q": Q_(-90.0, "GWh")}),
            io.InOp(w=120.0, q=-90_000),
            ValueError,
        ),
        # . unknown name -> KeyError
        (
            {"z": 28.0},
            KeyError,
            None,
        ),
        (
            pd.Series({"z": 28.0}),
            KeyError,
            None,
        ),
        (
            {"z": Q_(120.0, "MWh")},
            KeyError,
            None,
        ),
        (
            pd.Series({"z": Q_(120.0, "MWh")}),
            KeyError,
            None,
        ),
        # . mix of know and unknown names -> KeyError
        (
            {"w": 120.0, "z": 28.0},
            KeyError,
            None,
        ),
        (
            pd.Series({"w": 120.0, "z": 28.0}),
            KeyError,
            None,
        ),
        (
            {"w": 120.0, "p": 50.0, "z": 28.0},
            KeyError,
            None,
        ),
        (
            pd.Series({"w": 120.0, "p": 50.0, "z": 28.0}),
            KeyError,
            None,
        ),
        # . combination of name with incorrect unit -> error
        (
            {"w": Q_(90.0, "MWh")},
            DimensionalityError,
            None,
        ),
        (
            pd.Series({"w": Q_(90.0, "MWh")}),
            DimensionalityError,
            None,
        ),
        (
            pd.Series({"w": 90}).astype("pint[MWh]"),
            DimensionalityError,
            None,
        ),
        (
            {"p": Q_(90.0, "MWh")},
            DimensionalityError,
            None,
        ),
        (
            pd.Series({"p": Q_(90.0, "MWh")}),
            DimensionalityError,
            None,
        ),
        (
            {"p": 50.0, "w": Q_(90.0, "MWh")},
            DimensionalityError,
            None,
        ),
        (
            pd.Series({"p": 50.0, "w": Q_(90.0, "MWh")}),
            DimensionalityError,
            None,
        ),
        # One timeseries
        # . unit-agnostic
        (
            s1,
            io.InOp(agn=s1),
            io.InOp(agn=s1),
        ),
        # . unitless
        # (s1.astype("pint[dimensionless]"), io.InterOp(nodim=s1)), # TODO: fix
        # . known unit
        (
            s1.astype("pint[MW]"),
            io.InOp(w=s1),
            io.InOp(w=s1),
        ),
        (
            (s1 / 1000).astype("pint[GW]"),  # series with pint unit
            io.InOp(w=s1),
            io.InOp(w=s1),
        ),
        (
            pd.Series([Q_(v, "MW") for v in val1], idx1),  # series of Quantities
            io.InOp(w=s1),
            io.InOp(w=s1),
        ),
        (
            s1.astype("pint[GWh]"),
            io.InOp(q=s1 * 1000),
            io.InOp(q=s1 * 1000),
        ),
        (
            s1.astype("pint[Eur/MWh]"),
            io.InOp(p=s1),
            io.InOp(p=s1),
        ),
        (
            s1.astype("pint[MEur]"),
            io.InOp(r=s1 * 1e6),
            io.InOp(r=s1 * 1e6),
        ),
        # . unknown unit
        (
            s1.astype("pint[Wh/MEur]"),
            UndefinedUnitError,
            None,
        ),
        # One or several timeseries
        # . name but no unit
        (
            {"w": s1},
            io.InOp(w=s1),
            io.InOp(w=s1),
        ),
        (
            pd.DataFrame({"w": s1}),
            io.InOp(w=s1),
            io.InOp(w=s1),
        ),
        (
            {"q": -s1},
            io.InOp(q=-s1),
            io.InOp(q=-s1),
        ),
        (
            pd.DataFrame({"q": -s1}),
            io.InOp(q=-s1),
            io.InOp(q=-s1),
        ),
        (
            {"r": s1},
            io.InOp(r=s1),
            io.InOp(r=s1),
        ),
        (
            pd.DataFrame({"r": s1}),
            io.InOp(r=s1),
            io.InOp(r=s1),
        ),
        (
            {"w": s1, "q": -s2},
            io.InOp(w=s1, q=-s2),
            io.InOp(w=s1_i, q=-s2_i),
        ),
        (
            pd.DataFrame({"w": s1, "q": -s2}),
            io.InOp(w=s1_u, q=-s2_u),
            io.InOp(w=s1_u, q=-s2_u),
        ),
        (
            {"w": s1, "p": s2, "r": s1 * 4},
            io.InOp(w=s1, p=s2, r=s1 * 4),
            io.InOp(w=s1_i, p=s2_i, r=s1_i * 4),
        ),
        (
            pd.DataFrame({"w": s1, "p": s2, "r": s1 * 4}),
            io.InOp(w=s1_u, p=s2_u, r=s1_u * 4),
            io.InOp(w=s1_u, p=s2_u, r=s1_u * 4),
        ),
        # . name and correct unit
        (
            {"p": s1.astype("pint[Eur/MWh]")},
            io.InOp(p=s1),
            io.InOp(p=s1),
        ),
        (
            pd.DataFrame({"p": s1.astype("pint[Eur/MWh]")}),
            io.InOp(p=s1),
            io.InOp(p=s1),
        ),
        (
            pd.DataFrame({"p": [Q_(v, "Eur/MWh") for v in val1]}, idx1),
            io.InOp(p=s1),
            io.InOp(p=s1),
        ),
        (
            {"r": s1.astype("pint[MEur]")},
            io.InOp(r=s1 * 1e6),
            io.InOp(r=s1 * 1e6),
        ),
        (
            pd.DataFrame({"r": s1.astype("pint[MEur]")}),
            io.InOp(r=s1 * 1e6),
            io.InOp(r=s1 * 1e6),
        ),
        (
            {"w": s1.astype("pint[MW]"), "q": s2.astype("pint[MWh]")},
            io.InOp(w=s1, q=s2),
            io.InOp(w=s1_i, q=s2_i),
        ),
        (
            pd.DataFrame({"w": s1.astype("pint[MW]"), "q": s2.astype("pint[MWh]")}),
            io.InOp(w=s1_u, q=s2_u),
            io.InOp(w=s1_u, q=s2_u),
        ),
        (
            {"w": s1.astype("pint[MW]"), "q": s2.astype("pint[GWh]")},
            io.InOp(w=s1, q=s2 * 1000),
            io.InOp(w=s1_i, q=s2_i * 1000),
        ),
        (
            pd.DataFrame({"w": s1.astype("pint[MW]"), "q": s2.astype("pint[GWh]")}),
            io.InOp(w=s1_u, q=s2_u * 1000),
            io.InOp(w=s1_u, q=s2_u * 1000),
        ),
        # . unknown name -> KeyError
        (
            {"z": s1},
            KeyError,
            None,
        ),
        (
            pd.DataFrame({"z": s1}),
            KeyError,
            None,
        ),
        (
            {"z": s1.astype("pint[MW]")},
            KeyError,
            None,
        ),
        (
            pd.DataFrame({"z": s1.astype("pint[MW]")}),
            KeyError,
            None,
        ),
        # . mix of know and unknown names -> KeyError
        (
            {"w": s1, "z": s2},
            KeyError,
            None,
        ),
        (
            pd.DataFrame({"w": s1, "z": s2}),
            KeyError,
            None,
        ),
        (
            {"w": s1, "p": s2 * 10, "z": s2},
            KeyError,
            None,
        ),
        (
            pd.DataFrame({"w": s1, "p": s2 * 10, "z": s2}),
            KeyError,
            None,
        ),
        (
            pd.DataFrame({"w": s2.astype("pint[GW]"), "p": s2 * 10, "z": s2}),
            KeyError,
            None,
        ),
        # . combination of name with incorrect unit -> error
        (
            {"w": s1.astype("pint[MWh]")},
            DimensionalityError,
            None,
        ),
        (
            pd.DataFrame({"w": s1.astype("pint[MWh]")}),
            DimensionalityError,
            None,
        ),
        (
            {"p": s1.astype("pint[MWh]")},
            DimensionalityError,
            None,
        ),
        (
            pd.DataFrame({"p": s1.astype("pint[MWh]")}),
            DimensionalityError,
            None,
        ),
        (
            {"p": s2, "w": s1.astype("pint[MWh]")},
            DimensionalityError,
            None,
        ),
        (
            pd.DataFrame({"p": s2, "w": s1.astype("pint[MWh]")}),
            DimensionalityError,
            None,
        ),
        # Combinations of value(s) and timeseries.
        # . name but no unit
        (
            {"w": s1, "p": 50.0},
            io.InOp(w=s1, p=50),
            io.InOp(w=s1, p=pd.Series(50, idx1)),
        ),
        (
            {"q": -s1, "p": 50.0, "r": s2},
            io.InOp(q=-s1, p=50, r=s2),
            io.InOp(q=-s1_i, r=s2_i, p=pd.Series(50, idx_i)),
        ),
        # . name and correct unit
        (
            {"w": s1.astype("pint[MW]"), "p": 50.0},
            io.InOp(w=s1, p=50),
            io.InOp(w=s1, p=pd.Series(50, idx1)),
        ),
        (
            {"w": s1.astype("pint[MW]"), "q": s2.astype("pint[MWh ]"), "p": 50},
            io.InOp(w=s1, q=s2, p=50),
            io.InOp(w=s1_i, q=s2_i, p=pd.Series(50, idx_i)),
        ),
        (
            {"r": s1.astype("pint[MEur]"), "p": 50.0, "q": 90_000},
            io.InOp(r=s1 * 1e6, p=50, q=90_000),
            io.InOp(r=s1 * 1e6, p=pd.Series(50, idx1), q=pd.Series(90_000, idx1)),
        ),
        # . unknown name -> KeyError
        (
            {"z": s1, "xy": 50},
            KeyError,
            None,
        ),
        # . mix of know and unknown names -> KeyError
        (
            {"z": s1, "p": 50.0},
            KeyError,
            None,
        ),
        (
            {"z": s1.astype("pint[MW]"), "p": s2},
            KeyError,
            None,
        ),
        (
            {"w": s1.astype("pint[GW]"), "z": 28},
            KeyError,
            None,
        ),
        (
            {"w": s1, "p": s2 * 10, "z": 50},
            KeyError,
            None,
        ),
        # ( # exclude: not a valid dataframe contructor
        #    pd.DataFrame({"w": s1, "p": Q_(5.0, "ctEur/kWh"), "z": s2}),
        #    io.InterOp(w=s1, p=50, rest=({"z": s2},)),
        # ),
        (
            pd.DataFrame({"w": s1.astype("pint[GW]"), "p": 50.0, "z": s2}),
            KeyError,
            None,
        ),
        # . combination of name with incorrect unit -> error
        (
            {"w": s1.astype("pint[MWh]"), "p": Q_(50.0, "MW")},
            DimensionalityError,
            None,
        ),
        (
            {"p": s1.astype("pint[MWh]"), "w": 120.0},
            DimensionalityError,
            None,
        ),
        (
            {"z": 23.0, "p": s2, "w": s1.astype("pint[MWh]")},
            KeyError,
            None,
        ),
    ],
    ids=id_fn,
)
def test_interop(data_in, expected_io, expected_io2):
    """Test if random data creates the expected InterOp object."""
    if type(expected_io) is type and issubclass(expected_io, Exception):
        with pytest.raises(expected_io):
            _ = io.InOp.from_data(data_in)
        return

    result_io = io.InOp.from_data(data_in)
    assert result_io == expected_io

    if type(expected_io2) is type and issubclass(expected_io2, Exception):
        with pytest.raises(expected_io2):
            _ = result_io.to_timeseries()
        return

    result_io2 = result_io.to_timeseries()
    assert result_io2 == expected_io2

    result_io3 = result_io2.to_timeseries()
    assert result_io3 == result_io2  # repeated application of intersection does nothing
