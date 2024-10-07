import numpy as np
import pandas as pd
import pytest
from pint import DimensionalityError, UndefinedUnitError
from utils import id_fn  # relative to /tests

from portfolyo.core.pfline import interop as io
from portfolyo.tools.unit import Q_

np.random.seed(3)  # ensure same data all the time

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


@pytest.mark.parametrize(
    ("data_in", "expected_io", "expected_io2"),
    [
        # One value
        # . unitless
        (
            Q_(23.0, ""),
            io.InOp(nodim=23.0),
            ValueError,
        ),
        # . known unit
        (
            Q_(-120.0, "MW"),
            io.InOp(w=Q_(-120.0, "MW")),
            ValueError,
        ),
        (
            Q_(120e-3, "GW"),
            io.InOp(w=Q_(120.0, "MW")),
            ValueError,
        ),
        (
            Q_(432e9, "J/h"),
            io.InOp(w=Q_(120.0, "MW")),
            ValueError,
        ),
        (
            Q_(90_000.0, "MWh"),
            io.InOp(q=Q_(90_000.0, "MWh")),
            ValueError,
        ),
        (
            Q_(90.0, "GWh"),
            io.InOp(q=Q_(90_000.0, "MWh")),
            ValueError,
        ),
        (
            Q_(50.0, "Eur/MWh"),
            io.InOp(p=Q_(50.0, "Eur/MWh")),
            ValueError,
        ),
        (
            Q_(5.0, "ctEur/kWh"),
            io.InOp(p=Q_(50.0, "Eur/MWh")),
            ValueError,
        ),
        (
            Q_(4_500_000.0, "Eur"),
            io.InOp(r=Q_(4_500_000.0, "Eur")),
            ValueError,
        ),
        (
            Q_(4.5, "MEur"),
            io.InOp(r=Q_(4_500_000.0, "Eur")),
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
            DimensionalityError,
            None,
        ),
        (
            pd.Series({"w": 120.0}),
            DimensionalityError,
            None,
        ),
        (
            {"q": -90_000.0},
            DimensionalityError,  # should be value error we have don't have an object on line before it
            None,
        ),
        (
            pd.Series({"q": -90_000.0}),
            DimensionalityError,
            None,
        ),
        (
            {"p": 50.0},
            DimensionalityError,
            None,
        ),
        (
            pd.Series({"p": 50.0}),
            DimensionalityError,
            None,
        ),
        (
            {"r": 4.5e6},
            DimensionalityError,
            None,
        ),
        (
            pd.Series({"r": 4.5e6}),
            DimensionalityError,
            None,
        ),
        (
            {"w": 120.0, "q": -90_000},
            DimensionalityError,
            None,
        ),
        (
            pd.Series({"w": 120.0, "q": -90_000}),
            DimensionalityError,
            None,
        ),
        (
            {"w": 120.0, "p": 50},
            DimensionalityError,
            None,
        ),
        (
            pd.Series({"w": 120.0, "p": 50}),
            DimensionalityError,
            None,
        ),
        (
            {"w": 120.0, "p": 50.0, "r": 4.5e6},
            DimensionalityError,
            None,
        ),
        (
            pd.Series({"w": 120.0, "p": 50.0, "r": 4.5e6}),
            DimensionalityError,
            None,
        ),
        (
            {"w": 120.0, "p": 50.0, "r": 4.5e6},
            DimensionalityError,
            None,
        ),
        (
            pd.Series({"w": 120.0, "p": 50.0, "r": 4.5e6}),
            DimensionalityError,
            None,
        ),
        # . name and correct unit
        (
            {"p": Q_(50.0, "Eur/MWh")},
            io.InOp(p=Q_(50.0, "Eur/MWh")),
            ValueError,
        ),
        (
            pd.Series({"p": Q_(50.0, "Eur/MWh")}),
            io.InOp(p=Q_(50.0, "Eur/MWh")),
            ValueError,
        ),
        (
            pd.Series({"p": 50}).astype("pint[Eur/MWh]"),
            io.InOp(p=Q_(50.0, "Eur/MWh")),
            ValueError,
        ),
        (
            {"r": Q_(4.5, "MEur")},
            io.InOp(r=Q_(4.5, "MEur")),
            ValueError,
        ),
        (
            pd.Series({"r": Q_(4.5, "MEur")}),
            io.InOp(r=Q_(4.5, "MEur")),
            ValueError,
        ),
        (
            pd.Series({"r": 4.5}).astype("pint[MEur]"),
            io.InOp(r=Q_(4.5, "MEur")),
            ValueError,
        ),
        (
            {"w": 120.0, "q": Q_(-90_000.0, "MWh")},
            # io.InOp(w=120.0, q=-90_000),
            DimensionalityError,
            None,
        ),
        (
            pd.Series({"w": 120.0, "q": Q_(-90_000.0, "MWh")}),
            # io.InOp(w=120.0, q=-90_000),
            DimensionalityError,
            None,
        ),
        (
            pd.Series({"w": 120.0, "q": Q_(-90.0, "GWh")}),
            # io.InOp(w=120.0, q=-90_000),
            DimensionalityError,
            None,
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
            DimensionalityError,
            None,
        ),
        (  # depends on what columnt we get firat
            pd.Series({"w": 120.0, "z": 28.0}),
            DimensionalityError,
            None,
        ),
        (
            {"z": 28.0, "w": 120.0, "p": 50.0},
            KeyError,
            None,
        ),
        (
            pd.Series({"z": 28.0, "w": 120.0, "p": 50.0}),
            KeyError,
            None,
        ),
        # . combination of name with incorrect unit -> error
        (
            {"w": Q_(90.0, "MWh")},
            AttributeError,
            None,
        ),
        (
            pd.Series({"w": Q_(90.0, "MWh")}),
            AttributeError,
            None,
        ),
        (
            pd.Series({"w": 90}).astype("pint[MWh]"),
            AttributeError,
            None,
        ),
        (
            {"p": Q_(90.0, "MWh")},
            AttributeError,
            None,
        ),
        (
            pd.Series({"p": Q_(90.0, "MWh")}),
            AttributeError,
            None,
        ),
        # One timeseries
        # . unitless
        # (s1.astype("pint[dimensionless]"), io.InterOp(nodim=s1)), # TODO: fix
        # . known unit
        (
            s1.astype("pint[MW]"),
            io.InOp(w=s1.astype("pint[MW]")),
            io.InOp(w=s1.astype("pint[MW]")),
        ),
        (
            (s1 / 1000).astype("pint[GW]"),  # series with pint unit
            io.InOp(w=s1.astype("pint[MW]")),
            io.InOp(w=s1.astype("pint[MW]")),
        ),
        (
            pd.Series([Q_(v, "MW") for v in val1], idx1),  # series of Quantities
            io.InOp(w=s1.astype("pint[MW]")),
            io.InOp(w=s1.astype("pint[MW]")),
        ),
        (
            s1.astype("pint[GWh]"),
            io.InOp(q=s1.astype("pint[MWh]") * 1000),
            io.InOp(q=s1.astype("pint[MWh]") * 1000),
        ),
        (
            s1.astype("pint[Eur/MWh]"),
            io.InOp(p=s1.astype("pint[Eur/MWh]")),
            io.InOp(p=s1.astype("pint[Eur/MWh]")),
        ),
        (
            s1.astype("pint[MEur]"),
            io.InOp(r=s1.astype("pint[MEur]")),
            io.InOp(r=s1.astype("pint[MEur]")),
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
            DimensionalityError,
            None,
        ),
        (
            pd.DataFrame({"w": s1}),
            DimensionalityError,
            None,
        ),
        (
            {"q": -s1},
            DimensionalityError,
            None,
        ),
        (
            pd.DataFrame({"q": -s1}),
            DimensionalityError,
            None,
        ),
        (
            {"r": s1},
            DimensionalityError,
            None,
        ),
        (
            pd.DataFrame({"r": s1}),
            DimensionalityError,
            None,
        ),
        (
            {"w": s1, "q": -s2},
            DimensionalityError,
            None,
        ),
        (
            pd.DataFrame({"w": s1, "q": -s2}),
            DimensionalityError,
            None,
        ),
        (
            {"w": s1, "p": s2, "r": s1 * 4},
            DimensionalityError,
            None,
        ),
        (
            pd.DataFrame({"w": s1, "p": s2, "r": s1 * 4}),
            DimensionalityError,
            None,
        ),
        # . name and correct unit
        (
            {"p": s1.astype("pint[Eur/MWh]")},
            io.InOp(p=s1.astype("pint[Eur/MWh]")),
            io.InOp(p=s1.astype("pint[Eur/MWh]")),
        ),
        (
            pd.DataFrame({"p": s1.astype("pint[Eur/MWh]")}),
            io.InOp(p=s1.astype("pint[Eur/MWh]")),
            io.InOp(p=s1.astype("pint[Eur/MWh]")),
        ),
        (
            pd.DataFrame({"p": [Q_(v, "Eur/MWh") for v in val1]}, idx1),
            io.InOp(p=s1.astype("pint[Eur/MWh]")),
            io.InOp(p=s1.astype("pint[Eur/MWh]")),
        ),
        (
            {"r": s1.astype("pint[MEur]")},
            io.InOp(r=s1.astype("pint[MEur]")),
            io.InOp(r=s1.astype("pint[MEur]")),
        ),
        (
            pd.DataFrame({"r": s1.astype("pint[MEur]")}),
            io.InOp(r=s1.astype("pint[MEur]")),
            io.InOp(r=s1.astype("pint[MEur]")),
        ),
        (
            {"w": s1.astype("pint[MW]"), "q": s2.astype("pint[MWh]")},
            io.InOp(w=s1.astype("pint[MW]"), q=s2.astype("pint[MWh]")),
            io.InOp(w=s1_i.astype("pint[MW]"), q=s2_i.astype("pint[MWh]")),
        ),
        (
            pd.DataFrame({"w": s1.astype("pint[MW]"), "q": s2.astype("pint[MWh]")}),
            io.InOp(w=s1_u.astype("pint[MW]"), q=s2_u.astype("pint[MWh]")),
            io.InOp(w=s1_u.astype("pint[MW]"), q=s2_u.astype("pint[MWh]")),
        ),
        (
            {"w": s1.astype("pint[MW]"), "q": s2.astype("pint[GWh]")},
            io.InOp(w=s1.astype("pint[MW]"), q=s2.astype("pint[MWh]") * 1000),
            io.InOp(w=s1_i.astype("pint[MW]"), q=s2_i.astype("pint[MWh]") * 1000),
        ),
        (
            pd.DataFrame({"w": s1.astype("pint[MW]"), "q": s2.astype("pint[GWh]")}),
            io.InOp(w=s1_u.astype("pint[MW]"), q=s2_u.astype("pint[MWh]") * 1000),
            io.InOp(w=s1_u.astype("pint[MW]"), q=s2_u.astype("pint[MWh]") * 1000),
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
            {"z": s2, "w": s1},
            KeyError,
            None,
        ),
        (
            pd.DataFrame({"z": s2, "w": s1}),
            KeyError,
            None,
        ),
        (
            pd.DataFrame({"z": s2, "w": s1, "p": s2 * 10}),
            KeyError,
            None,
        ),
        (
            pd.DataFrame({"w": s2.astype("pint[GW]"), "z": s2}),
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
            DimensionalityError,
            None,
        ),
        (
            {"q": -s1, "p": 50.0, "r": s2},
            DimensionalityError,
            None,
        ),
        # . name and correct unit
        (
            {"w": s1.astype("pint[MW]"), "p": 50.0},
            DimensionalityError,
            None,
        ),
        (
            {"w": s1.astype("pint[MW]"), "q": s2.astype("pint[MWh ]"), "p": 50},
            DimensionalityError,
            None,
        ),
        (
            {"r": s1.astype("pint[MEur]"), "p": 50.0, "q": 90_000},
            DimensionalityError,
            None,
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
        # ( # exclude: not a valid dataframe contructor
        #    pd.DataFrame({"w": s1, "p": Q_(5.0, "ctEur/kWh"), "z": s2}),
        #    io.InterOp(w=s1, p=50, rest=({"z": s2},)),
        # ),
        (
            pd.DataFrame({"w": s1.astype("pint[GW]"), "z": s2}),
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
