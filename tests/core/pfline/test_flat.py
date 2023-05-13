import numpy as np
import pandas as pd
import pytest

from portfolyo import Kind, PfLine, create, dev, testing, tools

# @dataclass
# class InteropTestObject:
#     value: Any
#     has_name: bool
#     has_unit: bool
#     in_defunit: Any  # None if conversion impossible
#     error_name_unit_compatible: bool = False  # True if name and unit don't fit together
#     error_unit_unknown: bool = False


# def interoperability_test_object_generator():
#     # first unit is the default one.
#     name_unit = {
#         "": [(1, "")],
#         "w": [(1, "MW"), (1e6, "W"), (3.6e9, "J/h"), (1e-3, "GW"), (None, "Eur/MW")],
#         "q": [(1, "MWh"), (1e6, "Wh"), (3.6e9, "J"), (1e-3, "GWh"), (None, "MW")],
#         "p": [(1, "Eur/MWh"), (0.1, "ct/kWh"), (1 / 3.6e3, "Eur/MJ"), (None, "Eur")],
#         "r": [(1, "Eur"), (100, "ctEur"), (1e-3, "kEur"), (None, "Eur*MW")],
#     }
#     values = []
#     # Single float
#     values.append((50, False, False, None))
#     # Single quantity
#     for name, units in name_unit.items():
#         defunit = units[0][1]
#         for factor, unit in units:
#             if factor:
#                 indef, unitunknown = Q_(50 * factor, defunit), False
#             else:
#                 indef, unitunknown = None, True
#             values.append(
#                 InteropTestObject(Q_(50, uni$t), False, True, indef, False, unitunknown)
#             )
#     # Dictionary and Series of values


@pytest.mark.parametrize(
    "columns,available",
    [
        ("w", "wq"),
        ("q", "wq"),
        ("p", "p"),
        ("pr", "wqpr"),
        ("qr", "wqpr"),
        ("pq", "wqpr"),
        ("wp", "wqpr"),
        ("wr", "wqpr"),
    ],
)
@pytest.mark.parametrize("constructor", [create.flatpfline, PfLine])
def test_flatpfline_access(columns: str, available: str, constructor: type):
    """Test if core data can be accessed by item and attribute."""

    df_in = dev.get_dataframe(columns=columns)
    result = constructor(df_in)

    testing.assert_index_equal(result.index, df_in.index)
    # testing.assert_index_equal(result["index"], df_in.index)# access by item is deprecated

    for col in list("wqpr"):
        if col in available and col in df_in:
            expected = df_in[col]
            testing.assert_series_equal(getattr(result, col), expected)
            # testing.assert_series_equal(result[col], expected) # access by item is deprecated
            testing.assert_series_equal(result.df[col], expected)
        elif col not in available:
            with pytest.raises(Exception):
                _ = getattr(result, col)
            with pytest.raises(KeyError):
                _ = result.df[col]


series = {}
for freq in ["MS", "D", "15T"]:
    idx = pd.date_range(
        "2020", "2020-04", freq=freq, inclusive="left", tz="Europe/Berlin"
    )
    p = pd.Series(50, idx)
    w = pd.Series(20, idx)
    q = w * tools.duration.index(w.index).pint.m
    series[freq] = {"i": idx, "w": w, "q": q, "p": p, "r": q * p}


@pytest.mark.parametrize("freq_in", ["MS", "D", "15T"])
@pytest.mark.parametrize("freq_out", ["MS", "D", "15T"])
@pytest.mark.parametrize("columns", ["w", "q", "p", "pw", "wr"])
def test_flatpfline_asfreqcorrect1(freq_in: str, freq_out: str, columns: str):
    """Test if changing frequency is done correctly (when it's possible), for uniform pflines."""
    pfl_in = create.flatpfline({col: series[freq_in][col] for col in columns})
    expected_out = create.flatpfline({col: series[freq_out][col] for col in columns})
    pfl_out = pfl_in.asfreq(freq_out)
    assert pfl_out == expected_out


# . check correct working of attributes .asfreq().
@pytest.mark.only_on_pr
@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("freq", ["H", "D", "MS", "QS", "AS"])
@pytest.mark.parametrize("newfreq", ["H", "D", "MS", "QS", "AS"])
@pytest.mark.parametrize("columns", ["pr", "qr", "pq", "wp", "wr"])
def test_flatpfline_asfreqcorrect2(freq, newfreq, columns, tz):
    """Test if changing frequency is done correctly (when it's possible)."""

    # Includes at 2 full years
    a, m, d = np.array([2016, 1, 1]) + np.random.randint(0, 12, 3)  # each + 0..11
    start = f"{a}-{m}-{d}"
    a, (m, d) = a + 3, np.array([1, 1]) + np.random.randint(0, 12, 2)  # each + 0..11
    end = f"{a}-{m}-{d}"

    i = pd.date_range(start, end, freq=freq, tz=tz)
    df = dev.get_dataframe(i, columns)
    pfl1 = create.flatpfline(df)
    pfl2 = pfl1.asfreq(newfreq)

    # Compare the dataframes, only keep time intervals that are in both objects.
    if pfl1.kind is Kind.PRICE:
        df1, df2 = pfl1.p * pfl1.index.duration, pfl2.p * pfl2.index.duration
    else:
        cols = ["q"]
        if pfl1.kind is Kind.COMPLETE:
            cols.append("r")
        df1, df2 = pfl1.df[cols], pfl2.df[cols]

    mask1 = (df1.index >= df2.index[0]) & (df1.index.right <= df2.index.right[-1])
    mask2 = (df2.index >= df1.index[0]) & (df2.index.right <= df1.index.right[-1])
    df1, df2 = df1[mask1], df2[mask2]

    if df2.empty or df1.empty:
        return

    testing.assert_series_equal(df1.apply(np.sum), df2.apply(np.sum))


@pytest.mark.parametrize("freq", ["15T", "H", "D"])
@pytest.mark.parametrize("newfreq", ["MS", "QS", "AS"])
@pytest.mark.parametrize("kind", [Kind.COMPLETE, Kind.VOLUME, Kind.PRICE])
def test_flatpfline_asfreqimpossible(freq, newfreq, kind):
    """Test if changing frequency raises error if it's impossible."""

    periods = {"H": 200, "15T": 2000, "D": 20}[freq]
    i = pd.date_range("2020-04-06", freq=freq, periods=periods, tz="Europe/Berlin")
    pfl = dev.get_flatpfline(i, kind)
    with pytest.raises(ValueError):
        _ = pfl.asfreq(newfreq)


# @pytest.mark.parametrize("kind", [Kind.COMPLETE, Kind.VOLUME, Kind.PRICE, Kind.REVENUE])
# @pytest.mark.parametrize("col", ["w", "q", "p", "r"])
# def test_flatpfline_setseries(kind, col):
#     """Test if series can be set on existing pfline."""
#     pfl_in = dev.get_flatpfline(kind=kind)
#
#     s = dev.get_series(pfl_in.index, col)
#
#     if kind is Kind.COMPLETE:  # Expecting error
#         with pytest.raises(Exception):
#             _ = pfl_in.set_r(s)
#         return
#
#     result = getattr(pfl_in, f"set_{col}")(s)
#     testing.assert_series_equal(getattr(result, col), s)
#     assert col in result.kind.available
#     if kind is Kind.VOLUME and col in ["w", "q"]:
#         expectedkind = Kind.VOLUME
#     elif kind is Kind.PRICE and col == "p":
#         expectedkind = Kind.PRICE
#     elif kind is Kind.REVENUE and col == "r":
#         expectedkind = Kind.REVENUE
#     else:
#         expectedkind = Kind.COMPLETE
#     assert result.kind is expectedkind
