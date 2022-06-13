"""Test initialisation of PfLine, SinglePfLine, and MultiPfLine."""

from portfolyo import FREQUENCIES, testing, PfLine, SinglePfLine, MultiPfLine  # noqa
from portfolyo import dev
import pytest


@pytest.mark.parametrize("freq", FREQUENCIES[::2])
@pytest.mark.parametrize("columns", ["w", "q", "p", "pr", "qr", "pq", "wp", "wr"])
@pytest.mark.parametrize(
    "inputtype", ["df", "dict", "series", "singlepfline", "multipfline"]
)
@pytest.mark.parametrize("has_unit", [True, False])
def test_singlepfline_init(freq, columns, inputtype, has_unit):
    """Test if single pfline can be initialized correctly, and attributes return correct values."""

    must_raise = False

    i = dev.get_index(freq, "Europe/Berlin")
    df = dev.get_dataframe(i, columns, has_unit)
    if inputtype == "df":
        data_in = df
    elif inputtype == "dict":
        data_in = {name: s for name, s in df.items()}
    elif inputtype == "series":
        if len(columns) > 1:
            return  # cannot pass multiple series
        data_in = df[columns]
        if not has_unit:
            must_raise = True
    elif inputtype == "singlepfline":
        data_in = SinglePfLine(df)
    else:  # inputtype multipfline
        if columns in ["w", "q", "p", "qr", "wr"]:
            df1 = 0.4 * df
            df2 = 0.6 * df
        else:  # has price column
            othercol = columns.replace("p", "")
            df1 = df.mul({"p": 1, othercol: 0.4})
            df2 = df.mul({"p": 1, othercol: 0.6})
        data_in = MultiPfLine({"a": SinglePfLine(df1), "b": SinglePfLine(df2)})

    if must_raise:
        with pytest.raises(ValueError):
            _ = SinglePfLine(data_in)
        return

    result = SinglePfLine(data_in)
    result_df = result.df(columns)
    expected_df = df.rename_axis("ts_left")
    if columns in ["w", "q"]:  # kind == 'q'
        expectedkind = "q"
        expectedavailable = "wq"
        expectedsummable = "q"
    elif columns in ["p"]:  # kind == 'p'
        expectedkind = "p"
        expectedavailable = "p"
        expectedsummable = "p"
    else:  # kind == 'all'
        expectedkind = "all"
        expectedavailable = "wqpr"
        expectedsummable = "qr"

    assert type(result) is SinglePfLine
    testing.assert_frame_equal(result_df, expected_df)
    assert result.kind == expectedkind
    assert set(list(result.available)) == set(list(expectedavailable))
    assert set(list(result.summable)) == set(list(expectedsummable))
    assert result.children == {}


@pytest.mark.parametrize("freq", FREQUENCIES[::2])
@pytest.mark.parametrize("columns", ["w", "q", "p", "pr", "qr", "pq", "wp", "wr"])
@pytest.mark.parametrize(
    "inputtype", ["df", "dict", "series", "singlepfline", "multipfline"]
)
@pytest.mark.parametrize("has_unit", [True, False])
def test_multipfline_init(freq, columns, inputtype, has_unit):
    """Test if multi pfline can be initialized correctly, and attributes return correct values."""
    pass  # TODO after clear, how multipfline can be initialised.


@pytest.mark.parametrize("kind", ["all", "p", "q"])
@pytest.mark.parametrize("inputtype", ["df", "dict", "pfline"])
@pytest.mark.parametrize("expected_type", [SinglePfLine, MultiPfLine, None])
def test_pfline_init(inputtype, expected_type, kind):
    """Test if pfline can be initialized correctly."""
    pass  # TODO after clear, how multipfline can be initialised.

    # if expected_type is SinglePfLine:
    #     spfl = dev.get_singlepfline(kind=kind)
    #     if inputtype == "df":
    #         data_in = spfl.df()
    #     elif inputtype == "dict":
    #         data_in = {name: s for name, s in spfl.df().items()}
    #     else:  # inputtype == 'pfline'
    #         data_in = spfl
    # elif expected_type is MultiPfLine:
    #     mpfl = dev.get_multipfline(kind=kind)
    #     if inputtype == "df":
    #         return  # no way to call with dataframe
    #     elif inputtype == "dict":
    #         data_in = {name: pfl for name, pfl in mpfl.items()}
    #     else:  # inputtype == 'pfline'
    #         data_in = mpfl
    # else:  # Expect error
    #     cols = "pq" if kind == "all" else kind
    #     if inputtype == "df":
    #         # dataframe with columns that don't make sense
    #         data_in = dev.get_dataframe(columns=cols).rename(
    #             columns=dict(zip("wqpr", "abcd"))
    #         )
    #     elif inputtype == "dict":
    #         # dictionary with mix of series and pflines (not suitable for either)
    #         data_in = {"p": dev.get_series(name="p"), "partA": dev.get_singlepfline()}
    #     else:  # inputtype == 'pfline'
    #         return

    # if expected_type is None:  # expect error
    #     with pytest.raises(NotImplementedError):
    #         _ = PfLine(data_in)
    #     return

    # result = PfLine(data_in)
    # assert type(result) is expected_type
