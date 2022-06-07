from portfolyo.core.pfline.base import PfLine
from portfolyo.core.pfline.single import SinglePfLine
from portfolyo.core.pfline.multi import MultiPfLine
from portfolyo import dev
import pytest


@pytest.mark.parametrize("kind", ["all", "p", "q"])
@pytest.mark.parametrize("inputtype", ["df", "dict", "pfline"])
@pytest.mark.parametrize("expected_type", [SinglePfLine, MultiPfLine, None])
def test_pfline_init(inputtype, expected_type, kind):
    """Test if object can be initialized correctly."""

    if expected_type is SinglePfLine:
        spfl = dev.get_singlepfline(kind=kind)
        if inputtype == "df":
            data_in = spfl.df()
        elif inputtype == "dict":
            data_in = {name: s for name, s in spfl.df().items()}
        else:  # inputtype == 'pfline'
            data_in = spfl
    elif expected_type is MultiPfLine:
        mpfl = dev.get_multipfline(kind=kind)
        if inputtype == "df":
            return  # no way to call with dataframe
        elif inputtype == "dict":
            data_in = {name: pfl for name, pfl in mpfl.items()}
        else:  # inputtype == 'pfline'
            data_in = mpfl
    else:  # Expect error
        cols = "pq" if kind == "all" else kind
        if inputtype == "df":
            # dataframe with columns that don't make sense
            data_in = dev.get_dataframe(columns=cols).rename(
                columns=dict(zip("wqpr", "abcd"))
            )
        elif inputtype == "dict":
            # dictionary with mix of series and pflines (not suitable for either)
            data_in = {"p": dev.get_series(name="p"), "partA": dev.get_singlepfline()}
        else:  # inputtype == 'pfline'
            return

    if expected_type is None:  # expect error
        with pytest.raises(NotImplementedError):
            _ = PfLine(data_in)
        return

    result = PfLine(data_in)
    assert type(result) is expected_type
