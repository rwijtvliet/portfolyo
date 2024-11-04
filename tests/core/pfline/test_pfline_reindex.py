import portfolyo as pf
import pytest
import pandas as pd

idx = pd.date_range("2020", freq="MS", periods=3)
idx2 = pd.date_range("2020-02", freq="MS", periods=3)
s = pd.Series([10, 20, 30], idx)
s2 = pd.Series([20, 30, 0], idx2)


@pytest.mark.parametrize(
    "pfl,idx,expected",
    [
        (pf.PfLine(s.astype("pint[MWh]")), idx2, pf.PfLine(s2.astype("pint[MWh]"))),
        (pf.PfLine(s.astype("pint[MW]")), idx2, pf.PfLine(s2.astype("pint[MW]"))),
        (pf.PfLine(s.astype("pint[Wh]")), idx2, pf.PfLine(s2.astype("pint[Wh]"))),
    ],
)
def test_pfl_reindex(pfl: pf.PfLine, idx: pd.DatetimeIndex, expected: pf.PfLine):
    assert pfl.reindex(idx) == expected
