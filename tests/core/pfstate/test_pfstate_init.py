import numpy as np
import pandas as pd
import pytest

from portfolyo import PfLine, PfState, dev, testing  # noqa
from portfolyo.core.pfstate.pfstate_helper import make_pflines

# Assert correct working of _make_pflines
# . check unsourced and offtake are specified.
# . check that inconsistent data raises error.
# . check with keys having unequal indexes: unequal freq, timeperiod.
# . check if missing values have expected result.

i_ref = pd.date_range("2020", freq="D", periods=80)
s_ref = dev.get_series(i_ref, "")
i_less = pd.date_range("2020-01-15", freq="D", periods=60)
s_less = dev.get_series(i_less, "")
i_more = pd.date_range("2019-12-15", freq="D", periods=100)
s_more = dev.get_series(i_more, "")
i_difffreq = pd.date_range("2020", freq="H", periods=80)
s_difffreq = dev.get_series(i_difffreq, "")


@pytest.mark.parametrize(
    ("o", "u", "s", "o_exp", "u_exp", "s_exp"),
    [
        (  # Full package.
            PfLine({"w": s_ref}),
            PfLine({"p": s_ref * 10}),
            PfLine({"w": s_ref + 20, "p": s_ref * 20}),
            PfLine({"w": s_ref}),
            PfLine({"p": s_ref * 10}),
            PfLine({"w": s_ref + 20, "p": s_ref * 20}),
        ),
        (  # Too much information at offtake.
            PfLine({"w": s_ref, "p": s_ref + 1}),
            PfLine({"p": s_ref * 10}),
            PfLine({"w": s_ref + 20, "p": s_ref * 20}),
            PfLine({"w": s_ref}),
            PfLine({"p": s_ref * 10}),
            PfLine({"w": s_ref + 20, "p": s_ref * 20}),
        ),
        (  # Too much information at unsourcedprice.
            PfLine({"w": s_ref}),
            PfLine({"p": s_ref * 10, "w": s_ref + 10}),
            PfLine({"w": s_ref + 20, "p": s_ref * 20}),
            PfLine({"w": s_ref}),
            PfLine({"p": s_ref * 10}),
            PfLine({"w": s_ref + 20, "p": s_ref * 20}),
        ),
        (  # Incorrect kind at offtake.
            PfLine({"p": s_ref + 1}),
            PfLine({"p": s_ref * 10, "w": s_ref + 10}),
            PfLine({"w": s_ref + 20, "p": s_ref * 20}),
            ValueError,
            None,
            None,
        ),
        (  # Incorrect kind at unsourcedprice.
            PfLine({"w": s_ref}),
            PfLine({"w": s_ref + 10}),
            PfLine({"w": s_ref + 20, "p": s_ref * 20}),
            ValueError,
            None,
            None,
        ),
        (  # Incorrect kind at sourced.
            PfLine({"w": s_ref}),
            PfLine({"p": s_ref * 10}),
            PfLine({"w": s_ref + 20}),
            ValueError,
            None,
            None,
        ),
        (  # Incorrect kind at sourced.
            PfLine({"w": s_ref}),
            PfLine({"p": s_ref * 10}),
            PfLine({"p": s_ref * 20}),
            ValueError,
            None,
            None,
        ),
        (  # No sourcing yet.
            PfLine({"w": s_ref}),
            PfLine({"p": s_ref * 10}),
            None,
            PfLine({"w": s_ref}),
            PfLine({"p": s_ref * 10}),
            None,
        ),
        (  # Unequal periods; result is trimmed.
            PfLine({"w": s_ref}),
            PfLine({"p": s_more * 10}),
            PfLine({"w": s_more + 20, "p": s_more * 20}),
            PfLine({"w": s_ref}),
            PfLine({"p": s_more * 10}).loc[i_ref],
            PfLine({"w": s_more + 20, "p": s_more * 20}).loc[i_ref],
        ),
        (  # Unequal periods; error (intersection; not enough sourced volume).
            PfLine({"w": s_ref}),
            PfLine({"p": s_more * 10}),
            PfLine({"w": s_less + 20, "p": s_less * 20}),
            ValueError,
            None,
            None,
        ),
        (  # Unequal periods; error (intersection; not enough unsourced prices).
            PfLine({"w": s_ref}),
            PfLine({"p": s_less * 10}),
            PfLine({"w": s_more + 20, "p": s_more * 20}),
            ValueError,
            None,
            None,
        ),
        (  # Not passing PfLines.
            {"w": s_ref},
            {"p": s_ref * 10},
            {"w": s_ref + 20, "p": s_ref * 20},
            PfLine({"w": s_ref}),
            PfLine({"p": s_ref * 10}),
            PfLine({"w": s_ref + 20, "p": s_ref * 20}),
        ),
        (  # Not passing PfLines.
            {"w": s_ref},
            {"p": s_more * 10},
            None,
            PfLine({"w": s_ref}),
            PfLine({"p": s_more * 10}).loc[i_ref],
            None,
        ),
        (  # Not passing PfLines; error (intersection)
            {"w": s_more},
            {"p": s_ref * 10},
            None,
            ValueError,
            None,
            None,
        ),
        (  # Unequal frequencies.
            PfLine({"w": s_ref}),
            PfLine({"p": s_difffreq * 10}),
            None,
            ValueError,
            None,
            None,
        ),
        (  # Unequal frequencies.
            PfLine({"w": s_difffreq}),
            PfLine({"p": s_ref * 10}),
            None,
            ValueError,
            None,
            None,
        ),
    ],
)
def test_makepflines_initpfstate(o, u, s, o_exp, u_exp, s_exp):
    """Test that input values can be turned into three pflines and used to initialize a pfstate.

    If Exception is expected, set `o_exp` to the exception type.
    """
    if isinstance(o_exp, type) and issubclass(o_exp, Exception):
        with pytest.raises(o_exp):
            _ = make_pflines(o, u, s)
        return

    # Test helper.
    o_res, u_res, s_res = make_pflines(o, u, s)
    assert o_res == o_exp
    assert u_res == u_exp
    assert s_res == s_exp

    # Test init.
    result = PfState(o, u, s)
    assert result.offtakevolume == o_exp
    assert result.unsourcedprice == u_exp
    if s_exp is None:
        sourced_exp = PfLine(pd.DataFrame({"q": 0, "r": 0}, o_exp.index))
    else:
        sourced_exp = s_exp
    assert result.sourced == sourced_exp


@pytest.mark.parametrize(
    ("wo", "qo", "ws", "qs", "ps", "rs", "pu", "o_exp", "u_exp", "s_exp"),
    [
        (  # Full package with power and price.
            s_ref,
            None,
            s_ref + 20,
            None,
            s_ref * 20,
            None,
            s_ref * 10,
            PfLine({"w": s_ref}),
            PfLine({"p": s_ref * 10}),
            PfLine({"w": s_ref + 20, "p": s_ref * 20}),
        ),
        (  # Full package with energy and price.
            None,
            s_ref,
            None,
            s_ref + 20,
            s_ref * 20,
            None,
            s_ref * 10,
            PfLine({"q": s_ref}),
            PfLine({"p": s_ref * 10}),
            PfLine({"q": s_ref + 20, "p": s_ref * 20}),
        ),
        (  # Full package with power and revenue.
            s_ref,
            None,
            s_ref + 20,
            None,
            None,
            s_ref * 20,
            s_ref * 10,
            PfLine({"w": s_ref}),
            PfLine({"p": s_ref * 10}),
            PfLine({"w": s_ref + 20, "r": s_ref * 20}),
        ),
        (  # Full package with price and revenue.
            s_ref,
            None,
            None,
            None,
            s_ref + 20,
            s_ref * 20,
            s_ref * 10,
            PfLine({"w": s_ref}),
            PfLine({"p": s_ref * 10}),
            PfLine({"p": s_ref + 20, "r": s_ref * 20}),
        ),
        (  # Full package with energy and revenue.
            None,
            s_ref,
            None,
            s_ref + 20,
            None,
            s_ref * 20,
            s_ref * 10,
            PfLine({"q": s_ref}),
            PfLine({"p": s_ref * 10}),
            PfLine({"q": s_ref + 20, "r": s_ref * 20}),
        ),
        (  # Too little information at offtake.
            None,
            None,
            s_ref + 20,
            None,
            s_ref * 20,
            None,
            s_ref * 10,
            ValueError,
            None,
            None,
        ),
        (  # Too little information at unsourcedprice.
            s_ref,
            None,
            s_ref + 20,
            None,
            s_ref * 20,
            None,
            None,
            ValueError,
            None,
            None,
        ),
        (  # Too little information at sourced.
            s_ref,
            None,
            s_ref + 20,
            None,
            None,
            None,
            s_ref * 10,
            ValueError,
            None,
            None,
        ),
        (  # Too little information at sourced.
            s_ref,
            None,
            None,
            None,
            s_ref * 20,
            None,
            s_ref * 10,
            ValueError,
            None,
            None,
        ),
        (  # No sourcing yet.
            s_ref,
            None,
            None,
            None,
            None,
            None,
            s_ref * 10,
            PfLine({"w": s_ref}),
            PfLine({"p": s_ref * 10}),
            PfLine(pd.DataFrame({"q": 0.0, "r": 0.0}, i_ref)),
        ),
        (  # Unequal periods.
            s_ref,
            None,
            s_less + 20,
            None,
            s_less * 20,
            None,
            s_more * 10,
            PfLine({"w": s_ref}).loc[i_less],
            PfLine({"p": s_more * 10}),
            PfLine({"w": s_less + 20, "p": s_less * 20}),
        ),
        (  # Unequal periods.
            s_ref,
            None,
            s_more + 20,
            None,
            s_more * 20,
            None,
            s_more * 10,
            PfLine({"w": s_ref}),
            PfLine({"p": s_more * 10}),
            PfLine({"w": s_more + 20, "p": s_more * 20}).loc[i_ref],
        ),
        (  # Enough sourced prices even though looks like there are not.
            s_more,
            None,
            s_less + 20,
            None,
            s_less * 20,
            None,
            s_ref * 10,
            PfLine({"w": s_more}).loc[i_less],
            PfLine({"p": s_ref * 10}),
            PfLine({"w": s_less + 20, "p": s_less * 20}),
        ),
        (  # Error: not enough unsourced prices
            s_ref,
            None,
            s_more + 20,
            None,
            s_more * 20,
            None,
            s_less * 10,
            ValueError,
            None,
            None,
        ),
        (  # Unequal frequencies.
            s_ref,
            None,
            None,
            None,
            None,
            None,
            s_difffreq * 10,
            ValueError,
            None,
            None,
        ),
        (  # Unequal frequencies.
            s_difffreq,
            None,
            None,
            None,
            None,
            None,
            s_ref * 10,
            ValueError,
            None,
            None,
        ),
    ],
)
def test_initpfstate_fromseries(pu, qo, qs, rs, wo, ws, ps, o_exp, u_exp, s_exp):
    """Test that input values can be turned into three pflines and used to initialize a pfstate.

    If Exception is expected, set `o_exp` to the exception type.
    """
    if isinstance(o_exp, type) and issubclass(o_exp, Exception):
        with pytest.raises(o_exp):
            _ = PfState.from_series(pu=pu, qo=qo, qs=qs, rs=rs, wo=wo, ws=ws, ps=ps)
        return

    # Test init.
    result = PfState.from_series(pu=pu, qo=qo, qs=qs, rs=rs, wo=wo, ws=ws, ps=ps)
    assert result.offtakevolume == o_exp
    assert result.unsourcedprice == u_exp
    assert result.sourced == s_exp


def test_pfstate_consistency_uniformfreq():
    """Test if all values are consistent as expected."""
    # Starting values. (qo defined as being positive.)
    qo, qs, rs, pu = s_ref, s_ref + 20, s_ref * 20, s_ref * 10
    # Create PfState.
    result = PfState.from_series(qo=-qo, qs=qs, rs=rs, pu=pu)
    # Expected.
    o_exp = PfLine({"q": -qo})
    s_exp = PfLine({"q": qs, "r": rs})
    u_exp = PfLine({"p": pu})
    # Test.
    assert result.offtakevolume == o_exp
    assert result.unsourcedprice == u_exp
    assert result.sourced == s_exp
    testing.assert_series_equal(result.offtakevolume.q.pint.m, -qo, check_names=False)
    testing.assert_series_equal(result.unsourcedprice.p.pint.m, pu, check_names=False)
    testing.assert_series_equal(result.sourced.q.pint.m, qs, check_names=False)
    testing.assert_series_equal(result.sourced.r.pint.m, rs, check_names=False)
    testing.assert_series_equal(result.sourced.p.pint.m, rs / qs, check_names=False)
    testing.assert_series_equal(result.unsourced.p.pint.m, pu, check_names=False)
    testing.assert_series_equal(result.unsourced.q.pint.m, qo - qs, check_names=False)
    testing.assert_series_equal(
        result.unsourced.r.pint.m, pu * (qo - qs), check_names=False
    )
    testing.assert_series_equal(result.pnl_cost.q.pint.m, qo, check_names=False)
    testing.assert_series_equal(
        result.pnl_cost.r.pint.m, rs + pu * (qo - qs), check_names=False
    )
    testing.assert_series_equal(
        result.sourcedfraction,
        (qs / qo).astype("pint[dimensionless]"),
        check_names=False,
    )
    testing.assert_series_equal(
        result.unsourcedfraction,
        ((qo - qs) / qo).astype("pint[dimensionless]"),
        check_names=False,
    )


def test_pfstate_consistency_unequalfreq():
    """Test if all values are consistent as expected."""
    # Starting values. (qo defined as being positive.)
    qo, qs, rs, pu = s_ref, s_ref + 20, s_less * 20, s_more * 10
    # Create PfState.
    result = PfState.from_series(qo=-qo, qs=qs, rs=rs, pu=pu)
    # Expected.
    qo, qs, rs, pu = qo.loc[i_less], qs.loc[i_less], rs.loc[i_less], pu
    o_exp = PfLine({"q": -qo})
    s_exp = PfLine({"q": qs, "r": rs})
    u_exp = PfLine({"p": pu})
    # Test.
    assert result.offtakevolume == o_exp
    assert result.unsourcedprice == u_exp
    assert result.sourced == s_exp
    testing.assert_series_equal(result.offtakevolume.q.pint.m, -qo, check_names=False)
    testing.assert_series_equal(result.unsourcedprice.p.pint.m, pu, check_names=False)
    testing.assert_series_equal(result.sourced.q.pint.m, qs, check_names=False)
    testing.assert_series_equal(result.sourced.r.pint.m, rs, check_names=False)
    testing.assert_series_equal(result.sourced.p.pint.m, rs / qs, check_names=False)
    testing.assert_series_equal(
        result.unsourced.p.pint.m, pu.loc[i_less], check_names=False
    )
    testing.assert_series_equal(result.unsourced.q.pint.m, qo - qs, check_names=False)
    testing.assert_series_equal(
        result.unsourced.r.pint.m, pu.loc[i_less] * (qo - qs), check_names=False
    )
    testing.assert_series_equal(result.pnl_cost.q.pint.m, qo, check_names=False)
    testing.assert_series_equal(
        result.pnl_cost.r.pint.m, rs + pu.loc[i_less] * (qo - qs), check_names=False
    )
    testing.assert_series_equal(
        result.sourcedfraction,
        (qs / qo).astype("pint[dimensionless]"),
        check_names=False,
    )
    testing.assert_series_equal(
        result.unsourcedfraction,
        ((qo - qs) / qo).astype("pint[dimensionless]"),
        check_names=False,
    )


def test_pfstate_consistency_nosourcing():
    """Test if all values are consistent as expected."""
    # Starting values. (qo defined as being positive.)
    qo, pu = s_ref, s_more * 10
    # Create PfState.
    result = PfState.from_series(qo=-qo, pu=pu)
    # Expected.
    qo, qs, rs, pu = qo, pd.Series(0.0, i_ref), pd.Series(0.0, i_ref), pu
    o_exp = PfLine({"q": -qo})
    s_exp = PfLine({"q": qs, "r": rs})
    u_exp = PfLine({"p": pu})
    # Test.
    assert result.offtakevolume == o_exp
    assert result.unsourcedprice == u_exp
    assert result.sourced == s_exp
    testing.assert_series_equal(result.offtakevolume.q.pint.m, -qo, check_names=False)
    testing.assert_series_equal(result.unsourcedprice.p.pint.m, pu, check_names=False)
    testing.assert_series_equal(result.sourced.q.pint.m, qs, check_names=False)
    testing.assert_series_equal(result.sourced.r.pint.m, rs, check_names=False)
    testing.assert_series_equal(
        result.sourced.p.pint.m, pd.Series(np.nan, i_ref), check_names=False
    )
    testing.assert_series_equal(
        result.unsourced.p.pint.m, pu.loc[i_ref], check_names=False
    )
    testing.assert_series_equal(result.unsourced.q.pint.m, qo, check_names=False)
    testing.assert_series_equal(
        result.unsourced.r.pint.m, pu.loc[i_ref] * qo, check_names=False
    )
    testing.assert_series_equal(result.pnl_cost.q.pint.m, qo, check_names=False)
    testing.assert_series_equal(
        result.pnl_cost.r.pint.m, rs + pu.loc[i_ref] * (qo - qs), check_names=False
    )
    testing.assert_series_equal(
        result.sourcedfraction,
        pd.Series(0.0, i_ref, "pint[dimensionless]"),
        check_names=False,
    )
    testing.assert_series_equal(
        result.unsourcedfraction,
        pd.Series(1.0, i_ref, "pint[dimensionless]"),
        check_names=False,
    )
