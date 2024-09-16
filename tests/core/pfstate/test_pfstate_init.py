import numpy as np
import pandas as pd
from pint import DimensionalityError
import pytest

from portfolyo import PfState, dev, testing  # noqa
from portfolyo.core.pfline import create
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
i_difffreq = pd.date_range("2020", freq="h", periods=24 * 5)
s_difffreq = dev.get_series(i_difffreq, "")


# helper function to avoid setting the same dtypes everywhere
def wrapper_pfline(args_dict):
    if "w" in args_dict and args_dict["w"].dtype == "float64":
        args_dict["w"] = args_dict["w"].astype("pint[MW]")
    if "p" in args_dict and args_dict["p"].dtype == "float64":
        args_dict["p"] = args_dict["p"].astype("pint[Eur/MWh]")
    if "q" in args_dict and args_dict["q"].dtype == "float64":
        args_dict["q"] = args_dict["q"].astype("pint[MWh]")
    if "r" in args_dict and args_dict["r"].dtype == "float64":
        args_dict["r"] = args_dict["r"].astype("pint[Eur]")
    # pfl = create.flatpfline(args_dict)
    return create.flatpfline(args_dict)


@pytest.mark.parametrize(
    ("o,u,s,o_exp,u_exp,s_exp"),
    [
        (  # Full package.0.
            wrapper_pfline({"w": s_ref}),
            wrapper_pfline({"p": s_ref * 10}),
            wrapper_pfline({"w": s_ref + 20, "p": s_ref * 20}),
            wrapper_pfline({"w": s_ref}),
            wrapper_pfline({"p": s_ref * 10}),
            wrapper_pfline({"w": s_ref + 20, "p": s_ref * 20}),
        ),
        (  # Too much information at offtake.1.
            wrapper_pfline({"w": s_ref, "p": s_ref + 1}),
            wrapper_pfline({"p": s_ref * 10}),
            wrapper_pfline({"w": s_ref + 20, "p": s_ref * 20}),
            wrapper_pfline({"w": s_ref}),
            wrapper_pfline({"p": s_ref * 10}),
            wrapper_pfline({"w": s_ref + 20, "p": s_ref * 20}),
        ),
        (  # Too much information at unsourcedprice.2.
            wrapper_pfline({"w": s_ref}),
            wrapper_pfline({"p": s_ref * 10, "w": s_ref + 10}),
            wrapper_pfline({"w": s_ref + 20, "p": s_ref * 20}),
            wrapper_pfline({"w": s_ref}),
            wrapper_pfline({"p": s_ref * 10}),
            wrapper_pfline({"w": s_ref + 20, "p": s_ref * 20}),
        ),
        (  # Incorrect kind at offtake.3.
            wrapper_pfline({"p": s_ref + 1}),
            wrapper_pfline({"p": s_ref * 10, "w": s_ref + 10}),
            wrapper_pfline({"w": s_ref + 20, "p": s_ref * 20}),
            ValueError,
            None,
            None,
        ),
        (  # Incorrect kind at unsourcedprice.4.
            wrapper_pfline({"w": s_ref}),
            wrapper_pfline({"w": s_ref + 10}),
            wrapper_pfline({"w": s_ref + 20, "p": s_ref * 20}),
            ValueError,
            None,
            None,
        ),
        (  # Incorrect kind at sourced.5.
            wrapper_pfline({"w": s_ref}),
            wrapper_pfline({"p": s_ref * 10}),
            wrapper_pfline({"w": s_ref + 20}),
            ValueError,
            None,
            None,
        ),
        (  # Incorrect kind at sourced.6.
            wrapper_pfline({"w": s_ref}),
            wrapper_pfline({"p": s_ref * 10}),
            wrapper_pfline({"p": s_ref * 20}),
            ValueError,
            None,
            None,
        ),
        (  # No sourcing yet.7.
            wrapper_pfline({"w": s_ref}),
            wrapper_pfline({"p": s_ref * 10}),
            None,
            wrapper_pfline({"w": s_ref}),
            wrapper_pfline({"p": s_ref * 10}),
            wrapper_pfline({"w": s_ref * 0, "r": s_ref * 0}),
            # DimensionalityError,
            # None,
            # None,
        ),
        (  # Unequal periods; result is trimmed.8.
            wrapper_pfline({"w": s_ref}),
            wrapper_pfline({"p": s_more * 10}),
            wrapper_pfline({"w": s_more + 20, "p": s_more * 20}),
            wrapper_pfline({"w": s_ref}),
            wrapper_pfline({"p": s_more * 10}).loc[i_ref],
            wrapper_pfline({"w": s_more + 20, "p": s_more * 20}).loc[i_ref],
        ),
        (  # Unequal periods; error (intersection; not enough sourced volume).9.
            wrapper_pfline({"w": s_ref}),
            wrapper_pfline({"p": s_more * 10}),
            wrapper_pfline({"w": s_less + 20, "p": s_less * 20}),
            ValueError,
            None,
            None,
        ),
        (  # Unequal periods; error (intersection; not enough unsourced prices).10.
            wrapper_pfline({"w": s_ref}),
            wrapper_pfline({"p": s_less * 10}),
            wrapper_pfline({"w": s_more + 20, "p": s_more * 20}),
            ValueError,
            None,
            None,
        ),
        (  # Not passing PfLines.11.
            {"w": s_ref.astype("pint[MW]")},
            {"p": s_ref.astype("pint[Eur/MWh]") * 10},
            {
                "w": (s_ref + 20).astype("pint[MW]"),
                "p": (s_ref * 20).astype("pint[Eur/MWh]"),
            },
            wrapper_pfline({"w": s_ref.astype("pint[MW]")}),
            wrapper_pfline({"p": s_ref.astype("pint[Eur/MWh]") * 10}),
            wrapper_pfline(
                {
                    "w": (s_ref + 20).astype("pint[MW]"),
                    "p": (s_ref * 20).astype("pint[Eur/MWh]"),
                }
            ),
        ),
        (  # Not passing PfLines.12.
            {"w": s_ref.astype("pint[MW]")},
            {"p": s_more.astype("pint[Eur/MWh]") * 10},
            None,
            wrapper_pfline({"w": s_ref.astype("pint[MW]")}),
            wrapper_pfline({"p": s_more.astype("pint[Eur/MWh]") * 10}).loc[i_ref],
            wrapper_pfline({"w": s_ref * 0, "r": s_ref * 0}),
            # DimensionalityError,
            # None,
            # None,
        ),
        (  # Not passing PfLines; error (intersection).13
            {"w": s_more},
            {"p": s_ref * 10},
            None,
            ValueError,
            None,
            None,
        ),
        (  # Unequal frequencies.14.
            wrapper_pfline({"w": s_ref}),
            wrapper_pfline({"p": s_difffreq * 10}),
            None,
            ValueError,
            None,
            None,
        ),
        (  # Unequal frequencies.15.
            wrapper_pfline({"w": s_difffreq}),
            wrapper_pfline({"p": s_ref * 10}),
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
    # o_res, u_res, s_res = o[0],u[0],s[0]
    o_res, u_res, s_res = make_pflines(o, u, s)
    assert o_res == o_exp
    assert u_res == u_exp
    assert s_res == s_exp

    # Test init.
    result = PfState(o, u, s)
    assert result.offtakevolume == o_exp
    assert result.unsourcedprice == u_exp
    assert result.sourced == s_exp


@pytest.mark.parametrize(
    ("wo", "qo", "ws", "qs", "ps", "rs", "pu", "o_exp", "u_exp", "s_exp"),
    [
        (  # Full package with power and price.0
            s_ref,
            None,
            s_ref + 20,
            None,
            s_ref * 20,
            None,
            s_ref * 10,
            wrapper_pfline({"w": s_ref}),
            wrapper_pfline({"p": s_ref * 10}),
            wrapper_pfline({"w": s_ref + 20, "p": s_ref * 20}),
        ),
        (  # Full package with energy and price.1
            None,
            s_ref,
            None,
            s_ref + 20,
            s_ref * 20,
            None,
            s_ref * 10,
            wrapper_pfline({"q": s_ref}),
            wrapper_pfline({"p": s_ref * 10}),
            wrapper_pfline({"q": s_ref + 20, "p": s_ref * 20}),
        ),
        (  # Full package with power and revenue.2.
            s_ref,
            None,
            s_ref + 20,
            None,
            None,
            s_ref * 20,
            s_ref * 10,
            wrapper_pfline({"w": s_ref}),
            wrapper_pfline({"p": s_ref * 10}),
            wrapper_pfline({"w": s_ref + 20, "r": s_ref * 20}),
        ),
        (  # Full package with price and revenue.3.
            s_ref,
            None,
            None,
            None,
            s_ref + 20,
            s_ref * 20,
            s_ref * 10,
            wrapper_pfline({"w": s_ref}),
            wrapper_pfline({"p": s_ref * 10}),
            wrapper_pfline({"p": s_ref + 20, "r": s_ref * 20}),
        ),
        (  # Full package with energy and revenue.4.
            None,
            s_ref,
            None,
            s_ref + 20,
            None,
            s_ref * 20,
            s_ref * 10,
            wrapper_pfline({"q": s_ref}),
            wrapper_pfline({"p": s_ref * 10}),
            wrapper_pfline({"q": s_ref + 20, "r": s_ref * 20}),
        ),
        (  # Too little information at offtake.5.
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
        (  # Too little information at unsourcedprice.6.
            s_ref,
            None,
            s_ref + 20,
            None,
            s_ref * 20,
            None,
            None,
            DimensionalityError,
            None,
            None,
        ),
        (  # Too little information at sourced.7.
            s_ref,
            None,
            s_ref + 20,
            None,
            None,
            None,
            s_ref * 10,
            DimensionalityError,
            None,
            None,
        ),
        (  # Too little information at sourced.8.
            s_ref,
            None,
            None,
            None,
            s_ref * 20,
            None,
            s_ref * 10,
            DimensionalityError,
            None,
            None,
        ),
        (  # No sourcing yet.9.
            s_ref,
            None,
            None,
            None,
            None,
            None,
            s_ref * 10,
            wrapper_pfline({"w": s_ref}),
            wrapper_pfline({"p": s_ref * 10}),
            wrapper_pfline(pd.DataFrame({"q": 0.0, "r": 0.0}, i_ref)),
            # create.flatpfline(pd.DataFrame({"q": 0.0, "r": 0.0}, i_ref)),
        ),
        (  # Unequal periods, no problem.10.
            s_ref,
            None,
            s_more + 20,
            None,
            s_more * 20,
            None,
            s_more * 10,
            wrapper_pfline({"w": s_ref}),
            wrapper_pfline({"p": s_more * 10}).loc[i_ref],
            wrapper_pfline({"w": s_more + 20, "p": s_more * 20}).loc[i_ref],
        ),
        (  # Unequal periods, error (not enough sourced).11.
            s_ref,
            None,
            s_less + 20,
            None,
            s_less * 20,
            None,
            s_more * 10,
            DimensionalityError,
            None,
            None,
        ),
        (  # Unequal periods, error (not enough unsourced).12.
            s_ref,
            None,
            s_more + 20,
            None,
            s_more * 20,
            None,
            s_less * 10,
            DimensionalityError,
            None,
            None,
        ),
        (  # Unequal frequencies.13.
            s_ref,
            None,
            None,
            None,
            None,
            None,
            s_difffreq * 10,
            DimensionalityError,
            None,
            None,
        ),
        (  # Unequal frequencies.14.
            s_difffreq,
            None,
            None,
            None,
            None,
            None,
            s_ref * 10,
            DimensionalityError,
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
    # qo, qs, rs, pu = qo.astype("pint[MWh]"), qs.astype("pint[MWh]"),rs.astype("pint[Eur]"), pu.astype("pint[Eur/MWh]")
    qo, qs, rs, pu, wo, ws, ps = (
        x.astype(dtype) if x is not None else x
        for x, dtype in [
            (qo, "pint[MWh]"),
            (qs, "pint[MWh]"),
            (rs, "pint[Eur]"),
            (pu, "pint[Eur/MWh]"),
            (wo, "pint[MW]"),
            (ws, "pint[MW]"),
            (ps, "pint[Eur/MWh]"),
        ]
    )

    result = PfState.from_series(pu=pu, qo=qo, qs=qs, rs=rs, wo=wo, ws=ws, ps=ps)
    assert result.offtakevolume == o_exp
    assert result.unsourcedprice == u_exp
    assert result.sourced == s_exp


def test_pfstate_consistency_uniformfreq():
    """Test if all values are consistent as expected."""
    # Starting values. (qo defined as being positive.)
    qo, qs, rs, pu = (
        s_ref.astype("pint[MWh]"),
        (s_ref + 20).astype("pint[MWh]"),
        s_ref.astype("pint[Eur]") * 20,
        s_ref.astype("pint[Eur/MWh]") * 10,
    )
    # Create PfState.
    result = PfState.from_series(qo=-qo, qs=qs, rs=rs, pu=pu)
    # Expected.
    o_exp = wrapper_pfline({"q": -qo})
    s_exp = wrapper_pfline({"q": qs, "r": rs})
    u_exp = wrapper_pfline({"p": pu})
    # Test.
    assert result.offtakevolume == o_exp
    assert result.unsourcedprice == u_exp
    assert result.sourced == s_exp
    # !ATTN: before changes it was:
    # testing.assert_series_equal(result.offtakevolume.q.pint.m, -qo, check_names=False)
    testing.assert_series_equal(result.offtakevolume.q, -qo, check_names=False)
    testing.assert_series_equal(result.unsourcedprice.p, pu, check_names=False)
    testing.assert_series_equal(result.sourced.q, qs, check_names=False)
    testing.assert_series_equal(result.sourced.r, rs, check_names=False)
    testing.assert_series_equal(result.sourced.p, rs / qs, check_names=False)
    testing.assert_series_equal(result.unsourced.p, pu, check_names=False)
    testing.assert_series_equal(result.unsourced.q, qo - qs, check_names=False)
    testing.assert_series_equal(result.unsourced.r, pu * (qo - qs), check_names=False)
    testing.assert_series_equal(result.pnl_cost.q, qo, check_names=False)
    testing.assert_series_equal(
        result.pnl_cost.r, rs + pu * (qo - qs), check_names=False
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
    qo, qs, rs, pu = (
        s_less.astype("pint[MWh]"),
        (s_ref + 20).astype("pint[MWh]"),
        s_less.astype("pint[Eur]") * 20,
        s_more.astype("pint[Eur/MWh]") * 10,
    )
    # Create PfState.
    result = PfState.from_series(qo=-qo, qs=qs, rs=rs, pu=pu)
    # Expected.
    qo, qs, rs, pu = qo.loc[i_less], qs.loc[i_less], rs.loc[i_less], pu.loc[i_less]
    o_exp = wrapper_pfline({"q": -qo})
    s_exp = wrapper_pfline({"q": qs, "r": rs})
    u_exp = wrapper_pfline({"p": pu})
    # Test.
    assert result.offtakevolume == o_exp
    assert result.unsourcedprice == u_exp
    assert result.sourced == s_exp
    # !ATTN : same as above
    testing.assert_series_equal(result.offtakevolume.q, -qo, check_names=False)
    testing.assert_series_equal(result.unsourcedprice.p, pu, check_names=False)
    testing.assert_series_equal(result.sourced.q, qs, check_names=False)
    testing.assert_series_equal(result.sourced.r, rs, check_names=False)
    testing.assert_series_equal(result.sourced.p, rs / qs, check_names=False)
    testing.assert_series_equal(result.unsourced.p, pu.loc[i_less], check_names=False)
    testing.assert_series_equal(result.unsourced.q, qo - qs, check_names=False)
    testing.assert_series_equal(
        result.unsourced.r, pu.loc[i_less] * (qo - qs), check_names=False
    )
    testing.assert_series_equal(result.pnl_cost.q, qo, check_names=False)
    testing.assert_series_equal(
        result.pnl_cost.r, rs + pu.loc[i_less] * (qo - qs), check_names=False
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
    qo, pu = s_ref.astype("pint[MWh]"), s_more.astype("pint[Eur/MWh]") * 10
    # Create PfState.
    result = PfState.from_series(qo=-qo, pu=pu)
    # Expected.
    qo, qs, rs, pu = qo, pd.Series(0.0, i_ref), pd.Series(0.0, i_ref), pu.loc[i_ref]
    o_exp = wrapper_pfline({"q": -qo})
    s_exp = wrapper_pfline({"q": qs, "r": rs})
    u_exp = wrapper_pfline({"p": pu})
    # Test.
    assert result.offtakevolume == o_exp
    assert result.unsourcedprice == u_exp
    assert result.sourced == s_exp
    testing.assert_series_equal(result.offtakevolume.q, -qo, check_names=False)
    testing.assert_series_equal(result.unsourcedprice.p, pu, check_names=False)
    testing.assert_series_equal(result.sourced.q.pint.m, qs, check_names=False)
    testing.assert_series_equal(result.sourced.r.pint.m, rs, check_names=False)
    testing.assert_series_equal(
        result.sourced.p.pint.m, pd.Series(np.nan, i_ref), check_names=False
    )
    testing.assert_series_equal(result.unsourced.p, pu.loc[i_ref], check_names=False)
    testing.assert_series_equal(result.unsourced.q, qo, check_names=False)
    testing.assert_series_equal(
        result.unsourced.r, pu.loc[i_ref] * qo, check_names=False
    )
    testing.assert_series_equal(result.pnl_cost.q, qo, check_names=False)
    testing.assert_series_equal(
        result.pnl_cost.r, rs + pu.loc[i_ref] * (qo - qs), check_names=False
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


@pytest.mark.parametrize("inclusive", ["left", "both"])
@pytest.mark.parametrize("freq", ["15min", "h"])
def test_contain_whole_day(inclusive: str, freq: str):
    """An index must contain full days.
    For hourly-or-shorter values, this means that the start time of the first period () must equal the end time of the
    last period (), which is not the case."""
    index = pd.date_range(
        "2020-01-01", "2020-02-01", freq=freq, tz="Europe/Berlin", inclusive=inclusive
    )
    if inclusive == "left":
        # This should work without any error
        pfs = dev.get_pfstate(index)
        assert isinstance(pfs, PfState)
    else:
        # For "both" inclusive, it should raise an error
        with pytest.raises(ValueError):
            pfs = dev.get_pfstate(index)
