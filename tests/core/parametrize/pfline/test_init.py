import pandas as pd
import pytest

import portfolyo as pf

commodity = pf.power_ger
idx = pd.date_range("2020", "2021", freq="D", inclusive="left")
floats = {"w": 4.0, "q": 4.0 * 24, "p": 110.0, "r": 110.0 * 4 * 24}  # MW, MWh, Eur/MWh, Eur
floatseries = {col: pd.Series(fl, idx) for col, fl in floats.items()}
pintseries = {
    "w": (floatseries["w"] * 1000).astype("pint[kW]"),
    "q": (floatseries["q"] * 1000).astype("pint[kWh]"),
    "p": (floatseries["p"] / 10).astype("pint[ctEur/kWh]"),
    "r": (floatseries["r"] * 100).astype("pint[ctEur]"),
}
quantities = {c: s.iloc[0] for c, s in pintseries.items()}


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ([pd.DataFrame({"w": pintseries["w"]})], {"commodity": commodity}),
        (
            [pd.DataFrame({"w": floatseries["w"]})],
            {"commodity": commodity, "no_units": "imply"},
        ),
        (
            [{"w": floatseries["w"], "q": floatseries["q"]}],
            {"commodity": commodity, "no_units": "imply"},
        ),
        (
            [{"w": pintseries["w"], "q": quantities["q"]}],
            {"commodity": commodity},
        ),
    ],
)
def test_volume(args, kwargs):
    pfl = pf.PfLineb(*args, **kwargs)
    assert pfl.kind == pf.VOLUME
    assert all(col1 == col2 for col1, col2 in zip(pfl.columns, pfl.kind.available, strict=True))


@pytest.mark.parametrize(
    "args,kwargs",
    [
        (
            [{"w": floatseries["w"], "r": pintseries["r"]}],
            {"commodity": commodity, "no_units": "imply"},
        ),
        (
            [{"w": pintseries["w"], "r": pintseries["r"]}],
            {"commodity": commodity},
        ),
        (
            [{"w": pintseries["w"], "r": pintseries["r"]}],
            {"commodity": commodity, "no_units": "imply"},
        ),
    ],
)
def test_complete(args, kwargs):
    pfl = pf.PfLineb(*args, **kwargs)
    assert pfl.kind == pf.COMPLETE
    assert all(col1 == col2 for col1, col2 in zip(pfl.columns, pfl.kind.available, strict=True))


@pytest.mark.parametrize(
    "args,kwargs",
    [
        (  # no commodity, no units
            [pd.DataFrame({"w": floatseries["w"]})],
            {},
        ),
        (  # no units
            [pd.DataFrame({"w": floatseries["w"]})],
            {"commodity": commodity},
        ),
        (  # no units
            [{"w": floatseries["w"]}],
            {"commodity": commodity},
        ),
        (  # incompatible values
            [{"w": pintseries["w"], "q": floatseries["r"].astype("pint[MW]")}],
            {"commodity": commodity},
        ),
    ],
)
def test_init_nok(args, kwargs):
    with pytest.raises(Exception):
        _ = pf.PfLineb(*args, **kwargs)
