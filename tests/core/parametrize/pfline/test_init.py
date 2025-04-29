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
    ],
)
def test_init_ok(args, kwargs):
    _ = pf.PfLineb(*args, **kwargs)


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ([pd.DataFrame({"w": floatseries["w"]})], {}),
        ([pd.DataFrame({"w": floatseries["w"]})], {"commodity": commodity}),
        ([{"w": floatseries["w"]}], {"commodity": commodity}),
    ],
)
def test_init_nok(args, kwargs):
    with pytest.raises(Exception):
        _ = pf.PfLineb(*args, **kwargs)
