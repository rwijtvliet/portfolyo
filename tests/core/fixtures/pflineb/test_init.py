from typing import Iterable, Literal

import numpy as np
import pandas as pd
import pytest

import portfolyo as pf


@pytest.fixture(scope="module")
def idx():
    return pd.date_range("2020", "2021", freq="h", inclusive="left")


@pytest.fixture(scope="module")
def floatseries(idx):
    # 100...0...100 throughout idx
    return pd.Series(np.cos(1 + np.linspace(0, 2 * np.pi, len(idx), endpoint=False)) * 100, idx)


@pytest.fixture
def energyseries(floatseries):
    return floatseries.astype("pint[kWh]")


@pytest.fixture
def emissionsseries(floatseries):
    return floatseries.astype("pint[tCO2]")


@pytest.fixture
def energyrateseries(floatseries):
    return floatseries.astype("pint[W]")


@pytest.fixture
def emissionsrateseries(floatseries):
    return floatseries.astype("pint[tCO2/min]")


@pytest.fixture
def energypriceseries_eur(floatseries):
    return floatseries.astype("pint[Eur/MWh]")


@pytest.fixture
def emissionspriceseries_eur(floatseries):
    return floatseries.astype("pint[Eur/tCO2]")


@pytest.fixture
def energypriceseries_usd(floatseries):
    return floatseries.astype("pint[Usd/MWh]")


@pytest.fixture
def emissionspriceseries_usd(floatseries):
    return floatseries.astype("pint[Usd/tCO2]")


@pytest.fixture
def revenueseries_eur(floatseries):
    return floatseries.astype("pint[Eur]")


@pytest.fixture
def revenueseries_usd(floatseries):
    return floatseries.astype("pint[Usd]")


@pytest.fixture(
    params=[
        "energyseries",
        "emissionsseries",
        "energyrateseries",
        "emissionsrateseries",
        "energypriceseries_eur",
        "emissionspriceseries_eur",
        "energypriceseries_usd",
        "emissionspriceseries_usd",
        "revenueseries_eur",
        "revenueseries_usd",
    ]
)
def valid_pintseries(request):
    series = request.getfixturevalue(request.param)
    return series.rename(request.param)


@pytest.fixture(scope="session", params=["list", "tuple", "generator"])
def iterabletype(request):
    return request.param


def make_iterable(iterable, iterabletype: Literal["list", "tuple", "generator"]) -> Iterable:
    if iterabletype == "list":
        return list(iterable)
    elif iterabletype == "tuple":
        return tuple(iterable)
    elif iterabletype == "generator":
        return (element for element in iterable)


@pytest.fixture(
    params=[
        {"q": "energyseries", "p": "energypriceseries_eur"},
        {"q": "energyseries", "p": "energypriceseries_usd"},
        {"w": "energyrateseries", "p": "energypriceseries_eur"},
        {"w": "energyrateseries", "p": "energypriceseries_usd"},
        {"q": "energyseries", "r": "revenueseries_eur"},
        {"q": "energyseries", "r": "revenueseries_usd"},
        {"w": "energyrateseries", "r": "revenueseries_eur"},
        {"w": "energyrateseries", "r": "revenueseries_usd"},
        {"p": "energypriceseries_eur", "r": "revenueseries_eur"},
        {"p": "energypriceseries_usd", "r": "revenueseries_usd"},
        {"q": "emissionsseries", "p": "emissionspriceseries_eur"},
        {"q": "emissionsseries", "p": "emissionspriceseries_usd"},
    ]
)
def valid_pintseries_dict(request) -> dict[str, pd.Series]:
    return {
        col: request.getfixturevalue(seriesfixname) for col, seriesfixname in request.param.items()
    }


@pytest.fixture
def valid_pintseries_iterable(valid_pintseries_dict, iterabletype) -> Iterable[pd.Series]:
    return make_iterable(valid_pintseries_dict.values(), iterabletype)


def test_init_with_pintseries(valid_pintseries):
    _ = pf.PfLineb(valid_pintseries)


def test_init_with_pintseries_iterable(valid_pintseries_iterable):
    _ = pf.PfLineb(valid_pintseries_iterable)


def test_init_with_pintseries_dict(valid_pintseries_dict):
    _ = pf.PfLineb(valid_pintseries_dict)
