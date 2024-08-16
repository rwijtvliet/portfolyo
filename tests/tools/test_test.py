import pytest


@pytest.fixture(
    scope="session",
    params=[
        {
            "val": "a2",
            "wei": "a1",
            "exp": "a1",
            "axis": "a0",
        },
        {
            "val": "b2",
            "wei": "b2",
            "exp": "b1",
            "axis": "b1",
        },
        {
            "val": "c3",
            "wei": "c2",
            "exp": "c1",
            "axis": "c0",
        },
        {
            "val": "d2",
            "wei": "d2",
            "exp": "d1",
            "axis": "d1",
        },
    ],
)
def complexcases(request):
    return request.param


@pytest.fixture(
    scope="session",
    params=[
        {
            "val": "e1",
            "wei": "e1",
            "exp": "e0",
        },
        {
            "val": "f1",
            "wei": "f1",
            "exp": "f0",
        },
        {
            "val": "g1",
            "wei": "g1",
            "exp": "g0",
        },
        {
            "val": "h1",
            "wei": "h0",
            "exp": "h2",
        },
    ],
)
def easycases(request):
    return request.param


@pytest.fixture
def val2d(easycases, complexcases):
    allcases = [easycases] + [complexcases]
    values = [cas["val"] for cas in allcases]
    return [val for val in values if val[1] == "2"]


@pytest.fixture
def val1d(easycases, complexcases):
    allcases = [*easycases, *complexcases]
    return [case["val"] for case in allcases if case["val"][1] == "1"]


@pytest.fixture
def val0d(easycases, complexcases):
    allcases = [*easycases, *complexcases]
    return [case["val"] for case in allcases if case["val"][1] == "0"]


def test_two_d_cases(val2d):
    print(val2d)
