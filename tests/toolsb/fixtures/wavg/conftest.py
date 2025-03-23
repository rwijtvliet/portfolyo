import pytest


@pytest.fixture(scope="module", params=["list", "series", "dict"])
def weights_as(request) -> str:
    return request.param
