# I have a pytest integer fixture fix1 which is parametrized with the values 1,2,3, and another pytest integer fixture fix2 which is parametrized with the values 10,11, 20. I want to create a fixture fix3 which returns every value for fix1 and for fix2, one after the other, as long as the value is odd. So, fix3 should return the integer values 1, then 3, then 11. How do I do that?

import pytest


# Define fixture for `fix1`
@pytest.fixture(params=[[1], [2], [3]])
def alist(request):
    return request.param


# Define fixture for `fix2`
@pytest.fixture(params=[9, 10, 11])
def anelement(request):
    return request.param


def test_1(anelement):
    print(f"test_1: {anelement:=}")


@pytest.fixture
def longer_list(alist, anelement):
    alist.append(anelement)


def test_listcontent(longer_list):
    print("---".join(longer_list))
