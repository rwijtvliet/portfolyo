"""Test if concatenation of PfStates works properly with different test cases."""

import pandas as pd
import pytest
from portfolyo import dev
from portfolyo.core.shared import concat


TESTCASES2 = [  # whole idx, freq, where
    (
        ("2020-01-01", "2023-04-01"),
        "QS",
        "2022-04-01",
    ),
    (("2020", "2022"), "AS", "2021-01-01"),
    (
        ("2020-05-01", "2023-04-01"),
        "MS",
        "2022-11-01",
    ),
    (("2022-03-20", "2022-07-28"), "D", "2022-05-28"),
]

TESTCASES3 = [  # whole idx, freq, where
    (
        ("2020-01-01", "2023-04-01"),
        "QS",
        ("2022-04-01", "2023-01-01"),
    ),
    (("2020", "2023"), "AS", ("2021-01-01", "2022-01-01")),
    (
        ("2020-05-01", "2023-04-01"),
        "MS",
        ("2022-11-01", "2023-01-01"),
    ),
    (("2022-03-20", "2022-07-28"), "D", ("2022-04-28", "2022-05-15")),
]


def get_idx(
    startdate: str, starttime: str, tz: str, freq: str, enddate: str
) -> pd.DatetimeIndex:
    # Empty index.
    if startdate is None:
        return pd.DatetimeIndex([], freq=freq, tz=tz)
    # Normal index.
    ts_start = pd.Timestamp(f"{startdate} {starttime}", tz=tz)
    ts_end = pd.Timestamp(f"{enddate} {starttime}", tz=tz)
    return pd.date_range(ts_start, ts_end, freq=freq, inclusive="left")


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
@pytest.mark.parametrize(("whole_idx", "freq", "where"), TESTCASES2)
@pytest.mark.parametrize("test_fn", ["general", "concat_pfstates"])
def test_concat_pfstates(
    whole_idx: str,
    starttime: str,
    tz: str,
    freq: str,
    where: str,
    test_fn: str,
):
    """Test that two PfStates get concatenated properly."""
    idx = get_idx(whole_idx[0], starttime, tz, freq, whole_idx[1])
    whole_pfs = dev.get_pfstate(idx)
    pfs_a = whole_pfs.slice[:where]
    pfs_b = whole_pfs.slice[where:]
    fn = concat.general if test_fn == "general" else concat.concat_pfstates
    result = fn([pfs_a, pfs_b])
    result2 = fn([pfs_b, pfs_a])
    assert whole_pfs == result
    assert whole_pfs == result2


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
@pytest.mark.parametrize(("whole_idx", "freq", "where"), TESTCASES3)
@pytest.mark.parametrize("test_fn", ["general", "concat_pfstates"])
def test_concat_three_pfstates(
    whole_idx: str,
    starttime: str,
    tz: str,
    freq: str,
    where: str,
    test_fn: str,
):
    """Test that three PfStates get concatenated properly."""
    idx = get_idx(whole_idx[0], starttime, tz, freq, whole_idx[1])
    whole_pfs = dev.get_pfstate(idx)
    split_one = where[0]
    split_two = where[1]
    pfs_a = whole_pfs.slice[:split_one]
    pfs_b = whole_pfs.slice[split_one:split_two]
    pfs_c = whole_pfs.slice[split_two:]
    fn = concat.general if test_fn == "general" else concat.concat_pfstates
    result = fn([pfs_a, pfs_b, pfs_c])
    result2 = fn([pfs_b, pfs_c, pfs_a])
    assert whole_pfs == result
    assert whole_pfs == result2
