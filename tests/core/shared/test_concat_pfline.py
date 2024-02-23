import pandas as pd
import pytest


from portfolyo import dev, testing
from portfolyo.core.shared import concat


TESTCASES = [  # firts pfl, freq, second pfl
    (
        ("2020-01-01", "2023-04-01"),
        "QS",
        ("2023-04-01", "2024-07-01"),
    ),
    (("2020", "2022"), "AS", ("2022", "2024")),
    (("2022-03-20", "2022-07-28"), "D", ("2022-07-28", "2022-10-10")),
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
@pytest.mark.parametrize(("first_idx", "freq", "second_idx"), TESTCASES)
def test_concat_pfline(
    first_idx: str,
    starttime: str,
    tz: str,
    freq: str,
    second_idx: str,
):
    """Test if intersection of indices gives correct result."""
    idxs = [
        get_idx(first_idx[0], starttime, tz, freq, first_idx[1]),
        get_idx(second_idx[0], starttime, tz, freq, second_idx[1]),
    ]
    pfl = dev.get_flatpfline(idxs[0])
    pfl2 = dev.get_flatpfline(idxs[1])
    expected_idx = get_idx(first_idx[0], starttime, tz, freq, second_idx[1])
    expected_pfl = dev.get_flatpfline(expected_idx)
    result = concat.concat_pflines(pfl, pfl2)
    testing.assert_index_equal(result.index, expected_pfl.index)
