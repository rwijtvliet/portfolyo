import pandas as pd
import pytest

import portfolyo as pf


def get_idx(
    startdate: str,
    starttime: str,
    tz: str,
    freq: str,
    enddate: str,
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
def test_intersect_freq_ignore(
    starttime: str,
    tz: str,
):
    """Test that intersection works properly on PfLines and/or PfStates with ignore_freq."""
    idx1 = get_idx("2022-04-01", starttime, tz, "QS", "2024-07-01")
    s1 = pd.Series(range(len(idx1)), idx1)

    idx2 = get_idx("2021-01-01", starttime, tz, "MS", "2024-01-01")
    s2 = pd.Series(range(len(idx2)), idx2)

    # fn = pf.PfState if test_fn == "get_pfstate" else pf.PfLine
    pf_a = pf.PfLine({"w": s1})
    pf_b = pf.PfLine({"w": s2})
    # Do intersection
    intersect = pf.intersection(pf_a, pf_b, ignore_freq=True)
    # Expected results
    expected_s1 = s1.iloc[:7]
    expected_s2 = s2.iloc[15:48]
    output1 = pf.PfLine({"w": expected_s1})
    output2 = pf.PfLine({"w": expected_s2})
    assert output1 == intersect[0]
    assert output2 == intersect[1]


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
def test_intersect_sod(
    starttime: str,
    tz: str,
):
    """Test that intersection works properly on PfLines and/or PfStates with ignore_freq."""
    idx1 = get_idx("2022-04-01", starttime, tz, "QS", "2024-07-01")
    s1 = pd.Series(range(len(idx1)), idx1)

    otherstarttime = "00:00" if starttime == "06:00" else "06:00"
    idx2 = get_idx("2021-01-01", otherstarttime, tz, "QS", "2024-01-01")
    s2 = pd.Series(range(len(idx2)), idx2)

    # fn = pf.PfState if test_fn == "get_pfstate" else pf.PfLine
    pf_a = pf.PfLine({"w": s1})
    pf_b = pf.PfLine({"w": s2})
    # Do intersection
    intersect = pf.intersection(pf_a, pf_b, ignore_start_of_day=True)
    # Expected results
    expected_s1 = s1.iloc[:7]
    expected_s2 = s2.iloc[5:12]
    output1 = pf.PfLine({"w": expected_s1})
    output2 = pf.PfLine({"w": expected_s2})
    assert output1 == intersect[0]
    assert output2 == intersect[1]


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
def test_intersect_tz(
    starttime: str,
    tz: str,
):
    """Test that intersection works properly on PfLines and/or PfStates with ignore_freq."""
    idx1 = get_idx("2022-04-01", starttime, tz, "QS", "2024-07-01")
    s1 = pd.Series(range(len(idx1)), idx1)

    othertz = None if tz == "Europe/Berlin" else "Europe/Berlin"
    idx2 = get_idx("2021-01-01", starttime, othertz, "QS", "2024-01-01")
    s2 = pd.Series(range(len(idx2)), idx2)

    # fn = pf.PfState if test_fn == "get_pfstate" else pf.PfLine
    pf_a = pf.PfLine({"w": s1})
    pf_b = pf.PfLine({"w": s2})
    # Do intersection
    intersect = pf.intersection(pf_a, pf_b, ignore_tz=True)
    # Expected results
    expected_s1 = s1.iloc[:7]
    expected_s2 = s2.iloc[5:12]
    output1 = pf.PfLine({"w": expected_s1})
    output2 = pf.PfLine({"w": expected_s2})
    assert output1 == intersect[0]
    assert output2 == intersect[1]
