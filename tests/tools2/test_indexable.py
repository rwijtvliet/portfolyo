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


def create_obj(
    series: pd.Series, name_obj: str
) -> pd.DataFrame | pf.PfLine | pf.PfState:
    if name_obj == "pfline":
        return pf.PfLine({"w": series})
    elif name_obj == "pfstate":
        volume = pf.PfLine({"w": series})
        prices = pf.PfLine({"p": series})
        return pf.PfState(volume, prices)
    else:
        return pd.DataFrame({"col": series})


def t_function(objtype: str):
    if objtype == "series":
        return pd.testing.assert_series_equal
    elif objtype == "dataframe":
        return pd.testing.assert_frame_equal
    elif objtype == "pfline":
        return pf.PfLine.__eq__
    else:
        return pf.PfState.__eq__


@pytest.mark.parametrize("first_obj", ["pfstate", "pfline", "series", "dataframe"])
@pytest.mark.parametrize("second_obj", ["pfstate", "pfline", "series", "dataframe"])
def test_intersect_freq_ignore(
    first_obj: str,
    second_obj: str,
):
    """Test that intersection works properly on PfLines and/or PfStates with ignore_freq."""
    idx1 = get_idx("2022-04-01", "00:00", "Europe/Berlin", "QS", "2024-07-01")
    s1 = pd.Series(range(len(idx1)), idx1)

    idx2 = get_idx("2021-01-01", "00:00", "Europe/Berlin", "MS", "2024-01-01")
    s2 = pd.Series(range(len(idx2)), idx2)

    first = create_obj(s1, first_obj) if first_obj != "series" else s1
    second = create_obj(s2, second_obj) if second_obj != "series" else s2
    # Do intersection
    intersect = pf.intersection(first, second, ignore_freq=True)

    # Expected results
    expected_s1 = s1.iloc[:7]
    expected_s2 = s2.iloc[15:48]
    output_1 = (
        create_obj(expected_s1, first_obj) if first_obj != "series" else expected_s1
    )
    output_2 = (
        create_obj(expected_s2, second_obj) if second_obj != "series" else expected_s2
    )
    for a, b, objtype in zip([output_1, output_2], intersect, [first_obj, second_obj]):
        fn = t_function(objtype)
        fn(a, b)


@pytest.mark.parametrize("first_obj", ["pfstate", "pfline", "series", "dataframe"])
@pytest.mark.parametrize("second_obj", ["pfstate", "pfline", "series", "dataframe"])
def test_intersect_sod(
    first_obj: str,
    second_obj: str,
):
    """Test that intersection works properly on PfLines and/or PfStates with ignore_sod."""
    idx1 = get_idx("2022-04-01", "00:00", "Europe/Berlin", "QS", "2024-07-01")
    s1 = pd.Series(range(len(idx1)), idx1)

    idx2 = get_idx("2021-01-01", "06:00", "Europe/Berlin", "QS", "2024-01-01")
    s2 = pd.Series(range(len(idx2)), idx2)

    first = create_obj(s1, first_obj) if first_obj != "series" else s1
    second = create_obj(s2, second_obj) if second_obj != "series" else s2
    # Do intersection
    intersect = pf.intersection(first, second, ignore_start_of_day=True)

    # Expected results
    expected_s1 = s1.iloc[:7]
    expected_s2 = s2.iloc[5:12]
    output_1 = (
        create_obj(expected_s1, first_obj) if first_obj != "series" else expected_s1
    )
    output_2 = (
        create_obj(expected_s2, second_obj) if second_obj != "series" else expected_s2
    )
    for a, b, objtype in zip([output_1, output_2], intersect, [first_obj, second_obj]):
        fn = t_function(objtype)
        fn(a, b)


@pytest.mark.parametrize("first_obj", ["pfstate", "pfline", "series", "dataframe"])
@pytest.mark.parametrize("second_obj", ["pfstate", "pfline", "series", "dataframe"])
def test_intersect_tz(
    first_obj: str,
    second_obj: str,
):
    """Test that intersection works properly on PfLines and/or PfStates with ignore_tz."""
    idx1 = get_idx("2022-04-01", "00:00", "Europe/Berlin", "QS", "2024-07-01")
    s1 = pd.Series(range(len(idx1)), idx1)

    idx2 = get_idx("2021-01-01", "00:00", None, "QS", "2024-01-01")
    s2 = pd.Series(range(len(idx2)), idx2)

    first = create_obj(s1, first_obj) if first_obj != "series" else s1
    second = create_obj(s2, second_obj) if second_obj != "series" else s2
    # Do intersection
    intersect = pf.intersection(first, second, ignore_tz=True)

    # Expected results
    expected_s1 = s1.iloc[:7]
    expected_s2 = s2.iloc[5:12]
    output_1 = (
        create_obj(expected_s1, first_obj) if first_obj != "series" else expected_s1
    )
    output_2 = (
        create_obj(expected_s2, second_obj) if second_obj != "series" else expected_s2
    )
    for a, b, objtype in zip([output_1, output_2], intersect, [first_obj, second_obj]):
        fn = t_function(objtype)
        fn(a, b)


@pytest.mark.parametrize("first_obj", ["pfstate", "pfline", "series", "dataframe"])
@pytest.mark.parametrize("second_obj", ["pfstate", "pfline", "series", "dataframe"])
def test_intersect_ignore_all(
    first_obj: str,
    second_obj: str,
):
    """Test that intersection works properly on PfLines and/or PfStates with ignore_all."""
    idx1 = get_idx("2022-04-01", "00:00", "Europe/Berlin", "QS", "2024-07-01")
    s1 = pd.Series(range(len(idx1)), idx1)

    idx2 = get_idx("2021-01-01", "06:00", None, "MS", "2024-01-01")
    s2 = pd.Series(range(len(idx2)), idx2)

    first = create_obj(s1, first_obj) if first_obj != "series" else s1
    second = create_obj(s2, second_obj) if second_obj != "series" else s2
    # Do intersection
    intersect = pf.intersection(
        first, second, ignore_freq=True, ignore_tz=True, ignore_start_of_day=True
    )

    # Expected results
    expected_s1 = s1.iloc[:7]
    expected_s2 = s2.iloc[15:48]
    output_1 = (
        create_obj(expected_s1, first_obj) if first_obj != "series" else expected_s1
    )
    output_2 = (
        create_obj(expected_s2, second_obj) if second_obj != "series" else expected_s2
    )
    for a, b, objtype in zip([output_1, output_2], intersect, [first_obj, second_obj]):
        fn = t_function(objtype)
        fn(a, b)


@pytest.mark.parametrize("first_obj", ["pfstate", "pfline", "series", "dataframe"])
@pytest.mark.parametrize("second_obj", ["pfstate", "pfline", "series", "dataframe"])
@pytest.mark.parametrize("third_obj", ["pfstate", "pfline", "series", "dataframe"])
def test_intersect_ignore_all_3obj(
    first_obj: str,
    second_obj: str,
    third_obj: str,
):
    """Test that intersection works properly on PfLines and/or PfStates with ignore_all."""
    idx1 = get_idx("2022-04-01", "00:00", "Europe/Berlin", "QS", "2024-07-01")
    s1 = pd.Series(range(len(idx1)), idx1)

    idx2 = get_idx("2021-01-01", "06:00", None, "MS", "2024-01-01")
    s2 = pd.Series(range(len(idx2)), idx2)

    idx3 = get_idx("2023-01-01", "00:00", "Asia/Kolkata", "YS", "2025-01-01")
    s3 = pd.Series(range(len(idx3)), idx3)

    first = create_obj(s1, first_obj) if first_obj != "series" else s1
    second = create_obj(s2, second_obj) if second_obj != "series" else s2
    third = create_obj(s3, third_obj) if third_obj != "series" else s3

    # Do intersection
    intersect = pf.intersection(
        first, second, third, ignore_freq=True, ignore_tz=True, ignore_start_of_day=True
    )

    # Expected results
    expected_s1 = s1.iloc[3:7]
    expected_s2 = s2.iloc[24:36]
    expected_s3 = s3.iloc[:1]
    output_1 = (
        create_obj(expected_s1, first_obj) if first_obj != "series" else expected_s1
    )
    output_2 = (
        create_obj(expected_s2, second_obj) if second_obj != "series" else expected_s2
    )
    output_3 = (
        create_obj(expected_s3, third_obj) if third_obj != "series" else expected_s3
    )

    for a, b, objtype in zip(
        [output_1, output_2, output_3], intersect, [first_obj, second_obj, third_obj]
    ):
        fn = t_function(objtype)
        fn(a, b)
