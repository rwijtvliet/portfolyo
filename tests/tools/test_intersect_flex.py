import pandas as pd
import pytest

from portfolyo import testing, tools


COMMON_END = "2022-02-02"

TESTCASES = [  # startdates, freq, expected_startdate
    # One starts at first day of year.
    (("2020-01-01", "2020-01-20"), "15T", "2020-01-20"),
    (("2020-01-01", "2020-01-20"), "15T", "2020-01-20"),
    (("2020-01-01", "2020-01-20"), "H", "2020-01-20"),
    (("2020-01-01", "2020-01-20"), "H", "2020-01-20"),
    (("2020-01-01", "2020-01-20"), "D", "2020-01-20"),
    (("2020-01-01", "2020-01-20"), "D", "2020-01-20"),
    (("2020-01-01", "2020-03-01"), "MS", "2020-03-01"),
    (("2020-01-01", "2020-03-01"), "MS", "2020-03-01"),
    (("2020-01-01", "2020-04-01"), "QS", "2020-04-01"),
    (("2020-01-01", "2020-04-01"), "QS", "2020-04-01"),
    (("2020-01-01", "2021-01-01"), "AS", "2021-01-01"),
    (("2020-01-01", "2021-01-01"), "AS", "2021-01-01"),
    # Both start in middle of year.
    (("2020-04-21", "2020-06-20"), "15T", "2020-06-20"),
    (("2020-04-21", "2020-06-20"), "15T", "2020-06-20"),
    (("2020-04-21", "2020-06-20"), "H", "2020-06-20"),
    (("2020-04-21", "2020-06-20"), "H", "2020-06-20"),
    (("2020-04-21", "2020-06-20"), "D", "2020-06-20"),
    (("2020-04-21", "2020-06-20"), "D", "2020-06-20"),
]

COMMON_END_2 = "2023-01-01"
TESTCASES_2 = [  # startdates, freq, expected_dates
    # One starts at first day of year.
    (("2020-01-01", "2020-01-20"), ("15T", "H"), ("2020-01-20", "2023-01-01")),
    (("2020-01-01", "2020-01-20"), ("15T", "D"), ("2020-01-20", "2023-01-01")),
    (("2022-04-01", "2021-02-01"), ("H", "MS"), ("2022-04-01", "2023-01-01")),
    (("2020-01-01", "2020-04-01"), ("H", "QS"), ("2020-04-01", "2023-01-01")),
    (("2020-01-01", "2021-01-01"), ("D", "AS"), ("2021-01-01", "2023-01-01")),
    # Both start in middle of year.
    (("2020-04-21", "2020-06-20"), ("15T", "H"), ("2020-06-20", "2023-01-01")),
    (("2020-04-21", "2020-06-20"), ("15T", "D"), ("2020-06-20", "2023-01-01")),
    (("2020-04-21", "2020-07-01"), ("H", "MS"), ("2020-07-01", "2023-01-01")),
    (("2020-04-21", "2020-07-01"), ("H", "QS"), ("2020-07-01", "2023-01-01")),
    (("2020-04-21", "2021-01-01"), ("D", "AS"), ("2021-01-01", "2023-01-01")),
]


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(("startdates", "freq", "expected_startdate"), TESTCASES)
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
def test_intersect_flex_ignore_start_of_day(
    tz, startdates, freq, starttime, expected_startdate
):
    otherstarttime = "00:00" if starttime == "06:00" else "06:00"
    a = pd.date_range(
        f"{startdates[0]} {starttime}",
        f"{COMMON_END} {starttime}",
        freq=freq,
        tz=tz,
        inclusive="left",
    )
    b = pd.date_range(
        f"{startdates[1]} {otherstarttime}",
        f"{COMMON_END} {otherstarttime}",
        freq=freq,
        tz=tz,
        inclusive="left",
    )
    e = (
        pd.date_range(
            f"{expected_startdate} {starttime}",
            f"{COMMON_END} {starttime}",
            freq=freq,
            tz=tz,
            inclusive="left",
        ),
        pd.date_range(
            f"{expected_startdate} {otherstarttime}",
            f"{COMMON_END} {otherstarttime}",
            freq=freq,
            tz=tz,
            inclusive="left",
        ),
    )
    # Test error case.
    with pytest.raises(ValueError):
        _ = tools.intersect.indices_flex(a, b, ignore_start_of_day=False)
    # Test ok case.
    result = tools.intersect.indices_flex(a, b, ignore_start_of_day=True)

    for i in range(0, len(result)):
        testing.assert_index_equal(result[i], e[i])


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(("startdates", "freq", "expected_startdate"), TESTCASES)
@pytest.mark.parametrize("time_a", ["00:00", "06:00"])
def test_intersect_flex_ignore_tz(tz, startdates, freq, time_a, expected_startdate):
    othertz = None if tz == "Europe/Berlin" else "Europe/Berlin"
    a = pd.date_range(
        f"{startdates[0]} {time_a}",
        f"{COMMON_END} {time_a}",
        freq=freq,
        tz=tz,
        inclusive="left",
    )
    b = pd.date_range(
        f"{startdates[1]} {time_a}",
        f"{COMMON_END} {time_a}",
        freq=freq,
        tz=othertz,
        inclusive="left",
    )
    e = (
        pd.date_range(
            f"{expected_startdate} {time_a}",
            f"{COMMON_END} {time_a}",
            freq=freq,
            tz=tz,
            inclusive="left",
        ),
        pd.date_range(
            f"{expected_startdate} {time_a}",
            f"{COMMON_END} {time_a}",
            freq=freq,
            tz=othertz,
            inclusive="left",
        ),
    )
    # Test error case.
    with pytest.raises(ValueError):
        _ = tools.intersect.indices_flex(a, b, ignore_tz=False)
    # Test ok case.
    result = tools.intersect.indices_flex(a, b, ignore_tz=True)

    for i in range(0, len(result)):
        testing.assert_index_equal(result[i], e[i])


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(("startdates", "freq", "expected_dates"), TESTCASES_2)
@pytest.mark.parametrize("time_a", ["00:00", "06:00"])
def test_intersect_flex_ignore_freq(tz, startdates, freq, time_a, expected_dates):
    """Test if intersection of indices with distinct frequencies gives correct result."""
    a = pd.date_range(
        f"{startdates[0]} {time_a}",
        f"{COMMON_END_2} {time_a}",
        freq=freq[0],
        tz=tz,
        inclusive="left",
    )
    b = pd.date_range(
        f"{startdates[1]} {time_a}",
        f"{COMMON_END_2} {time_a}",
        freq=freq[1],
        tz=tz,
        inclusive="left",
    )
    expected_a = pd.date_range(
        f"{expected_dates[0]} {time_a}",
        f"{expected_dates[1]} {time_a}",
        freq=freq[0],
        tz=tz,
        inclusive="left",
    )
    expected_b = pd.date_range(
        f"{expected_dates[0]} {time_a}",
        f"{expected_dates[1]} {time_a}",
        freq=freq[1],
        tz=tz,
        inclusive="left",
    )
    # Test error case.
    with pytest.raises(ValueError):
        _ = tools.intersect.indices_flex(a, b, ignore_freq=False)

    result_a, result_b = tools.intersect.indices_flex(a, b, ignore_freq=True)

    testing.assert_index_equal(result_a, expected_a)
    testing.assert_index_equal(result_b, expected_b)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(("startdates", "freq", "expected_dates"), TESTCASES_2)
@pytest.mark.parametrize("starttime", ["00:00", "06:00"])
def test_ignore_all(tz, startdates, freq, starttime, expected_dates):
    otherstarttime = "00:00" if starttime == "06:00" else "06:00"
    othertz = None if tz == "Europe/Berlin" else "Europe/Berlin"
    a = pd.date_range(
        f"{startdates[0]} {starttime}",
        f"{COMMON_END_2} {starttime}",
        freq=freq[0],
        tz=tz,
        inclusive="left",
    )
    b = pd.date_range(
        f"{startdates[1]} {otherstarttime}",
        f"{COMMON_END_2} {otherstarttime}",
        freq=freq[1],
        tz=othertz,
        inclusive="left",
    )
    expected_a = pd.date_range(
        f"{expected_dates[0]} {starttime}",
        f"{expected_dates[1]} {starttime}",
        freq=freq[0],
        tz=tz,
        inclusive="left",
    )
    expected_b = pd.date_range(
        f"{expected_dates[0]} {otherstarttime}",
        f"{expected_dates[1]} {otherstarttime}",
        freq=freq[1],
        tz=othertz,
        inclusive="left",
    )
    # Test error cases.
    with pytest.raises(ValueError):
        _ = tools.intersect.indices_flex(
            a, b, ignore_freq=False, ignore_start_of_day=False, ignore_tz=False
        )

    with pytest.raises(ValueError):
        _ = tools.intersect.indices_flex(
            a, b, ignore_freq=False, ignore_start_of_day=True, ignore_tz=True
        )

    with pytest.raises(ValueError):
        _ = tools.intersect.indices_flex(
            a, b, ignore_freq=True, ignore_start_of_day=False, ignore_tz=True
        )

    with pytest.raises(ValueError):
        _ = tools.intersect.indices_flex(
            a, b, ignore_freq=True, ignore_start_of_day=True, ignore_tz=False
        )

    # Test ok case.
    out_a, out_b = tools.intersect.indices_flex(
        a, b, ignore_freq=True, ignore_start_of_day=True, ignore_tz=True
    )
    testing.assert_index_equal(out_a, expected_a)
    testing.assert_index_equal(out_b, expected_b)
