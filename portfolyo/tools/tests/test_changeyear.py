import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

import numpy as np
import pandas as pd
import pytest
import yaml

from portfolyo import testing, tools

TESTCASES_DATA = Path(__file__).parent / "test_changeyear_data.yaml"
TESTCASES_DATA_2YEARS = Path(__file__).parent / "test_changeyear_data_2years.yaml"


@dataclass
class GermanyYearchar:
    year: int
    startisoweekday: int  # isoweekday that the year starts with
    count: int  # number of days in year
    dst: Iterable[Tuple[int, int]]  # (month, day) of days that have dst changeover
    holidays: Iterable[Tuple[int, int]]  # (month, day) of days that are holidays


def germanyearchars() -> Iterator[GermanyYearchar]:
    """Generator to get the characteristics of a few years."""
    yield GermanyYearchar(
        2020,
        3,
        366,
        ((3, 29), (10, 25)),
        [
            (1, 1),
            (4, 10),
            (4, 13),
            (5, 1),
            (5, 21),
            (6, 1),
            (10, 3),
            (12, 25),
            (12, 26),
        ],
    )
    yield GermanyYearchar(
        2021,
        5,
        365,
        ((3, 28), (10, 31)),
        [
            (1, 1),
            (4, 2),
            (4, 5),
            (5, 1),
            (5, 13),
            (5, 24),
            (10, 3),
            (12, 25),
            (12, 26),
        ],
    )
    yield GermanyYearchar(
        2022,
        6,
        365,
        ((3, 27), (10, 30)),
        [
            (1, 1),
            (4, 15),
            (4, 18),
            (5, 1),
            (5, 26),
            (6, 6),
            (10, 3),
            (12, 25),
            (12, 26),
        ],
    )


@dataclass
class TestCase:
    idx_source: pd.DatetimeIndex  # source index
    idx_target: pd.DatetimeIndex  # target index
    holiday_country: str
    expected_mapping: pd.Series  # for every day in the target index, get position in source


def create_testcase_factory():
    def add_testcase(tc: dict, several_years: bool):

        # Testcase with daily frequency.
        year_s_start, year_t_start = tc["source_year"], tc["target_year"]
        year_s_end, year_t_end = year_s_start + 1, year_t_start + 1
        country, tz = tc["holiday_country"], tc["tz"]
        mapping = np.array(tc["mapping"])
        if several_years:
            if year_s_start == 2021:
                year_s_end += 1
            else:
                year_t_end += 1

        idx_s = pd.date_range(
            str(year_s_start), str(year_s_end), freq="D", inclusive="left", tz=tz
        )
        idx_t = pd.date_range(
            str(year_t_start), str(year_t_end), freq="D", inclusive="left", tz=tz
        )
        key = (year_s_start, year_t_start, tz, country, "D", several_years)
        testcases[key] = TestCase(idx_s, idx_t, country, mapping)

        # Testcase with hourly frequency.
        idx_s2 = pd.date_range(
            str(year_s_start), str(year_s_end), freq="H", inclusive="left", tz=tz
        )
        idx_t2 = pd.date_range(
            str(year_t_start), str(year_t_end), freq="H", inclusive="left", tz=tz
        )
        count_s = pd.Series(0, idx_s2).resample("D").count()
        count_t = pd.Series(0, idx_t2).resample("D").count()
        mapping_hourly = []
        for count_t_here, m in zip(count_t, mapping):
            count_s_start = count_s[:m].sum()
            mapping_hourly.extend(range(count_s_start, count_s_start + count_t_here))
        key = (year_s_start, year_t_start, tz, country, "H", several_years)

        testcases[key] = TestCase(idx_s2, idx_t2, country, mapping_hourly)
        # Testcase for identical mapping (days).
        key = (year_s_start, year_s_start, tz, country, "D", several_years)
        testcases[key] = TestCase(idx_s, idx_s, country, range(len(idx_s)))

        # Testcase for identical mapping (hours).
        key = (year_s_start, year_s_start, tz, country, "H", several_years)
        testcases[key] = TestCase(idx_s2, idx_s2, country, range(len(idx_s2)))

    testcases: Dict[Tuple, TestCase] = {}
    for tc in yaml.load(open(TESTCASES_DATA), Loader=yaml.FullLoader):
        # 2020 <-> 2021
        add_testcase(tc, False)
    for tc in yaml.load(open(TESTCASES_DATA_2YEARS), Loader=yaml.FullLoader):
        # 2020 <-> 2021/2022
        add_testcase(tc, True)

    def testcase(
        year_s: int,
        year_t: int,
        tz: str,
        holiday_country: str,
        freq: str,
        several_years: bool,
    ) -> TestCase:
        return testcases.get((year_s, year_t, tz, holiday_country, freq, several_years))

    return testcase


get_testcase = functools.lru_cache(100)(create_testcase_factory())

# ---


@pytest.mark.parametrize("freq", ["15T", "H", "MS", "QS", "AS"])
@pytest.mark.parametrize("tz", ["Europe/Berlin", None])
def test_characterizeindex_error(freq: str, tz: str):
    """Test if characterization is only possible for daily indices."""
    i = pd.date_range("2020", "2022", inclusive="left", freq=freq, tz=tz)
    with pytest.raises(ValueError):
        _ = tools.changeyear.characterize_index(i)


@pytest.mark.parametrize("yearchar", germanyearchars())
@pytest.mark.parametrize("tz", ["Europe/Berlin", None])
@pytest.mark.parametrize("holiday_country", [None, "DE"])
def test_characterizeindex_weekdays(
    yearchar: GermanyYearchar, tz: str, holiday_country: str
):
    """Test if the weekdays of an index are correctly characterized."""
    idx = pd.date_range(
        str(yearchar.year), str(yearchar.year + 1), freq="D", tz=tz, inclusive="left"
    )
    char = tools.changeyear.characterize_index(idx, holiday_country)
    # Test index.
    for i1, i2 in zip(char.index, idx):
        assert i1 == i2
    assert len(char.index) == yearchar.count
    # Test weekday.
    expected = yearchar.startisoweekday
    for isoweekday in char.isoweekday:
        assert isoweekday == expected
        expected = (expected % 7) + 1


@pytest.mark.parametrize("yearchar", germanyearchars())
@pytest.mark.parametrize("tz", ["Europe/Berlin", None])
@pytest.mark.parametrize("holiday_country", [None, "DE"])
def test_characterizeindex_dst(
    yearchar: GermanyYearchar, tz: str, holiday_country: str
):
    """Test if the DST days of an index are correctly characterized."""
    idx = pd.date_range(
        str(yearchar.year), str(yearchar.year + 1), freq="D", tz=tz, inclusive="left"
    )
    char = tools.changeyear.characterize_index(idx, holiday_country)
    # Test dst.
    if tz is None:
        assert not any(char.dst_change)
    else:
        has_change = char[char.dst_change]
        assert len(has_change) == len(yearchar.dst)
        for i, (mm, dd) in zip(has_change.index, yearchar.dst):
            assert i.month == mm
            assert i.day == dd


@pytest.mark.parametrize("yearchar", germanyearchars())
@pytest.mark.parametrize("tz", ["Europe/Berlin", None])
@pytest.mark.parametrize("holiday_country", [None, "DE"])
def test_characterizeindex_holidays(
    yearchar: GermanyYearchar, tz: str, holiday_country: str
):
    """Test if the holidays of an index are correctly characterized."""
    idx = pd.date_range(
        str(yearchar.year), str(yearchar.year + 1), freq="D", tz=tz, inclusive="left"
    )
    char = tools.changeyear.characterize_index(idx, holiday_country)
    # Test holiday.
    if holiday_country is None:
        assert not any(char.holiday)
    else:
        is_holiday = char[char.holiday != ""]
        assert len(is_holiday) == len(yearchar.holidays)
        for i, (mm, dd) in zip(is_holiday.index, yearchar.holidays):
            assert i.month == mm
            assert i.day == dd


@pytest.mark.parametrize(
    ("idx_source", "idx_target"),
    [
        # Distinct timezones
        (
            pd.date_range("2020", "2021", freq="D", inclusive="left", tz=None),
            pd.date_range(
                "2021", "2022", freq="D", inclusive="left", tz="Europe/Berlin"
            ),
        ),
        (
            pd.date_range(
                "2020", "2021", freq="D", inclusive="left", tz="Europe/Berlin"
            ),
            pd.date_range("2021", "2022", freq="D", inclusive="left", tz=None),
        ),
        (
            pd.date_range(
                "2020", "2021", freq="D", inclusive="left", tz="Asia/Kolkata"
            ),
            pd.date_range(
                "2021", "2022", freq="D", inclusive="left", tz="Europe/Berlin"
            ),
        ),
        # Distinct frequencies
        (
            pd.date_range("2020", "2021", freq="D", inclusive="left", tz=None),
            pd.date_range("2021", "2022", freq="15T", inclusive="left", tz=None),
        ),
        (
            pd.date_range("2020", "2021", freq="15T", inclusive="left", tz=None),
            pd.date_range("2021", "2022", freq="D", inclusive="left", tz=None),
        ),
        (
            pd.date_range(
                "2020", "2021", freq="D", inclusive="left", tz="Europe/Berlin"
            ),
            pd.date_range(
                "2021", "2022", freq="H", inclusive="left", tz="Europe/Berlin"
            ),
        ),
        # Too few source data
        (
            pd.date_range("2020", "2020-06", freq="D", inclusive="left", tz=None),
            pd.date_range("2021", "2022", freq="D", inclusive="left", tz=None),
        ),
    ],
)
@pytest.mark.parametrize("holiday_country", ["DE", None])
def test_mapindextoindex_error(
    idx_source: pd.DatetimeIndex, idx_target: pd.DatetimeIndex, holiday_country: str
):
    """Test if error cases are correctly identified."""
    with pytest.raises(Exception):
        _ = tools.changeyear.map_index_to_index(
            idx_source, idx_target, holiday_country=holiday_country
        )


@pytest.mark.parametrize("tz", ["Europe/Berlin", None])
@pytest.mark.parametrize("holiday_country", ["DE", None])
@pytest.mark.parametrize("numyears", [1, 3])
@pytest.mark.parametrize("freq", ["MS", "D", "H"])
@pytest.mark.parametrize("partial", [True, False])
def test_mapindextoindex_identical(
    tz: str, holiday_country: str, numyears: int, partial: bool, freq: str
):
    """Test if the (trivial) mapping is done correctly from index to itself."""
    idx = pd.date_range(
        "2020", str(2020 + numyears), freq=freq, inclusive="left", tz=tz
    )
    if partial:
        idx_target = idx[: len(idx) // 2]
    else:
        idx_target = idx
    result = tools.changeyear.map_index_to_index(idx, idx_target, holiday_country)
    expected = pd.Series(idx_target, idx_target)
    testing.assert_series_equal(result, expected, check_names=False)


@pytest.mark.parametrize(
    ("freq", "start_s", "start_t", "periods_s", "periods_t", "expected_mapping"),
    [
        ("MS", "2020", "2021", 7, 7, range(7)),
        ("MS", "2020", "2021", 14, 14, [12, 13, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1]),
        ("MS", "2020", "2021", 12, 14, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1]),
        ("MS", "2020", "2021", 14, 12, [12, 13, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        ("MS", "2020-03", "2021", 12, 12, [10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ("QS", "2020", "2021", 4, 4, range(4)),
        ("QS", "2020", "2021", 7, 7, [4, 5, 6, 3, 0, 1, 2]),
        ("QS", "2020", "2021", 4, 7, [0, 1, 2, 3, 0, 1, 2]),
        ("QS", "2020", "2021", 7, 4, [4, 5, 6, 3]),
        ("AS", "2020", "2024", 4, 4, range(4)),
        ("AS", "2020", "2024", 7, 7, [4, 5, 6, 0, 1, 2, 3]),
        ("AS", "2020", "2024", 4, 7, [0, 1, 2, 3, 0, 1, 2]),
        ("AS", "2020", "2024", 7, 4, [4, 5, 6, 0]),
    ],
)
@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("holiday_country", [None, "DE"])
def test_mapindextoindex_monthlyandlonger(
    start_s: str,
    start_t: str,
    periods_s: int,
    periods_t: int,
    freq: str,
    tz: str,
    holiday_country: str,
    expected_mapping: Iterable[int],
):
    """Test if the mapping is done correctly between monthly indices."""
    idx_source = pd.date_range(
        start_s, periods=periods_s, tz=tz, freq=freq, inclusive="left"
    )
    idx_target = pd.date_range(
        start_t, periods=periods_t, tz=tz, freq=freq, inclusive="left"
    )

    expected = pd.Series(idx_source[expected_mapping], idx_target)
    result = tools.changeyear.map_index_to_index(
        idx_source, idx_target, holiday_country
    )
    testing.assert_series_equal(result, expected, check_names=False)


@pytest.mark.parametrize("year_s", [2020, 2021])
@pytest.mark.parametrize("year_t", [2020, 2021])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("holiday_country", [None, "DE"])
@pytest.mark.parametrize("partial", [True, False])
@pytest.mark.parametrize("freq", ["D", "H"])
@pytest.mark.parametrize("several_years", [True, False])
def test_mapindextoindex_daysandhours(
    year_s: int,
    year_t: int,
    tz: str,
    holiday_country: str,
    partial: bool,
    freq: str,
    several_years: bool,
):
    """Test if the mapping is done correctly between daily and hourly indices."""
    tc = get_testcase(year_s, year_t, tz, holiday_country, freq, several_years)
    if tc is None:
        pytest.skip("This test case is not found.")

    count = len(tc.idx_target) // (2 if partial else 1)
    idx_target = tc.idx_target[:count]
    expected_mapping = tc.expected_mapping[:count]

    expected = pd.Series(tc.idx_source[expected_mapping], idx_target)
    result = tools.changeyear.map_index_to_index(
        tc.idx_source, idx_target, holiday_country
    )
    testing.assert_series_equal(result, expected, check_names=False)


@pytest.mark.parametrize("year_s", [2020, 2021])
@pytest.mark.parametrize("year_t", [2020, 2021])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("holiday_country", [None, "DE"])
@pytest.mark.parametrize("partial", [True, False])
@pytest.mark.parametrize("series_or_df", ["series", "df"])
@pytest.mark.parametrize("freq", ["D", "H"])
@pytest.mark.parametrize("several_years", [True, False])
def test_mapframetoindex(
    year_s: int,
    year_t: int,
    tz: str,
    holiday_country: str,
    partial: bool,
    series_or_df: str,
    freq: str,
    several_years: bool,
):
    """Test if the mapping is done correctly between frames."""
    tc = get_testcase(year_s, year_t, tz, holiday_country, freq, several_years)

    if tc is None:
        pytest.skip("This test case is not found.")

    values_in = np.random.rand(len(tc.idx_source))

    count = len(tc.idx_target) // (2 if partial else 1)
    idx_target = tc.idx_target[:count]
    expected_mapping = tc.expected_mapping[:count]

    if series_or_df == "series":
        frame_in = pd.Series(values_in, tc.idx_source)
        expected = pd.Series(values_in[expected_mapping], idx_target)
        result = tools.changeyear.map_frame_to_index(
            frame_in, idx_target, holiday_country
        )
        testing.assert_series_equal(result, expected, check_names=False)
    else:
        frame_in = pd.DataFrame({"a": values_in, "b": values_in + 1}, tc.idx_source)
        expected = pd.DataFrame(
            {
                "a": values_in[expected_mapping],
                "b": (values_in + 1)[expected_mapping],
            },
            idx_target,
        )
        result = tools.changeyear.map_frame_to_index(
            frame_in, idx_target, holiday_country
        )
        testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("year_s", [2020, 2021])
@pytest.mark.parametrize("year_t", [2020, 2021])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("holiday_country", [None, "DE"])
@pytest.mark.parametrize("series_or_df", ["series", "df"])
@pytest.mark.parametrize("freq", ["D", "H"])
@pytest.mark.parametrize("several_years_source", [True, False])
def test_mapframetoyear_oneyear(
    year_s: int,
    year_t: int,
    tz: str,
    holiday_country: str,
    series_or_df: str,
    freq: str,
    several_years_source: bool,
):
    """Test if the mapping is done correctly between frames."""

    if several_years_source:
        if year_s != 2021 or year_t != 2021:
            pytest.skip("Only test mapping several years when mapping to itself.")
        tc = get_testcase(year_s, year_t, tz, holiday_country, freq, True)
    else:
        tc = get_testcase(year_s, year_t, tz, holiday_country, freq, False)

    if tc is None:
        pytest.skip("This test case is not found.")

    values_in = np.random.rand(len(tc.idx_source))

    if series_or_df == "series":
        frame_in = pd.Series(values_in, tc.idx_source)
        expected = pd.Series(values_in[tc.expected_mapping], tc.idx_target)
        result = tools.changeyear.map_frame_to_year(frame_in, year_t, holiday_country)
        testing.assert_series_equal(result, expected, check_names=False)
    else:
        frame_in = pd.DataFrame({"a": values_in, "b": values_in + 1}, tc.idx_source)
        expected = pd.DataFrame(
            {
                "a": values_in[tc.expected_mapping],
                "b": (values_in + 1)[tc.expected_mapping],
            },
            tc.idx_target,
        )
        result = tools.changeyear.map_frame_to_year(frame_in, year_t, holiday_country)
        testing.assert_frame_equal(result, expected)
