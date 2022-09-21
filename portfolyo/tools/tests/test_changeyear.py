import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Tuple

import numpy as np
import pandas as pd
import pytest
import yaml
from portfolyo import testing
from portfolyo.tools import changeyear

TESTOBJECTFILE = pathlib.Path(__file__).parent / "test_changeyear_data.yaml"


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
class Yearmapping:
    idx_source: pd.DatetimeIndex  # source index
    idx_target: pd.DatetimeIndex  # target index
    holiday_country: str
    expected_mapping: pd.Series  # for every day in the target index, get position in source


def get_yearmapping_factory():

    yearmappings: Dict[Tuple, Yearmapping] = {}
    for tc in yaml.load(open(TESTOBJECTFILE), Loader=yaml.FullLoader):
        # Testcase with daily frequency
        source_year, target_year = tc["source_year"], tc["target_year"]
        country, tz = tc["holiday_country"], tc["tz"]
        mapping = np.array(tc["mapping"])
        idx_s = pd.date_range(
            str(source_year), str(source_year + 1), freq="D", closed="left", tz=tz
        )
        idx_t = pd.date_range(
            str(target_year), str(target_year + 1), freq="D", closed="left", tz=tz
        )
        key = (source_year, target_year, tz, country, "D")
        yearmappings[key] = Yearmapping(idx_s, idx_t, country, mapping)

        # Testcase with hourly frequency
        idx_s2 = pd.date_range(
            str(source_year), str(source_year + 1), freq="H", closed="left", tz=tz
        )
        idx_t2 = pd.date_range(
            str(target_year), str(target_year + 1), freq="H", closed="left", tz=tz
        )
        count_s = pd.Series(0, idx_s2).resample("D").count()
        count_t = pd.Series(0, idx_t2).resample("D").count()
        mapping_hourly = []
        for count_t_here, m in zip(count_t, mapping):
            count_s_start = count_s[:m].sum()
            mapping_hourly.extend(range(count_s_start, count_s_start + count_t_here))
        key = (source_year, target_year, tz, country, "H")
        yearmappings[key] = Yearmapping(idx_s2, idx_t2, country, mapping_hourly)

        # Testcase for identical mapping (days)
        key = (source_year, source_year, tz, country, "D")
        yearmappings[key] = Yearmapping(idx_s, idx_s, country, range(len(idx_s)))

        # Testcase for identical mapping (hours)
        key = (source_year, source_year, tz, country, "H")
        yearmappings[key] = Yearmapping(idx_s2, idx_s2, country, range(len(idx_s2)))

    def get_yearmapping(
        source_year: int, target_year: int, tz: str, holiday_country: str, freq: str
    ) -> Yearmapping:
        return yearmappings.get((source_year, target_year, tz, holiday_country, freq))

    return get_yearmapping


get_yearmapping = get_yearmapping_factory()

# ---


@pytest.mark.parametrize("freq", ["15T", "H", "MS", "QS", "AS"])
@pytest.mark.parametrize("tz", ["Europe/Berlin", None])
def test_characterize_error(freq: str, tz: str):
    """Test if characterization is only possible for daily indices."""
    i = pd.date_range("2020", "2022", closed="left", freq=freq, tz=tz)
    with pytest.raises(ValueError):
        _ = changeyear.characterize_index(i)


@pytest.mark.parametrize("yearchar", germanyearchars())
@pytest.mark.parametrize("tz", ["Europe/Berlin", None])
@pytest.mark.parametrize("holiday_country", [None, "DE"])
def test_characterize_weekdays(
    yearchar: GermanyYearchar, tz: str, holiday_country: str
):
    """Test if the weekdays of an index are correctly characterized."""
    idx = pd.date_range(
        str(yearchar.year), str(yearchar.year + 1), freq="D", tz=tz, closed="left"
    )
    char = changeyear.characterize_index(idx, holiday_country)
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
def test_characterize_dst(yearchar: GermanyYearchar, tz: str, holiday_country: str):
    """Test if the DST days of an index are correctly characterized."""
    idx = pd.date_range(
        str(yearchar.year), str(yearchar.year + 1), freq="D", tz=tz, closed="left"
    )
    char = changeyear.characterize_index(idx, holiday_country)
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
def test_characterize_holidays(
    yearchar: GermanyYearchar, tz: str, holiday_country: str
):
    """Test if the holidays of an index are correctly characterized."""
    idx = pd.date_range(
        str(yearchar.year), str(yearchar.year + 1), freq="D", tz=tz, closed="left"
    )
    char = changeyear.characterize_index(idx, holiday_country)
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
        (
            pd.date_range("2020", "2021", freq="D", closed="left", tz=None),
            pd.date_range("2021", "2022", freq="D", closed="left", tz="Europe/Berlin"),
        ),
        (
            pd.date_range("2020", "2021", freq="D", closed="left", tz="Europe/Berlin"),
            pd.date_range("2021", "2022", freq="D", closed="left", tz=None),
        ),
        (
            pd.date_range("2020", "2021", freq="D", closed="left", tz="Asia/Kolkata"),
            pd.date_range("2021", "2022", freq="D", closed="left", tz="Europe/Berlin"),
        ),
        (
            pd.date_range("2020", "2021", freq="D", closed="left", tz=None),
            pd.date_range("2021", "2022", freq="15T", closed="left", tz=None),
        ),
        (
            pd.date_range("2020", "2021", freq="15T", closed="left", tz=None),
            pd.date_range("2021", "2022", freq="D", closed="left", tz=None),
        ),
        (
            pd.date_range("2020", "2021", freq="D", closed="left", tz="Europe/Berlin"),
            pd.date_range(
                "2021", "2022", freq="15T", closed="left", tz="Europe/Berlin"
            ),
        ),
    ],
)
@pytest.mark.parametrize("holiday_country", ["DE", None])
def test_map_index_error(
    idx_source: pd.DatetimeIndex, idx_target: pd.DatetimeIndex, holiday_country: str
):
    """Test if error cases are correctly identified."""
    with pytest.raises(Exception):
        _ = changeyear.map_index(
            idx_source, idx_target, holiday_country=holiday_country
        )


@pytest.mark.parametrize("tz", ["Europe/Berlin", None])
@pytest.mark.parametrize("holiday_country", ["DE", None])
@pytest.mark.parametrize("numyears", [1, 3])
@pytest.mark.parametrize("partial", [True, False])
def test_map_index_identical(
    tz: str, holiday_country: str, numyears: int, partial: bool
):
    """Test if the (trivial) mapping is done correctly from index to itself."""
    idx = pd.date_range("2020", str(2020 + numyears), freq="D", closed="left", tz=tz)
    if partial:
        idx_target = idx[: len(idx) // 2]
    else:
        idx_target = idx
    result = changeyear.map_index(idx, idx_target, holiday_country)
    expected = pd.Series(idx_target, idx_target)
    testing.assert_series_equal(result, expected, check_names=False)


@pytest.mark.parametrize("source_year", [2020, 2021])
@pytest.mark.parametrize("target_year", [2020, 2021])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("holiday_country", [None, "DE"])
@pytest.mark.parametrize("partial", [True, False])
@pytest.mark.parametrize("freq", ["D", "H"])
def test_map_index_general(
    source_year: int,
    target_year: int,
    tz: str,
    holiday_country: str,
    partial: bool,
    freq: str,
):
    """Test if the mapping is done correctly between indices."""
    ym = get_yearmapping(source_year, target_year, tz, holiday_country, freq)
    if ym is None:
        pytest.skip("This test case is not found.")

    count = len(ym.idx_target) // (2 if partial else 1)
    idx_target = ym.idx_target[:count]
    expected_mapping = ym.expected_mapping[:count]

    expected = pd.Series(ym.idx_source[expected_mapping], idx_target)
    result = changeyear.map_index(ym.idx_source, idx_target, holiday_country)
    testing.assert_series_equal(result, expected, check_names=False)


@pytest.mark.parametrize("source_year", [2020, 2021])
@pytest.mark.parametrize("target_year", [2020, 2021])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("holiday_country", [None, "DE"])
@pytest.mark.parametrize("partial", [True, False])
@pytest.mark.parametrize("series_or_df", ["series", "df"])
@pytest.mark.parametrize("freq", ["D", "H"])
def test_frame_set_index(
    source_year: int,
    target_year: int,
    tz: str,
    holiday_country: str,
    partial: bool,
    series_or_df: str,
    freq: str,
):
    """Test if the mapping is done correctly between frames."""
    ym = get_yearmapping(source_year, target_year, tz, holiday_country, freq)

    if ym is None:
        pytest.skip("This test case is not found.")

    values_in = np.random.rand(len(ym.idx_source))

    count = len(ym.idx_target) // (2 if partial else 1)
    idx_target = ym.idx_target[:count]
    expected_mapping = ym.expected_mapping[:count]

    if series_or_df == "series":
        frame_in = pd.Series(values_in, ym.idx_source)
        expected = pd.Series(values_in[expected_mapping], idx_target)
        result = changeyear.frame_set_index(frame_in, idx_target, holiday_country)
        testing.assert_series_equal(result, expected, check_names=False)
    else:
        frame_in = pd.DataFrame({"a": values_in, "b": values_in + 1}, ym.idx_source)
        expected = pd.DataFrame(
            {
                "a": values_in[expected_mapping],
                "b": (values_in + 1)[expected_mapping],
            },
            idx_target,
        )
        result = changeyear.frame_set_index(frame_in, idx_target, holiday_country)
        testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("source_year", [2020, 2021])
@pytest.mark.parametrize("target_year", [2020, 2021])
@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("holiday_country", [None, "DE"])
@pytest.mark.parametrize("partial", [True, False])
@pytest.mark.parametrize("series_or_df", ["series", "df"])
@pytest.mark.parametrize("freq", ["D", "H"])
def test_frame_set_year(
    source_year: int,
    target_year: int,
    tz: str,
    holiday_country: str,
    partial: bool,
    series_or_df: str,
    freq: str,
):
    """Test if the mapping is done correctly between frames."""
    ym = get_yearmapping(source_year, target_year, tz, holiday_country, freq)

    if ym is None:
        pytest.skip("This test case is not found.")

    values_in = np.random.rand(len(ym.idx_source))

    count = len(ym.idx_target) // (2 if partial else 1)
    idx_target = ym.idx_target[:count]
    expected_mapping = ym.expected_mapping[:count]

    if series_or_df == "series":
        frame_in = pd.Series(values_in, ym.idx_source)
        expected = pd.Series(values_in[expected_mapping], idx_target)
        result = changeyear.frame_set_year(frame_in, target_year, holiday_country)
        testing.assert_series_equal(result, expected, check_names=False)
    else:
        frame_in = pd.DataFrame({"a": values_in, "b": values_in + 1}, ym.idx_source)
        expected = pd.DataFrame(
            {
                "a": values_in[expected_mapping],
                "b": (values_in + 1)[expected_mapping],
            },
            idx_target,
        )
        result = changeyear.frame_set_year(frame_in, target_year, holiday_country)
        testing.assert_frame_equal(result, expected)
