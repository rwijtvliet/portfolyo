from functools import lru_cache
from typing import Any, Mapping

import pandas as pd
import pytest

from portfolyo import Kind, dev, testing, tools
from portfolyo.core.pfline import multi_helper


@pytest.mark.parametrize("freq", ["MS", "D"])
@pytest.mark.parametrize("kind1", [Kind.ALL, Kind.VOLUME_ONLY, Kind.PRICE_ONLY])
@pytest.mark.parametrize("kind2", [Kind.ALL, Kind.VOLUME_ONLY, Kind.PRICE_ONLY, None])
@pytest.mark.parametrize("kind3", [Kind.ALL, Kind.VOLUME_ONLY, Kind.PRICE_ONLY, None])
def test_verifydict_kindconsistency(freq, kind1, kind2, kind3):
    """Test if conversions are done correctly and inconsistent kind raises error."""

    i = dev.get_index(freq, "Europe/Berlin")

    kinds, dic = [], {}
    for k, kind in enumerate([kind1, kind2, kind3]):
        if kind is not None:
            kinds.append(kind)
            dic[f"part_{k}"] = dev.get_singlepfline(i, kind)

    if len(dic) == 1:
        pass

    elif len(dic) == 2:
        kset = set(kinds)
        if len(kset) != 1 and kset != set([Kind.PRICE_ONLY, Kind.VOLUME_ONLY]):
            # Can only combine 2 pflines if they have the same kind or are 'q' and 'p'
            with pytest.raises(ValueError):
                _ = multi_helper.verify_and_trim_dict(dic)
            return

    elif len(dic) == 3:
        if len(set(kinds)) != 1:
            # Can only combine 3 pflines if they have the same kind.
            with pytest.raises(ValueError):
                _ = multi_helper.verify_and_trim_dict(dic)
            return

    result = multi_helper.verify_and_trim_dict(dic)
    assert result == dic


@lru_cache()
def pfl_for_frequencyconsistency(freq: str):
    i = pd.date_range("2020", "2021", inclusive="left", tz="Europe/Berlin", freq=freq)
    return dev.get_singlepfline(i, "pri", _random=False)


@pytest.mark.slow
@pytest.mark.parametrize("freq1", ["15T", "D", "MS", "QS"])  # don't do all - many!
@pytest.mark.parametrize("freq2", ["15T", "H", "D", "MS", "QS"])
def test_verifydict_frequencyconsistency(freq1, freq2):
    """Test if error is raised when creating a dictionary from pflines with unequal frequencies."""

    dic = {
        f"part_{i}": pfl_for_frequencyconsistency(freq)
        for i, freq in enumerate([freq1, freq2])
    }

    if freq1 != freq2:
        # Expect error.
        with pytest.raises(ValueError):
            _ = multi_helper.verify_and_trim_dict(dic)
        return
    else:
        # Expect no error.
        _ = multi_helper.verify_and_trim_dict(dic)


@lru_cache()
def pfl_for_unequaltimeperiods_1(freq: str):
    i = pd.date_range(
        "2020-01-01", "2020-06-01", freq=freq, inclusive="left", tz="Europe/Berlin"
    )
    return dev.get_singlepfline(i, "all", _random=False)


@lru_cache()
def pfl_for_unequaltimeperiods_2(start: str, freq: str):
    i = pd.date_range(
        start, "2020-09-01", freq=freq, inclusive="left", tz="Europe/Berlin"
    )
    return dev.get_singlepfline(i, "all", _random=False)


@pytest.mark.slow
@pytest.mark.parametrize("freq", ["15T", "H", "D", "MS"])
@pytest.mark.parametrize("overlap", [True, False])
def test_verifydict_unequaltimeperiods(freq, overlap):
    """Test if only intersection is kept for overlapping pflines, and error is raised
    for non-overlapping pflines."""

    start = "2020-03-01" if overlap else "2020-07-01"
    spfl1 = pfl_for_unequaltimeperiods_1(freq)
    spfl2 = pfl_for_unequaltimeperiods_2(start, freq)
    dic = {"PartA": spfl1, "PartB": spfl2}

    intersection = tools.intersect.indices(spfl1.index, spfl2.index)

    if not overlap:
        # raise error (two portfoliolines do not have anything in common.)
        with pytest.raises(ValueError):
            result = multi_helper.verify_and_trim_dict(dic)
        return

    result = multi_helper.verify_and_trim_dict(dic)
    for name, child in result.items():
        testing.assert_series_equal(child.q, dic[name].loc[intersection].q)
        testing.assert_series_equal(child.r, dic[name].loc[intersection].r)
        testing.assert_index_equal(child.index, intersection)


input_simple = {"a": 2, "b": 34, "c": -1234.3}
input_complicated = {"a": 2, "b": None, "c": "bla", None: [1, 32, -4.5, "x"], 5: "test"}

i = [1, 4, -5, 60]
vals1, vals2, vals3 = [12, 34, 45, 78], [-12, 23.2, 18, "test"], [None, 5, -3, 44]
s1, s2, s3 = pd.Series(vals1, i), pd.Series(vals2, i), pd.Series(vals3, i)
input_df_1simple = pd.DataFrame({"a": s1, "b": s2})
input_df_2simple = pd.DataFrame({("a", "a1"): s1, ("a", "a2"): s2, ("b", "b1"): s3})
input_df_2complicated = pd.DataFrame({(22, "a1"): s1, (22, 5): s2, ("b", "b1"): s3})
input_df_3 = pd.concat(
    [input_df_2simple, input_df_2complicated], axis=1, keys=["AA", "CC"]
)


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        (input_simple, input_simple),
        (input_complicated, input_complicated),
        (input_df_1simple, {"a": s1, "b": s2}),
        (
            input_df_2simple,
            {"a": pd.DataFrame({"a1": s1, "a2": s2}), "b": pd.DataFrame({"b1": s3})},
        ),
        (
            input_df_2simple,
            {"a": pd.DataFrame({"a1": s1, "a2": s2}), "b": pd.DataFrame({"b1": s3})},
        ),
        (
            input_df_2complicated,
            {22: pd.DataFrame({"a1": s1, 5: s2}), "b": pd.DataFrame({"b1": s3})},
        ),
        (input_df_3, {"AA": input_df_2simple, "CC": input_df_2complicated}),
    ],
)
def test_makemapping(input: Any, expected: Mapping):
    result = multi_helper.make_mapping(input)
    assert isinstance(result, Mapping)
    assert set(key for key in result) == set(key for key in expected)
    for key, result_val in result.items():
        expected_val = expected[key]
        if isinstance(result_val, pd.Series) or isinstance(result_val, pd.DataFrame):
            assert all(result_val == expected_val)
        else:
            assert result_val == expected_val
