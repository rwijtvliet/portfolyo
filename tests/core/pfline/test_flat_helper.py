from typing import Any

import pandas as pd
import pytest

from portfolyo import dev, testing, tools
from portfolyo.core.pfline import flat_helper


@pytest.mark.parametrize("tz", ["Europe/Berlin", None])
@pytest.mark.parametrize("freq", tools.freq.FREQUENCIES)
def test_makedataframe_freqtz(freq, tz):
    """Test if dataframe can made from data with various timezones and frequencies."""

    i = dev.get_index(freq, tz)
    q = dev.get_series(i, "q")
    w = q / q.index.duration
    result1 = flat_helper._dataframe({"q": q})

    expected = pd.DataFrame({"q": q, "w": w})
    expected.index.freq = freq

    testing.assert_frame_equal(result1, expected, check_names=False)


i = pd.date_range("2020-01-01", freq="MS", periods=2)
DF = pd.DataFrame(
    {
        "w": pd.Series([1.0, 2], i, dtype="pint[MW]"),
        "q": pd.Series([744.0, 1392], i, dtype="pint[MWh]"),
        "p": pd.Series([100.0, 50], i, dtype="pint[Eur/MWh]"),
        "r": pd.Series([74400.0, 69600], i, dtype="pint[Eur]"),
    }
)
TESTCASES_INPUTTYPES = [  # data,expected
    (DF[["w"]], DF[["w", "q"]]),
    (DF[["q"]], DF[["w", "q"]]),
    (DF[["w", "q"]], DF[["w", "q"]]),
    ({"w": DF["w"]}, DF[["w", "q"]]),
    ({"w": DF["w"], "q": DF["q"]}, DF[["w", "q"]]),
    ({"w": DF["w"].pint.m}, DF[["w", "q"]]),
    (DF["w"], DF[["w", "q"]]),
    ([DF["w"], DF["q"]], DF[["w", "q"]]),
    (DF[["p"]], DF[["p"]]),
    ({"p": DF["p"].pint.m}, DF[["p"]]),
    (DF[["r"]], DF[["r"]]),
    ({"r": DF["r"].pint.m}, DF[["r"]]),
    (DF, DF),
    (DF[["w", "p"]], DF),
    (DF[["w", "p", "q"]], DF),
    (DF[["r", "w"]], DF),
    ({"r": DF["r"].pint.m, "w": DF["w"]}, DF),
    ([DF["r"], DF["w"]], DF),
]


@pytest.mark.parametrize("data,expected", TESTCASES_INPUTTYPES)
def test_makedataframe_inputtypes(data: Any, expected: pd.DataFrame):
    """Test if dataframe can be created from various input types."""
    result = flat_helper._dataframe(data)
    testing.assert_frame_equal(result, expected)


TESTCASES_COLUMNS = [
    "r",
    "p",
    "q",
    "w",
    "wq",
    "pr",
    "wp",
    "qp",
    "qr",
    "wr",
    "wqp",
    "qpr",
    "wqr",
    "wpr",
    "wqpr",
]


@pytest.mark.parametrize("inputtype", ["dict", "df"])
@pytest.mark.parametrize("tz", ["Europe/Berlin", None])
@pytest.mark.parametrize("freq", ["MS", "D"])
@pytest.mark.parametrize("columns", TESTCASES_COLUMNS)
def test_makedataframe_consistency(tz, freq, columns, inputtype):
    """Test if conversions are done correctly and inconsistent data raises error."""

    i = dev.get_index(freq, tz)
    df = dev.get_dataframe(i, columns)
    dic = {key: df[key] for key in columns}

    if columns in ["wq", "wqp", "wqr", "wpr", "qpr", "wqpr"]:  # error cases
        with pytest.raises(ValueError):
            if inputtype == "dict":
                _ = flat_helper._dataframe(dic)
            else:
                _ = flat_helper._dataframe(df)
        return

    # Actual result.
    if inputtype == "dict":
        result = flat_helper._dataframe(dic)
    else:
        result = flat_helper._dataframe(df)

    # Expected result and testing.
    df = df.rename_axis("ts_left")
    if columns == "p":  # kind is Kind.PRICE
        expected = df[["p"]]

    elif columns in ["q", "w"]:  # kind is Kind.VOLUME
        if columns == "w":
            df["q"] = df.w * df.index.duration
        else:
            df["w"] = df.q / df.index.duration
        expected = df[["w", "q"]]

    elif columns == "r":  # kind is Kind.REVENUE
        expected = df[["r"]]

    else:  # kind is Kind.ALL
        # fill dataframe first.
        expected = df

        if columns == "wp":
            expected["q"] = expected.w * expected.index.duration
            expected["r"] = expected.p * expected.q

        elif columns == "pr":
            expected["q"] = expected.r / expected.p
            expected["w"] = expected.q / expected.index.duration

        elif columns == "qp":
            expected["r"] = expected.p * expected.q
            expected["w"] = expected.q / expected.index.duration

        elif columns == "wr":
            expected["q"] = expected.w * expected.index.duration
            expected["p"] = expected.r / expected.q

        elif columns == "qr":
            expected["p"] = expected.r / expected.q
            expected["w"] = expected.q / expected.index.duration

    testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("freq1", ["15T", "MS", "AS"])  # don't do all - many!
@pytest.mark.parametrize("freq2", ["H", "D", "QS"])
@pytest.mark.parametrize("columns", ["rp", "wp", "pq", "qr", "wr"])
def test_makedataframe_unequalfrequencies(freq1, freq2, columns):
    """Test if error is raised when creating a dataframe from series with unequal frequencies."""

    if freq1 == freq2:
        pytest.skip("Only testing unequal frequencies.")

    kwargs = {
        "start": "2020",
        "end": "2021",
        "inclusive": "left",
        "tz": "Europe/Berlin",
    }
    i1 = pd.date_range(**kwargs, freq=freq1)
    i2 = pd.date_range(**kwargs, freq=freq2)

    s1 = dev.get_series(i1, columns[0], _seed=1)
    s2 = dev.get_series(i2, columns[1], _seed=1)

    dic = {columns[0]: s1, columns[1]: s2}
    with pytest.raises(ValueError):
        _ = flat_helper._dataframe(dic)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("freq", ["15T", "H", "D", "MS"])
def test_makedataframe_unequalstartofday(freq: str, tz: str):
    """Test if error is raised for series with unequal starttimes."""

    kwargs = {"freq": freq, "inclusive": "left", "tz": tz}
    i1 = pd.date_range(start="2020-01-01 00:00", end="2020-06-01", **kwargs)
    i2 = pd.date_range(start="2020-03-01 06:00", end="2020-09-01", **kwargs)
    s1 = dev.get_series(i1, "q")
    s2 = dev.get_series(i2, "r")

    # raise ValueError("The two timeseries have distinct start-of-days.")
    with pytest.raises(ValueError):
        _ = flat_helper._dataframe({"q": s1, "r": s2})


@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("freq", ["15T", "H", "D", "MS"])
@pytest.mark.parametrize("overlap", [True, False])
def test_makedataframe_unequaltimeperiods(freq: str, overlap: bool, tz: str):
    """Test if only intersection is kept for overlapping series, and error is raised
    for non-overlapping series."""

    kwargs = {"freq": freq, "inclusive": "left", "tz": tz}
    start2 = "2020-03-01" if overlap else "2020-07-01"
    i1 = pd.date_range(start="2020-01-01", end="2020-06-01", **kwargs)
    i2 = pd.date_range(start=start2, end="2020-09-01", **kwargs)
    s1 = dev.get_series(i1, "q")
    s2 = dev.get_series(i2, "r")

    intersection_values = [i for i in s1.index if i in s2.index]
    intersection = pd.DatetimeIndex(intersection_values, freq=freq, name="ts_left")

    if not overlap:
        # raise ValueError("The two timeseries do not have anything in common.")
        with pytest.raises(ValueError):
            result = flat_helper._dataframe({"q": s1, "r": s2})
        return

    result = flat_helper._dataframe({"q": s1, "r": s2})
    testing.assert_index_equal(result.index, intersection)
    testing.assert_series_equal(result.q, s1.loc[intersection])
    testing.assert_series_equal(result.r, s2.loc[intersection])
