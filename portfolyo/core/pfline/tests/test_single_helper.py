import numpy as np
import pandas as pd
import pytest

from portfolyo import dev, testing
from portfolyo.core.pfline import single_helper
from portfolyo.tools.nits import Q_
from portfolyo.tools.stamps import FREQUENCIES


def assert_w_q_compatible(freq, w, q):
    if freq == "15T":
        testing.assert_series_equal(q, w * Q_(0.25, "h"), check_names=False)
    elif freq == "H":
        testing.assert_series_equal(q, w * Q_(1, "h"), check_names=False)
    elif freq == "D":
        assert (q >= w * Q_(22.99, "h")).all()
        assert (q <= w * Q_(25.01, "h")).all()
    elif freq == "MS":
        assert (q >= w * 27 * Q_(24, "h")).all()
        assert (q <= w * 32 * Q_(24, "h")).all()
    elif freq == "QS":
        assert (q >= w * 89 * Q_(24, "h")).all()
        assert (q <= w * 93 * Q_(24, "h")).all()
    elif freq == "AS":
        assert (q >= w * Q_(8759.9, "h")).all()
        assert (q <= w * Q_(8784.1, "h")).all()
    else:
        raise ValueError("Uncaught value for freq: {freq}.")


def assert_p_q_r_compatible(r, p, q):
    testing.assert_series_equal(r, q * p, check_names=False)


@pytest.mark.parametrize("tz", ["Europe/Berlin", None])
@pytest.mark.parametrize("freq", FREQUENCIES)
def test_makedataframe_freqtz(freq, tz):
    """Test if dataframe can made from data with various timezones and frequencies."""

    i = dev.get_index(freq, tz)
    q = dev.get_series(i, "q")
    result1 = single_helper.make_dataframe({"q": q})

    expected = pd.DataFrame({"q": q})
    expected.index.freq = freq

    testing.assert_frame_equal(result1, expected, check_names=False)

    if tz:
        w = q / q.index.duration
        result2 = single_helper.make_dataframe({"w": w})
        testing.assert_frame_equal(
            result2, expected, check_names=False, check_dtype=False
        )


@pytest.mark.parametrize("inputtype", ["dict", "df"])
@pytest.mark.parametrize("tz", ["Europe/Berlin", None])
@pytest.mark.parametrize("freq", ["MS", "D"])
@pytest.mark.parametrize(
    "columns",
    [
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
    ],
)
def test_makedataframe_consistency(tz, freq, columns, inputtype):
    """Test if conversions are done correctly and inconsistent data raises error."""

    i = dev.get_index(freq, tz)
    df = dev.get_dataframe(i, columns)
    dic = {key: df[key] for key in columns}

    if columns in ["wq", "wqp", "wqr", "wpr", "qpr", "wqpr"]:  # error cases
        with pytest.raises(ValueError):
            if inputtype == "dict":
                _ = single_helper.make_dataframe(dic)
            else:
                _ = single_helper.make_dataframe(df)
        return

    # Actual result.
    if inputtype == "dict":
        result = single_helper.make_dataframe(dic)
    else:
        result = single_helper.make_dataframe(df)

    # Expected result and testing.
    df = df.rename_axis("ts_left")
    if columns == "p":  # kind is Kind.PRICE_ONLY
        expected = df[["p"]]
        testing.assert_frame_equal(result, expected)

    elif columns in ["q", "w"]:  # kind is Kind.VOLUME_ONLY
        if columns == "w":
            df["q"] = df.w * df.w.index.duration
        expected = df[["q"]]
        testing.assert_frame_equal(result, expected)

    elif columns in ["pr", "qp", "wp", "qr", "wr", "r"]:  # kind is Kind.ALL
        # fill dataframe first.
        if columns == "wp":
            df["q"] = df.w * df.w.index.duration
            df["r"] = df.p * df.q

        elif columns == "pr":
            df["q"] = df.r / df.p
            df["w"] = df.q / df.index.duration

        elif columns == "qp":
            df["r"] = df.p * df.q
            df["w"] = df.q / df.index.duration

        elif columns == "wr":
            df["q"] = df.w * df.w.index.duration
            df["p"] = df.r / df.q

        elif columns == "qr":
            df["p"] = df.r / df.q
            df["w"] = df.q / df.index.duration

        elif columns == "r":
            df["w"] = pd.Series(0, df.index, "pint[MW]")
            df["q"] = pd.Series(0, df.index, "pint[MWh]")
            df["p"] = pd.Series(np.nan, df.index, "pint[Eur/MWh]")

        # Ensure we will be expecting the correct values.
        assert_w_q_compatible(freq, df.w, result.q)
        if columns != "r":
            assert_p_q_r_compatible(result.r, df.p, result.q)
        expected = df[["q", "r"]].dropna()
        testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("freq1", ["15T", "D", "MS", "QS"])  # don't do all - many!
@pytest.mark.parametrize("freq2", ["15T", "H", "D", "MS", "QS"])
@pytest.mark.parametrize("columns", ["rp", "wp", "pq", "qr", "wr"])
def test_makedataframe_unequalfrequencies(freq1, freq2, columns):
    """Test if error is raised when creating a dataframe from series with unequal frequencies."""

    if freq1 == freq2:
        return

    kwargs = {"start": "2020", "end": "2021", "closed": "left", "tz": "Europe/Berlin"}
    i1 = pd.date_range(**kwargs, freq=freq1)
    i2 = pd.date_range(**kwargs, freq=freq2)

    s1 = dev.get_series(i1, columns[0])
    s2 = dev.get_series(i2, columns[1])

    dic = {columns[0]: s1, columns[1]: s2}
    with pytest.raises(ValueError):
        _ = single_helper.make_dataframe(dic)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
@pytest.mark.parametrize("freq", ["15T", "H", "D", "MS"])
@pytest.mark.parametrize("overlap", [True, False])
def test_makedataframe_unequaltimeperiods(freq, overlap, tz):
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
            result = single_helper.make_dataframe({"q": s1, "r": s2})
        return

    result = single_helper.make_dataframe({"q": s1, "r": s2})
    testing.assert_index_equal(result.index, intersection)
    testing.assert_series_equal(result.q, s1.loc[intersection])
    testing.assert_series_equal(result.r, s2.loc[intersection])
