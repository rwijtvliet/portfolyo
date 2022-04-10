from portfolyo.tools import stamps, nits
from portfolyo import testing
import pandas as pd
import numpy as np
import pytest

freqs_small_to_large = ["T", "5T", "15T", "30T", "H", "2H", "D", "MS", "QS", "AS"]


@pytest.mark.parametrize(
    ("idxs", "expected"),
    [
        # Days, with and without timezone.
        (
            [
                pd.date_range("2020", freq="D", periods=31),
                pd.date_range("2020-01-20", freq="D", periods=40),
            ],
            pd.date_range("2020-01-20", freq="D", periods=12),
        ),
        (
            [
                pd.date_range("2020", freq="D", periods=31, tz="Europe/Berlin"),
                pd.date_range("2020-01-20", freq="D", periods=40, tz="Europe/Berlin"),
            ],
            pd.date_range("2020-01-20", freq="D", periods=12, tz="Europe/Berlin"),
        ),
        # Error: incompatible timezones.
        (
            [
                pd.date_range("2020", freq="D", periods=31),
                pd.date_range("2020-01-20", freq="D", periods=40, tz="Europe/Berlin"),
            ],
            None,
        ),
        # Error: distinct frequencies.
        (
            [
                pd.date_range("2020", freq="H", periods=31),
                pd.date_range("2020-01-20", freq="D", periods=40),
            ],
            None,
        ),
        # No overlap.
        (
            [
                pd.date_range("2020", freq="H", periods=24),
                pd.date_range("2020-01-20", freq="H", periods=72),
            ],
            pd.date_range("2020", freq="H", periods=0),
        ),
        # Months, with and without timezone.
        (
            [
                pd.date_range("2020", freq="MS", periods=31),
                pd.date_range("2020-05-01", freq="MS", periods=40),
            ],
            pd.date_range("2020-05-01", freq="MS", periods=27),
        ),
        (
            [
                pd.date_range("2020", freq="MS", periods=31, tz="Europe/Berlin"),
                pd.date_range("2020-05-01", freq="MS", periods=40, tz="Europe/Berlin"),
            ],
            pd.date_range("2020-05-01", freq="MS", periods=27, tz="Europe/Berlin"),
        ),
        # Test if names retained.
        (
            [
                pd.date_range("2020", freq="MS", periods=31, name="ts_left"),
                pd.date_range("2020-05-01", freq="MS", periods=40),
            ],
            pd.date_range("2020-05-01", freq="MS", periods=27, name="ts_left"),
        ),
        # DST.
        (
            [
                pd.date_range("2020-03-28", freq="H", periods=71, tz="Europe/Berlin"),
                pd.date_range("2020-03-29", freq="H", periods=71, tz="Europe/Berlin"),
            ],
            pd.date_range("2020-03-29", freq="H", periods=47, tz="Europe/Berlin"),
        ),
        (
            [
                pd.date_range("2020-03-28", freq="H", periods=71, tz="Europe/Berlin"),
                pd.date_range("2020-03-30", freq="H", periods=72, tz="Europe/Berlin"),
            ],
            pd.date_range("2020-03-30", freq="H", periods=24, tz="Europe/Berlin"),
        ),
        (
            [
                pd.date_range("2020-03-28", freq="H", periods=72),
                pd.date_range("2020-03-29", freq="H", periods=72),
            ],
            pd.date_range("2020-03-29", freq="H", periods=48),
        ),
        (
            [
                pd.date_range("2020-03-28", freq="H", periods=72),
                pd.date_range("2020-03-30", freq="H", periods=72),
            ],
            pd.date_range("2020-03-30", freq="H", periods=24),
        ),
        (
            [
                pd.date_range("2020-10-24", freq="H", periods=73, tz="Europe/Berlin"),
                pd.date_range("2020-10-25", freq="H", periods=73, tz="Europe/Berlin"),
            ],
            pd.date_range("2020-10-25", freq="H", periods=49, tz="Europe/Berlin"),
        ),
        (
            [
                pd.date_range("2020-10-24", freq="H", periods=73, tz="Europe/Berlin"),
                pd.date_range("2020-10-26", freq="H", periods=72, tz="Europe/Berlin"),
            ],
            pd.date_range("2020-10-26", freq="H", periods=24, tz="Europe/Berlin"),
        ),
        (
            [
                pd.date_range("2020-10-24", freq="H", periods=72),
                pd.date_range("2020-10-25", freq="H", periods=72),
            ],
            pd.date_range("2020-10-25", freq="H", periods=48),
        ),
        (
            [
                pd.date_range("2020-10-24", freq="H", periods=72),
                pd.date_range("2020-10-26", freq="H", periods=72),
            ],
            pd.date_range("2020-10-26", freq="H", periods=24),
        ),
        # Distinct timezones.
        (
            [
                pd.date_range("2020-01-01", freq="15T", periods=96, tz="Europe/Berlin"),
                pd.date_range("2020-01-01", freq="15T", periods=96, tz="Asia/Kolkata"),
            ],
            pd.date_range("2020-01-01", freq="15T", periods=78, tz="Europe/Berlin"),
        ),
        (
            [
                pd.date_range("2020-01-01", freq="15T", periods=96, tz="Asia/Kolkata"),
                pd.date_range("2020-01-01", freq="15T", periods=96, tz="Europe/Berlin"),
            ],
            pd.date_range(
                "2020-01-01 04:30", freq="15T", periods=78, tz="Asia/Kolkata"
            ),
        ),
        (
            [
                pd.date_range("2020-01-01", freq="H", periods=24, tz="Asia/Kolkata"),
                pd.date_range("2020-01-01", freq="H", periods=24, tz="Europe/Berlin"),
            ],
            pd.date_range("2020-01-01", freq="H", periods=0, tz="Asia/Kolkata"),
        ),
    ],
)
def test_intersection(idxs, expected):
    """Test if intersection works correctly."""
    if expected is None:
        with pytest.raises(ValueError):
            _ = stamps.intersection(*idxs)
    else:
        result = stamps.intersection(*idxs)
        testing.assert_index_equal(result, expected)


@pytest.mark.parametrize("iterable", [False, True])  # make iterable or not
@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("ts", "fut", "freq", "expected_floor", "expected_ceil"),
    [
        ("2020", 0, "D", "2020", None),
        ("2020", 0, "MS", "2020", None),
        ("2020", 0, "QS", "2020", None),
        ("2020", 0, "AS", "2020", None),
        ("2020", 1, "D", "2020-01-02", None),
        ("2020", 1, "MS", "2020-02", None),
        ("2020", 1, "QS", "2020-04", None),
        ("2020", 1, "AS", "2021", None),
        ("2020", -1, "D", "2019-12-31", None),
        ("2020", -1, "MS", "2019-12", None),
        ("2020", -1, "QS", "2019-10", None),
        ("2020", -1, "AS", "2019", None),
        ("2020-02-01", 0, "D", "2020-02-01", None),
        ("2020-02-01", 0, "MS", "2020-02-01", None),
        ("2020-02-01", 0, "QS", "2020-01-01", "2020-04-01"),
        ("2020-02-01", 0, "AS", "2020", "2021"),
        ("2020-01-01 23:55", 0, "D", "2020", "2020-01-02"),
        ("2020-01-24 1:32", 0, "MS", "2020", "2020-02"),
        ("2020-03-03 3:33", 0, "QS", "2020", "2020-04"),
        ("2020-10-11 12:34", 0, "AS", "2020", "2021"),
        ("2020-01-01 23:55", 1, "D", "2020-01-02", "2020-01-03"),
        ("2020-01-24 1:32", 1, "MS", "2020-02", "2020-03"),
        ("2020-03-03 3:33", 1, "QS", "2020-04", "2020-07"),
        ("2020-10-11 12:34", 1, "AS", "2021", "2022"),
        ("2020-01-01 23:55", -1, "D", "2019-12-31", "2020-01-01"),
        ("2020-01-24 1:32", -1, "MS", "2019-12", "2020"),
        ("2020-03-03 3:33", -1, "QS", "2019-10", "2020"),
        ("2020-10-11 12:34", -1, "AS", "2019", "2020"),
        ("2020-03-29 00:00", 0, "H", "2020-03-29 00:00", None),
        ("2020-10-25 00:00", 0, "H", "2020-10-25 00:00", None),
    ],
)
@pytest.mark.parametrize("function", ["floor", "ceil"])
def test_floorceilts_1(
    function: str,
    ts: str,
    fut: int,
    freq: str,
    expected_floor: str,
    expected_ceil: str,
    tz: str,
    iterable: bool,
):
    """Test if timestamp or iterable of timestamps is correctly floored and ceiled."""
    # Prepare.
    fn = stamps.floor_ts if function == "floor" else stamps.ceil_ts
    if function == "floor" or expected_ceil is None:
        expected_single = expected_floor
    else:
        expected_single = expected_ceil

    ts = pd.Timestamp(ts)
    expected_single = pd.Timestamp(expected_single)

    if tz:
        ts = ts.tz_localize(tz)
        expected_single = expected_single.tz_localize(tz)

    # Test.
    if not iterable:
        # Test single value.
        assert fn(ts, freq, fut) == expected_single

    else:
        # Test index.
        periods = 10
        index = pd.date_range(ts, periods=periods, freq=freq)  # causes rounding of ts
        index += ts - index[0]  # undoes the rounding

        result_iter = fn(index, freq, fut)
        result_iter.freq = None  # disregard checking frequencies here
        expected_iter = pd.date_range(expected_single, periods=periods, freq=freq)
        expected_iter.freq = None  # disregard checking frequencies here
        pd.testing.assert_index_equal(result_iter, expected_iter)


@pytest.mark.parametrize(
    ("ts", "tz", "freq", "expected_floor", "expected_ceil"),
    [
        ("2020-04-21 15:25", None, "H", "2020-04-21 15:00", "2020-04-21 16:00"),
        (
            "2020-04-21 15:25",
            "Europe/Berlin",
            "H",
            "2020-04-21 15:00",
            "2020-04-21 16:00",
        ),
        (
            "2020-04-21 15:25+02:00",
            "Europe/Berlin",
            "H",
            "2020-04-21 15:00+02:00",
            "2020-04-21 16:00+02:00",
        ),
        (
            "2020-04-21 15:25",
            "Asia/Kolkata",
            "H",
            "2020-04-21 15:00",
            "2020-04-21 16:00",
        ),
        ("2020-03-29 01:50", None, "15T", "2020-03-29 01:45", "2020-03-29 02:00"),
        ("2020-03-29 03:05", None, "15T", "2020-03-29 03:00", "2020-03-29 03:15"),
        (
            "2020-03-29 01:50+01:00",
            "Europe/Berlin",
            "15T",
            "2020-03-29 01:45+01:00",
            "2020-03-29 03:00+02:00",
        ),
        (
            "2020-03-29 03:05+02:00",
            "Europe/Berlin",
            "15T",
            "2020-03-29 03:00+02:00",
            "2020-03-29 03:15+02:00",
        ),
        (
            "2020-03-29 01:50",
            "Europe/Berlin",
            "15T",
            "2020-03-29 01:45",
            "2020-03-29 03:00",
        ),
        ("2020-03-29 03:05", None, "15T", "2020-03-29 03:00", "2020-03-29 03:15"),
        ("2020-10-25 02:50", None, "15T", "2020-10-25 02:45", "2020-10-25 03:00"),
        ("2020-10-25 02:05", None, "15T", "2020-10-25 02:00", "2020-10-25 02:15"),
        (
            "2020-10-25 02:50+02:00",
            "Europe/Berlin",
            "15T",
            "2020-10-25 02:45+02:00",
            "2020-10-25 02:00+01:00",
        ),
        (
            "2020-10-25 02:05+02:00",
            "Europe/Berlin",
            "15T",
            "2020-10-25 02:00+02:00",
            "2020-10-25 02:15+02:00",
        ),
        (
            "2020-10-25 02:50+01:00",
            "Europe/Berlin",
            "15T",
            "2020-10-25 02:45+01:00",
            "2020-10-25 03:00+01:00",
        ),
        (
            "2020-10-25 02:05+01:00",
            "Europe/Berlin",
            "15T",
            "2020-10-25 02:00+01:00",
            "2020-10-25 02:15+01:00",
        ),
        (
            "2020-10-25 02:30+02:00",
            "Europe/Berlin",
            "H",
            "2020-10-25 02:00+02:00",
            "2020-10-25 02:00+01:00",
        ),
        (
            "2020-10-25 02:30+01:00",
            "Europe/Berlin",
            "H",
            "2020-10-25 02:00+01:00",
            "2020-10-25 03:00+01:00",
        ),
    ],
)
@pytest.mark.parametrize("function", ["floor", "ceil"])
def test_floorceilts_2(function, ts, tz, freq, expected_floor, expected_ceil):
    """Test flooring and ceiling during DST transitions."""
    ts_in = pd.Timestamp(ts, tz=tz)
    if function == "floor":
        result = stamps.floor_ts(ts_in, freq)
        expected = pd.Timestamp(expected_floor, tz=tz)
    else:
        result = stamps.ceil_ts(ts_in, freq)
        expected = pd.Timestamp(expected_ceil, tz=tz)

    assert result == expected


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("ts", "freq", "is_boundary"),
    [
        ("2020", "H", True),
        ("2020", "D", True),
        ("2020", "MS", True),
        ("2020", "AS", True),
        ("2020-04-01", "H", True),
        ("2020-04-01", "D", True),
        ("2020-04-01", "MS", True),
        ("2020-04-01", "AS", False),
        ("2020-01-01 15:00", "H", True),
        ("2020-01-01 15:00", "D", False),
        ("2020-01-01 15:00", "MS", False),
        ("2020-01-01 15:00", "AS", False),
        ("2020-01-01 15:45", "H", False),
        ("2020-01-01 15:45", "D", False),
        ("2020-01-01 15:45", "MS", False),
        ("2020-01-01 15:45", "AS", False),
    ],
)
def test_assertboundary(ts, freq, is_boundary, tz):
    """Test if boundary timestamps are correctly identified."""
    ts = pd.Timestamp(ts, tz=tz)
    if is_boundary:
        _ = stamps.assert_boundary_ts(ts, freq)
    else:
        with pytest.raises(ValueError):
            _ = stamps.assert_boundary_ts(ts, freq)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("tss", "freq", "are_boundary"),
    [
        (["2020", "2021"], "H", True),
        (["2020", "2021"], "D", True),
        (["2020", "2021"], "MS", True),
        (["2020", "2021"], "AS", True),
        (["2020-04-01", "2021"], "H", True),
        (["2020-04-01", "2021"], "D", True),
        (["2020-04-01", "2021"], "MS", True),
        (["2020-04-01", "2021"], "AS", False),
        (["2020-01-01 15:00", "2021"], "H", True),
        (["2020-01-01 15:00", "2021"], "D", False),
        (["2020-01-01 15:00", "2021"], "MS", False),
        (["2020-01-01 15:00", "2021"], "AS", False),
        (["2020-01-01 15:45", "2021"], "H", False),
        (["2020-01-01 15:45", "2021"], "D", False),
        (["2020-01-01 15:45", "2021"], "MS", False),
        (["2020-01-01 15:45", "2021"], "AS", False),
    ],
)
def test_assertboundary_asindex(tss, freq, are_boundary, tz):
    """Test if boundary timestamps are correctly identified."""
    i = pd.Index([pd.Timestamp(ts, tz=tz) for ts in tss])
    if are_boundary:
        _ = stamps.assert_boundary_ts(i, freq)
    else:
        with pytest.raises(ValueError):
            _ = stamps.assert_boundary_ts(i, freq)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("start", "end", "freq", "trimfreq", "expected_start", "expected_end"),
    [
        ("2020-01-05", "2020-12-21", "D", "H", "2020-01-05", "2020-12-21"),
        ("2020-01-05", "2020-12-21", "D", "D", "2020-01-05", "2020-12-21"),
        ("2020-01-05", "2020-12-21", "D", "MS", "2020-02", "2020-12"),
        ("2020-01-05", "2020-12-21", "D", "QS", "2020-04", "2020-10"),
        ("2020-01-05", "2020-12-21", "D", "AS", None, None),
    ],
)
def test_trimindex(start, end, freq, trimfreq, expected_start, expected_end, tz):
    """Test if indices are correctly trimmed."""
    i = pd.date_range(start, end, freq=freq, inclusive="left", tz=tz)
    result = stamps.trim_index(i, trimfreq)
    if expected_start is not None:
        expected = pd.date_range(
            expected_start, expected_end, freq=freq, inclusive="left", tz=tz
        )
    else:
        expected = pd.DatetimeIndex([], freq=freq, tz=tz)
    pd.testing.assert_index_equal(result, expected)


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("freq_as_attr", [True, False])
@pytest.mark.parametrize(
    ("ts_left", "freq", "expected_ts_right"),
    [
        ("2020", "15T", "2020-01-01 00:15"),
        ("2020", "H", "2020-01-01 01:00"),
        ("2020", "D", "2020-01-02"),
        ("2020", "MS", "2020-02-01"),
        ("2020", "QS", "2020-04"),
        ("2020", "AS", "2021"),
        ("2020-04-21", "15T", "2020-04-21 00:15"),
        ("2020-04-21", "H", "2020-04-21 01:00"),
        ("2020-04-21", "D", "2020-04-22"),
        ("2020-04-21 15:00", "15T", "2020-04-21 15:15"),
        ("2020-04-21 15:00", "H", "2020-04-21 16:00"),
    ],
)
def test_tsright(ts_left, freq, tz, freq_as_attr, expected_ts_right):
    """Test if right timestamp is correctly calculated."""
    if freq_as_attr:
        ts = pd.Timestamp(ts_left, freq=freq, tz=tz)
        result = stamps.ts_right(ts)
    else:
        ts = pd.Timestamp(ts_left, tz=tz)
        result = stamps.ts_right(ts, freq)
    expected = pd.Timestamp(expected_ts_right, tz=tz)
    assert result == expected


@pytest.mark.parametrize("tz_left", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize("tz_right", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("tss", "expected_tss"),
    [
        (("2020-01-01", "2020-02-02"), ("2020-01-01", "2020-02-02")),
        (("2020-02-02", "2020-01-01"), ("2020-01-01", "2020-02-02")),
        (("2020-01-01", None), ("2020-01-01", "2021-01-01")),
        ((None, "2020-02-02"), ("2020-01-01", "2020-02-02")),
        (("2020-01-01", "2020-01-01"), ("2020-01-01", "2020-01-01")),
        (("2020-03-03 3:33", "2021-10-09"), ("2020-03-03 3:33", "2021-10-09")),
        (("2021-10-09", "2020-03-03 3:33"), ("2020-03-03 3:33", "2021-10-09")),
        (("2020-03-03 3:33", "2021-10-09"), ("2020-03-03 3:33", "2021-10-09")),
        (("2020-03-03 3:33", None), ("2020-03-03 3:33", "2021-01-01")),
        ((None, "2021-10-09"), ("2021-01-01 0:00", "2021-10-09")),
        (
            (None, None),
            (
                pd.Timestamp(pd.Timestamp.today().year + 1, 1, 1),
                pd.Timestamp(pd.Timestamp.today().year + 2, 1, 1),
            ),
        ),
    ],
)
def test_tsleftright(tss: tuple, expected_tss: tuple, tz_left: str, tz_right: str):
    """Test if start and end of interval are correctly calculated."""
    tzs = [tz_left, tz_right]
    tss = [pd.Timestamp(ts) for ts in tss]  # turn into timestamps for comparison
    if tss[0] == tss[1] and tz_left != tz_right:
        return  # too complicated to test; would have to recreate function here.
    swap = tss[0] > tss[1]
    tss = [ts.tz_localize(tz) for ts, tz in zip(tss, tzs)]  # add timezone info

    result = stamps.ts_leftright(*tss)

    exp_tzs = [tz for ts, tz in zip(tss, tzs) if tz is not None and ts is not pd.NaT]
    if swap:
        exp_tzs.reverse()
    if not len(exp_tzs):
        exp_tzs = ["Europe/Berlin"]
    if len(exp_tzs) == 1:
        exp_tzs = exp_tzs * 2
    exp_result = [
        pd.Timestamp(ts).tz_localize(tz) for ts, tz in zip(expected_tss, exp_tzs)
    ]

    for a, b in zip(result, exp_result):
        assert a == b


@pytest.mark.parametrize(
    ("ts", "freq", "hours"),
    [
        (pd.Timestamp("2020-01-01"), "D", 24),
        (pd.Timestamp("2020-01-01"), "MS", 744),
        (pd.Timestamp("2020-01-01"), "QS", 2184),
        (pd.Timestamp("2020-01-01"), "AS", 8784),
        (pd.Timestamp("2020-03-01"), "D", 24),
        (pd.Timestamp("2020-03-01"), "MS", 744),
        (pd.Timestamp("2020-03-29"), "D", 24),
        (pd.Timestamp("2020-01-01", tz="Europe/Berlin"), "D", 24),
        (pd.Timestamp("2020-01-01", tz="Europe/Berlin"), "MS", 744),
        (pd.Timestamp("2020-01-01", tz="Europe/Berlin"), "QS", 2183),
        (pd.Timestamp("2020-01-01", tz="Europe/Berlin"), "AS", 8784),
        (pd.Timestamp("2020-03-01", tz="Europe/Berlin"), "D", 24),
        (pd.Timestamp("2020-03-01", tz="Europe/Berlin"), "MS", 743),
        (pd.Timestamp("2020-03-29", tz="Europe/Berlin"), "D", 23),
    ],
)
def test_duration(ts, freq, hours):
    """Test if duration in correctly calculated."""
    assert stamps.duration(ts, freq) == nits.Q_(hours, "h")


@pytest.mark.parametrize("freq1", freqs_small_to_large)
@pytest.mark.parametrize("freq2", freqs_small_to_large)
def test_frequpordown(freq1, freq2):
    i1 = freqs_small_to_large.index(freq1)
    i2 = freqs_small_to_large.index(freq2)
    outcome = np.sign(i1 - i2)
    assert stamps.freq_up_or_down(freq1, freq2) == outcome


@pytest.mark.parametrize("count", range(1, 30))
def test_longestshortestfreq(count):
    indices = np.random.randint(0, len(freqs_small_to_large), count)
    freqs = np.array(freqs_small_to_large)[indices]
    assert stamps.freq_longest(*freqs) == freqs_small_to_large[max(indices)]
    assert stamps.freq_shortest(*freqs) == freqs_small_to_large[min(indices)]


# TODO: add tests for other functions
