import pandas as pd
import pytest
from utils import id_fn  # relative to /tests

from portfolyo import toolsb

LEFT_FREQ_PERIODS_RIGHT_DURATION = [
    ("2020", "min", 5 * 24 * 60, "2020-01-01 00:01", 1 / 60),
    ("2020", "5min", 5 * 24 * 12, "2020-01-01 00:05", 5 / 60),
    ("2020", "15min", 5 * 24 * 4, "2020-01-01 00:15", 15 / 60),
    ("2020", "h", 5 * 24, "2020-01-01 01:00", 1),
    ("2020", "D", 5, "2020-01-02", 24),
    ("2020", "MS", 3, "2020-02", [31 * 24, 29 * 24, 31 * 24]),
    ("2020", "QS-JAN", 3, "2020-04", [91 * 24, 91 * 24, 92 * 24]),
    ("2020", "QS-APR", 3, "2020-04", [91 * 24, 91 * 24, 92 * 24]),
    ("2020", "QS-JUL", 3, "2020-04", [91 * 24, 91 * 24, 92 * 24]),
    ("2020", "QS-OCT", 3, "2020-04", [91 * 24, 91 * 24, 92 * 24]),
    ("2020", "YS", 3, "2021", [8784, 8760, 8760]),
    ("2020-05", "min", 5 * 24 * 60, "2020-05-01 00:01", 1 / 60),
    ("2020-05", "5min", 5 * 24 * 12, "2020-05-01 00:05", 5 / 60),
    ("2020-05", "15min", 5 * 24 * 4, "2020-05-01 00:15", 15 / 60),
    ("2020-05", "h", 5 * 24, "2020-05-01 01:00", 1),
    ("2020-05", "D", 5, "2020-05-02", 24),
    ("2020-05", "MS", 3, "2020-06", [31 * 24, 30 * 24, 31 * 24]),
    ("2020-05", "QS-FEB", 3, "2020-08", [92 * 24, 92 * 24, 92 * 24]),
    ("2020-05", "QS-MAY", 3, "2020-08", [92 * 24, 92 * 24, 92 * 24]),
    ("2020-05", "QS-AUG", 3, "2020-08", [92 * 24, 92 * 24, 92 * 24]),
    ("2020-05", "QS-NOV", 3, "2020-08", [92 * 24, 92 * 24, 92 * 24]),
    ("2020-05", "YS-MAY", 3, "2021-05", [8760, 8760, 8760]),
    ("2020-04-21 15:00", "min", 5 * 24 * 60, "2020-04-21 15:01", 1 / 60),
    ("2020-04-21 15:00", "5min", 5 * 24 * 12, "2020-04-21 15:05", 5 / 60),
    ("2020-04-21 15:00", "15min", 5 * 24 * 4, "2020-04-21 15:15", 15 / 60),
    ("2020-04-21 15:00", "h", 5 * 24, "2020-04-21 16:00", 1),
]


@pytest.mark.parametrize(
    "idx,right",
    [
        (
            pd.date_range(left, freq=freq, periods=periods),
            pd.date_range(right, freq=freq, periods=periods),
        )
        for (left, freq, periods, right, _) in LEFT_FREQ_PERIODS_RIGHT_DURATION
    ],
)
def test_index_toright(idx, right):
    toolsb.testing.assert_index_equal(toolsb.index.to_right(idx), right)


@pytest.mark.parametrize(
    "idx,durations",
    [
        (pd.date_range(left, freq=freq, periods=periods), durations)
        for (left, freq, periods, _, durations) in LEFT_FREQ_PERIODS_RIGHT_DURATION
    ],
)
def test_index_duration(idx, durations):
    expected = pd.Series(durations, idx, dtype=float).astype("pint[h]")
    toolsb.testing.assert_series_equal(toolsb.index.duration(idx), expected)


LEFTRIGHT_FREQ_TRIMFREQ_TRIMMEDLEFTRIGHT = [
    # 2020
    # . 15min -> other
    (("2020", "2021"), "15min", "h", ("2020", "2021")),
    (("2020", "2021"), "15min", "MS", ("2020", "2021")),
    (("2020", "2021"), "15min", "QS-JAN", ("2020", "2021")),
    (("2020", "2021"), "15min", "QS-FEB", ("2020-02", "2020-11")),
    (("2020", "2021"), "15min", "YS-JAN", ("2020", "2021")),
    (("2020", "2021"), "15min", "YS-FEB", None),
    # . h -> other
    (("2020", "2021"), "h", "15min", ("2020", "2021")),
    (("2020", "2021"), "h", "MS", ("2020", "2021")),
    (("2020", "2021"), "h", "QS-JAN", ("2020", "2021")),
    (("2020", "2021"), "h", "QS-FEB", ("2020-02", "2020-11")),
    (("2020", "2021"), "h", "YS-JAN", ("2020", "2021")),
    (("2020", "2021"), "h", "YS-FEB", None),
    # . MS -> other
    (("2020", "2021"), "MS", "15min", ("2020", "2021")),
    (("2020", "2021"), "MS", "h", ("2020", "2021")),
    (("2020", "2021"), "MS", "QS-JAN", ("2020", "2021")),
    (("2020", "2021"), "MS", "QS-FEB", ("2020-02", "2020-11")),
    (("2020", "2021"), "MS", "YS-JAN", ("2020", "2021")),
    (("2020", "2021"), "MS", "YS-FEB", None),
    # . QS-JAN -> other
    (("2020", "2021"), "QS-JAN", "15min", ("2020", "2021")),
    (("2020", "2021"), "QS-JAN", "h", ("2020", "2021")),
    (("2020", "2021"), "QS-JAN", "MS", ("2020", "2021")),
    (("2020", "2021"), "QS-JAN", "QS-FEB", Exception),
    (("2020", "2021"), "QS-JAN", "YS-JAN", ("2020", "2021")),
    (("2020", "2021"), "QS-JAN", "YS-FEB", Exception),
    # . YS-JAN -> other
    (("2020", "2021"), "YS-JAN", "15min", ("2020", "2021")),
    (("2020", "2021"), "YS-JAN", "h", ("2020", "2021")),
    (("2020", "2021"), "YS-JAN", "MS", ("2020", "2021")),
    (("2020", "2021"), "YS-JAN", "QS-JAN", ("2020", "2021")),
    (("2020", "2021"), "YS-JAN", "QS-FEB", Exception),
    (("2020", "2021"), "YS-JAN", "YS-FEB", Exception),
    # 2020-02-01 06:00
    # . 15min -> other
    (
        ("2020-02-01 06:00", "2021-02-01 06:00"),
        "15min",
        "h",
        ("2020-02-01 06:00", "2021-02-01 06:00"),
    ),
    (
        ("2020-02-01 06:00", "2021-02-01 06:00"),
        "15min",
        "MS",
        ("2020-02-01 06:00", "2021-02-01 06:00"),
    ),
    (
        ("2020-02-01 06:00", "2021-02-01 06:00"),
        "15min",
        "QS-JAN",
        ("2020-04-01 06:00", "2021-01-01 06:00"),
    ),
    (
        ("2020-02-01 06:00", "2021-02-01 06:00"),
        "15min",
        "QS-FEB",
        ("2020-02-01 06:00", "2021-02-01 06:00"),
    ),
    (("2020-02-01 06:00", "2021-02-01 06:00"), "15min", "YS-JAN", None),
    (
        ("2020-02-01 06:00", "2021-02-01 06:00"),
        "15min",
        "YS-FEB",
        ("2020-02-01 06:00", "2021-02-01 06:00"),
    ),
    # . h -> other
    (
        ("2020-02-01 06:00", "2021-02-01 06:00"),
        "h",
        "15min",
        ("2020-02-01 06:00", "2021-02-01 06:00"),
    ),
    (("2020-02-01 06:00", "2021-02-01 06:00"), "h", "MS", ("2020-02-01 06:00", "2021-02-01 06:00")),
    (
        ("2020-02-01 06:00", "2021-02-01 06:00"),
        "h",
        "QS-JAN",
        ("2020-04-01 06:00", "2021-01-01 06:00"),
    ),
    (
        ("2020-02-01 06:00", "2021-02-01 06:00"),
        "h",
        "QS-FEB",
        ("2020-02-01 06:00", "2021-02-01 06:00"),
    ),
    (("2020-02-01 06:00", "2021-02-01 06:00"), "h", "YS-JAN", None),
    (
        ("2020-02-01 06:00", "2021-02-01 06:00"),
        "h",
        "YS-FEB",
        ("2020-02-01 06:00", "2021-02-01 06:00"),
    ),
    # . MS -> other
    (
        ("2020-02-01 06:00", "2021-02-01 06:00"),
        "MS",
        "15min",
        ("2020-02-01 06:00", "2021-02-01 06:00"),
    ),
    (("2020-02-01 06:00", "2021-02-01 06:00"), "MS", "h", ("2020-02-01 06:00", "2021-02-01 06:00")),
    (
        ("2020-02-01 06:00", "2021-02-01 06:00"),
        "MS",
        "QS-JAN",
        ("2020-04-01 06:00", "2021-01-01 06:00"),
    ),
    (
        ("2020-02-01 06:00", "2021-02-01 06:00"),
        "MS",
        "QS-FEB",
        ("2020-02-01 06:00", "2021-02-01 06:00"),
    ),
    (("2020-02-01 06:00", "2021-02-01 06:00"), "MS", "YS-JAN", None),
    (
        ("2020-02-01 06:00", "2021-02-01 06:00"),
        "MS",
        "YS-FEB",
        ("2020-02-01 06:00", "2021-02-01 06:00"),
    ),
    # . QS-FEB -> other
    (
        ("2020-02-01 06:00", "2021-02-01 06:00"),
        "QS-FEB",
        "15min",
        ("2020-02-01 06:00", "2021-02-01 06:00"),
    ),
    (
        ("2020-02-01 06:00", "2021-02-01 06:00"),
        "QS-FEB",
        "h",
        ("2020-02-01 06:00", "2021-02-01 06:00"),
    ),
    (
        ("2020-02-01 06:00", "2021-02-01 06:00"),
        "QS-FEB",
        "MS",
        ("2020-02-01 06:00", "2021-02-01 06:00"),
    ),
    (("2020-02-01 06:00", "2021-02-01 06:00"), "QS-FEB", "QS-JAN", Exception),
    (("2020-02-01 06:00", "2021-02-01 06:00"), "QS-FEB", "YS-JAN", Exception),
    (
        ("2020-02-01 06:00", "2021-02-01 06:00"),
        "QS-FEB",
        "YS-FEB",
        ("2020-02-01 06:00", "2021-02-01 06:00"),
    ),
    # . YS-FEB -> other
    (
        ("2020-02-01 06:00", "2021-02-01 06:00"),
        "YS-FEB",
        "15min",
        ("2020-02-01 06:00", "2021-02-01 06:00"),
    ),
    (
        ("2020-02-01 06:00", "2021-02-01 06:00"),
        "YS-FEB",
        "h",
        ("2020-02-01 06:00", "2021-02-01 06:00"),
    ),
    (
        ("2020-02-01 06:00", "2021-02-01 06:00"),
        "YS-FEB",
        "MS",
        ("2020-02-01 06:00", "2021-02-01 06:00"),
    ),
    (("2020-02-01 06:00", "2021-02-01 06:00"), "YS-FEB", "QS-JAN", Exception),
    (
        ("2020-02-01 06:00", "2021-02-01 06:00"),
        "YS-FEB",
        "QS-FEB",
        ("2020-02-01 06:00", "2021-02-01 06:00"),
    ),
    (("2020-02-01 06:00", "2021-02-01 06:00"), "YS-FEB", "YS-JAN", Exception),
    # 2020-04-21 15:00
    # . 15min -> other
    (
        ("2020-04-21 15:00", "2021-08-21 15:00"),
        "15min",
        "h",
        ("2020-04-21 15:00", "2021-08-21 15:00"),
    ),
    (
        ("2020-04-21 15:00", "2021-08-21 15:00"),
        "15min",
        "MS",
        ("2020-05-01 15:00", "2021-08-01 15:00"),
    ),
    (
        ("2020-04-21 15:00", "2021-08-21 15:00"),
        "15min",
        "QS-JAN",
        ("2020-07-01 15:00", "2021-07-01 15:00"),
    ),
    (
        ("2020-04-21 15:00", "2021-08-21 15:00"),
        "15min",
        "QS-FEB",
        ("2020-05-01 15:00", "2021-08-01 15:00"),
    ),
    (("2020-04-21 15:00", "2021-08-21 15:00"), "15min", "YS-JAN", None),
    (("2020-04-21 15:00", "2021-08-21 15:00"), "15min", "YS-FEB", None),
    # . h -> other
    (
        ("2020-04-21 15:00", "2021-08-21 15:00"),
        "h",
        "15min",
        ("2020-04-21 15:00", "2021-08-21 15:00"),
    ),
    (("2020-04-21 15:00", "2021-08-21 15:00"), "h", "MS", ("2020-05-01 15:00", "2021-08-01 15:00")),
    (
        ("2020-04-21 15:00", "2021-08-21 15:00"),
        "h",
        "QS-JAN",
        ("2020-07-01 15:00", "2021-07-01 15:00"),
    ),
    (
        ("2020-04-21 15:00", "2021-08-21 15:00"),
        "h",
        "QS-FEB",
        ("2020-05-01 15:00", "2021-08-01 15:00"),
    ),
    (("2020-04-21 15:00", "2021-08-21 15:00"), "h", "YS-JAN", None),
    (("2020-04-21 15:00", "2021-08-21 15:00"), "h", "YS-FEB", None),
]


@pytest.mark.parametrize(
    "idx,trimfreq,trimmed",
    [
        (
            pd.date_range(*leftright, freq=freq, inclusive="left"),
            trimfreq,
            pd.date_range(*trimmedleftright, freq=freq, inclusive="left"),
        )
        for (
            leftright,
            freq,
            trimfreq,
            trimmedleftright,
        ) in LEFTRIGHT_FREQ_TRIMFREQ_TRIMMEDLEFTRIGHT
        if trimmedleftright is not None and trimmedleftright is not Exception
    ],
)
def test_index_trim(idx, trimfreq, trimmed):
    pd.testing.assert_index_equal(toolsb.index.trim(idx, trimfreq), trimmed)


@pytest.mark.parametrize(
    "idx,trimfreq,trimmed",
    [
        (
            pd.date_range(*leftright, freq=freq, inclusive="left"),
            trimfreq,
            pd.DatetimeIndex([], freq=freq),
        )
        for (
            leftright,
            freq,
            trimfreq,
            trimmedleftright,
        ) in LEFTRIGHT_FREQ_TRIMFREQ_TRIMMEDLEFTRIGHT
        if trimmedleftright is None
    ],
)
def test_index_trim_empty(idx, trimfreq, trimmed):
    pd.testing.assert_index_equal(toolsb.index.trim(idx, trimfreq), trimmed)


@pytest.mark.parametrize(
    "idx,trimfreq",
    [
        (pd.date_range(*leftright, freq=freq, inclusive="left"), trimfreq)
        for (
            leftright,
            freq,
            trimfreq,
            trimmedleftright,
        ) in LEFTRIGHT_FREQ_TRIMFREQ_TRIMMEDLEFTRIGHT
        if trimmedleftright is Exception
    ],
)
def test_index_trim_error(idx, trimfreq):
    with pytest.raises(ValueError):
        toolsb.index.trim(idx, trimfreq)


START_END_FREQ = [
    ("2020", "2022", "min"),
    ("2020", "2022", "h"),
    ("2020", "2022", "D"),
    ("2020", "2022", "MS"),
    ("2020", "2022", "QS-JAN"),
    ("2020", "2022", "YS-JAN"),
    ("2020-02-01", "2022-02-01", "min"),
    ("2020-02-01", "2022-02-01", "h"),
    ("2020-02-01", "2022-02-01", "D"),
    ("2020-02-01", "2022-02-01", "MS"),
    ("2020-02-01", "2022-02-01", "QS-FEB"),
    ("2020-02-01", "2022-02-01", "YS-FEB"),
    ("2020-04-21", "2022-06-10", "min"),
    ("2020-04-21", "2022-06-10", "h"),
    ("2020-04-21", "2022-06-10", "D"),
]


@pytest.mark.parametrize(
    "idx",
    [
        pd.date_range(f"{date1} {sod}", f"{date2} {sod}", inclusive="left", freq=freq, tz=tz)
        for date1, date2, freq in START_END_FREQ
        for sod in ["00:00", "06:00", "15:00"]
        for tz in [None, "Europe/Berlin", "Asia/Kolkata"]
    ],
)
@pytest.mark.parametrize("times", range(1, 4))
def test_index_intersect_identical(idx, times):
    pd.testing.assert_index_equal(toolsb.index.intersect(idx for _ in range(times)), idx)
    for idx2 in toolsb.index.intersect_flex(idx for _ in range(times)):
        pd.testing.assert_index_equal(idx2, idx)


# Starting with identical indices, we can vary (a) start/end, (b) freq (c) tz (d) sod.
# For .intersect, only (a) works. For .intersect_flex, all work.


STARTENDFREQ1_STARTENDFREQ2_ISCSTARTEND = [
    # startdate, enddate, freq of index1; startdate, enddate, freq of index2; startdate, enddate of intersection (if possible).
    (("2020", "2023", "min"), ("2021-04-21", "2023-02", "min"), ("2021-04-21", "2023")),
    (("2020", "2023", "min"), ("2021-04-21", "2023-02", "h"), ("2021-04-21", "2023")),
    (("2020", "2023", "min"), ("2021-04-21", "2023-02", "D"), ("2021-04-21", "2023")),
    (("2020", "2023", "min"), ("2021-04", "2023-02", "MS"), ("2021-04", "2023")),
    (("2020", "2023", "min"), ("2021-04", "2023-04", "QS-JAN"), ("2021-04", "2023")),
    (("2020", "2023", "min"), ("2021-04", "2023-04", "QS-APR"), ("2021-04", "2023")),
    (("2020", "2023", "min"), ("2021", "2023", "YS-JAN"), ("2021", "2023")),
    (("2020", "2023", "min"), ("2021-02", "2023-02", "QS-FEB"), ("2021-02", "2022-11")),
    (("2020", "2023", "min"), ("2021-02", "2023-02", "YS-FEB"), ("2021-02", "2022-02")),
    (("2020", "2023", "h"), ("2021-04-21", "2023-02", "min"), ("2021-04-21", "2023")),
    (("2020", "2023", "h"), ("2021-04-21", "2023-02", "h"), ("2021-04-21", "2023")),
    (("2020", "2023", "h"), ("2021-04-21", "2023-02", "D"), ("2021-04-21", "2023")),
    (("2020", "2023", "h"), ("2021-04", "2023-02", "MS"), ("2021-04", "2023")),
    (("2020", "2023", "h"), ("2021-04", "2023-04", "QS-JAN"), ("2021-04", "2023")),
    (("2020", "2023", "h"), ("2021-04", "2023-04", "QS-APR"), ("2021-04", "2023")),
    (("2020", "2023", "h"), ("2021", "2023", "YS-JAN"), ("2021", "2023")),
    (("2020", "2023", "h"), ("2021-02", "2023-02", "QS-FEB"), ("2021-02", "2022-11")),
    (("2020", "2023", "h"), ("2021-02", "2023-02", "YS-FEB"), ("2021-02", "2022-02")),
    (("2020", "2023", "D"), ("2021-04-21", "2023-02", "min"), ("2021-04-21", "2023")),
    (("2020", "2023", "D"), ("2021-04-21", "2023-02", "h"), ("2021-04-21", "2023")),
    (("2020", "2023", "D"), ("2021-04-21", "2023-02", "D"), ("2021-04-21", "2023")),
    (("2020", "2023", "D"), ("2021-04", "2023-02", "MS"), ("2021-04", "2023")),
    (("2020", "2023", "D"), ("2021-04", "2023-04", "QS-JAN"), ("2021-04", "2023")),
    (("2020", "2023", "D"), ("2021-04", "2023-04", "QS-APR"), ("2021-04", "2023")),
    (("2020", "2023", "D"), ("2021", "2023", "YS-JAN"), ("2021", "2023")),
    (("2020", "2023", "D"), ("2021-02", "2023-02", "QS-FEB"), ("2021-02", "2022-11")),
    (("2020", "2023", "D"), ("2021-02", "2023-02", "YS-FEB"), ("2021-02", "2022-02")),
    (("2020", "2023", "MS"), ("2021-04-21", "2023-02", "min"), ("2021-05", "2023")),
    (("2020", "2023", "MS"), ("2021-04-21", "2023-02", "h"), ("2021-05", "2023")),
    (("2020", "2023", "MS"), ("2021-04-21", "2023-02", "D"), ("2021-05", "2023")),
    (("2020", "2023", "MS"), ("2021-04", "2023-02", "MS"), ("2021-04", "2023")),
    (("2020", "2023", "MS"), ("2021-04", "2023-04", "QS-JAN"), ("2021-04", "2023")),
    (("2020", "2023", "MS"), ("2021-04", "2023-04", "QS-APR"), ("2021-04", "2023")),
    (("2020", "2023", "MS"), ("2021", "2023", "YS-JAN"), ("2021", "2023")),
    (("2020", "2023", "MS"), ("2021-02", "2023-02", "QS-FEB"), ("2021-02", "2022-11")),
    (("2020", "2023", "MS"), ("2021-02", "2023-02", "YS-FEB"), ("2021-02", "2022-02")),
    (("2020", "2023", "QS-JAN"), ("2021-04-21", "2023-02", "min"), ("2021-07", "2023")),
    (("2020", "2023", "QS-JAN"), ("2021-04-21", "2023-02", "h"), ("2021-07", "2023")),
    (("2020", "2023", "QS-JAN"), ("2021-04-21", "2023-02", "D"), ("2021-07", "2023")),
    (("2020", "2023", "QS-JAN"), ("2021-04", "2023-02", "MS"), ("2021-04", "2023")),
    (("2020", "2023", "QS-JAN"), ("2021-04", "2023-04", "QS-JAN"), ("2021-04", "2023")),
    (("2020", "2023", "QS-JAN"), ("2021-04", "2023-04", "QS-APR"), ("2021-04", "2023")),
    (("2020", "2023", "QS-JAN"), ("2021", "2023", "YS-JAN"), ("2021", "2023")),
    (("2020", "2023", "QS-JAN"), ("2021-02", "2023-02", "QS-FEB"), (Exception, Exception)),
    (("2020", "2023", "QS-JAN"), ("2021-02", "2023-02", "YS-FEB"), (Exception, Exception)),
    (("2020", "2023", "YS-JAN"), ("2021-04-21", "2023-02", "min"), ("2022", "2023")),
    (("2020", "2023", "YS-JAN"), ("2021-04-21", "2023-02", "h"), ("2022", "2023")),
    (("2020", "2023", "YS-JAN"), ("2021-04-21", "2023-02", "D"), ("2022", "2023")),
    (("2020", "2023", "YS-JAN"), ("2021-04", "2023-02", "MS"), ("2022", "2023")),
    (("2020", "2023", "YS-JAN"), ("2021-04", "2023-04", "QS-JAN"), ("2022", "2023")),
    (("2020", "2023", "YS-JAN"), ("2021-04", "2023-04", "QS-APR"), ("2022", "2023")),
    (("2020", "2023", "YS-JAN"), ("2021", "2023", "YS-JAN"), ("2021", "2023")),
    (("2020", "2023", "YS-JAN"), ("2021-02", "2023-02", "QS-FEB"), (Exception, Exception)),
    (("2020", "2023", "YS-JAN"), ("2021-02", "2023-02", "YS-FEB"), (Exception, Exception)),
    (("2020-02", "2022-11", "QS-FEB"), ("2021-04-21", "2023-02", "min"), ("2021-05", "2022-11")),
    (("2020-02", "2022-11", "QS-FEB"), ("2021-04-21", "2023-02", "h"), ("2021-05", "2022-11")),
    (("2020-02", "2022-11", "QS-FEB"), ("2021-04-21", "2023-02", "D"), ("2021-05", "2022-11")),
    (("2020-02", "2022-11", "QS-FEB"), ("2021-04", "2023-02", "MS"), ("2021-05", "2022-11")),
    (("2020-02", "2022-11", "QS-FEB"), ("2021-04", "2023-04", "QS-JAN"), (Exception, Exception)),
    (("2020-02", "2022-11", "QS-FEB"), ("2021-04", "2023-04", "QS-APR"), (Exception, Exception)),
    (("2020-02", "2022-11", "QS-FEB"), ("2021", "2023", "YS-JAN"), (Exception, Exception)),
    (("2020-02", "2022-11", "QS-FEB"), ("2021-02", "2023-02", "QS-FEB"), ("2021-02", "2022-11")),
    (("2020-02", "2023-02", "YS-FEB"), ("2021-04-21", "2023-02-15", "min"), ("2022-02", "2023-02")),
    (("2020-02", "2023-02", "YS-FEB"), ("2021-04-21", "2023-02-15", "h"), ("2022-02", "2023-02")),
    (("2020-02", "2023-02", "YS-FEB"), ("2021-04-21", "2023-02-15", "D"), ("2022-02", "2023-02")),
    (("2020-02", "2023-02", "YS-FEB"), ("2021-04", "2023-03", "MS"), ("2022-02", "2023-02")),
    (("2020-02", "2023-02", "YS-FEB"), ("2021-04", "2023-04", "QS-JAN"), (Exception, Exception)),
    (("2020-02", "2023-02", "YS-FEB"), ("2021-04", "2023-04", "QS-APR"), (Exception, Exception)),
    (("2020-02", "2023-02", "YS-FEB"), ("2021", "2023", "YS-JAN"), (Exception, Exception)),
    (("2020-02", "2023-02", "YS-FEB"), ("2021-05", "2023-05", "QS-FEB"), ("2022-02", "2023-02")),
    (("2020-02", "2023-02", "YS-FEB"), ("2021-02", "2024-02", "YS-FEB"), ("2021-02", "2023-02")),
]


def _index(startdate, enddate, sod, freq, tz):
    return pd.date_range(
        f"{startdate} {sod}", f"{enddate} {sod}", freq=freq, inclusive="left", tz=tz
    )


@pytest.mark.parametrize(
    "idx1,idx2,iscidx",
    [
        (
            _index(start1, end1, sod, freq1, tz),
            _index(start2, end2, sod, freq2, tz),
            _index(iscstart, iscend, sod, freq1, tz),
        )
        for (
            (start1, end1, freq1),
            (start2, end2, freq2),
            (iscstart, iscend),
        ) in STARTENDFREQ1_STARTENDFREQ2_ISCSTARTEND
        if freq1 == freq2 or set([freq1, freq2]) == set(["QS-JAN", "QS-APR"])
        for sod in ["00:00"]
        for tz in [None, "Europe/Berlin"]
    ],
    ids=id_fn,
)
def test_normalintersect_ok(idx1, idx2, iscidx):
    toolsb.testing.assert_index_equal(toolsb.index.intersect([idx1, idx2]), iscidx)


@pytest.mark.parametrize(
    "idx1,idx2",
    [
        (_index(start1, end1, sod1, freq1, tz1), _index(start2, end2, sod2, freq2, tz2))
        for (
            (start1, end1, freq1),
            (start2, end2, freq2),
            _,
        ) in STARTENDFREQ1_STARTENDFREQ2_ISCSTARTEND
        for sod1 in ["00:00", "06:00"]
        for sod2 in ["06:00"]
        for tz1 in [None, "Europe/Berlin", "Asia/Kolkata"]
        for tz2 in ["Europe/Berlin"]
        if (freq1 != freq2 and set([freq1, freq2]) != set(["QS-JAN", "QS-APR"]))
        or sod1 != sod2
        or tz1 != tz2
    ],
    ids=id_fn,
)
def test_normalintersect_nok(idx1, idx2):
    with pytest.raises(ValueError):
        _ = toolsb.index.intersect([idx1, idx2])


@pytest.fixture(
    scope="module",
    params=[pytest.param(True, id="ignorefreq"), pytest.param(False, id="dontignorefreq")],
)
def ignorefreq(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[pytest.param(True, id="ignoresod"), pytest.param(False, id="dontignoresod")],
)
def ignoresod(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[pytest.param(True, id="ignoretz"), pytest.param(False, id="dontignoretz")],
)
def ignoretz(request):
    return request.param


@pytest.mark.parametrize(
    "idx1,idx2,iscidx1,iscidx2,mustignorefreq,mustignoresod,mustignoretz",
    [
        (
            _index(start1, end1, sod1, freq1, tz1),
            _index(start2, end2, sod2, freq2, tz2),
            _index(iscstart, iscend, sod1, freq1, tz1),
            _index(iscstart, iscend, sod2, freq2, tz2),
            freq1 != freq2 and set([freq1, freq2]) != set(["QS-JAN", "QS-APR"]),
            sod1 != sod2,
            tz1 != tz2,
        )
        for (
            (start1, end1, freq1),
            (start2, end2, freq2),
            (iscstart, iscend),
        ) in STARTENDFREQ1_STARTENDFREQ2_ISCSTARTEND
        if iscstart is not Exception
        for sod1 in ["00:00", "06:00"]
        for sod2 in ["06:00"]
        for tz1 in [None, "Europe/Berlin", "Asia/Kolkata"]
        for tz2 in ["Europe/Berlin"]
        if None in set([tz1, tz2]) or tz1 == tz2
    ],
    ids=id_fn,
)
def test_flexintersect_ok(
    idx1,
    idx2,
    iscidx1,
    iscidx2,
    mustignorefreq,
    mustignoresod,
    mustignoretz,
    ignorefreq,
    ignoresod,
    ignoretz,
):
    def testfn():
        return toolsb.index.intersect_flex(
            [idx1, idx2], ignore_freq=ignorefreq, ignore_startofday=ignoresod, ignore_tz=ignoretz
        )

    if (
        mustignorefreq
        and not ignorefreq
        or mustignoresod
        and not ignoresod
        or mustignoretz
        and not ignoretz
    ):
        with pytest.raises(ValueError):
            testfn()
    else:
        resultidx1, resultidx2 = testfn()
        toolsb.testing.assert_index_equal(resultidx1, iscidx1)
        toolsb.testing.assert_index_equal(resultidx2, iscidx2)


@pytest.mark.parametrize(
    "idx1,idx2",
    [
        (_index(start1, end1, sod1, freq1, tz1), _index(start2, end2, sod2, freq2, tz2))
        for (
            (start1, end1, freq1),
            (start2, end2, freq2),
            (iscstart, _),
        ) in STARTENDFREQ1_STARTENDFREQ2_ISCSTARTEND
        if iscstart is Exception
        for sod1 in ["00:00", "06:00"]
        for sod2 in ["06:00"]
        for tz1 in [None, "Europe/Berlin", "Asia/Kolkata"]
        for tz2 in ["Europe/Berlin"]
    ],
    ids=id_fn,
)
def test_flexintersect_nok(idx1, idx2, ignorefreq, ignoresod, ignoretz):
    with pytest.raises(ValueError):
        toolsb.index.intersect_flex(
            [idx1, idx2], ignore_freq=ignorefreq, ignore_startofday=ignoresod, ignore_tz=ignoretz
        )
