import dataclasses

import pandas as pd
import pytest
from utils import id_fn


@dataclasses.dataclass
class _Case1:
    stamp: str
    freqstr: str
    periods: int
    rightstamp: str
    duration: int | float | list[int | float]


@pytest.fixture(
    scope="class",
    params=[
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
    ],
    ids=id_fn,
)
def _case1(request) -> _Case1:
    return _Case1(*request.param)


@pytest.fixture(scope="class")
def case1_idx(_case1: _Case1) -> pd.DatetimeIndex:
    return pd.date_range(_case1.stamp, freq=_case1.freqstr, periods=_case1.periods)


@pytest.fixture(scope="class")
def case1_rightidx(_case1: _Case1) -> pd.DatetimeIndex:
    return pd.date_range(_case1.rightstamp, freq=_case1.freqstr, periods=_case1.periods)


@pytest.fixture(scope="class")
def case1_duration(case1_idx, _case1: _Case1) -> pd.Series:
    return pd.Series(_case1.duration, case1_idx, dtype=float).astype("pint[h]")


# ---


@dataclasses.dataclass
class _Case2:
    stamp: str
    freqstr: str
    periods: int
    rightstamp: str
    duration: list[int]


@pytest.fixture(
    scope="class",
    params=[
        ("2020-03-27", "D", 3, "2020-03-28", [24, 24, 23]),
        ("2020-03-27 01:00", "D", 3, "2020-03-28 01:00", [24, 24, 23]),
        ("2020-03-27 03:00", "D", 3, "2020-03-28 03:00", [24, 23, 24]),
        ("2020-10-24", "D", 3, "2020-10-25", [24, 25, 24]),
        ("2020-10-24 01:00", "D", 3, "2020-10-25 01:00", [24, 25, 24]),
        ("2020-10-24 03:00", "D", 3, "2020-10-25 03:00", [25, 24, 24]),
        ("2020-02", "MS", 3, "2020-03", [29 * 24, 31 * 24 - 1, 30 * 24]),
        ("2020-09", "MS", 3, "2020-10", [30 * 24, 31 * 24 + 1, 30 * 24]),
        ("2020-01", "QS-JAN", 4, "2020-04", [91 * 24 - 1, 91 * 24, 92 * 24, 92 * 24 + 1]),
    ],
    ids=id_fn,
)
def _case2(request) -> _Case2:
    return _Case2(*request.param)


@pytest.fixture(scope="class", ids=id_fn)
def case2_idx(_case2: _Case2) -> pd.DatetimeIndex:
    return pd.date_range(
        _case2.stamp, freq=_case2.freqstr, periods=_case2.periods, tz="Europe/Berlin"
    )


@pytest.fixture(scope="class")
def case2_rightidx(_case2: _Case2) -> pd.DatetimeIndex:
    return pd.date_range(
        _case2.rightstamp, freq=_case2.freqstr, periods=_case2.periods, tz="Europe/Berlin"
    )


@pytest.fixture(scope="class", ids=id_fn)
def case2_duration(case2_idx, _case2: _Case2) -> pd.Series:
    return pd.Series(_case2.duration, case2_idx, dtype=float).astype("pint[h]")


# ---


@dataclasses.dataclass
class _Case3:
    leftright: tuple[str, str]
    freq: str
    trimfreq: str
    trimmedleftright: tuple[str, str] | None | Exception


@pytest.fixture(
    scope="class",
    params=[
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
        (
            ("2020-02-01 06:00", "2021-02-01 06:00"),
            "h",
            "MS",
            ("2020-02-01 06:00", "2021-02-01 06:00"),
        ),
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
        (
            ("2020-02-01 06:00", "2021-02-01 06:00"),
            "MS",
            "h",
            ("2020-02-01 06:00", "2021-02-01 06:00"),
        ),
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
        (
            ("2020-04-21 15:00", "2021-08-21 15:00"),
            "h",
            "MS",
            ("2020-05-01 15:00", "2021-08-01 15:00"),
        ),
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
    ],
    ids=id_fn,
)
def _case3(request) -> _Case3:
    return _Case3(*request.param)


@pytest.fixture(scope="class")
def case3_idx(_case3: _Case3, tz) -> pd.DatetimeIndex:
    return pd.date_range(*_case3.leftright, freq=_case3.freq, inclusive="left", tz=tz)


@pytest.fixture(scope="class")
def case3_trimfreq(_case3: _Case3) -> str:
    return _case3.trimfreq


@pytest.fixture(scope="class")
def case3_isok(_case3: _Case3) -> bool:
    return _case3.trimmedleftright is not Exception


@pytest.fixture(scope="class")
def case3_trimmedidx(_case3: _Case3, tz, case3_isok) -> pd.DatetimeIndex:
    if not case3_isok:
        pytest.skip("Error case.")

    if _case3.trimmedleftright is None:  # empty trim
        return pd.DatetimeIndex([], freq=_case3.freq, tz=tz)
    else:
        return pd.date_range(*_case3.trimmedleftright, freq=_case3.freq, inclusive="left", tz=tz)


# ---


@dataclasses.dataclass
class _Case4:
    startendfreq1: tuple[str, str, str]
    startendfreq2: tuple[str, str, str]
    itsstartend: tuple[str, str]


@pytest.fixture(
    scope="class",
    params=[
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
        (("2020", "2023", "QS-JAN"), ("2021-02", "2023-02", "QS-FEB"), Exception),
        (("2020", "2023", "QS-JAN"), ("2021-02", "2023-02", "YS-FEB"), Exception),
        (("2020", "2023", "YS-JAN"), ("2021-04-21", "2023-02", "min"), ("2022", "2023")),
        (("2020", "2023", "YS-JAN"), ("2021-04-21", "2023-02", "h"), ("2022", "2023")),
        (("2020", "2023", "YS-JAN"), ("2021-04-21", "2023-02", "D"), ("2022", "2023")),
        (("2020", "2023", "YS-JAN"), ("2021-04", "2023-02", "MS"), ("2022", "2023")),
        (("2020", "2023", "YS-JAN"), ("2021-04", "2023-04", "QS-JAN"), ("2022", "2023")),
        (("2020", "2023", "YS-JAN"), ("2021-04", "2023-04", "QS-APR"), ("2022", "2023")),
        (("2020", "2023", "YS-JAN"), ("2021", "2023", "YS-JAN"), ("2021", "2023")),
        (("2020", "2023", "YS-JAN"), ("2021-02", "2023-02", "QS-FEB"), Exception),
        (("2020", "2023", "YS-JAN"), ("2021-02", "2023-02", "YS-FEB"), Exception),
        (
            ("2020-02", "2022-11", "QS-FEB"),
            ("2021-04-21", "2023-02", "min"),
            ("2021-05", "2022-11"),
        ),
        (("2020-02", "2022-11", "QS-FEB"), ("2021-04-21", "2023-02", "h"), ("2021-05", "2022-11")),
        (("2020-02", "2022-11", "QS-FEB"), ("2021-04-21", "2023-02", "D"), ("2021-05", "2022-11")),
        (("2020-02", "2022-11", "QS-FEB"), ("2021-04", "2023-02", "MS"), ("2021-05", "2022-11")),
        (("2020-02", "2022-11", "QS-FEB"), ("2021-04", "2023-04", "QS-JAN"), Exception),
        (("2020-02", "2022-11", "QS-FEB"), ("2021-04", "2023-04", "QS-APR"), Exception),
        (("2020-02", "2022-11", "QS-FEB"), ("2021", "2023", "YS-JAN"), Exception),
        (
            ("2020-02", "2022-11", "QS-FEB"),
            ("2021-02", "2023-02", "QS-FEB"),
            ("2021-02", "2022-11"),
        ),
        (
            ("2020-02", "2022-11", "QS-FEB"),
            ("2021-02", "2023-02", "YS-FEB"),
            ("2021-02", "2022-02"),
        ),
        (
            ("2020-02", "2023-02", "YS-FEB"),
            ("2021-04-21", "2023-02-15", "min"),
            ("2022-02", "2023-02"),
        ),
        (
            ("2020-02", "2023-02", "YS-FEB"),
            ("2021-04-21", "2023-02-15", "h"),
            ("2022-02", "2023-02"),
        ),
        (
            ("2020-02", "2023-02", "YS-FEB"),
            ("2021-04-21", "2023-02-15", "D"),
            ("2022-02", "2023-02"),
        ),
        (("2020-02", "2023-02", "YS-FEB"), ("2021-04", "2023-03", "MS"), ("2022-02", "2023-02")),
        (("2020-02", "2023-02", "YS-FEB"), ("2021-04", "2023-04", "QS-JAN"), Exception),
        (("2020-02", "2023-02", "YS-FEB"), ("2021-04", "2023-04", "QS-APR"), Exception),
        (("2020-02", "2023-02", "YS-FEB"), ("2021", "2023", "YS-JAN"), Exception),
        (
            ("2020-02", "2023-02", "YS-FEB"),
            ("2021-05", "2023-05", "QS-FEB"),
            ("2022-02", "2023-02"),
        ),
        (
            ("2020-02", "2023-02", "YS-FEB"),
            ("2021-02", "2024-02", "YS-FEB"),
            ("2021-02", "2023-02"),
        ),
        # No overlap.
        (("2020", "2021", "min"), ("2022-04-21", "2023-02", "min"), None),
        (("2020", "2021", "min"), ("2022-04-21", "2023-02", "h"), None),
        (("2020", "2021", "min"), ("2022-04-21", "2023-02", "D"), None),
        (("2020", "2021", "min"), ("2022-04", "2023-02", "MS"), None),
        (("2020", "2021", "min"), ("2022-04", "2023-04", "QS-JAN"), None),
        (("2020", "2021", "min"), ("2022-04", "2023-04", "QS-APR"), None),
        (("2020", "2021", "min"), ("2022", "2023", "YS-JAN"), None),
        (("2020", "2021", "min"), ("2022-02", "2023-02", "QS-FEB"), None),
        (("2020", "2021", "min"), ("2022-02", "2023-02", "YS-FEB"), None),
        (("2020", "2021", "QS-JAN"), ("2021-04", "2023-02", "MS"), None),
        (("2020", "2021", "QS-JAN"), ("2021-04", "2023-04", "QS-JAN"), None),
        (("2020", "2021", "QS-JAN"), ("2021-04", "2023-04", "QS-APR"), None),
        (("2020", "2021", "QS-JAN"), ("2021", "2023", "YS-JAN"), None),
        (("2020", "2021", "QS-JAN"), ("2021-02", "2023-02", "QS-FEB"), Exception),
        (("2020", "2021", "QS-JAN"), ("2021-02", "2023-02", "YS-FEB"), Exception),
        (("2020", "2021", "YS-JAN"), ("2021-04", "2023-04", "QS-JAN"), None),
        (("2020", "2021", "YS-JAN"), ("2021-04", "2023-04", "QS-APR"), None),
        (("2020", "2021", "YS-JAN"), ("2021", "2023", "YS-JAN"), None),
        (("2020", "2021", "YS-JAN"), ("2021-02", "2023-02", "QS-FEB"), Exception),
        (("2020", "2021", "YS-JAN"), ("2021-02", "2023-02", "YS-FEB"), Exception),
        (("2020-02", "2020-11", "QS-FEB"), ("2021-04", "2023-04", "QS-JAN"), Exception),
        (("2020-02", "2020-11", "QS-FEB"), ("2021-04", "2023-04", "QS-APR"), Exception),
        (("2020-02", "2020-11", "QS-FEB"), ("2021", "2023", "YS-JAN"), Exception),
        (("2020-02", "2020-11", "QS-FEB"), ("2021-02", "2022-02", "QS-FEB"), None),
        (("2020-02", "2020-11", "QS-FEB"), ("2021-02", "2023-02", "YS-FEB"), None),
        (("2020-02", "2021-02", "YS-FEB"), ("2021-04", "2023-04", "QS-JAN"), Exception),
        (("2020-02", "2021-02", "YS-FEB"), ("2021-04", "2023-04", "QS-APR"), Exception),
        (("2020-02", "2021-02", "YS-FEB"), ("2021", "2023", "YS-JAN"), Exception),
        (("2020-02", "2021-02", "YS-FEB"), ("2021-05", "2023-05", "QS-FEB"), None),
        (("2020-02", "2021-02", "YS-FEB"), ("2021-02", "2024-02", "YS-FEB"), None),
    ],
    ids=id_fn,
)
def _case4(request) -> _Case4:
    return _Case4(*request.param)


def _index(startdate, enddate, freq, sodstr, tz) -> pd.DatetimeIndex:
    return pd.date_range(
        f"{startdate} {sodstr}", f"{enddate} {sodstr}", freq=freq, inclusive="left", tz=tz
    )


@pytest.fixture(scope="class")
def case4_idx1(_case4: _Case4, sod_asstr, tz) -> pd.DatetimeIndex:
    return _index(*_case4.startendfreq1, sod_asstr, tz)


@pytest.fixture(scope="class")
def case4_idx2(_case4: _Case4, sod2_asstr, tz2) -> pd.DatetimeIndex:
    return _index(*_case4.startendfreq2, sod2_asstr, tz2)


@pytest.fixture(scope="class")
def _case4_equivalentfreq(_case4: _Case4) -> bool:
    freqs = set([_case4.startendfreq1[-1], _case4.startendfreq2[-1]])
    if len(freqs) != 1 and freqs != set(["QS-JAN", "QS-APR"]):
        return False
    return True


@pytest.fixture(scope="class")
def _case4_itsidx_isempty(_case4: _Case4) -> bool:
    return _case4.itsstartend is None


@pytest.fixture(scope="class")
def case4_normalintersect_isok(_case4_equivalentfreq, equalsod, equaltz) -> bool:
    return _case4_equivalentfreq and equalsod and equaltz


@pytest.fixture(scope="class")
def case4_normalintersect_idx(
    _case4: _Case4, sod_asstr, tz, case4_normalintersect_isok, _case4_itsidx_isempty
) -> pd.DatetimeIndex:
    if not case4_normalintersect_isok:
        pytest.skip("Error case.")

    freq = _case4.startendfreq1[-1]
    if _case4_itsidx_isempty:
        return pd.DatetimeIndex([], freq=freq, tz=tz)
    else:
        return _index(*_case4.itsstartend, freq, sod_asstr, tz)


@pytest.fixture(
    scope="class",
    params=[pytest.param(True, id="ignorefreq"), pytest.param(False, id="dontignorefreq")],
)
def case4_ignorefreq(request, _case4_equivalentfreq) -> bool:
    ignore = request.param
    if _case4_equivalentfreq and ignore:
        pytest.skip("Don't test ignoring if values equal.")
    return ignore


@pytest.fixture(
    scope="class",
    params=[pytest.param(True, id="ignoretz"), pytest.param(False, id="dontignoretz")],
)
def case4_ignoretz(request, equaltz) -> bool:
    ignore = request.param
    if equaltz and ignore:
        pytest.skip("Don't test ignoring if values equal.")
    return ignore


@pytest.fixture(
    scope="class",
    params=[pytest.param(True, id="ignoresod"), pytest.param(False, id="dontignoresod")],
)
def case4_ignoresod(request, equalsod) -> bool:
    ignore = request.param
    if equalsod and ignore:
        pytest.skip("Don't test ignoring if values equal.")
    return ignore


@pytest.fixture(scope="class")
def case4_flexintersect_isok(
    _case4: _Case4,
    case4_ignorefreq,
    _case4_equivalentfreq,
    case4_ignoresod,
    equalsod,
    case4_ignoretz,
    equaltz,
) -> bool:
    return (
        _case4.itsstartend is not Exception
        and (case4_ignorefreq or _case4_equivalentfreq)
        and (case4_ignoresod or equalsod)
        and (case4_ignoretz or equaltz)
    )


@pytest.fixture(scope="class")
def case4_flexintersect_idxs(
    _case4: _Case4, sod_asstr, sod2_asstr, tz, tz2, case4_flexintersect_isok, _case4_itsidx_isempty
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    if not case4_flexintersect_isok:
        pytest.skip("Error case.")

    freq1, freq2 = _case4.startendfreq1[-1], _case4.startendfreq2[-1]
    if _case4_itsidx_isempty:
        return (pd.DatetimeIndex([], freq=freq1, tz=tz), pd.DatetimeIndex([], freq=freq2, tz=tz2))
    else:
        return (
            _index(*_case4.itsstartend, freq1, sod_asstr, tz),
            _index(*_case4.itsstartend, freq2, sod2_asstr, tz2),
        )
