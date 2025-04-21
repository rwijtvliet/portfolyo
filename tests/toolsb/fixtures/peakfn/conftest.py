from dataclasses import dataclass
from typing import Callable

import pandas as pd
import pytest
from pandas.tseries.frequencies import MONTH_ALIASES

from portfolyo import toolsb

f_germanpower = toolsb.peakfn.factory("08:00", "20:00")
f_everyday_13half = toolsb.peakfn.factory("08:00", "21:30", [1, 2, 3, 4, 5, 6, 7])
f_workingdays_full = toolsb.peakfn.factory(None, None, [1, 2, 3, 4, 5])
f_everyday_until6 = toolsb.peakfn.factory(None, "06:00", [1, 2, 3, 4, 5, 6, 7])


@dataclass
class _Case1:
    peakfn: toolsb.peakfn.PeakFunction
    idx: pd.DatetimeIndex
    peakperiodcount: int
    peakonlystretch: pd.DatetimeIndex


@pytest.fixture(
    scope="class",
    params=[
        (f_germanpower, "2020", "2020-01-08", "15min", 5 * 12 * 4, (32, 79)),
        (f_germanpower, "2020", "2020-01-08", "h", 5 * 12, (8, 19)),
        (f_germanpower, "2020", "2020-04-01", "15min", 65 * 12 * 4, (-64, -17)),
        (f_germanpower, "2020", "2020-04-01", "h", 65 * 12, (-16, -5)),
        (f_germanpower, "2020", "2021", "15min", 262 * 12 * 4, (-64, -17)),
        (f_germanpower, "2020", "2021", "h", 262 * 12, (-16, -5)),
        (f_everyday_13half, "2020", "2020-01-08", "15min", 7 * 13.5 * 4, (32, 85)),
        (f_everyday_13half, "2020", "2020-04-01", "15min", 91 * 13.5 * 4, (-64, -11)),
        (f_everyday_13half, "2020", "2021", "15min", 366 * 13.5 * 4, (-64, -11)),
        (f_workingdays_full, "2020", "2020-01-08", "15min", 5 * 24 * 4, (0, 72 * 4 - 1)),
        (f_workingdays_full, "2020", "2020-01-08", "h", 5 * 24, (0, 72 - 1)),
        (f_workingdays_full, "2020", "2020-01-08", "D", 5, (0, 3 - 1)),
        (
            f_workingdays_full,
            "2020",
            "2020-04-01",
            "15min",
            65 * 24 * 4,
            (0, 72 * 4 - 1),
        ),  # avoid DST transition
        (
            f_workingdays_full,
            "2020",
            "2020-04-01",
            "h",
            65 * 24,
            (0, 72 - 1),
        ),  # avoid DST transition
        (f_workingdays_full, "2020", "2020-04-01", "D", 65, (-9, -4 - 1)),
        (
            f_workingdays_full,
            "2020",
            "2021",
            "15min",
            262 * 24 * 4,
            (-11 * 24 * 4, -6 * 24 * 4 - 1),
        ),
        (f_workingdays_full, "2020", "2021", "h", 262 * 24, (-11 * 24, -6 * 24 - 1)),
        (f_workingdays_full, "2020", "2021", "D", 262, (-11, -6 - 1)),
        (f_everyday_until6, "2020", "2020-02", "15min", 31 * 6 * 4, (24 * 4, 30 * 4 - 1)),
        (f_everyday_until6, "2020", "2020-02", "h", 31 * 6, (24, 30 - 1)),
    ],
)
def _case1(request, tz) -> _Case1:
    fn, start, end, freq, count, stretch = request.param
    idx = pd.date_range(start, end, freq=freq, inclusive="left", tz=tz)
    peakonlystretch = idx[slice(*stretch)]
    return _Case1(fn, idx, count, peakonlystretch)


@pytest.fixture(scope="class")
def case1_peakfn(_case1: _Case1) -> toolsb.peakfn.PeakFunction:
    return _case1.peakfn


@pytest.fixture()
def case1_idx(_case1: _Case1) -> pd.DatetimeIndex:
    return _case1.idx


@pytest.fixture(scope="class")
def case1_peakperiodcount(_case1: _Case1) -> int:
    return _case1.peakperiodcount


@pytest.fixture(scope="class")
def case1_peakonlystretch(_case1: _Case1) -> pd.DatetimeIndex:
    return _case1.peakonlystretch


# ---


@pytest.fixture(
    scope="class", params=[f_germanpower, f_everyday_13half, f_workingdays_full, f_everyday_until6]
)
def case2_peakfn(request) -> toolsb.peakfn.PeakFunction:
    return request.param


@pytest.fixture(
    scope="class", params=["h", "D", "MS", "QS-JAN", "QS-FEB", "QS-APR", "YS-JAN", "YS-FEB"]
)
def case2_idx(request, case2_peakfn, tz) -> pd.DatetimeIndex:
    freqstr = request.param
    if freqstr == "h" and case2_peakfn is not f_everyday_13half:
        pytest.skip("This is not an error case.")
    elif freqstr == "D" and case2_peakfn is f_workingdays_full:
        pytest.skip("This is not an error case.")
    return pd.date_range("2020", freq=freqstr, periods=2, tz=tz)


# ---


@dataclass
class _Case3:
    month: int
    peakperiodcount: int
    peakonlystartendpos: tuple[int, int]


@pytest.fixture(scope="class", params=[(3, 31 * 6, (-72, -66)), (10, 31 * 6, (-168, -162))])
def _case3(request) -> _Case3:
    return _Case3(*request.param)


@pytest.fixture(scope="class", params=["15min", "h"])
def _case3_freqstr(request) -> str:
    return request.param


@pytest.fixture(scope="class")
def case3_peakfn() -> toolsb.peakfn.PeakFunction:
    return f_everyday_until6


@pytest.fixture(scope="class")
def case3_idx(_case3, tz, _case3_freqstr) -> pd.DatetimeIndex:
    month, freq = _case3.month, _case3_freqstr
    return pd.date_range(f"2020-{month}", f"2020-{month+1}", freq=freq, inclusive="left", tz=tz)


@pytest.fixture(scope="class")
def case3_peakperiodcount(_case3, _case3_freqstr, tz) -> int:
    count = _case3.peakperiodcount
    if tz == "Europe/Berlin":
        count += -1 if _case3.month == 3 else 1
    if _case3_freqstr == "15min":
        count *= 4
    return count


@pytest.fixture(scope="class")
def case3_peakonlystretch(_case3, _case3_freqstr, tz, case3_idx) -> int:
    startpos, endpos = _case3.peakonlystartendpos
    if tz == "Europe/Berlin":
        startpos += 1 if _case3.month == 3 else -1
    if _case3_freqstr == "15min":
        startpos *= 4
        endpos *= 4
    return case3_idx[startpos : endpos - 1]


# ---


@pytest.fixture(scope="class", params=[2020, 2021, 2022])
def _case4_year(request) -> int:
    return request.param


@pytest.fixture(scope="class", params=[1, 2, 3], ids=MONTH_ALIASES.get)
def _case4_startmonth(request) -> str:
    return request.param


@pytest.fixture(scope="class", params=["15min", "h", "D", "MS", "QS", "YS"])
def _case4_freqstr(request, _case4_startmonth) -> str:
    freqstr = request.param
    if freqstr in ["QS", "YS"]:
        return f"{freqstr}-{MONTH_ALIASES[_case4_startmonth]}"
    else:
        return freqstr


@pytest.fixture(scope="class")
def _case4_days(_case4_freqstr, _case4_startmonth, _case4_year) -> int | None:
    if _case4_freqstr in ["15min", "h", "D"]:
        return 1
    elif _case4_freqstr == "MS":
        return 31 if _case4_startmonth != 2 else 28 if _case4_year != 2020 else 29
    elif _case4_freqstr.startswith("QS"):
        if _case4_startmonth == 1:
            return 91 if _case4_year == 2020 else 90
        elif _case4_startmonth == 2:
            return 90 if _case4_year == 2020 else 89
        else:
            return 92
    elif _case4_freqstr.startswith("YS"):
        return (366 if _case4_year == 2020 and _case4_startmonth <= 2 else 365) * 24


@pytest.fixture(scope="class")
def _case4_peakdays(_case4_freqstr, _case4_startmonth, _case4_year) -> int | None:
    if _case4_freqstr in ["15min", "h", "D"]:
        if _case4_startmonth == 1:
            return 0 if _case4_year == 2022 else 1
        else:
            return 0 if _case4_year == 2020 else 1
    elif _case4_freqstr == "MS":
        if _case4_startmonth == 1:
            return 23 if _case4_year == 2020 else 21
        elif _case4_startmonth == 2:
            return 20
        else:
            return 22 if _case4_year == 2020 else 23
    elif _case4_freqstr.startswith("QS"):
        if _case4_startmonth == 1:
            return 65 if _case4_year == 2020 else 64
        elif _case4_startmonth == 2:
            return 65 if _case4_year == 2021 else 64
        else:
            return 65 if _case4_year == 2020 else 66
    elif _case4_freqstr.startswith("YS"):
        if _case4_startmonth == 1:
            return 262 if _case4_year == 2020 else 261 if _case4_year == 2021 else 260
        elif _case4_startmonth == 2:
            return 260 if _case4_year == 2020 else 261
        else:
            return 260 if _case4_year == 2020 else 261 if _case4_year == 2021 else 261


@pytest.fixture(scope="class")
def _case4_baseduration(
    _case4_freqstr, _case4_startmonth, _case4_year, tz, _case4_days
) -> int | float | list[int] | list[float]:
    if _case4_freqstr == "15min":
        return [0.25] * 96
    elif _case4_freqstr == "h":
        return [1] * 24
    elif _case4_freqstr == "D":
        return 24
    elif _case4_freqstr == "MS":
        return _case4_days * 24 + (-1 if (_case4_startmonth == 3 and tz == "Europe/Berlin") else 0)
    elif _case4_freqstr.startswith("QS"):
        return _case4_days * 24 + (-1 if tz == "Europe/Berlin" else 0)
    else:  # _case4_freqstr.startswith('YS')
        return (366 if _case4_year == 2020 and _case4_startmonth <= 2 else 365) * 24


@pytest.fixture(scope="class")
def _case4_peakduration(_case4_freqstr, _case4_peakdays) -> int | float | list[int] | list[float]:
    if _case4_freqstr == "15min":
        return [0] * 96 if _case4_peakdays == 0 else [*[0] * 32, *[0.25] * 48, *[0] * 16]
    elif _case4_freqstr == "h":
        return [0] * 24 if _case4_peakdays == 0 else [*[0] * 8, *[1] * 12, *[0] * 4]
    else:
        return _case4_peakdays * 12


@pytest.fixture(scope="class")
def _case4_offpeakduration(
    _case4_baseduration, _case4_peakduration
) -> int | float | list[int] | list[float]:
    if isinstance(_case4_baseduration, list):
        return [b - p for b, p in zip(_case4_baseduration, _case4_peakduration, strict=True)]
    else:
        return _case4_baseduration - _case4_peakduration


@pytest.fixture(scope="class", params=["baseduration", "offpeakduration", "peakduration"])
def _case4_durationfnstr(request) -> str:
    return request.param


@pytest.fixture(scope="class")
def case4_durationfn1param(_case4_durationfnstr) -> Callable:
    return {
        "baseduration": lambda idx: toolsb.peakfn.base_duration(idx),
        "peakduration": lambda idx: toolsb.peakfn.peak_duration(idx, f_germanpower),
        "offpeakduration": lambda idx: toolsb.peakfn.offpeak_duration(idx, f_germanpower),
    }[_case4_durationfnstr]


@pytest.fixture(scope="class")
def case4_peakfn() -> toolsb.peakfn.PeakFunction:
    return f_germanpower


@pytest.fixture(scope="class")
def case4_idx(_case4_year, _case4_startmonth, _case4_freqstr, tz) -> pd.DatetimeIndex:
    periods = {"15min": 96, "h": 24}.get(_case4_freqstr, 1)
    return pd.date_range(
        f"{_case4_year}-{_case4_startmonth}", freq=_case4_freqstr, periods=periods, tz=tz
    )


@pytest.fixture(scope="class")
def case4_duration(
    case4_idx,
    _case4_durationfnstr,
    _case4_baseduration,
    _case4_peakduration,
    _case4_offpeakduration,
) -> pd.Series:
    duration = {
        "baseduration": _case4_baseduration,
        "peakduration": _case4_peakduration,
        "offpeakduration": _case4_offpeakduration,
    }[_case4_durationfnstr]
    return pd.Series(duration, case4_idx, dtype=float, name="duration").astype("pint[h]")
