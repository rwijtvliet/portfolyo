import pandas as pd
import datetime as dt
import pytest
from pandas.tseries.frequencies import MONTHS, to_offset


@pytest.fixture(scope="session", params=range(3))
def seed(request) -> int:
    request.param


# Freq ---


_SHORTERTHANDAILY = ("min", "5min", "15min", "h")
_ALL = (*_SHORTERTHANDAILY, "D", "MS", *(f"QS-{m}" for m in MONTHS), *(f"YS-{m}" for m in MONTHS))


@pytest.fixture(scope="session", params=_ALL)
def freq_asstr(request) -> str:
    return request.param


@pytest.fixture(scope="session", params=_ALL)
def freq2_asstr(request) -> str:
    return request.param


@pytest.fixture(scope="session")
def equivalentfreq(freq_asstr, freq2_asstr) -> bool:
    if freq_asstr == freq2_asstr:
        return True  # same
    elif freq_asstr.startswith("QS") and freq2_asstr.startswith("QS"):
        month1, month2 = freq_asstr[-3:], freq2_asstr[-3:]
        for group in [("JAN", "APR", "JUL", "OCT"), ("FEB", "MAY", "AUG", "NOV")]:
            if month1 in group and month2 not in group:
                return False  # both quarters but not equivalent
        return True  # both quarters and equivalent
    return False  # different length


@pytest.fixture(scope="session", params=["2min", "4h", "7D", "3MS"])
def freq_nok_asstr(request) -> str:
    return request.param


@pytest.fixture(scope="session")
def freq_is_shorterthandaily(freq_asstr: str) -> bool:
    return freq_asstr in _SHORTERTHANDAILY


@pytest.fixture(scope="session")
def freq(freq_asstr: str) -> pd.tseries.offsets.BaseOffset:
    return to_offset(freq_asstr)


@pytest.fixture(scope="session")
def freq_nok(freq_nok_asstr: str) -> pd.tseries.offsets.BaseOffset:
    return to_offset(freq_nok_asstr)


# Start-of-day ---


@pytest.fixture(
    scope="session",
    params=[
        (dt.time(hour=0), "00:00:00", dt.timedelta(seconds=0)),
        (dt.time(hour=6), "06:00:00", dt.timedelta(hours=6)),
    ],
    ids=lambda t: t[1],
)
def _sod_time_str_tdelta(request) -> tuple[dt.time, str, dt.timedelta]:
    return request.param


@pytest.fixture(scope="session")
def sod(_sod_time_str_tdelta) -> dt.time:
    return _sod_time_str_tdelta[0]


@pytest.fixture(scope="session")
def sod_asstr(_sod_time_str_tdelta) -> str:
    return _sod_time_str_tdelta[1]


@pytest.fixture(scope="session")
def sod_astdelta(_sod_time_str_tdelta) -> dt.timedelta:
    return _sod_time_str_tdelta[2]


@pytest.fixture(
    scope="session",
    params=["00:00:00", "06:00:00"],
)
def sod2_asstr(request) -> str:
    return request.param


@pytest.fixture(scope="session")
def equalsod(sod_asstr, sod2_asstr) -> bool:
    return sod_asstr == sod2_asstr


@pytest.fixture(
    scope="session",
    params=[
        (
            dt.time(hour=6, minute=39, second=51),
            "06:39:51",
            dt.timedelta(hours=6, minutes=39, seconds=51),
        ),
        (dt.time(hour=6, minute=30), "06:30:00", dt.timedelta(hours=6, minutes=30)),
    ],
    ids=lambda t: t[1],
)
def _sod_time_str_tdelta_nok(request) -> tuple[dt.time, str, dt.timedelta]:
    return request.param


@pytest.fixture(scope="session")
def sod_nok(_sod_time_str_tdelta_nok) -> dt.time:
    return _sod_time_str_tdelta_nok[0]


@pytest.fixture(scope="session")
def sod_nok_asstr(_sod_time_str_tdelta_nok) -> str:
    return _sod_time_str_tdelta_nok[1]


@pytest.fixture(scope="session")
def sod_nok_astdelta(_sod_time_str_tdelta_nok) -> dt.timedelta:
    return _sod_time_str_tdelta_nok[2]


# Timezone ---


@pytest.fixture(
    scope="session",
    params=[
        pytest.param("Europe/Berlin", id="Berlin"),
        pytest.param("Asia/Kolkata", id="Kolkata"),
        None,
    ],
)
def tz(request) -> str | None:
    return request.param


@pytest.fixture(
    scope="session",
    params=[
        pytest.param("Europe/Berlin", id="Berlin"),
        pytest.param("Asia/Kolkata", id="Kolkata"),
        None,
    ],
)
def tz2(request) -> str | None:
    return request.param


@pytest.fixture(scope="session")
def equaltz(tz, tz2) -> bool:
    return tz == tz2


# Year/month/day ---


@pytest.fixture(scope="session", params=[2020, 2021])
def year(request) -> int:
    return request.param


@pytest.fixture(scope="session", params=["01-01", "02-01", "04-21"])
def monthday(request) -> str:
    return request.param


#
# @pytest.fixture(scope="session", params=["01-01", "02-01", "04-21"])
# def monthday2(request) -> str:
#     return request.param


# Date ---


# @pytest.fixture(scope="session")
# def date2022_othermonthday(other_monthday: str) -> str:
#     return f"2022-{other_monthday}"


# Stamp ---


@pytest.fixture(scope="session")
def stamp(year, monthday, sod_asstr, tz) -> pd.Timestamp:
    return pd.Timestamp(f"{year}-{monthday} {sod_asstr}", tz=tz)


@pytest.fixture(scope="session")
def stamp_on_freqboundary(monthday, freq_asstr) -> bool:
    if monthday == "01-01":
        if freq_asstr.startswith("YS"):
            return freq_asstr[3:] == "JAN"
        elif freq_asstr.startswith("QS"):
            return freq_asstr[3:] in ["JAN", "APR", "JUL", "OCT"]
        return True
    elif monthday == "02-01":
        if freq_asstr.startswith("YS"):
            return freq_asstr[3:] == "FEB"
        elif freq_asstr.startswith("QS"):
            return freq_asstr[3:] in ["FEB", "MAY", "AUG", "NOV"]
        return True
    elif monthday == "04-21":
        return freq_asstr in ("min", "5min", "15min", "h", "D")
    raise ValueError()


# Index ---


@pytest.fixture(scope="session")
def idx(stamp_on_freqboundary, monthday, sod_asstr, tz, freq) -> pd.DatetimeIndex:
    if not stamp_on_freqboundary:
        pytest.skip("Can't create index if first stamp does not fit the frequency.")

    return pd.date_range(
        f"2020-{monthday} {sod_asstr}",
        f"2022-{monthday} {sod_asstr}",
        freq=freq,
        inclusive="left",
        tz=tz,
    )


#
#
# def index2020to2022(date2020, date2022_other
#
#
# @dataclasses.dataclass
# class Date:
#     date: str
#     onfreqboundary: tuple[str, ...]
#
#     def __str__(self) -> str:
#         return self.date
#
#
# _DAILYANDSHORTER = ("min", "5min", "15min", "h", "D")
#
#
# @pytest.fixture(
#     scope="session",
#     params=[
#         Date(
#             "2020",
#             (*_DAILYANDSHORTER, "MS", *[f"QS-{MONTH_ALIASES[m]}" for m in [1, 4, 7, 10]], "YS-JAN"),
#         ),
#         Date(
#             "2020-02",
#             (*_DAILYANDSHORTER, "MS", *[f"QS-{MONTH_ALIASES[m]}" for m in [2, 5, 8, 11]], "YS-FEB"),
#         ),
#         Date(
#             "2020-04-21",
#             _DAILYANDSHORTER,
#         ),
#     ],
# )
# def date2020(request) -> Date:  # first half of 2020
#     return request.param
#
#
# @pytest.fixture(
#     scope="session",
#     params=[
#         Date(
#             "2022",
#             (*_DAILYANDSHORTER, "MS", *[f"QS-{MONTH_ALIASES[m]}" for m in [1, 4, 7, 10]], "YS-JAN"),
#         ),
#         Date(
#             "2022-02",
#             (*_DAILYANDSHORTER, "MS", *[f"QS-{MONTH_ALIASES[m]}" for m in [2, 5, 8, 11]], "YS-FEB"),
#         ),
#         Date(
#             "2022-04-21",
#             _DAILYANDSHORTER,
#         ),
#     ],
# )
# def date2022(request) -> Date:  # first half of 2022
#     return request.param
#
#
# # Timestamp ---
#
#
# @pytest.fixture(scope="session")
# def stamp2020(date2020: Date, sod: str, tz: str | None) -> pd.Timestamp:
#     return pd.Timestamp(f"{date2020} {sod}", tz=tz)
#
#
# # @pytest.fixture(scope="session")
# # def stamp2020_othersod_sametz(date2020: Date, other_sod: str, tz: str) -> pd.Timestamp:
# #     return pd.Timestamp(f"{date2020} {other_sod}", tz=tz)
# #
# #
# # @pytest.fixture(scope="session")
# # def stamp2020_samesod_othertz(date2020: Date, sod: str, other_tz: str) -> pd.Timestamp:
# #     return pd.Timestamp(f"{date2020} {sod}", tz=other_tz)
# #
# #
# # @pytest.fixture(scope="session")
# # def stamp2020_othersod_othertz(date2020: Date, other_sod: str, other_tz: str) -> pd.Timestamp:
# #     return pd.Timestamp(f"{date2020} {other_sod}", tz=other_tz)
#
#
# @pytest.fixture(scope="session")
# def stamp2022(date2022: Date, sod: str, tz: str) -> pd.Timestamp:
#     return pd.Timestamp(f"{date2022} {sod}", tz=tz)
#
#
# @pytest.fixture(scope="session")
# def stamp2022_othersod_sametz(date2022: Date, other_sod: str, tz: str) -> pd.Timestamp:
#     return pd.Timestamp(f"{date2022} {other_sod}", tz=tz)
#
#
# @pytest.fixture(scope="session")
# def stamp2022_samesod_othertz(date2022: Date, sod: str, other_tz: str) -> pd.Timestamp:
#     return pd.Timestamp(f"{date2022} {sod}", tz=other_tz)
#
#
# @pytest.fixture(scope="session")
# def stamp2022_othersod_othertz(date2022: Date, other_sod: str, other_tz: str) -> pd.Timestamp:
#     return pd.Timestamp(f"{date2022} {other_sod}", tz=other_tz)
#
#
# # Index ---
#
#
# @pytest.fixture(scope="session")
# def index2020to2022(
#     date2020: Date,
#     date2022: Date,
#     stamp2020: pd.DatetimeIndex,
#     stamp2022: pd.DatetimeIndex,
#     _freq: pd.DateOffset,
# ) -> pd.DatetimeIndex:
#     if (
#         _freq.freqstr not in date2020.onfreqboundary
#         or _freq.freqstr not in date2022.onfreqboundary
#     ):
#         pytest.skip(f"Can't create index from {stamp2020} to {stamp2022} with frequency {_freq}.")
#     return pd.date_range(
#         stamp2020, stamp2022, freq=_freq, inclusive="left"
#     )  # tz already specified in stamps
