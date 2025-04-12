import numpy as np
from typing import Callable
import pytest
import pandas as pd
from portfolyo import toolsb


@pytest.fixture(scope="module", params=["ones", "allnumbers"])
def complexity(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def shortfreq(freq):
    return freq


@pytest.fixture(scope="module")
def longfreq(freq2_that_is_longer_than_freq):
    return freq2_that_is_longer_than_freq


@pytest.fixture(scope="module")
def startdate_asstr(shortfreq) -> str:
    if shortfreq == "15min" or shortfreq == "h" or shortfreq == "D":
        return "2019-12-15"
    elif shortfreq == "MS":
        return "2019-12-01"
    elif shortfreq == "QS-JAN" or shortfreq == "QS-APR":
        return "2020-04-01"
    elif shortfreq == "QS-FEB":
        return "2020-02-01"
    elif shortfreq == "YS-JAN":
        return "2020-01-01"
    elif shortfreq == "YS-FEB":
        return "2020-02-01"
    raise ValueError("Unexpected value for freq")


@pytest.fixture(scope="module")
def idx_shortfreq_untrimmed(shortfreq, tz, startdate_asstr, sod_asstr) -> pd.DatetimeIndex:
    return pd.date_range(
        f"{startdate_asstr} {sod_asstr}", "2022-02-15", tz=tz, freq=shortfreq, inclusive="left"
    )


@pytest.fixture(scope="module")
def idx_longfreq_untrimmed(longfreq, idx_shortfreq_untrimmed, sod) -> pd.DatetimeIndex:
    # Generous index with long frequency to cover at least idx_shortfreq_untrimmed.
    return pd.date_range(
        toolsb.stamp.floor(idx_shortfreq_untrimmed[0], longfreq, sod),
        toolsb.stamp.ceil(idx_shortfreq_untrimmed[-1], longfreq, sod),
        freq=longfreq,
    )


@pytest.fixture(scope="module")
def identifier_fn(longfreq, sod) -> Callable[[pd.DatetimeIndex], pd.Index]:
    # Identifiers that allow for 1-to-n mapping between idx_longfreq_untrimmed and idx_shortfreq_untrimmed

    def utc(idx):
        if not idx.tz:
            return np.repeat(0, len(idx))
        else:
            return [ts.utcoffset().total_seconds() for ts in idx]

    def prevday(idx) -> pd.Index:  # [bool]:
        return idx.time < sod

    def prevmonth(idx) -> pd.Index:  # [bool]:
        return (idx.day == 1) & prevday(idx)

    def prevquarter(idx, startmonth=1) -> pd.Index:  # [bool]:
        return ((idx.month - startmonth) % 3 == 0) & prevmonth(idx)

    def prevyear(idx, startmonth=1) -> pd.Index:  # [bool]:
        return (idx.month < startmonth) | (idx.month == startmonth) & prevmonth(idx)

    def numofdaysinmonth(idx):
        days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        m0 = month0(idx)
        return days.loc[m0] + 1 * (idx.year == 2020) * (m0 == 1)

    def day0(idx) -> pd.Index:  # [int]:
        return (idx.day - 1 * prevday(idx) - 1) % numofdaysinmonth(idx)  # 0..27/28/29/30

    def month0(idx) -> pd.Index:  # [int]:
        return (idx.month - 1 * prevmonth(idx) - 1) % 12  # 0..11

    def quarter0(idx, startmonth=1) -> pd.Index:  # [int]:
        return (((month0(idx) - startmonth) // 3) - 1 * prevquarter(idx)) % 4  # 0..3

    def year(idx, startmonth=1) -> pd.Index:  # [int]:
        return idx.year - 1 * prevyear(idx, startmonth)

    if longfreq == "15min":
        return lambda idx: pd.Index(
            zip(idx.year, idx.month, idx.day, idx.hour, idx.minute, utc(idx))
        )
    elif longfreq == "h":
        return lambda idx: pd.Index(zip(idx.year, idx.month, idx.day, idx.hour, utc(idx)))
    elif longfreq == "D":
        return lambda idx: pd.Index(zip(year(idx), month0(idx), day0(idx)))
    elif longfreq == "MS":
        return lambda idx: pd.Index(zip(year(idx), month0(idx)))
    elif longfreq == "QS-JAN" or longfreq == "QS-APR":
        return lambda idx: pd.Index(zip(year(idx), quarter0(idx)))
    elif longfreq == "QS-FEB":
        return lambda idx: pd.Index(zip(year(idx, 2), quarter0(idx, 2)))
    elif longfreq == "YS-JAN":
        return lambda idx: year(idx)
    elif longfreq == "YS-FEB":
        return lambda idx: year(idx, 2)
    raise ValueError("Unexpected value for `freq2_that_is_longer_than_freq`.")


@pytest.fixture(scope="module")
def mapping_longshort(
    idx_shortfreq_untrimmed,
    idx_longfreq_untrimmed,
    identifier_fn,
) -> pd.DataFrame:
    # Find mapping. Index: timestamps of long trimmed index. Columns: ['fraction', 'ts_short'].
    df_long = pd.DataFrame(
        {
            "dur_long": toolsb.index.duration(idx_longfreq_untrimmed).pint.m.values,
            "ts_long": idx_longfreq_untrimmed,
        },
        identifier_fn(idx_longfreq_untrimmed),
    )
    df_shrt = pd.DataFrame(
        {
            "dur_shrt": toolsb.index.duration(idx_shortfreq_untrimmed).pint.m.values,
            "ts_shrt": idx_shortfreq_untrimmed,
        },
        identifier_fn(idx_shortfreq_untrimmed),
    )
    df_mapping = df_long.merge(df_shrt, left_index=True, right_index=True)
    df_mapping["fraction"] = df_mapping["dur_shrt"] / df_mapping["dur_long"]
    df_mapping = df_mapping.drop(columns=["dur_long", "dur_shrt"])
    # . Reject periods that are not fully present.
    reject = df_mapping["fraction"].groupby(df_mapping.index).sum() < 0.99
    df_mapping = df_mapping[~reject]
    # df_mapping = df_mapping.drop(reject[reject].index)
    return df_mapping.set_index("ts_long")


@pytest.fixture(scope="module")
def idx_shortfreq(mapping_longshort, shortfreq, tz):
    # indices that will actually be result of changing frequency
    return pd.DatetimeIndex(
        mapping_longshort["ts_shrt"].sort_values(), freq=shortfreq, tz=tz
    ).rename(None)


@pytest.fixture(scope="module")
def idx_longfreq(mapping_longshort, longfreq, tz):
    # indices that will actually be result of changing frequency
    return pd.DatetimeIndex(mapping_longshort.index.sort_values(), freq=longfreq, tz=tz).rename(
        None
    )
