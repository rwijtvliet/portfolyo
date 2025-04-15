import numpy as np
from typing import Callable
from pandas.tseries.offsets import BaseOffset
import pytest
import pandas as pd
from portfolyo import toolsb


@pytest.fixture(scope="module", params=["uniform", "distinct"])
def complexity(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def shortfreq(freq):
    return freq


@pytest.fixture(scope="module")
def longfreq(freq2_that_is_longer_or_same_length_as_freq):
    return freq2_that_is_longer_or_same_length_as_freq


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
        f"{startdate_asstr} {sod_asstr}",
        f"2022-02-15 {sod_asstr}",
        tz=tz,
        freq=shortfreq,
        inclusive="left",
    )


@pytest.fixture(scope="module")
def idx_longfreq_untrimmed(longfreq, idx_shortfreq_untrimmed, sod) -> pd.DatetimeIndex:
    # Generous index with long frequency to cover at least idx_shortfreq_untrimmed.
    return pd.date_range(
        toolsb.stamp.floor(idx_shortfreq_untrimmed[0], longfreq, sod),
        toolsb.stamp.ceil(idx_shortfreq_untrimmed[-1], longfreq, sod),
        freq=longfreq,
        inclusive="left",
    )


@pytest.fixture(scope="module")
def _identifier_fn(longfreq, sod) -> Callable[[pd.DatetimeIndex], pd.Index]:
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
        startmonth0 = startmonth - 1
        return (((month0(idx) - startmonth0) // 3) - 1 * prevquarter(idx)) % 4  # 0..3

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


# @pytest.fixture(scope="module")
# def mapping_unfiltered(
#     idx_shortfreq_untrimmed,
#     idx_longfreq_untrimmed,
#     _identifier_fn
# ) -> pd.Series:
#     # Find mapping.
#     # - Index: timestamps of (trimmed) short index (unique values).
#     # - Values: timestamps of long index.
#     long = pd.Series(_identifier_fn(idx_longfreq_untrimmed), idx_longfreq_untrimmed)
#     short = pd.Series(_identifier_fn(idx_shortfreq_untrimmed), idx_shortfreq_untrimmed)
#     return pd.Series(long[short], short.index)
#
# def mapping(mapping_unfiltered):
#     # Merge by aligning on the identifier-tuple.
#     df_mapping = df_long.merge(df_shrt, left_index=True, right_index=True)
#     # Calculate additional information.
#     df_mapping["fraction"] = df_mapping["dur_shrt"] / df_mapping["dur_long"]
#     # Put indices in index, keep only fraction as values.
#     s_mapping = df_mapping.set_index(["idx_longfreq", "idx_shortfreq"])["fraction"]
#     # . Reject periods that are not fully present.
#     reject = s_mapping.groupby(level=0).sum() < 0.99
#     return s_mapping[~reject]
#


@pytest.fixture(scope="module")
def mapping_and_fractions(
    idx_shortfreq_untrimmed,
    idx_longfreq_untrimmed,
    _identifier_fn,
) -> pd.Series:
    # Find mapping.
    # - Index level 0: timestamps of long index (multiple rows with same timestamp).
    # - Index level 1: timestamps of short (trimmed) index.
    # - Values: fraction of long timestamp spent in each of the shorter timestamps.
    df_long = pd.DataFrame(
        {
            "idx_longfreq": idx_longfreq_untrimmed,
            "dur_long": toolsb.index.duration(idx_longfreq_untrimmed).pint.m.values,
        },
        _identifier_fn(idx_longfreq_untrimmed),
    )
    df_shrt = pd.DataFrame(
        {
            "idx_shortfreq": idx_shortfreq_untrimmed,
            "dur_shrt": toolsb.index.duration(idx_shortfreq_untrimmed).pint.m.values,
        },
        _identifier_fn(idx_shortfreq_untrimmed),
    )
    # Merge by aligning on the identifier-tuple.
    df_mapping = df_long.merge(df_shrt, left_index=True, right_index=True)
    # Calculate additional information.
    df_mapping["fraction"] = df_mapping["dur_shrt"] / df_mapping["dur_long"]
    # Put indices in index, keep only fraction as values.
    s_mapping = df_mapping.set_index(["idx_longfreq", "idx_shortfreq"])["fraction"]
    # . Reject periods that are not fully present.
    reject = s_mapping.groupby(level=0).sum() < 0.99
    return s_mapping[~reject]


@pytest.fixture(scope="module")
def mapping(mapping_and_fractions) -> pd.Series:
    return pd.Series(
        mapping_and_fractions.index.get_level_values("idx_longfreq"),
        mapping_and_fractions.index.get_level_values("idx_shortfreq"),
    )


@pytest.fixture(scope="module")
def fractions(mapping_and_fractions, shortfreq) -> pd.Series:
    return mapping_and_fractions.droplevel("idx_longfreq")


@pytest.fixture(scope="module")
def idx_shortfreq(mapping, shortfreq, tz):
    # indices that will actually be result of changing frequency
    return pd.DatetimeIndex(mapping.index, freq=shortfreq, tz=tz).rename(None)


@pytest.fixture(scope="module")
def idx_longfreq(mapping, longfreq, tz):
    # indices that will actually be result of changing frequency
    return pd.DatetimeIndex(mapping.unique(), freq=longfreq, tz=tz).rename(None)


# Calculate values for each case: upsample/downsample x summable/avgable


@pytest.fixture(scope="module", params=["summable", "averagable"])
def avg_or_sum(request) -> str:
    return request.param


@pytest.fixture(scope="module", params=["upsample", "downsample"])
def up_or_down(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def resample_fn(avg_or_sum) -> Callable:
    return {"sum": toolsb.changefreq.summable, "avg": toolsb.changefreq.averagable}[avg_or_sum]


@pytest.fixture(scope="module")
def targetfreq(up_or_down, longfreq, shortfreq) -> BaseOffset:
    return {"down": longfreq, "up": shortfreq}[up_or_down]


@pytest.fixture(scope="module")
def inputidx(up_or_down, idx_longfreq, idx_shortfreq_untrimmed) -> pd.DatetimeIndex:
    return {"down": idx_shortfreq_untrimmed, "up": idx_longfreq}[up_or_down]


@pytest.fixture(scope="module")
def outputidx(up_or_down, idx_longfreq, idx_shortfreq) -> pd.DatetimeIndex:
    return {"down": idx_longfreq, "up": idx_shortfreq}[up_or_down]


@pytest.fixture(scope="module")
def sin_and_sout_uniform(
    up_or_down,
    avg_or_sum,
    mapping,
    fractions,
    idx_longfreq,
    idx_shortfreq,
    idx_shortfreq_untrimmed,
) -> tuple[pd.Series, pd.Series]:
    # NB: inputidx may be untrimmed and therefore have more values than in mapping.index
    if (avg_or_sum, up_or_down) == ("avg", "up"):
        sin, sout = pd.Series(1.0, idx_longfreq), pd.Series(1.0, idx_shortfreq)
    elif (avg_or_sum, up_or_down) == ("avg", "down"):
        sin, sout = pd.Series(1.0, idx_shortfreq_untrimmed), pd.Series(1.0, idx_longfreq)
    elif (avg_or_sum, up_or_down) == ("sum", "up"):
        sin, sout = pd.Series(1.0, idx_longfreq), pd.Series(fractions, idx_shortfreq)
    else:
        sin = pd.Series(1.0, idx_shortfreq_untrimmed)
        sout = pd.Series(mapping.groupby(mapping).count() * 1.0, idx_longfreq)
    return sin, sout


@pytest.fixture(scope="module")
def sin_and_sout_distinct(
    up_or_down,
    avg_or_sum,
    mapping,
    fractions,
    idx_longfreq,
    idx_shortfreq,
    idx_shortfreq_untrimmed,
) -> tuple[pd.Series, pd.Series]:
    np.random.seed(1)  # to ensure always same

    def get_inputframe(idx):
        return pd.Series(np.random.random(len(idx)) * 100, idx)

    if (avg_or_sum, up_or_down) == ("avg", "up"):
        sin = get_inputframe(idx_longfreq)
        sout = sin[mapping].set_axis(idx_shortfreq)  # repeats value
    elif (avg_or_sum, up_or_down) == ("avg", "down"):
        sin = get_inputframe(idx_shortfreq_untrimmed)
        sout = (
            (sin * fractions).dropna().set_axis(mapping.values).groupby(level=0).sum()
        )  # TODO: is there an easier way to aggregate over index?
    elif (avg_or_sum, up_or_down) == ("sum", "up"):
        sin = get_inputframe(idx_longfreq)
        sout = fractions * sin[mapping.values].values
    else:
        sin = get_inputframe(idx_shortfreq_untrimmed)
        sout = sin[mapping].set_axis(mapping.values).groupby(level=0).sum()

    return sin, sout


@pytest.fixture(scope="module")
def sin(sin_and_sout_uniform, sin_and_sout_distinct, complexity):
    return (sin_and_sout_uniform if complexity == "uniform" else sin_and_sout_distinct)[0].rename(
        "testseries"
    )


@pytest.fixture(scope="module")
def sout(sin_and_sout_uniform, sin_and_sout_distinct, complexity):
    return (sin_and_sout_uniform if complexity == "uniform" else sin_and_sout_distinct)[1].rename(
        "testseries"
    )


@pytest.fixture(scope="module")
def inputframe(sin, frame_type, with_units) -> pd.Series | pd.DataFrame:
    # Add units.
    if with_units:
        sin = sin.astype("pint[MW]")

    # Change to dataframe if wanted.
    return sin.to_frame() if frame_type is pd.DataFrame else sin


@pytest.fixture(scope="module")
def outputframe(sout, frame_type, with_units) -> pd.Series | pd.DataFrame:
    # Add units.
    if with_units:
        sout = sout.astype("pint[MW]")

    # Change to dataframe if wanted.
    return sout.to_frame() if frame_type is pd.DataFrame else sout
