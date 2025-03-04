from portfolyo import toolsb
import pytest
import pandas as pd
from utils import id_fn  # relative to /tests


def test_index_case1_toright(case1_idx, case1_rightidx):
    toolsb.testing.assert_index_equal(toolsb.index.to_right(case1_idx), case1_rightidx)


def test_index_case1_duration(case1_idx, case1_duration):
    toolsb.testing.assert_series_equal(toolsb.index.duration(case1_idx), case1_duration)


def test_index_case2_toright(case2_idx, case2_rightidx):
    toolsb.testing.assert_index_equal(toolsb.index.to_right(case2_idx), case2_rightidx)


def test_index_case2_duration(case2_idx, case2_duration):
    toolsb.testing.assert_series_equal(toolsb.index.duration(case2_idx), case2_duration)


def test_index_case3_trim(case3_idx, case3_trimfreq, case3_trimmedidx):
    if case3_trimmedidx is Exception:
        with pytest.raises(ValueError):
            _ = toolsb.index.trim(case3_idx, case3_trimfreq)
    else:
        toolsb.testing.assert_index_equal(
            toolsb.index.trim(case3_idx, case3_trimfreq), case3_trimmedidx
        )


# ---


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
    (("2020-02", "2022-11", "QS-FEB"), ("2021-02", "2023-02", "YS-FEB"), ("2021-02", "2022-11")),
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
        for sod2 in ["00:00", "06:00"]
        for tz1 in [None, "Europe/Berlin", "Asia/Kolkata"]
        for tz2 in [None, "Europe/Berlin", "Asia/Kolkata"]
        if (freq1 != freq2 and set([freq1, freq2]) != set(["QS-JAN", "QS-APR"]))
        or sod1 != sod2
        or tz1 != tz2
    ],
    ids=id_fn,
)
def test_normalintersect_nok(idx1, idx2):
    with pytest.raises(ValueError):
        _ = toolsb.index.intersect([idx1, idx2])


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
        for sod2 in ["00:00", "06:00"]
        for tz1 in [None, "Europe/Berlin", "Asia/Kolkata"]
        for tz2 in [None, "Europe/Berlin", "Asia/Kolkata"]
        if None in set([tz1, tz2]) or tz1 == tz2
    ],
    ids=id_fn,
)
@pytest.mark.parametrize("ignorefreq", [True, False])
@pytest.mark.parametrize("ignoresod", [True, False])
@pytest.mark.parametrize("ignoretz", [True, False])
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
        and not ignoresod
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
        for sod2 in ["00:00", "06:00"]
        for tz1 in [None, "Europe/Berlin", "Asia/Kolkata"]
        for tz2 in [None, "Europe/Berlin", "Asia/Kolkata"]
    ],
    ids=id_fn,
)
@pytest.mark.parametrize("ignorefreq", [True, False])
@pytest.mark.parametrize("ignoresod", [True, False])
@pytest.mark.parametrize("ignoretz", [True, False])
def test_flexintersect_nok(
    idx1,
    idx2,
    ignorefreq,
    ignoresod,
    ignoretz,
):
    with pytest.raises(ValueError):
        toolsb.index.intersect_flex(
            [idx1, idx2], ignore_freq=ignorefreq, ignore_startofday=ignoresod, ignore_tz=ignoretz
        )


# LEFT_FREQ_PERIODS_RIGHT_DURATION = [
#     ("2020", "min", 5 * 24 * 60, "2020-01-01 00:01", 1 / 60),
#     ("2020", "5min", 5 * 24 * 12, "2020-01-01 00:05", 5 / 60),
#     ("2020", "15min", 5 * 24 * 4, "2020-01-01 00:15", 15 / 60),
#     ("2020", "h", 5 * 24, "2020-01-01 01:00", 1),
#     ("2020", "D", 5, "2020-01-02", 24),
#     ("2020", "MS", 3, "2020-02", [31 * 24, 29 * 24, 31 * 24]),
#     ("2020", "QS-JAN", 3, "2020-04", [91 * 24, 91 * 24, 92 * 24]),
#     ("2020", "QS-APR", 3, "2020-04", [91 * 24, 91 * 24, 92 * 24]),
#     ("2020", "QS-JUL", 3, "2020-04", [91 * 24, 91 * 24, 92 * 24]),
#     ("2020", "QS-OCT", 3, "2020-04", [91 * 24, 91 * 24, 92 * 24]),
#     ("2020", "YS", 3, "2021", [8784, 8760, 8760]),
#     ("2020-05", "min", 5 * 24 * 60, "2020-05-01 00:01", 1 / 60),
#     ("2020-05", "5min", 5 * 24 * 12, "2020-05-01 00:05", 5 / 60),
#     ("2020-05", "15min", 5 * 24 * 4, "2020-05-01 00:15", 15 / 60),
#     ("2020-05", "h", 5 * 24, "2020-05-01 01:00", 1),
#     ("2020-05", "D", 5, "2020-05-02", 24),
#     ("2020-05", "MS", 3, "2020-06", [31 * 24, 30 * 24, 31 * 24]),
#     ("2020-05", "QS-FEB", 3, "2020-08", [92 * 24, 92 * 24, 92 * 24]),
#     ("2020-05", "QS-MAY", 3, "2020-08", [92 * 24, 92 * 24, 92 * 24]),
#     ("2020-05", "QS-AUG", 3, "2020-08", [92 * 24, 92 * 24, 92 * 24]),
#     ("2020-05", "QS-NOV", 3, "2020-08", [92 * 24, 92 * 24, 92 * 24]),
#     ("2020-05", "YS-MAY", 3, "2021-05", [8760, 8760, 8760]),
#     ("2020-04-21 15:00", "min", 5 * 24 * 60, "2020-04-21 15:01", 1 / 60),
#     ("2020-04-21 15:00", "5min", 5 * 24 * 12, "2020-04-21 15:05", 5 / 60),
#     ("2020-04-21 15:00", "15min", 5 * 24 * 4, "2020-04-21 15:15", 15 / 60),
#     ("2020-04-21 15:00", "h", 5 * 24, "2020-04-21 16:00", 1),
# ]
