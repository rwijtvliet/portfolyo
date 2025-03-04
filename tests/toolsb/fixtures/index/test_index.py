from portfolyo import toolsb
import pytest


def test_index_case1_toright(case1_idx, case1_rightidx):
    toolsb.testing.assert_index_equal(toolsb.index.to_right(case1_idx), case1_rightidx)


def test_index_case1_duration(case1_idx, case1_duration):
    toolsb.testing.assert_series_equal(toolsb.index.duration(case1_idx), case1_duration)


def test_index_case2_toright(case2_idx, case2_rightidx):
    toolsb.testing.assert_index_equal(toolsb.index.to_right(case2_idx), case2_rightidx)


def test_index_case2_duration(case2_idx, case2_duration):
    toolsb.testing.assert_series_equal(toolsb.index.duration(case2_idx), case2_duration)


def test_index_case3_trim_ok(case3_idx, case3_trimfreq, case3_trimmedidx):
    toolsb.testing.assert_index_equal(
        toolsb.index.trim(case3_idx, case3_trimfreq), case3_trimmedidx
    )


def test_index_case3_trim_nok(case3_idx, case3_trimfreq, case3_isok):
    if case3_isok:  # we want to avoid cases where trimming is possible
        pytest.skip("Not an error case.")
    with pytest.raises(ValueError):
        _ = toolsb.index.trim(case3_idx, case3_trimfreq)


def test_index_intersect_identical(idx):
    toolsb.testing.assert_index_equal(toolsb.index.intersect([idx, idx]), idx)


def test_index_case4_normalintersect_ok(case4_idx1, case4_idx2, case4_normalintersect_idx):
    toolsb.testing.assert_index_equal(
        toolsb.index.intersect([case4_idx1, case4_idx2]), case4_normalintersect_idx
    )


def test_index_case4_normalintersect_nok(case4_idx1, case4_idx2, case4_normalintersect_isok):
    if case4_normalintersect_isok:
        pytest.skip("Not an error case.")
    with pytest.raises(ValueError):
        toolsb.index.intersect([case4_idx1, case4_idx2])


def test_index_case4_flexintersect_ok(
    case4_idx1,
    case4_idx2,
    case4_ignorefreq,
    case4_ignoresod,
    case4_ignoretz,
    case4_flexintersect_idxs,
):
    resultidxs = toolsb.index.intersect_flex(
        [case4_idx1, case4_idx2],
        ignore_freq=case4_ignorefreq,
        ignore_startofday=case4_ignoresod,
        ignore_tz=case4_ignoretz,
    )
    for resultidx, expectedidx in zip(resultidxs, case4_flexintersect_idxs):
        toolsb.testing.assert_index_equal(resultidx, expectedidx)


def test_index_case4_flexintersect_nok(
    case4_idx1,
    case4_idx2,
    case4_ignorefreq,
    case4_ignoresod,
    case4_ignoretz,
    case4_flexintersect_isok,
):
    if case4_flexintersect_isok:
        pytest.skip("Not an error case.")
    with pytest.raises(ValueError):
        toolsb.index.intersect_flex(
            [case4_idx1, case4_idx2],
            ignore_freq=case4_ignorefreq,
            ignore_startofday=case4_ignoresod,
            ignore_tz=case4_ignoretz,
        )


# ---


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
