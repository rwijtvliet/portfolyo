import pytest

from portfolyo import toolsb


def test_index_intersect_identical(idx):
    toolsb.testing.assert_index_equal(toolsb.index.intersect([idx, idx]), idx)


class TestCase1:
    def test_index_case1_toright(case1_idx, case1_rightidx):
        toolsb.testing.assert_index_equal(toolsb.index.to_right(case1_idx), case1_rightidx)

    def test_index_case1_duration(case1_idx, case1_duration):
        toolsb.testing.assert_series_equal(toolsb.index.duration(case1_idx), case1_duration)


class TestCase2:
    def test_index_case2_toright(case2_idx, case2_rightidx):
        toolsb.testing.assert_index_equal(toolsb.index.to_right(case2_idx), case2_rightidx)

    def test_index_case2_duration(case2_idx, case2_duration):
        toolsb.testing.assert_series_equal(toolsb.index.duration(case2_idx), case2_duration)


class TestCase3:
    def test_index_case3_trim_ok(case3_idx, case3_trimfreq, case3_trimmedidx):
        toolsb.testing.assert_index_equal(
            toolsb.index.trim(case3_idx, case3_trimfreq), case3_trimmedidx
        )

    def test_index_case3_trim_nok(case3_idx, case3_trimfreq, case3_isok):
        if case3_isok:  # we want to avoid cases where trimming is possible
            pytest.skip("Not an error case.")
        with pytest.raises(ValueError):
            _ = toolsb.index.trim(case3_idx, case3_trimfreq)


class TestCase4:
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
