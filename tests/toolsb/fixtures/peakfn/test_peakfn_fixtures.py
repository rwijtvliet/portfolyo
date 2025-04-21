import pytest

from portfolyo import toolsb


class TestPeakfnCase1:
    def test_peakperiods(
        self, case1_peakfn, case1_idx, case1_peakperiodcount, case1_peakonlystretch
    ):
        result = case1_peakfn(case1_idx)
        assert result.sum() == case1_peakperiodcount
        assert all(result[case1_peakonlystretch])


class TestPeakfnCase2:
    def test_peakfn_nok(self, case2_peakfn, case2_idx):
        with pytest.raises(ValueError):
            _ = case2_peakfn(case2_idx)


class TestPeakfnCase3:
    def test_peakfn_dst(
        self, case3_peakfn, case3_idx, case3_peakperiodcount, case3_peakonlystretch
    ):
        result = case3_peakfn(case3_idx)
        assert result.sum() == case3_peakperiodcount
        assert all(result[case3_peakonlystretch])


class TestPeakfnCase4:
    def test_peakfn_duration(self, case4_durationfn1param, case4_idx, case4_duration):
        result = case4_durationfn1param(case4_idx)
        toolsb.testing.assert_series_equal(result, case4_duration)
