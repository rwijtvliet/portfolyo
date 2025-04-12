from portfolyo import toolsb


class TestStamp:
    def test_duration(self, stamp, freq, stamp_duration):
        assert toolsb.stamp.duration(stamp, freq) == stamp_duration

    def test_isboundary(self, stamp, freq, sod, stamp_on_freqboundary):
        assert toolsb.stamp.is_boundary(stamp, freq, sod) == stamp_on_freqboundary

    def test_floor_isboundary(self, stamp, freq, sod, stamp_on_freqboundary):
        assert (toolsb.stamp.floor(stamp, freq, sod) == stamp) == stamp_on_freqboundary

    def test_ceil_isboundary(self, stamp, freq, sod, stamp_on_freqboundary):
        assert (toolsb.stamp.ceil(stamp, freq, sod) == stamp) == stamp_on_freqboundary


class TestStampCase1:
    def test_case1_right(self, case1_stamp, case1_freq, case1_right):
        assert toolsb.stamp.to_right(case1_stamp, case1_freq) == case1_right

    def test_case1_duration(self, case1_stamp, case1_freq, case1_duration):
        assert toolsb.stamp.duration(case1_stamp, case1_freq) == case1_duration


class TestStampCase2:
    def test_case2_right(self, case2_stamp, case2_freq, case2_right):
        assert toolsb.stamp.to_right(case2_stamp, case2_freq) == case2_right

    def test_case2_duration(self, case2_stamp, case2_freq, case2_duration):
        assert toolsb.stamp.duration(case2_stamp, case2_freq) == case2_duration


class TestStampCase3:
    def test_case3_floor(self, case3_stamp, case3_freq, case3_sodstr, case3_floored):
        assert toolsb.stamp.floor(case3_stamp, case3_freq, case3_sodstr) == case3_floored

    def test_case3_ceil(self, case3_stamp, case3_freq, case3_sodstr, case3_ceiled):
        assert toolsb.stamp.ceil(case3_stamp, case3_freq, case3_sodstr) == case3_ceiled


class TestStampCase4:
    def test_case4_replacetime(self, case4_stamp, case4_newsodstr, case4_newstamp):
        assert toolsb.stamp.replace_time(case4_stamp, case4_newsodstr) == case4_newstamp
