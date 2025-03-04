from portfolyo import toolsb


def test_stamp_duration(stamp, freq, stamp_duration):
    assert toolsb.stamp.duration(stamp, freq) == stamp_duration


def test_stamp_isboundary(stamp, freq, sod, stamp_on_freqboundary):
    assert toolsb.stamp.is_boundary(stamp, freq, sod) == stamp_on_freqboundary


def test_stamp_floor_isboundary(stamp, freq, sod, stamp_on_freqboundary):
    assert (toolsb.stamp.floor(stamp, freq, sod) == stamp) == stamp_on_freqboundary


def test_stamp_ceil_isboundary(stamp, freq, sod, stamp_on_freqboundary):
    assert (toolsb.stamp.ceil(stamp, freq, sod) == stamp) == stamp_on_freqboundary


def test_stamp_case1_right(case1_stamp, case1_freq, case1_right):
    assert toolsb.stamp.to_right(case1_stamp, case1_freq) == case1_right


def test_stamp_case1_duration(case1_stamp, case1_freq, case1_duration):
    assert toolsb.stamp.duration(case1_stamp, case1_freq) == case1_duration


def test_stamp_case2_right(case2_stamp, case2_freq, case2_right):
    assert toolsb.stamp.to_right(case2_stamp, case2_freq) == case2_right


def test_stamp_case2_duration(case2_stamp, case2_freq, case2_duration):
    assert toolsb.stamp.duration(case2_stamp, case2_freq) == case2_duration


def test_stamp_case3_floor(case3_stamp, case3_freq, case3_sodstr, case3_floored):
    assert toolsb.stamp.floor(case3_stamp, case3_freq, case3_sodstr) == case3_floored


def test_stamp_case3_ceil(case3_stamp, case3_freq, case3_sodstr, case3_ceiled):
    assert toolsb.stamp.ceil(case3_stamp, case3_freq, case3_sodstr) == case3_ceiled


def test_stamp_case4_replacetime(case4_stamp, case4_newsodstr, case4_newstamp):
    assert toolsb.stamp.replace_time(case4_stamp, case4_newsodstr) == case4_newstamp
