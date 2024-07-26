import numpy as np
import pandas as pd
import pytest

from portfolyo import tools

freqs_small_to_large = [
    "min",
    "5min",
    "15min",
    "30min",
    "h",
    "2h",
    "D",
    "MS",
    "QS",
    "YS",
]
freqs_small_to_large_valid = [
    "15min",
    "h",
    "D",
    "MS",
    "QS",
    "QS-FEB",
    "YS",
    "YS-APR",
]
invalid_freq = ["min", "5min", "2h", "5D", "3MS"]


@pytest.mark.parametrize("count", range(1, 30))
def test_longestshortestfreq(count):
    indices = np.random.randint(0, len(freqs_small_to_large), count)
    freqs = np.array(freqs_small_to_large)[indices]
    assert tools.freq.longest(*freqs) == freqs_small_to_large[max(indices)]
    assert tools.freq.shortest(*freqs) == freqs_small_to_large[min(indices)]


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ("2020", "2021", "YS"),
        ("2020", "2020-04", "QS"),
        ("2020", "2020-02", "MS"),
        ("2020", "2020-01-02", "D"),
        ("2020", "2020-01-01 01:00", "h"),
        ("2020", "2020-01-01 00:15", "15min"),
        ("2020-03-29", "2020-03-30", "D"),
        ("2020-03-01", "2020-04-01", "MS"),
        ("2020-10-25", "2020-10-26", "D"),
        ("2020-10-01", "2020-11-01", "MS"),
    ],
)
def test_fromtdelta(start, end, expected, tz):
    """Test if correct frequency is guessed from start and end timestamp."""
    tdelta = pd.Timestamp(end, tz=tz) - pd.Timestamp(start, tz=tz)
    if expected is None:
        with pytest.raises(ValueError):
            _ = tools.freq.from_tdelta(tdelta)
    else:
        result = tools.freq.from_tdelta(tdelta)
        assert result == expected


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        # Hourly.
        ("2020-03-29 01:00", "2020-03-29 03:00", "h"),
        ("2020-10-25 01:00", "2020-10-25 02:00+0200", "h"),
        ("2020-10-25 01:00", "2020-10-25 02:00+0100", None),
        ("2020-10-25 02:00+0200", "2020-10-25 02:00+0100", "h"),
        ("2020-10-25 02:00+0200", "2020-10-25 03:00", None),
        ("2020-10-25 02:00+0100", "2020-10-25 03:00", "h"),
        # Quarterhourly.
        ("2020-03-29 01:45", "2020-03-29 03:00", "15min"),
        ("2020-10-25 01:45", "2020-10-25 02:00+0200", "15min"),
        ("2020-10-25 01:45", "2020-10-25 02:00+0100", None),
        ("2020-10-25 02:45+0200", "2020-10-25 02:00+0100", "15min"),
        ("2020-10-25 02:45+0200", "2020-10-25 03:00", None),
        ("2020-10-25 02:45+0100", "2020-10-25 03:00", "15min"),
    ],
)
def test_fromtdelta_dst(start, end, expected):
    """Test if correct frequency is guessed from start and end timestamp."""
    tz = "Europe/Berlin"
    tdelta = pd.Timestamp(end, tz=tz) - pd.Timestamp(start, tz=tz)
    if expected is None:
        with pytest.raises(ValueError):
            _ = tools.freq.from_tdelta(tdelta)
    else:
        result = tools.freq.from_tdelta(tdelta)
        assert result == expected


@pytest.mark.parametrize(
    ("freq", "num", "wanted", "expected"),
    [
        # D
        # . enough
        ("D", 10, "MS", ValueError),
        ("D", 10, "D", "D"),
        # . too few
        ("D", 2, "MS", ValueError),
        ("D", 2, "D", "D"),
        # 15T, too few
        ("15min", 2, "MS", ValueError),
        ("15min", 2, "15min", "15min"),
        # invalid freq
        # . enough
        ("2D", 10, "MS", ValueError),
        ("2D", 10, "2D", "2D"),
        # . too few
        ("2D", 2, "MS", ValueError),
        ("2D", 2, "2D", "2D"),
        # uncommon freq
        # . enough
        ("QS-APR", 10, "MS", ValueError),
        ("QS-APR", 10, "QS", "QS"),
        ("QS-APR", 10, "QS-FEB", ValueError),
        ("QS", 10, "QS-APR", "QS-APR"),
        # . too few
        ("QS-APR", 2, "MS", ValueError),
        ("QS-APR", 2, "QS", "QS"),
        ("QS-APR", 2, "QS-FEB", ValueError),
        ("QS", 2, "QS-APR", "QS-APR"),
    ],
)
@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
def test_setfreq(
    freq: str,
    num: int,
    wanted: str,
    tz: str,
    expected: str | Exception,
):
    i = pd.date_range("2020", periods=num, freq=freq, tz=tz)
    i.freq = None

    inputvalue = pd.Series(np.random.rand(num), i)
    # Test.
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            _ = tools.freq.set_to_frame(inputvalue, wanted)
        return
    result = tools.freq.set_to_frame(inputvalue, wanted)

    outputfreq = result.index.freq
    assert outputfreq == expected


@pytest.mark.parametrize(
    ("freq", "num", "expected"),
    [
        # D
        # . enough
        ("D", 10, "D"),
        # . too few
        ("D", 2, None),
        # 15T, too few
        ("15min", 2, None),
        # invalid freq
        # . enough
        ("2D", 10, "2D"),
        # . too few
        ("2D", 2, None),
        # uncommon freq
        # . enough
        ("QS-APR", 10, "QS"),
        ("QS", 10, "QS"),
        ("QS-FEB", 10, "QS-FEB"),
        ("QS-MAY", 10, "QS-FEB"),
        # . too few
        ("QS-APR", 2, None),
        ("QS", 2, None),
        ("YS-FEB", 10, "YS-FEB"),
    ],
)
@pytest.mark.parametrize("tz", [None, "Europe/Berlin"])
def test_guessfreq(
    freq: str,
    num: int,
    tz: str,
    expected: str | Exception,
):
    i = pd.date_range("2020", periods=num, freq=freq, tz=tz)
    i.freq = None

    inputvalue = pd.Series(np.random.rand(num), i)
    # Test.
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            _ = tools.freq.guess_to_frame(inputvalue)
        return
    result = tools.freq.guess_to_frame(inputvalue)

    outputfreq = result.index.freq
    assert outputfreq == expected


# Define your frequencies and their validity
freqs_with_validity = [
    ("15min", True),
    ("30min", False),
    ("D", True),
    ("h", True),
    ("MS", True),
    ("QS", True),
    ("YS", True),
    ("YS-APR", True),
    ("QS-FEB", True),
    ("min", False),
    ("5min", False),
    ("2h", False),
    ("5D", False),
    ("3MS", False),
]


@pytest.mark.parametrize("freq, is_valid", freqs_with_validity)
def test_freq_validity(freq: str, is_valid: bool):
    if is_valid:
        # No exception should be raised for valid frequencies
        tools.freq.assert_freq_valid(freq)
    else:
        # ValueError should be raised for invalid frequencies
        with pytest.raises(ValueError):
            _ = tools.freq.assert_freq_valid(freq)


@pytest.mark.parametrize(
    ("freq1", "freq2", "strict", "is_supposed_to_fail"),
    [
        ("15min", "15min", False, False),
        ("15min", "15min", True, True),
        ("h", "15min", True, False),
        ("15min", "h", True, True),
        ("15min", "h", False, True),
        ("MS", "MS", True, True),
        ("MS", "MS", False, False),
        ("MS", "QS-APR", False, True),
        ("QS", "YS", True, True),
        ("QS", "QS-APR", False, False),
        ("QS-FEB", "QS-APR", True, True),
        ("QS-FEB", "QS-APR", False, False),
        ("YS", "QS", False, False),
        ("QS-APR", "YS-APR", False, True),
    ],
)
def test_freq_sufficiently_long(
    freq1: str, freq2: str, strict: bool, is_supposed_to_fail: bool
):
    if is_supposed_to_fail:
        with pytest.raises(AssertionError):
            _ = tools.freq.assert_freq_sufficiently_long(freq1, freq2, strict)
    else:
        tools.freq.assert_freq_sufficiently_long(freq1, freq2, strict)


@pytest.mark.parametrize(
    ("freq1", "freq2", "is_supposed_to_fail"),
    [
        ("15min", "15min", False),
        ("h", "15min", True),
        ("15min", "h", True),
        ("MS", "MS", False),
        ("MS", "QS-APR", True),
        ("QS", "YS", True),
        ("QS", "QS-APR", False),
        ("QS-FEB", "QS-APR", False),
        ("YS", "QS", True),
        ("QS-APR", "YS-APR", True),
        ("YS-APR", "YS-FEB", False),
    ],
)
def test_freq_equally_long(freq1: str, freq2: str, is_supposed_to_fail: bool):
    if is_supposed_to_fail:
        with pytest.raises(AssertionError):
            _ = tools.freq.assert_freq_equally_long(freq1, freq2)
    else:
        tools.freq.assert_freq_equally_long(freq1, freq2)


@pytest.mark.parametrize(
    ("freq1", "freq2", "strict", "is_supposed_to_fail"),
    [
        ("15min", "15min", False, False),
        ("15min", "15min", True, True),
        ("h", "15min", True, True),
        ("15min", "h", True, False),
        ("15min", "h", False, False),
        ("MS", "MS", True, True),
        ("MS", "MS", False, False),
        ("MS", "QS-APR", False, False),
        ("QS", "YS", True, False),
        ("QS", "QS-APR", False, False),
        ("QS-FEB", "QS-APR", True, True),
        ("QS-FEB", "QS-APR", False, False),
        ("YS", "QS", False, True),
        ("QS-APR", "YS-APR", False, False),
    ],
)
def test_freq_sufficiently_short(
    freq1: str, freq2: str, strict: bool, is_supposed_to_fail: bool
):
    if is_supposed_to_fail:
        with pytest.raises(AssertionError):
            _ = tools.freq.assert_freq_sufficiently_short(freq1, freq2, strict)
    else:
        tools.freq.assert_freq_sufficiently_short(freq1, freq2, strict)


@pytest.mark.parametrize(
    ("source_freq", "ref_freq", "expected"),
    [
        # downsampling
        ("D", "MS", -1),
        ("MS", "QS", -1),
        ("MS", "QS-APR", -1),
        ("QS", "YS-APR", -1),
        ("QS", "YS", -1),
        # upsampling
        ("QS", "D", 1),
        ("YS-APR", "QS", 1),
        # the same
        ("MS", "MS", 0),
        ("QS", "QS", 0),
        ("QS", "QS-APR", 0),
        ("QS", "QS-JAN", 0),
        # ValueError
        ("QS", "QS-FEB", ValueError),
        ("QS", "YS-FEB", ValueError),
        ("YS-APR", "YS", ValueError),
        ("YS-FEB", "QS", ValueError),
    ],
)
def test_up_pr_down2(source_freq: str, ref_freq: str, expected: int | Exception):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            tools.freq.up_or_down(source_freq, ref_freq)
    else:
        result = tools.freq.up_or_down(source_freq, ref_freq)
        assert result == expected
