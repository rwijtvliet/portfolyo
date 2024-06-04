import numpy as np
import pandas as pd
import pytest

from portfolyo import tools

freqs_small_to_large = ["T", "5T", "15T", "30T", "H", "2H", "D", "MS", "QS", "AS"]
freqs_small_to_large_valid = [
    "15T",
    "30T",
    "H",
    "D",
    "MS",
    "QS",
    "QS-FEB",
    "AS",
    "AS-APR",
]
invalid_freq = ["T", "5T", "2H", "5D", "3MS"]


@pytest.mark.parametrize("count", range(1, 30))
def test_longestshortestfreq(count):
    indices = np.random.randint(0, len(freqs_small_to_large), count)
    freqs = np.array(freqs_small_to_large)[indices]
    assert tools.freq.longest(*freqs) == freqs_small_to_large[max(indices)]
    assert tools.freq.shortest(*freqs) == freqs_small_to_large[min(indices)]


@pytest.mark.parametrize("freq1", freqs_small_to_large)
@pytest.mark.parametrize("freq2", freqs_small_to_large)
def test_frequpordown(freq1, freq2):
    i1 = freqs_small_to_large.index(freq1)
    i2 = freqs_small_to_large.index(freq2)
    outcome = np.sign(i1 - i2)
    assert tools.freq.up_or_down(freq1, freq2) == outcome


@pytest.mark.parametrize("tz", [None, "Europe/Berlin", "Asia/Kolkata"])
@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ("2020", "2021", "AS"),
        ("2020", "2020-04", "QS"),
        ("2020", "2020-02", "MS"),
        ("2020", "2020-01-02", "D"),
        ("2020", "2020-01-01 01:00", "H"),
        ("2020", "2020-01-01 00:15", "15T"),
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
        ("2020-03-29 01:00", "2020-03-29 03:00", "H"),
        ("2020-10-25 01:00", "2020-10-25 02:00+0200", "H"),
        ("2020-10-25 01:00", "2020-10-25 02:00+0100", None),
        ("2020-10-25 02:00+0200", "2020-10-25 02:00+0100", "H"),
        ("2020-10-25 02:00+0200", "2020-10-25 03:00", None),
        ("2020-10-25 02:00+0100", "2020-10-25 03:00", "H"),
        # Quarterhourly.
        ("2020-03-29 01:45", "2020-03-29 03:00", "15T"),
        ("2020-10-25 01:45", "2020-10-25 02:00+0200", "15T"),
        ("2020-10-25 01:45", "2020-10-25 02:00+0100", None),
        ("2020-10-25 02:45+0200", "2020-10-25 02:00+0100", "15T"),
        ("2020-10-25 02:45+0200", "2020-10-25 03:00", None),
        ("2020-10-25 02:45+0100", "2020-10-25 03:00", "15T"),
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


@pytest.mark.parametrize("indexorframe", ["index", "series"])
@pytest.mark.parametrize(
    ("freq", "num", "wanted", "strict", "expected"),
    [
        # D
        # . enough
        ("D", 10, None, False, "D"),
        ("D", 10, None, True, "D"),
        ("D", 10, "MS", False, ValueError),
        ("D", 10, "MS", True, ValueError),
        ("D", 10, "D", False, "D"),
        ("D", 10, "D", True, "D"),
        # . too few
        ("D", 2, None, False, None),
        ("D", 2, None, True, ValueError),
        ("D", 2, "MS", False, ValueError),
        ("D", 2, "MS", True, ValueError),
        ("D", 2, "D", False, "D"),
        ("D", 2, "D", True, "D"),
        # 15T, too few
        ("15T", 2, None, False, None),
        ("15T", 2, None, True, ValueError),
        ("15T", 2, "MS", False, ValueError),
        ("15T", 2, "MS", True, ValueError),
        ("15T", 2, "15T", False, "15T"),
        ("15T", 2, "15T", True, "15T"),
        # invalid freq, not correctable
        # . enough
        ("2D", 10, None, False, "2D"),
        ("2D", 10, None, True, ValueError),
        ("2D", 10, "MS", False, ValueError),
        ("2D", 10, "MS", True, ValueError),
        ("2D", 10, "2D", False, "2D"),
        ("2D", 10, "2D", True, ValueError),
        # . too few
        ("2D", 2, None, False, None),
        ("2D", 2, None, True, ValueError),
        ("2D", 2, "MS", False, ValueError),
        ("2D", 2, "MS", True, ValueError),
        ("2D", 2, "2D", False, "2D"),
        ("2D", 2, "2D", True, ValueError),
        # invalid freq, correctable
        # . enough
        ("QS-APR", 10, None, False, "QS"),
        ("QS-APR", 10, None, True, "QS"),
        ("QS-APR", 10, "MS", False, ValueError),
        ("QS-APR", 10, "MS", True, ValueError),
        ("QS-APR", 10, "QS", False, "QS"),
        ("QS-APR", 10, "QS", True, "QS"),
        # . too few
        ("QS-APR", 2, None, False, None),
        ("QS-APR", 2, None, True, ValueError),
        ("QS-APR", 2, "MS", False, ValueError),
        ("QS-APR", 2, "MS", True, ValueError),
        ("QS-APR", 2, "QS", False, "QS"),
        ("QS-APR", 2, "QS", True, "QS"),
    ],
)
def test_setfreq(
    freq: str,
    num: int,
    wanted: str,
    strict: bool,
    expected: str | Exception,
    indexorframe: str,
):
    i = pd.date_range("2020", periods=num, freq=freq)
    i.freq = None

    if indexorframe == "index":
        inputvalue = i
        fn = tools.freq.set_to_index
    else:
        inputvalue = pd.Series(np.random.rand(num), i)
        fn = tools.freq.set_to_frame

    # Test.
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            _ = fn(inputvalue, wanted, strict)
        return
    result = fn(inputvalue, wanted, strict)
    if indexorframe == "index":
        outputfreq = result.freq
    else:
        outputfreq = result.index.freq
    assert outputfreq == expected


# Define your frequencies and their validity
freqs_with_validity = [
    ("15T", True),
    ("30T", True),
    ("D", True),
    ("H", True),
    ("MS", True),
    ("QS", True),
    ("AS", True),
    ("AS-APR", True),
    ("QS-FEB", True),
    ("T", False),
    ("5T", False),
    ("2H", False),
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
            tools.freq.assert_freq_valid(freq)


@pytest.mark.parametrize(
    ("freq1", "freq2", "strict", "isSupposedToFail"),
    [
        ("15T", "15T", False, False),
        ("15T", "15T", True, True),
        ("30T", "15T", True, False),
        ("30T", "30T", True, True),
        ("QS", "AS", True, True),
        ("QS", "QS-APR", False, False),
        ("AS", "QS", False, False),
    ],
)
def test_freq_sufficiently_long(
    freq1: str, freq2: str, strict: bool, isSupposedToFail: bool
):
    if isSupposedToFail:
        with pytest.raises(AssertionError):
            _ = tools.freq.assert_freq_sufficiently_long(freq1, freq2, strict)
    else:
        tools.freq.assert_freq_sufficiently_long(freq1, freq2, strict)
