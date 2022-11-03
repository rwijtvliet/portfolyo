import numpy as np
import pandas as pd
import pytest
from portfolyo import tools

freqs_small_to_large = ["T", "5T", "15T", "30T", "H", "2H", "D", "MS", "QS", "AS"]


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
