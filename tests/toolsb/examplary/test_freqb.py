import numpy as np
import pytest

from portfolyo import toolsb


@pytest.mark.parametrize(
    "freq",
    ["15min", "h", "D", "MS", "QS", "QS-FEB", "QS-APR", "YS", "YS-FEB", "YS-APR"],
)
def test_freq_conversionvalidation(freq):
    freq = toolsb.freq.convert(freq)
    toolsb.freq.validate(freq)


@pytest.mark.parametrize("freq", ["3min", "2h", "7D", "ME", "2QS", "BS", "W-MON"])
def test_freq_conversionvalidation_error(freq):
    with pytest.raises(ValueError):
        freq = toolsb.freq.convert(freq)
        toolsb.freq.validate(freq)


@pytest.mark.parametrize("freq", ["min", "15min", "h"])
def test_freq_isshorterthandaily_True(freq):
    assert toolsb.freq.is_shorter_than_daily(freq)


@pytest.mark.parametrize("freq", ["D", "MS", "QS", "QS-FEB", "YS-APR"])
def test_freq_isshorterthandaily_False(freq):
    assert not toolsb.freq.is_shorter_than_daily(freq)


@pytest.mark.parametrize(
    "sourcefreq,targetfreq",
    [
        ("min", "5min"),
        ("min", "D"),
        ("min", "QS"),
        ("min", "QS-FEB"),
        ("D", "QS"),
        ("D", "QS-FEB"),
        ("D", "YS-FEB"),
    ],
)
def test_freq_upordown_1(sourcefreq, targetfreq):
    assert toolsb.freq.up_or_down(sourcefreq, targetfreq) == -1
    assert toolsb.freq.up_or_down(targetfreq, sourcefreq) == 1


@pytest.mark.parametrize(
    "sourcefreq,targetfreq",
    [
        ("min", "min"),
        ("D", "D"),
        ("QS", "QS"),
        ("QS", "QS-APR"),
        ("QS-FEB", "QS-MAY"),
        ("YS", "YS-JAN"),
    ],
)
def test_freq_upordown_0(sourcefreq, targetfreq):
    assert toolsb.freq.up_or_down(sourcefreq, targetfreq) == 0


@pytest.mark.parametrize(
    "sourcefreq,targetfreq",
    [
        ("QS", "QS-FEB"),
        ("QS", "QS-MAR"),
        ("YS", "YS-FEB"),
        ("YS", "YS-MAR"),
        ("YS", "YS-APR"),
    ],
)
def test_freq_upordown_error(sourcefreq, targetfreq):
    with pytest.raises(ValueError):
        _ = toolsb.freq.up_or_down(sourcefreq, targetfreq)


@pytest.mark.parametrize(
    "sorted",
    [
        ("min", "5min", "D", "MS", "QS", "YS"),
        ("15min", "h", "QS", "YS-APR"),
        ("h", "D", "QS-APR", "YS"),
        ("min", "min", "5min", "D", "D", "MS", "QS", "YS"),  # repeated entries
    ],
)
@pytest.mark.parametrize("seed", range(5))
def test_sortedshortedlongest_ok(seed, sorted):
    np.random.seed(seed)
    unsorted = np.random.default_rng().permutation(sorted)
    assert toolsb.freq.sorted(unsorted) == sorted
    assert toolsb.freq.shortest(unsorted) == sorted[0]
    assert toolsb.freq.longest(unsorted) == sorted[-1]


@pytest.mark.parametrize(
    "nestedsorted",
    [
        (["min"], ["5min"], ["D"], ["MS"], {"QS", "QS-APR"}),
        (["15min"], ["h"], {"QS", "QS-APR", "QS-JUL"}, ["YS-APR"]),
    ],
)
@pytest.mark.parametrize("seed", range(5))
def test_sortedshortedlongest_withequivalent(seed, nestedsorted):
    def assert_sorted(result):
        for subsorted in nestedsorted:
            for _ in range(len(subsorted)):
                r, *result = result
                assert r in subsorted

    np.random.seed(seed)
    unsorted = np.random.default_rng().permutation([f for ff in nestedsorted for f in ff])
    assert_sorted(toolsb.freq.sorted(unsorted))
    assert toolsb.freq.shortest(unsorted) in nestedsorted[0]
    assert toolsb.freq.longest(unsorted) in nestedsorted[-1]


@pytest.mark.parametrize(
    "unsortable",
    [
        ("min", "5min", "D", "MS", "QS", "QS-FEB"),
        ("min", "5min", "D", "MS", "QS", "YS-FEB"),
        ("min", "5min", "D", "MS", "YS", "YS-FEB"),
    ],
)
@pytest.mark.parametrize("seed", range(5))
def test_sortedshortestlongest_incompatible(seed, unsortable):
    np.random.seed(seed)
    unsorted = np.random.default_rng().permutation(unsortable)
    with pytest.raises(ValueError):
        _ = toolsb.freq.sorted(unsorted)
    with pytest.raises(ValueError):
        _ = toolsb.freq.shortest(unsortable)
    with pytest.raises(ValueError):
        _ = toolsb.freq.longest(unsortable)
