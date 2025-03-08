import pytest

from portfolyo import toolsb


class TestFreq:
    def test_conversion_ok(self, freq_asstr, freq):
        assert toolsb.freq.convert(freq_asstr) == freq

    def test_conversion_nok(self, freq_nok_asstr, freq_nok):
        assert toolsb.freq.convert(freq_nok_asstr) == freq_nok

    def test_validation_ok(self, freq):
        toolsb.freq.validate(freq)

    def test_validation_nok(self, freq_nok):
        with pytest.raises(ValueError):
            toolsb.freq.validate(freq_nok)

    def test_conversionvalidation_error(self, freq_nok_asstr):
        with pytest.raises(ValueError):
            freq_nok_asstr = toolsb.freq.convert(freq_nok_asstr)
            toolsb.freq.validate(freq_nok_asstr)

    def test_isshorterthandaily(self, freq_asstr, freq_is_shorterthandaily):
        assert toolsb.freq.is_shorter_than_daily(freq_asstr) == freq_is_shorterthandaily

    def test_upordown_distinct(self, freqs_shortlong_asstr):
        short, long = freqs_shortlong_asstr
        assert toolsb.freq.up_or_down(long, short) == 1
        assert toolsb.freq.up_or_down(short, long) == -1

    def test_upordown_equivalent(self, freqs_equivalent_asstr):
        freq1, freq2 = freqs_equivalent_asstr
        assert toolsb.freq.up_or_down(freq1, freq2) == 0
        assert toolsb.freq.up_or_down(freq2, freq1) == 0

    def test_upordown_equal(self, freq_asstr):
        assert toolsb.freq.up_or_down(freq_asstr, freq_asstr) == 0

    def test_upordown_incompatible(self, freqs_incompatible_asstr):
        freq1, freq2 = freqs_incompatible_asstr
        with pytest.raises(ValueError):
            _ = toolsb.freq.up_or_down(freq1, freq2)

    def test_sorted(self, freqs_unsorted_asstr, freqs_sorted_asstr):
        assert toolsb.freq.sorted(freqs_unsorted_asstr) == freqs_sorted_asstr
        assert toolsb.freq.shortest(freqs_unsorted_asstr) == freqs_sorted_asstr[0]
        assert toolsb.freq.longest(freqs_unsorted_asstr) == freqs_sorted_asstr[-1]

    def test_sorted_unsortable(self, freqs_unsortable_asstr):
        with pytest.raises(ValueError):
            _ = toolsb.freq.sorted(freqs_unsortable_asstr)
        with pytest.raises(ValueError):
            _ = toolsb.freq.shortest(freqs_unsortable_asstr)
        with pytest.raises(ValueError):
            _ = toolsb.freq.longest(freqs_unsortable_asstr)

    def test_sorted_inclcompatible(
        self, freqs_unsorted_inclcompatible_asstr, freqs_sorted_inclcompatible_asstr
    ):
        assert (
            toolsb.freq.sorted(freqs_unsorted_inclcompatible_asstr)
            in freqs_sorted_inclcompatible_asstr
        )
        assert toolsb.freq.shortest(freqs_unsorted_inclcompatible_asstr) in (
            freqs[0] for freqs in freqs_sorted_inclcompatible_asstr
        )
        assert toolsb.freq.longest(freqs_unsorted_inclcompatible_asstr) in (
            freqs[-1] for freqs in freqs_sorted_inclcompatible_asstr
        )
