import pytest

from portfolyo import toolsb


class TestStartofday:
    def test_conversion_ok_str(self, sod_asstr, sod):
        assert toolsb.startofday.convert(sod_asstr) == sod

    def test_conversion_ok_tdelta(self, sod_astdelta, sod):
        assert toolsb.startofday.convert(sod_astdelta) == sod

    def test_conversion_nok_str(self, sod_nok_asstr, sod_nok):
        assert toolsb.startofday.convert(sod_nok_asstr) == sod_nok

    def test_conversion_nok_tdelta(self, sod_nok_astdelta, sod_nok):
        assert toolsb.startofday.convert(sod_nok_astdelta) == sod_nok

    def test_validation_ok(self, sod):
        toolsb.startofday.validate(sod)

    def test_validation_nok(self, sod_nok):
        with pytest.raises(ValueError):
            toolsb.startofday.validate(sod_nok)

    def test_tostring(self, sod, sod_asstr):
        assert toolsb.startofday.to_string(sod) == sod_asstr

    def test_totdelta(self, sod, sod_astdelta):
        assert toolsb.startofday.to_tdelta(sod) == sod_astdelta
