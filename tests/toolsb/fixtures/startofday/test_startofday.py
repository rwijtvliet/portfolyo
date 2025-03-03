import pytest
from portfolyo import toolsb


def test_startofday_conversion_ok_str(sod_asstr, sod):
    assert toolsb.startofday.convert(sod_asstr) == sod


def test_startofday_conversion_ok_tdelta(sod_astdelta, sod):
    assert toolsb.startofday.convert(sod_astdelta) == sod


def test_startofday_conversion_nok_str(sod_nok_asstr, sod_nok):
    assert toolsb.startofday.convert(sod_nok_asstr) == sod_nok


def test_startofday_conversion_nok_tdelta(sod_nok_astdelta, sod_nok):
    assert toolsb.startofday.convert(sod_nok_astdelta) == sod_nok


def test_startofday_validation_ok(sod):
    toolsb.startofday.validate(sod)


def test_startofday_validation_nok(sod_nok):
    with pytest.raises(ValueError):
        toolsb.startofday.validate(sod_nok)


def test_startofday_tostring(sod, sod_asstr):
    assert toolsb.startofday.to_string(sod) == sod_asstr


def test_startofday_totdelta(sod, sod_astdelta):
    assert toolsb.startofday.to_tdelta(sod) == sod_astdelta
