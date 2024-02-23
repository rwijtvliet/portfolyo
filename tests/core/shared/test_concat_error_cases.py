import pandas as pd
import pytest


from portfolyo import dev
from portfolyo.core.pfline.enums import Kind
from portfolyo.core.shared import concat


def test_general():
    """Test if concatenating PfLine with PfState raises error."""
    index = pd.date_range("2020", "2024", freq="QS", inclusive="left")
    index2 = pd.date_range("2024", "2025", freq="QS", inclusive="left")
    pfl = dev.get_flatpfline(index)
    pfs = dev.get_pfstate(index2)
    with pytest.raises(NotImplementedError):
        _ = concat.general(pfl, pfs)


def test_diff_freq():
    index = pd.date_range("2020", "2024", freq="QS", inclusive="left")
    index2 = pd.date_range("2024", "2025", freq="AS", inclusive="left")
    pfl = dev.get_flatpfline(index)
    pfl2 = dev.get_flatpfline(index2)
    with pytest.raises(TypeError):
        _ = concat.concat_pflines(pfl, pfl2)


def test_diff_sod():
    index = pd.date_range("2020-01-01 00:00", "2024", freq="QS", inclusive="left")
    index2 = pd.date_range("2024-01-01 06:00", "2025", freq="QS", inclusive="left")
    pfl = dev.get_flatpfline(index)
    pfl2 = dev.get_flatpfline(index2)
    with pytest.raises(TypeError):
        _ = concat.concat_pflines(pfl, pfl2)


def test_diff_tz():
    index = pd.date_range(
        "2020-01-01", "2024", freq="QS", tz="Europe/Berlin", inclusive="left"
    )
    index2 = pd.date_range("2024-01-01", "2025", freq="QS", tz=None, inclusive="left")
    pfl = dev.get_flatpfline(index)
    pfl2 = dev.get_flatpfline(index2)
    with pytest.raises(TypeError):
        _ = concat.concat_pflines(pfl, pfl2)


def test_diff_kind():
    index = pd.date_range("2020-01-01", "2024", freq="QS", inclusive="left")
    index2 = pd.date_range("2024-01-01", "2025", freq="QS", inclusive="left")
    pfl = dev.get_flatpfline(index, kind=Kind.COMPLETE)
    pfl2 = dev.get_flatpfline(index2, kind=Kind.VOLUME)
    with pytest.raises(TypeError):
        _ = concat.concat_pflines(pfl, pfl2)


def test_app_lenght():
    index = pd.date_range("2020-01-01", "2024", freq="QS", inclusive="left")
    pfl = dev.get_flatpfline(index)
    with pytest.raises(NotImplementedError):
        _ = concat.concat_pflines(pfl)
