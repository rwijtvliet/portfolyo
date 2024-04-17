"""Test different error cases for concatenation of PfStates and PfLines."""

import pandas as pd
import pytest


from portfolyo import dev
from portfolyo.core.pfline.enums import Kind
from portfolyo.core.pfstate.pfstate import PfState
from portfolyo.utilities import concat


def test_general():
    """Test if concatenating PfLine with PfState raises error."""
    index = pd.date_range("2020", "2024", freq="QS", inclusive="left")
    index2 = pd.date_range("2024", "2025", freq="QS", inclusive="left")
    pfl = dev.get_flatpfline(index)
    pfs = dev.get_pfstate(index2)
    with pytest.raises(NotImplementedError):
        _ = concat.general([pfl, pfs])


def test_diff_freq():
    """Test if concatenating of two flat PfLines with different freq raises error."""
    index = pd.date_range("2020", "2024", freq="QS", inclusive="left")
    index2 = pd.date_range("2024", "2025", freq="AS", inclusive="left")
    pfl = dev.get_flatpfline(index)
    pfl2 = dev.get_flatpfline(index2)
    with pytest.raises(TypeError):
        _ = concat.concat_pflines([pfl, pfl2])


def test_diff_sod():
    """Test if concatenating of two flat PfLines with different sod raises error."""
    index = pd.date_range("2020-01-01 00:00", "2024", freq="QS", inclusive="left")
    index2 = pd.date_range("2024-01-01 06:00", "2025", freq="QS", inclusive="left")
    pfl = dev.get_flatpfline(index)
    pfl2 = dev.get_flatpfline(index2)
    with pytest.raises(TypeError):
        _ = concat.concat_pflines([pfl, pfl2])


def test_slice_not_sod():
    """Test if concatenating of two flat PfLines with different sod raises error."""
    index = pd.date_range("2020-01-01 00:00", "2020-03-01", freq="H", inclusive="left")
    index2 = pd.date_range(
        "2020-02-01 06:00", "2020-04-01 06:00", freq="H", inclusive="left"
    )
    pfl_a = dev.get_flatpfline(index)
    pfl_b = dev.get_flatpfline(index2)
    with pytest.raises(TypeError):
        _ = concat.concat_pflines([pfl_a, pfl_b])


def test_diff_tz():
    """Test if concatenating of two flat PfLines with different tz raises error."""
    index = pd.date_range(
        "2020-01-01", "2024", freq="QS", tz="Europe/Berlin", inclusive="left"
    )
    index2 = pd.date_range("2024-01-01", "2025", freq="QS", tz=None, inclusive="left")
    pfl = dev.get_flatpfline(index)
    pfl2 = dev.get_flatpfline(index2)
    with pytest.raises(TypeError):
        _ = concat.concat_pflines([pfl, pfl2])


def test_diff_kind():
    """Test if concatenating of two flat PfLines with different kind raises error."""
    index = pd.date_range("2020-01-01", "2024", freq="QS", inclusive="left")
    index2 = pd.date_range("2024-01-01", "2025", freq="QS", inclusive="left")
    pfl = dev.get_flatpfline(index, kind=Kind.COMPLETE)
    pfl2 = dev.get_flatpfline(index2, kind=Kind.VOLUME)
    with pytest.raises(TypeError):
        _ = concat.concat_pflines([pfl, pfl2])


def test_app_lenght():
    """Test if concatenatination raises error if we pass only one parameter."""
    index = pd.date_range("2020-01-01", "2024", freq="QS", inclusive="left")
    pfl = dev.get_flatpfline(index)
    with pytest.raises(NotImplementedError):
        _ = concat.concat_pflines([pfl])


def test_concat_with_overlap():
    """Test if concatenatination raises error if there is overlap in indices of PfLines."""
    index = pd.date_range("2020-01-01", "2024", freq="QS", inclusive="left")
    index2 = pd.date_range("2020-01-01", "2023", freq="QS", inclusive="left")
    pfl = dev.get_flatpfline(index)
    pfl2 = dev.get_flatpfline(index2)
    with pytest.raises(ValueError):
        _ = concat.concat_pflines([pfl, pfl2])


def test_concat_with_gaps():
    """Test if concatenatination raises error if there is a gap in indices of PfLines."""
    index = pd.date_range("2020-01-01", "2023", freq="QS", inclusive="left")
    index2 = pd.date_range("2024-01-01", "2025", freq="QS", inclusive="left")
    pfl = dev.get_flatpfline(index)
    pfl2 = dev.get_flatpfline(index2)
    with pytest.raises(ValueError):
        _ = concat.concat_pflines([pfl, pfl2])


def test_concat_children():
    """Test if concatenating of flat PfLine with nested PfLine raises error."""
    index = pd.date_range("2020-01-01", "2024", freq="QS", inclusive="left")
    index2 = pd.date_range("2024-01-01", "2025", freq="QS", inclusive="left")
    pfl = dev.get_flatpfline(index)
    pfl2 = dev.get_nestedpfline(index2)
    with pytest.raises(TypeError):
        _ = concat.concat_pflines([pfl, pfl2])


def test_concat_diff_children():
    """Test if concatenating of two nested PfLines with different children raises error."""
    index = pd.date_range("2020-01-01", "2024", freq="QS", inclusive="left")
    index2 = pd.date_range("2024-01-01", "2025", freq="QS", inclusive="left")
    pfl = dev.get_nestedpfline(index)
    pfl2 = dev.get_nestedpfline(index2).drop_child(name="a")
    with pytest.raises(TypeError):
        _ = concat.concat_pflines([pfl, pfl2])


def test_concat_pfss():
    """Test if concatenating of Pfstate with "nested" PfState
    (meaning that offtakevolume, sourced and unsourcedprice are nested Pflines) raises error.
    """
    index = pd.date_range("2020-01-01", "2024", freq="QS", inclusive="left")
    index2 = pd.date_range("2024-01-01", "2025", freq="QS", inclusive="left")
    pfs1 = dev.get_pfstate(index)
    offtakevolume = dev.get_nestedpfline(index2, kind=Kind.VOLUME)
    sourced = dev.get_nestedpfline(index2, kind=Kind.COMPLETE)
    unsourcedprice = dev.get_nestedpfline(index2, kind=Kind.PRICE)
    pfs2 = PfState(offtakevolume, unsourcedprice, sourced)
    with pytest.raises(TypeError):
        _ = concat.concat_pfstates([pfs1, pfs2])
