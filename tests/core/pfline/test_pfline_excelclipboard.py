"""Test if portfolio line can be exported."""

import pytest

import portfolyo as pf


@pytest.mark.parametrize("levels", [1, 2, 3])
def test_pfline_toexcel(levels: int):
    """Test if data can be exported to excel."""
    pfl = pf.dev.get_randompfline(max_nlevels=levels)
    pfl.to_excel("test.xlsx")


@pytest.mark.parametrize("levels", [1, 2, 3])
def test_pfline_toclipboard(levels: int):
    """Test if data can be copied to clipboard."""
    pfl = pf.dev.get_randompfline(max_nlevels=levels)
    pfl.to_clipboard()