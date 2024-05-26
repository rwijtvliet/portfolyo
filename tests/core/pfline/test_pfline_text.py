"""Test if portfolio line can be printed."""

import pytest

import portfolyo as pf


@pytest.mark.parametrize("levels", [1, 2, 3])
@pytest.mark.parametrize("kind", pf.Kind)
@pytest.mark.parametrize("flatten", [True, False])
def test_pfline_print(levels: int, kind: pf.Kind, flatten: bool):
    """Test if portfolio line can be printed."""
    pfl = pf.dev.get_pfline(kind=kind, nlevels=levels)
    pfl.print(flatten)
